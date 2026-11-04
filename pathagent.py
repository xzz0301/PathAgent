import os
import gc
import sys
import json
import torch
import warnings
import argparse
import numpy as np
from copy import deepcopy

# Suppress warnings
warnings.filterwarnings("ignore")

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="PathAgent inference with navigation trajectory logging")
    parser.add_argument("--config", help="JSON defaults; CLI options override config values")
    parser.add_argument("--profile-name", dest="profile_name", default="custom")
    parser.add_argument('--method-profile', choices=['public_code', 'paper_aligned'], default='public_code')
    parser.add_argument("--model-dtype", dest="model_dtype", choices=["auto", "bfloat16", "float16", "float32"], default="auto")
    parser.add_argument("--quantization", choices=["none", "nf4"], default="none")
    parser.add_argument("--executor-device-map", default="auto")
    parser.add_argument("--perceptor-device-map", default="auto")
    parser.add_argument("--plip-device", default="auto")
    parser.add_argument("--executor-max-memory", help="Optional JSON mapping of device to memory budget")
    parser.add_argument("--perceptor-max-memory", help="Optional JSON mapping of device to memory budget")
    parser.add_argument("--allow-empty-descriptions", action=argparse.BooleanOptionalAction, default=False,
                        help="Explicitly allow incomplete generic descriptions for smoke tests")
    parser.add_argument("--check-inputs", action="store_true", help="Validate files without loading model weights")
    parser.add_argument("--capture-observations", action=argparse.BooleanOptionalAction, default=True,
                        help="Save exact Perceptor input images for offline evidence localization")
    parser.add_argument('--image-backend',choices=['patch','wsi'],default='patch')
    parser.add_argument('--wsi-manifest',help='Case-keyed WSI path, quickhash, dimensions and native nominal magnification')

    # --- Library Paths ---
    parser.add_argument("--plip_lib_path", type=str, required=False,
                        help="Path to the PLIP library directory")

    # --- Model Checkpoints ---
    parser.add_argument("--qwen_ckpt", type=str, required=False,
                        help="Path to Qwen checkpoint")
    parser.add_argument("--plip_ckpt", type=str, required=False,
                        help="Path to PLIP checkpoint")
    parser.add_argument("--patho_r1_ckpt", type=str, required=False,
                        help="Path to Patho-R1 checkpoint")

    # --- Data Files ---
    parser.add_argument("--descriptions_file", type=str, required=False,
                        help="Path to patch descriptions JSON file")
    parser.add_argument("--questions_file", type=str, required=False,
                        help="Path to questions/VQA dataset JSON file")
    parser.add_argument("--feature_dir", type=str, required=False,
                        help="Directory containing image features")
    parser.add_argument("--patch_root", type=str, required=False,
                        help="Root directory for image patches")
    parser.add_argument("--save_dir", type=str, required=False,
                        help="Directory to save results")

    # --- Settings ---
    parser.add_argument("--dataset_name", type=str, default="wsi_vqa", 
                        help="Name of the dataset (e.g., wsi_vqa, slidechat)")
    parser.add_argument("--geometry_file", type=str,
                        help="Case-keyed coordinate scale and native magnification JSON")
    parser.add_argument("--seed", type=int, default=128)
    parser.add_argument("--initial_sample_count", type=int, default=0,
                        help="Optional fixed initial observation count; 0 uses original 10%% rule")
    parser.add_argument("--max_attempts", type=int, default=5)
    parser.add_argument("--perceptor_max_new_tokens", type=int, default=1024)
    parser.add_argument("--min_observation_rounds", type=int, default=1,
                        help="Exploratory coverage guard; default 1 preserves original stopping behavior")

    preliminary, _ = parser.parse_known_args(argv)
    if preliminary.config:
        from pathlib import Path
        config = json.loads(Path(preliminary.config).read_text(encoding="utf-8"))
        known = {action.dest for action in parser._actions}
        unknown = set(config) - known
        if unknown:
            parser.error(f"Unknown config keys: {sorted(unknown)}")
        for action in parser._actions:
            if action.dest in config and action.choices and config[action.dest] not in action.choices:
                parser.error(f"Invalid config value for {action.dest}: {config[action.dest]}")
        path_keys = ["plip_lib_path", "qwen_ckpt", "plip_ckpt", "patho_r1_ckpt",
                     "descriptions_file", "questions_file", "feature_dir", "patch_root",
                     "save_dir", "geometry_file", "wsi_manifest"]
        for key in path_keys:
            if config.get(key) and not Path(config[key]).is_absolute():
                config[key] = str(Path(__file__).resolve().parent / config[key])
        parser.set_defaults(**config)
    args = parser.parse_args(argv)
    required = ["plip_lib_path", "qwen_ckpt", "plip_ckpt", "patho_r1_ckpt",
                "descriptions_file", "questions_file", "feature_dir", "patch_root", "save_dir"]
    missing = [key for key in required if not getattr(args, key)]
    if missing:
        parser.error(f"Missing required inputs: {', '.join(missing)}")
    if args.max_attempts < 1 or args.min_observation_rounds < 1 or args.perceptor_max_new_tokens < 1 or args.initial_sample_count < 0:
        parser.error("Attempt/token/round limits must be positive; initial_sample_count must be >= 0")
    if args.min_observation_rounds > args.max_attempts:
        parser.error("min_observation_rounds cannot exceed max_attempts")
    if args.image_backend=='wsi' and (not args.wsi_manifest or not args.geometry_file):
        parser.error('WSI backend requires wsi-manifest and geometry_file')
    return args


from data_processing.utils import *
from models.inference import evaluate_with_llm_chain, slide_llm_answer, patho_r1_describe, summarize_patches_in_chunks
from models.runtime import load_models, runtime_metadata
from models.input_validation import validate_inputs
from trajectory import TraceRecorder, export_trajectory_artifacts, split_zoom_patch
from wsi_regions import WSIReader


def main(args=None):
    args = args or parse_args()
    preflight = validate_inputs(args)
    if args.check_inputs:
        print(json.dumps(preflight, ensure_ascii=False, indent=2))
        return preflight

    # Ensure save directory exists
    os.makedirs(args.save_dir, exist_ok=True)

    print("="*40)
    print(f"PLIP Lib:   {args.plip_lib_path}")
    print(f"Model Qwen: {args.qwen_ckpt}")
    print(f"Model PLIP: {args.plip_ckpt}")
    print(f"Model PR1:  {args.patho_r1_ckpt}")
    print(f"Dataset:    {args.dataset_name}")
    print(f"Results to: {args.save_dir}")
    print("="*40)

    # Load VQA Pairs
    print(f"Loading VQA pairs from {args.questions_file}...")
    pairs = load_all_vqa_pairs(
        args.questions_file, 
        dataset_name=args.dataset_name, 
        image_dir=args.patch_root
    )
    print(f"Successfully loaded {len(pairs)} VQA pairs.")

    initial_sample_ratio = 0.10
    replenish_ratio = 0.05
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    geometries = {}
    if args.geometry_file:
        with open(args.geometry_file, encoding="utf-8") as handle:
            geometries = json.load(handle)
    wsi_manifests={}
    if args.image_backend=='wsi':
        with open(args.wsi_manifest,encoding='utf-8') as handle:wsi_manifests=json.load(handle)

    tokenizer, model, plip, patho_r1_model, patho_r1_processor = load_models(args)

    # --- Data Preparation ---
    finished_cases = set()
    if os.path.exists(args.save_dir):
        for f in os.listdir(args.save_dir):
            if f.endswith(".json"):
                finished_cases.add(f.replace(".json", ""))

    with open(args.descriptions_file, "r", encoding="utf-8") as f:
        all_descriptions = json.load(f)
    all_keys = list(all_descriptions.keys())
    print(f"Pre-loaded descriptions file, containing {len(all_keys)} keys.")

    all_case_results = []
    max_attempts = args.max_attempts

    for idx, pair in enumerate(pairs):
        TARGET_LONG_ID = pair["long_id"]
        
        matching_keys = ([TARGET_LONG_ID] if TARGET_LONG_ID in all_descriptions else
                         [k for k in all_keys if k.startswith(TARGET_LONG_ID)])
        if len(matching_keys) == 1:
            TARGET_LONG_ID = matching_keys[0]
        elif len(matching_keys) > 1:
            raise ValueError(f"Ambiguous description keys for {TARGET_LONG_ID}: {matching_keys}")
        else:
            print(f"Target ID '{TARGET_LONG_ID}' not found in description file, keeping as is.")

        question_text = pair["question"]
        question_choices = pair["choices"]
        correct_answer = pair["answer"]

        # Assuming make_unique_id is imported from data_processing.utils
        unique_id = make_unique_id(TARGET_LONG_ID, question_text)

        save_path = os.path.join(args.save_dir, f"{unique_id}.json")

        # Resume from breakpoint
        if unique_id in finished_cases:
            print(f"Skipping Case {idx+1}/{len(pairs)}: {unique_id} (Already completed)")
            continue

        print(f"\nStarting Case {idx+1}/{len(pairs)}: {unique_id}")
        geometry = geometries.get(TARGET_LONG_ID, {
            "coordinate_scale": 1,
            "base_magnification": 5,
            "coordinate_space": "patch_filename_unverified",
            "level0_verified": False,
        })
        if args.geometry_file and TARGET_LONG_ID not in geometries:
            raise ValueError(f"Missing geometry for case {TARGET_LONG_ID}")
        base_magnification = geometry["base_magnification"]
        coordinate_scale = geometry["coordinate_scale"]
        reader=None
        if args.image_backend=='wsi':
            if TARGET_LONG_ID not in wsi_manifests:raise ValueError('Missing WSI identity for '+TARGET_LONG_ID)
            reader=WSIReader(wsi_manifests[TARGET_LONG_ID])
            if list(reader.slide.dimensions)!=geometry['slide_dimensions_level0'] or not geometry.get('level0_verified'):
                reader.close();raise ValueError('Candidate geometry and WSI disagree')
            geometry={**geometry,'image_backend':'wsi','wsi_quickhash':reader.manifest['quickhash']}
        zoom_level_val = base_magnification
        recorder = TraceRecorder(args.save_dir, unique_id, capture_images=args.capture_observations)
        recorder.event("run_started", geometry=geometry, seed=args.seed)

        print("="*80)
        print(f"Question: {question_text}")
        print(f"Choices: {question_choices}")
        print(f"Ground Truth: {correct_answer}")

        # --- Step 1: Extract Data ---
        descriptions_dict = deepcopy(all_descriptions[TARGET_LONG_ID])
        print(f"Extracted {len(descriptions_dict)} patch descriptions.")

        # --- Step 2: Preload Features ---
        feature_cache = {}
        all_patch_names = list(descriptions_dict.keys())
        for patch_name in all_patch_names:
            # Assuming filename format aligns with your logic
            npy_path = os.path.join(args.feature_dir, TARGET_LONG_ID, f"{patch_name.split('.')[0]}.npy")
            if os.path.exists(npy_path):
                feature_cache[patch_name] = np.load(npy_path)
        print(f"Successfully cached {len(feature_cache)} / {len(all_patch_names)} patch features")

        # --- Step 3: Initialization ---
        q_emb = plip.encode_text([question_text], batch_size=1)
        q_emb = q_emb / np.linalg.norm(q_emb, axis=-1, keepdims=True)
        available_patches = [p for p in all_patch_names if p in feature_cache]

        if not available_patches:
            print("No available patches with features found. Skipping case.")
            continue

        feats = np.stack([feature_cache[p] for p in available_patches], axis=0)
        # feats = feats / np.linalg.norm(feats, axis=-1, keepdims=True)
        sims_all = np.atleast_1d((feats @ q_emb.T).squeeze())

        K = min(len(available_patches), args.initial_sample_count) if args.initial_sample_count > 0 else max(1, int(len(all_patch_names) * initial_sample_ratio))
        if args.method_profile == 'paper_aligned':
            if base_magnification != 5 or args.initial_sample_count or args.min_observation_rounds != 1:
                raise ValueError('paper_aligned requires real 5x geometry, proportional sampling, and natural stopping')
            if len(available_patches) != len(all_patch_names):
                raise ValueError('paper_aligned requires features for every candidate')
            K = max(1, __import__('math').ceil(len(all_patch_names) * initial_sample_ratio))
        topk_idx = np.argsort(sims_all)[-K:][::-1]
        initial_sample_names = [available_patches[i] for i in topk_idx]
        recorder.event("navigation_selected", round=1, action="initial_selection",
                       patch_names=initial_sample_names, scores=[float(sims_all[i]) for i in topk_idx])

        accumulated_patch_names = list(initial_sample_names)
        patches_to_evaluate_names = list(initial_sample_names)
        remaining_patch_names = [p for p in all_patch_names if p not in accumulated_patch_names]

        print(f"Initial sample size: {len(initial_sample_names)}; Remaining: {len(remaining_patch_names)}")

        # --- Step 4: Loop Evaluation ---
        attempt = 0
        final_answer = None
        all_results = []
        trajectory = recorder.points

        if args.method_profile == 'paper_aligned':
            from navigation import run_paper_case
            case_result = run_paper_case(
                args=args, case_id=TARGET_LONG_ID, result_id=unique_id, question=question_text,
                choices=question_choices, descriptions=descriptions_dict, features=feature_cache,
                initial_names=initial_sample_names, geometry=geometry, reader=reader, recorder=recorder,
                executor=model, tokenizer=tokenizer, navigator=plip,
                perceptor=patho_r1_model, processor=patho_r1_processor)
            with open(save_path, 'w', encoding='utf-8') as handle:
                json.dump(case_result, handle, ensure_ascii=False, indent=2)
            all_case_results.append(case_result)
            if reader:
                reader.close()
            gc.collect()
            torch.cuda.empty_cache()
            continue

        while attempt < max_attempts:
            attempt += 1
            print(f"\nLoop attempt {attempt}... Evaluating {len(patches_to_evaluate_names)} patches this round.")
            if not patches_to_evaluate_names:
                print("No new patches available for evaluation this round, stopping.")
                break

            # ------------- Step A: Generate question-specific descriptions for top N patches -------------
            num_top_patches_to_describe = 5
            num_top_patches_to_describe = min(num_top_patches_to_describe, len(patches_to_evaluate_names))
            for i in range(num_top_patches_to_describe):
                patch_name = patches_to_evaluate_names[i]
                patch_path = get_patch_fullpath(args.patch_root, TARGET_LONG_ID, patch_name)
                patch_img = load_image(patch_path)

                x, y = extract_coords_from_name(patch_name)
                roi=dict(x0=x,y0=y,width0=patch_img.width*coordinate_scale,height0=patch_img.height*coordinate_scale)
                acquisition={'image_source':'saved_patch','roi':roi}
                if reader:patch_img,acquisition=reader.read(roi,base_magnification)
                input_audit={}
                started = recorder.start_observation(
                    image=patch_img,
                    round=attempt,
                    action="initial_observation" if attempt == 1 else "find_missing_observation",
                    patch_name=patch_name, x=x, y=y, width=patch_img.width, height=patch_img.height,
                    magnification=base_magnification,
                    roi=acquisition['roi'],acquisition=acquisition,
                )
                question_specific_desc = patho_r1_describe(
                    patch_img,
                    question=question_text,
                    patho_r1_processor=patho_r1_processor,
                    patho_r1_model=patho_r1_model,
                    coords=(x, y),
                    magnification=base_magnification,
                    choices=question_choices,
                    max_new_tokens=args.perceptor_max_new_tokens,input_audit=input_audit
                )
                started[0]['model_input']=input_audit
                recorder.complete_observation(started, question_specific_desc)
                
                orig = descriptions_dict.get(patch_name, "")
                descriptions_dict[patch_name] = (orig + " " + question_specific_desc).strip()
                print(f"Appended question-specific description for {patch_name} (Total length {len(descriptions_dict[patch_name])})")


            # ------------- Step B: Construct meta-descriptions and submit to patch-evaluator -------------
            current_items_to_evaluate = [(name, descriptions_dict[name]) for name in patches_to_evaluate_names]
            desc_for_evaluation = build_descriptions_with_meta(current_items_to_evaluate, mag_level=zoom_level_val, include_header=True, include_coords=True)

            evaluation_result = evaluate_with_llm_chain(model, tokenizer, desc_for_evaluation, question_text, question_choices)
            recorder.event("executor_evaluation", round=attempt, result=evaluation_result)
            can_now = str(evaluation_result.get("sufficient", "")).strip().lower() == "yes"
            can_zoom = str(evaluation_result.get("zoom_recommendation", "")).strip().lower() == "yes"
            if can_now and attempt < args.min_observation_rounds and remaining_patch_names:
                # A configured coverage experiment, independent of ground truth.
                # Preserve the original executor judgment and log the intervention.
                recorder.event("minimum_rounds_guard", round=attempt,
                               original_sufficient=evaluation_result.get("sufficient"),
                               minimum_rounds=args.min_observation_rounds,
                               reason="configured spatial coverage experiment")
                can_now = False
                can_zoom = False
                evaluation_result = dict(evaluation_result)
                evaluation_result["coverage_guard_applied"] = True
                evaluation_result["missing_info"] = question_text

            # ------------- Handle Branches -------------
            if can_now:
                print("\npatch-llm judgment: Can answer now.")
                final_desc_text = summarize_patches_in_chunks(
                    model, tokenizer,
                    descriptions_dict, accumulated_patch_names,
                    question_text=question_text,
                    chunk_size=10, threshold=50,
                    magnification=zoom_level_val
                )
                final_answer = slide_llm_answer(
                    model, tokenizer, 
                    final_desc_text, 
                    question_text, 
                    question_choices, 
                    magnification=zoom_level_val, 
                    case_name=unique_id, 
                ) 

                all_results.append({
                    "attempt": attempt,
                    "mode": "can_answer_now",
                    "evaluated_patches_this_round": patches_to_evaluate_names,
                    "total_accumulated_patches": len(accumulated_patch_names),
                    "evaluation_result": evaluation_result,
                    "answer": final_answer["answer"],
                    "explanation": final_answer["explanation"]
                })
                break

            elif can_zoom:

                final_desc_text = summarize_patches_in_chunks(
                    model, tokenizer,
                    descriptions_dict, accumulated_patch_names,
                    question_text=question_text,
                    chunk_size=10, threshold=50,
                    magnification=zoom_level_val
                )
                zoom_reason = evaluation_result.get("zoom_reason", "").strip()
                print(f"\npatch-evaluator judgment: Need zoom. Reason for zoom is {zoom_reason}")
                # parse zoom level from evaluator
                zoom_level_raw = evaluation_result.get("zoom_level", zoom_level_val)
                try:
                    zoom_level_val = int(zoom_level_raw)
                except (ValueError, TypeError):
                    zoom_level_val = zoom_level_val  # keep current

                print(f"  -> Zoom level: {zoom_level_val}x")

                patch_feats = []
                patch_names_valid = []
                for p in patches_to_evaluate_names:
                    if p in feature_cache:
                        patch_feats.append(feature_cache[p])
                        patch_names_valid.append(p)
                if not patch_feats:
                    print("No valid patch features, skipping zoom process.")
                else:
                    patch_feats = np.stack(patch_feats, axis=0)
                    # patch_feats = patch_feats / np.linalg.norm(patch_feats, axis=-1, keepdims=True)
                    text_emb = plip.encode_text([question_text], batch_size=1)
                    text_emb = text_emb / np.linalg.norm(text_emb, axis=-1, keepdims=True)
                    sims = np.atleast_1d(np.dot(patch_feats, text_emb.T).squeeze())

                    top2_idx = np.argsort(sims)[-2:][::-1]
                    top2_names = [patch_names_valid[i] for i in top2_idx]
                    print(f"  -> Selected Top2 patches: {top2_names}")

                    # split top2 patches into sub-patches at zoom_level_val
                    candidate_images = []
                    candidate_names = []
                    candidate_acquisitions = []
                    for patch_name in top2_names:
                        patch_path = get_patch_fullpath(args.patch_root, TARGET_LONG_ID, patch_name)
                        sub_patches = split_zoom_patch(patch_path, zoom_level_val, base_magnification, coordinate_scale)
                        for sub_img, (x, y) in sub_patches:
                            roi=dict(x0=x,y0=y,width0=sub_img.width*coordinate_scale,height0=sub_img.height*coordinate_scale)
                            acquisition={'image_source':'crop_existing_patch','roi':roi}
                            if reader:sub_img,acquisition=reader.read(roi,zoom_level_val)
                            candidate_images.append(sub_img)
                            candidate_names.append(f"{x:g}_{y:g}")
                            candidate_acquisitions.append(acquisition)

                    if not candidate_images:
                        print("No sub-patches generated, skipping zoom process.")
                    else:
                        image_embs = plip.encode_images(candidate_images, batch_size=4)
                        image_embs = image_embs / np.linalg.norm(image_embs, axis=-1, keepdims=True)
                        sims_sub = np.atleast_1d(np.dot(image_embs, text_emb.T).squeeze())
                        best_idx = int(np.argmax(sims_sub))
                        best_img = candidate_images[best_idx]
                        best_name = candidate_names[best_idx]
                        acquisition=candidate_acquisitions[best_idx]
                        print(f"Selected zoomed sub-patch: {best_name}")
                        recorder.event("navigation_selected", round=attempt, action="zoom_selection",
                                       patch_names=[best_name], candidate_count=len(candidate_names),
                                       scores=[float(sims_sub[best_idx])])

                        x, y = best_name.split("_")
                        started = recorder.start_observation(
                            image=best_img,
                            round=attempt, action="zoom_observation", patch_name=best_name,
                            x=float(x), y=float(y), width=best_img.width, height=best_img.height,
                            magnification=zoom_level_val,
                            roi=acquisition['roi'],acquisition=acquisition,
                        )
                        input_audit={}
                        zoomed_patch_desc = patho_r1_describe(
                            best_img,
                            question=question_text,
                            patho_r1_processor=patho_r1_processor,
                            patho_r1_model=patho_r1_model,
                            coords=(x, y),
                            magnification=zoom_level_val,
                            choices=question_choices,
                            max_new_tokens=args.perceptor_max_new_tokens,input_audit=input_audit
                        )
                        started[0]['model_input']=input_audit
                        recorder.complete_observation(started, zoomed_patch_desc)

                        extra_note = (
                            f"\n\n[NOTE] A key sub-patch was identified after zooming.\n"
                            f"Patch=({x},{y}), Magnification={zoom_level_val}x\n"
                            f"Detail: {zoomed_patch_desc}"
                        )
                        final_desc_text = final_desc_text + extra_note

                        final_answer = slide_llm_answer(
                            model, tokenizer,
                            final_desc_text,
                            question_text, question_choices,
                            magnification=zoom_level_val,
                            case_name=unique_id,
                        )

                        all_results.append({
                            "attempt": attempt,
                            "mode": "zoom_then_select",
                            "evaluated_patches_this_round": patches_to_evaluate_names,
                            "total_accumulated_patches": len(accumulated_patch_names),
                            "evaluation_result": evaluation_result,
                            "selected_zoom_patch": best_name,
                            "zoom_patch_desc": zoomed_patch_desc,
                            "answer": final_answer["answer"],
                            "explanation": final_answer["explanation"]
                        })
                        break

            else:
                # cannot answer and cannot zoom -> find additional patches relevant to missing_info
                missing_info = evaluation_result.get("missing_info", "pathology details").strip()
                all_results.append({
                    "attempt": attempt, "mode": "find_missing",
                    "evaluated_patches_this_round": list(patches_to_evaluate_names),
                    "evaluation_result": evaluation_result,
                })
                print(f"patch-evaluator identifies missing info: '{missing_info}' -> Searching for new patches.")

                if not remaining_patch_names:
                    print("No more remaining patches available, loop terminated.")
                    break

                valid_remaining_names = [name for name in remaining_patch_names if name in feature_cache]
                if not valid_remaining_names:
                    print("No valid feature cache for remaining patches, loop terminated.")
                    break

                remaining_embs = np.stack([feature_cache[name] for name in valid_remaining_names], axis=0)
                text_emb = plip.encode_text([missing_info], batch_size=1)

                # remaining_embs = remaining_embs / np.linalg.norm(remaining_embs, axis=-1, keepdims=True)
                text_emb = text_emb / np.linalg.norm(text_emb, axis=-1, keepdims=True)
                sims = np.atleast_1d(np.dot(remaining_embs, text_emb.T).squeeze())

                num_to_add = max(1, int(len(all_patch_names) * replenish_ratio))
                best_indices = np.argsort(sims)[::-1][:num_to_add]

                newly_selected_patches = [valid_remaining_names[i] for i in best_indices]
                recorder.event("navigation_selected", round=attempt + 1, action="find_missing_selection",
                               patch_names=newly_selected_patches, scores=[float(sims[i]) for i in best_indices],
                               query=missing_info)
                print(f"Selected {len(newly_selected_patches)} new relevant patches.")

                patches_to_evaluate_names = newly_selected_patches
                accumulated_patch_names.extend(newly_selected_patches)
                remaining_patch_names = [p for p in remaining_patch_names if p not in newly_selected_patches]

                print(f"   Total accumulated patches: {len(accumulated_patch_names)}, Remaining available: {len(remaining_patch_names)}")

        if final_answer is None:
            print("Answer not found, entering fallback...")
            final_desc_text = summarize_patches_in_chunks(
                model, tokenizer,
                descriptions_dict, accumulated_patch_names,
                question_text=question_text,
                chunk_size=10, threshold=50,
                magnification=zoom_level_val
            )
            final_answer = slide_llm_answer(
                model, tokenizer, 
                final_desc_text, 
                question_text, 
                question_choices, 
                magnification=zoom_level_val, 
                case_name=unique_id, 
            )

            all_results.append({
                "attempt": "final_fallback",
                "mode": "fallback_final_attempt",
                "total_accumulated_patches": len(accumulated_patch_names),
                "answer": final_answer["answer"],
                "explanation": final_answer["explanation"]
            })
        print('single_case_result:', final_answer["answer"])
        print('ground_truth:', correct_answer)
        
        # --- Save Single Case Result ---
        case_result = {
            "long_id": TARGET_LONG_ID,
            "question": question_text,
            "choices": question_choices,
            "ground_truth": correct_answer,
            "pred_answer": final_answer["answer"],
            "explanation": final_answer["explanation"],
            "process": all_results,
            "trajectory": trajectory,
            "geometry": geometry,
            "runtime": runtime_metadata(args, model, patho_r1_model, plip),
        }

        trace_paths = export_trajectory_artifacts(
            trajectory=trajectory,
            patch_dir=os.path.join(args.patch_root, TARGET_LONG_ID),
            output_dir=args.save_dir,
            case_id=TARGET_LONG_ID,
            result_id=unique_id,
            geometry=geometry,
        )
        trace_paths["events"] = str(recorder.path)
        case_result["trajectory_files"] = trace_paths
        recorder.event("run_completed", answer=final_answer["answer"], observations=len(trajectory))

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(case_result, f, indent=2, ensure_ascii=False)

        all_case_results.append(case_result)
        if reader:reader.close()

        gc.collect()
        torch.cuda.empty_cache()

    # --- Final Summary Output ---
    print("\n\n========== All Results ==========")
    print(json.dumps(all_case_results, indent=2, ensure_ascii=False))
    return all_case_results
    
if __name__ == "__main__":
    main()

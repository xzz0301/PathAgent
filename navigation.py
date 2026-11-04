"""Paper-aligned navigation selected explicitly through the single Agent entrypoint.

Uses Fig. 3's regional plans and high-magnification re-evaluation. These choices
resolve documented ambiguities in Algorithm 1; they are recorded in every run.
"""
import math
from pathlib import Path

import numpy as np

from data_processing.utils import get_patch_fullpath, load_image, extract_coords_from_name
from models.inference import evaluate_with_llm_chain, patho_r1_describe, slide_llm_answer, summarize_patches_in_chunks
from models.runtime import runtime_metadata


def sample_count(n, ratio):
    if n < 1:
        return 0
    return min(n, max(1, math.ceil(n * ratio)))


def valid_region_plans(decision, regions, native):
    """Accept only actual, current ROI IDs and calibrated higher magnifications."""
    valid, rejected, used = [], [], set()
    plans = decision.get('region_plans', [])
    if not isinstance(plans, list):
        return [], [{'reason': 'region_plans_not_list'}]
    for plan in plans:
        try:
            if not isinstance(plan, dict):
                raise ValueError('not_object')
            name = plan['patch_name']
            if name not in regions or name in used:
                raise ValueError('unknown_or_duplicate_roi')
            if str(plan.get('zoom_recommendation', '')).strip().lower() != 'yes':
                continue
            target = float(plan['recommended_zoom_level'])
            current = float(regions[name]['magnification'])
            factor = target / current
            if target not in [10, 20, 40] or target > native or factor <= 1 or not factor.is_integer():
                raise ValueError('invalid_or_nonincreasing_magnification')
            query = str(plan.get('missing_info') or decision.get('missing_info') or '').strip()
            if not query or query.lower() in ['uncertain', 'none']:
                raise ValueError('missing_retrieval_query')
            valid.append({**plan, 'patch_name': name, 'target': target, 'query': query})
            used.add(name)
        except (KeyError, TypeError, ValueError) as exc:
            rejected.append({'plan': plan, 'reason': str(exc)})
    return valid, rejected


def rank_names(names, features, query, navigator, count):
    emb = navigator.encode_text([query], batch_size=1)
    emb = emb / np.linalg.norm(emb, axis=-1, keepdims=True)
    scores = np.atleast_1d((np.stack([features[n] for n in names]) @ emb.T).squeeze())
    indices = np.argsort(scores, kind='stable')[::-1][:count]
    return [names[i] for i in indices], [float(scores[i]) for i in indices]


def run_paper_case(*, args, case_id, result_id, question, choices, descriptions, features,
                   initial_names, geometry, reader, recorder, executor, tokenizer, navigator, perceptor, processor):
    if reader is None:
        raise ValueError('paper_aligned requires original WSI reads and verified geometry')
    regions, accumulated, process = {}, [], []
    visited_base = set()
    current_names, pending_base = list(initial_names), list(initial_names)
    native = reader.native
    stop_reason = None

    def observe(name, image, roi, mag, round_id, action, acquisition, query=None):
        start = recorder.start_observation(image=image, round=round_id, action=action, patch_name=name,
            x=roi['x0'], y=roi['y0'], width=image.width, height=image.height,
            magnification=mag, roi=roi, acquisition=acquisition)
        audit = {}
        text = patho_r1_describe(image, question=question, choices=None,
            patho_r1_processor=processor, patho_r1_model=perceptor,
            coords=(roi['x0'], roi['y0']), magnification=mag,
            max_new_tokens=args.perceptor_max_new_tokens, input_audit=audit,
            prompt_mode='paper_morphology')
        start[0]['model_input'] = audit
        recorder.complete_observation(start, text)
        return text

    def state_descriptions(names):
        return {name: (f"[Magnification={regions[name]['magnification']:g}x; "
                       f"ROI_level0={regions[name]['roi']}]\n" + regions[name]['description']) for name in names}

    for round_id in range(1, args.max_attempts + 1):
        # Every selected patch contributes its generic caption; only the Top-5
        # get online question-related visual descriptions (§3.1).
        for index, name in enumerate(pending_base):
            path = get_patch_fullpath(args.patch_root, case_id, name)
            saved = load_image(path)
            x, y = extract_coords_from_name(name)
            roi = dict(x0=x, y0=y, width0=saved.width * geometry['coordinate_scale'],
                       height0=saved.height * geometry['coordinate_scale'])
            text = descriptions[name]
            if index < 5:
                image, acquisition = reader.read(roi, 5)
                extra = observe(name, image, acquisition['roi'], 5, round_id,
                                'initial_observation' if round_id == 1 else 'find_missing_observation', acquisition)
                text = text + '\n[Question-related Patho-R1 morphology]\n' + extra
            regions[name] = {'roi': roi, 'magnification': 5., 'description': text,
                             'source': 'Quilt generic' + (' + Patho-R1' if index < 5 else '')}
            visited_base.add(name)
            if name not in accumulated:
                accumulated.append(name)
        pending_base = []
        state = state_descriptions(current_names)
        inputs = '\n\n'.join(f'[ROI_ID={name}]\n{state[name]}' for name in current_names)
        options = [{'patch_name': name, 'roi_level0': regions[name]['roi'],
                    'current_magnification': regions[name]['magnification'],
                    'allowed_magnifications': [m for m in [10, 20, 40]
                                              if regions[name]['magnification'] < m <= native]}
                   for name in current_names]
        decision = evaluate_with_llm_chain(executor, tokenizer, inputs, question, choices, region_options=options)
        record = {'attempt': round_id, 'state': {name: regions[name] for name in current_names},
                  'evaluation_input': inputs, 'evaluation_result': decision}
        process.append(record)
        recorder.event('executor_evaluation', round=round_id, result=decision, state_ids=current_names)
        if str(decision.get('sufficient', '')).strip().lower() == 'yes':
            record['mode'] = 'can_answer_now'
            stop_reason = 'evidence_sufficient'
            break
        if round_id == args.max_attempts:
            record['mode'] = 'budget_exhausted'
            stop_reason = 'max_evaluation_rounds'
            break
        plans, rejected = valid_region_plans(decision, {n: regions[n] for n in current_names}, native)
        record['validated_region_plans'], record['rejected_region_plans'] = plans, rejected
        if plans:
            new_names = []
            record['mode'] = 'regional_zoom_then_reassess'
            for plan_index, plan in enumerate(plans):
                parent = regions[plan['patch_name']]
                roi = parent['roi']
                factor = int(plan['target'] / parent['magnification'])
                candidates, acquisitions, candidate_rois = [], [], []
                # Partition actual level-0 bounds, preserving remainder pixels.
                for iy in range(factor):
                    for ix in range(factor):
                        left = roi['x0'] + math.floor(ix * roi['width0'] / factor)
                        top = roi['y0'] + math.floor(iy * roi['height0'] / factor)
                        right = roi['x0'] + math.floor((ix + 1) * roi['width0'] / factor)
                        bottom = roi['y0'] + math.floor((iy + 1) * roi['height0'] / factor)
                        sub = dict(x0=left, y0=top, width0=right-left, height0=bottom-top)
                        image, acquisition = reader.read(sub, plan['target'])
                        candidates.append(image); acquisitions.append(acquisition); candidate_rois.append(acquisition['roi'])
                text_emb = navigator.encode_text([plan['query']], batch_size=1)
                text_emb /= np.linalg.norm(text_emb, axis=-1, keepdims=True)
                image_emb = navigator.encode_images(candidates, batch_size=4)
                image_emb /= np.linalg.norm(image_emb, axis=-1, keepdims=True)
                scores = np.atleast_1d((image_emb @ text_emb.T).squeeze())
                best = int(np.argmax(scores))
                selected = candidate_rois[best]
                name = f"{selected['x0']}_{selected['y0']}@{plan['target']:g}x_r{round_id}_{plan_index}"
                text = observe(name, candidates[best], selected, plan['target'], round_id + 1,
                               'zoom_observation', acquisitions[best], plan['query'])
                regions[name] = {'roi': selected, 'magnification': plan['target'], 'description': text,
                                 'source': 'Patho-R1', 'parent_id': plan['patch_name'], 'retrieval_query': plan['query']}
                accumulated.append(name); new_names.append(name)
                recorder.event('navigation_selected', round=round_id+1, action='regional_zoom_selection',
                               patch_names=[name], parent_id=plan['patch_name'], query=plan['query'],
                               target_magnification=plan['target'], scores=[float(scores[best])])
            # Re-evaluate accumulated evidence including actual new high-power inputs.
            current_names = list(accumulated)
            continue
        remaining = [n for n in descriptions if n not in visited_base]
        if not remaining:
            record['mode'] = 'no_remaining_candidates'
            stop_reason = 'no_remaining_candidates'
            break
        query = str(decision.get('missing_info') or '').strip()
        if not query or query.lower() in ['none', 'uncertain']:
            record['mode'] = 'invalid_missing_information'
            stop_reason = 'invalid_missing_information'
            break
        current_names, scores = rank_names(remaining, features, query, navigator,
                                           min(len(remaining), sample_count(len(descriptions), .05)))
        pending_base = list(current_names)
        record['mode'] = 'find_missing'
        recorder.event('navigation_selected', round=round_id+1, action='find_missing_selection',
                       query=query, patch_names=current_names, scores=scores)

    final_state = state_descriptions(accumulated)
    final_text = summarize_patches_in_chunks(executor, tokenizer, final_state, accumulated,
                                            question_text=question, chunk_size=10, threshold=50, magnification=None)
    final = slide_llm_answer(executor, tokenizer, final_text, question, choices, case_name=result_id)
    recorder.event('run_completed', answer=final['answer'], termination_reason=stop_reason,
                   observations=len(recorder.points), evidence_ids=accumulated)
    return {'long_id': case_id, 'question': question, 'choices': choices, 'ground_truth': 'WITHHELD',
            'pred_answer': final['answer'], 'explanation': final['explanation'], 'process': process,
            'trajectory': recorder.points, 'evidence_regions': regions, 'final_evidence_ids': accumulated,
            'termination_reason': stop_reason, 'geometry': geometry,
            'runtime': {**runtime_metadata(args, executor, perceptor, navigator),
                        'method_profile': 'paper_aligned', 'sampling': 'ceil_10pct_5pct',
                        'prompt_mode': 'paper_morphology', 'zoom_query': 'missing_info',
                        'regional_plans': True, 'post_zoom_reassessment': True},
            'trajectory_files': {'events': str(recorder.path)}}

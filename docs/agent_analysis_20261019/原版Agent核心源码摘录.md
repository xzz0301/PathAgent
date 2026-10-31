# 原版 Agent 核心源码摘录

原版提交：`8c2abee2cd18ca2e2c839742c9296bd8a41738c5`。以下仅摘录原版文件，保留源码原文及源文件行号范围；代码块内未加入行号。各段控制流讲解见[原版分析](D:/PathCode/docs/agent_analysis_20261019/01_原版PathAgent代码与论文对应分析.md)。

## 1. 初选、候选集合、逐块观察与状态构造

来源：[pathagent.py:171](D:/PathCode/pathagent.py:171)，原文件第171—236行。

```python
        # --- Step 3: Initialization ---
        q_emb = plip.encode_text([question_text], batch_size=1)
        q_emb = q_emb / np.linalg.norm(q_emb, axis=-1, keepdims=True)
        available_patches = [p for p in all_patch_names if p in feature_cache]

        if not available_patches:
            print("No available patches with features found. Skipping case.")
            continue

        feats = np.stack([feature_cache[p] for p in available_patches], axis=0)
        # feats = feats / np.linalg.norm(feats, axis=-1, keepdims=True)
        sims_all = (feats @ q_emb.T).squeeze()

        K = max(1, int(len(all_patch_names) * initial_sample_ratio))
        topk_idx = np.argsort(sims_all)[-K:][::-1]
        initial_sample_names = [available_patches[i] for i in topk_idx]

        accumulated_patch_names = list(initial_sample_names)
        patches_to_evaluate_names = list(initial_sample_names)
        remaining_patch_names = [p for p in all_patch_names if p not in accumulated_patch_names]

        print(f"Initial sample size: {len(initial_sample_names)}; Remaining: {len(remaining_patch_names)}")

        # --- Step 4: Loop Evaluation ---
        attempt = 0
        final_answer = None
        all_results = []

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
                question_specific_desc = patho_r1_describe(
                    patch_img,
                    question=question_text,
                    patho_r1_processor=patho_r1_processor,
                    patho_r1_model=patho_r1_model,
                    coords=(x, y),
                    magnification=zoom_level_val,
                    choices=question_choices
                )

                orig = descriptions_dict.get(patch_name, "")
                descriptions_dict[patch_name] = (orig + " " + question_specific_desc).strip()
                print(f"Appended question-specific description for {patch_name} (Total length {len(descriptions_dict[patch_name])})")


            # ------------- Step B: Construct meta-descriptions and submit to patch-evaluator -------------
            current_items_to_evaluate = [(name, descriptions_dict[name]) for name in patches_to_evaluate_names]
            desc_for_evaluation = build_descriptions_with_meta(current_items_to_evaluate, mag_level=zoom_level_val, include_header=True, include_coords=True)

            evaluation_result = evaluate_with_llm_chain(model, tokenizer, desc_for_evaluation, question_text, question_choices)
            can_now = str(evaluation_result.get("sufficient", "")).strip().lower() == "yes"
            can_zoom = str(evaluation_result.get("zoom_recommendation", "")).strip().lower() == "yes"
```

## 2. 充分分支：最终整合并退出

来源：[pathagent.py:239](D:/PathCode/pathagent.py:239)，原文件第239—266行。

```python
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

                all_results = [{
                    "attempt": attempt,
                    "mode": "can_answer_now",
                    "evaluated_patches_this_round": patches_to_evaluate_names,
                    "total_accumulated_patches": len(accumulated_patch_names),
                    "evaluation_result": evaluation_result,
                    "answer": final_answer["answer"],
                    "explanation": final_answer["explanation"]
                }]
                break
```

## 3. zoom分支：父块/子块检索与终结

来源：[pathagent.py:268](D:/PathCode/pathagent.py:268)，原文件第268—366行。

```python
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
                    sims = np.dot(patch_feats, text_emb.T).squeeze()

                    top2_idx = np.argsort(sims)[-2:][::-1]
                    top2_names = [patch_names_valid[i] for i in top2_idx]
                    print(f"  -> Selected Top2 patches: {top2_names}")

                    # split top2 patches into sub-patches at zoom_level_val
                    candidate_images = []
                    candidate_names = []
                    for patch_name in top2_names:
                        patch_path = get_patch_fullpath(args.patch_root, TARGET_LONG_ID, patch_name)
                        sub_patches = split_patch_for_zoom(patch_path, zoom_level_val)
                        for sub_img, (x, y) in sub_patches:
                            candidate_images.append(sub_img)
                            candidate_names.append(f"{x}_{y}")

                    if not candidate_images:
                        print("No sub-patches generated, skipping zoom process.")
                    else:
                        image_embs = plip.encode_images(candidate_images, batch_size=4)
                        image_embs = image_embs / np.linalg.norm(image_embs, axis=-1, keepdims=True)
                        sims_sub = np.dot(image_embs, text_emb.T).squeeze()
                        best_idx = int(np.argmax(sims_sub))
                        best_img = candidate_images[best_idx]
                        best_name = candidate_names[best_idx]
                        print(f"Selected zoomed sub-patch: {best_name}")

                        x, y = best_name.split("_")

                        zoomed_patch_desc = patho_r1_describe(
                            best_img,
                            question=question_text,
                            patho_r1_processor=patho_r1_processor,
                            patho_r1_model=patho_r1_model,
                            coords=(x, y),
                            magnification=zoom_level_val,
                            choices=question_choices
                        )

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

                        all_results = [{
                            "attempt": attempt,
                            "mode": "zoom_then_select",
                            "evaluated_patches_this_round": patches_to_evaluate_names,
                            "total_accumulated_patches": len(accumulated_patch_names),
                            "evaluation_result": evaluation_result,
                            "selected_zoom_patch": best_name,
                            "zoom_patch_desc": zoomed_patch_desc,
                            "answer": final_answer["answer"],
                            "explanation": final_answer["explanation"]
                        }]
                        break
```

## 4. 缺失信息补检与预算耗尽后的回答

来源：[pathagent.py:368](D:/PathCode/pathagent.py:368)，原文件第368—425行。

```python
            else:
                # cannot answer and cannot zoom -> find additional patches relevant to missing_info
                missing_info = evaluation_result.get("missing_info", "pathology details").strip()
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
                sims = np.dot(remaining_embs, text_emb.T).squeeze()

                num_to_add = max(1, int(len(all_patch_names) * replenish_ratio))
                best_indices = np.argsort(sims)[::-1][:num_to_add]

                newly_selected_patches = [valid_remaining_names[i] for i in best_indices]
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
```

## 5. JSON生成调用的基础封装

来源：[models/inference.py:10](D:/PathCode/models/inference.py:10)，原文件第10—78行。

```python
def _call_llm_return_json_simple(model, tokenizer, messages, max_new_tokens=256, retries=1):
    """
    Call the qwen LLM and parse JSON.
    """
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )

    def extract_json_block(s: str):
        """Extract the first complete { ... } block from the generated text (based on brace counting)."""
        start = s.find("{")
        if start == -1:
            return None
        depth = 0
        for i in range(start, len(s)):
            if s[i] == "{":
                depth += 1
            elif s[i] == "}":
                depth -= 1
                if depth == 0:
                    return s[start:i + 1]
        return None

    for attempt in range(retries + 1):
        with torch.no_grad():
            model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens
            )

        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        full_output = tokenizer.decode(output_ids, skip_special_tokens=True).strip()

        # Attempt to parse directly
        try:
            parsed = json.loads(full_output)
            return full_output, parsed
        except Exception:
            # Attempt to extract the first JSON block
            json_block = extract_json_block(full_output)
            if json_block:
                try:
                    parsed = json.loads(json_block)
                    return full_output, parsed
                except Exception:
                    pass

        # If failed, add a reminder prompt and try again
        if attempt < retries:
            messages_retry = deepcopy(messages)
            messages_retry.append({
                "role": "system",
                "content": "Reminder: Respond with only a single valid JSON object containing the requested keys."
            })
            text = tokenizer.apply_chat_template(
                messages_retry,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
            continue

        return full_output, None

    return None, None
```

## 6. Executor：A/B/C判断链

来源：[models/inference.py:80](D:/PathCode/models/inference.py:80)，原文件第80—200行。

```python
def evaluate_with_llm_chain(model, tokenizer, description, question, choices=None,
                               max_new_tokens_a=512, max_new_tokens_b=256, max_new_tokens_c=256,
                               retries=1):
    """
    Three-step logic chain (simplified):
      Step A: Generate answer + thinking_steps
      Step B: Judge if sufficient (Yes / No)
      Step C: If insufficient, analyze missing info and zoom strategy
    """
    # --- Step A ---
    system_a = (
        "You are an expert AI pathology assistant.\n"
        "Task: Based on the patch descriptions, try to answer the question step-by-step.\n"
        "Output ONLY a JSON object:\n"
        "{\n"
        '  "answer": "the final predicted answer (string)",\n'
        '  "thinking_steps": "your detailed reasoning, step-by-step (string)"\n'
        "}"
    )

    choices_text = f"\nChoices: {choices}" if choices else ""
    user_a = f"--- Patch Descriptions ---\n{description}\n--- End ---\nQuestion: {question}{choices_text}\nNow output the JSON with 'answer' and 'thinking_steps'."

    messages_a = [
        {"role": "system", "content": system_a},
        {"role": "user", "content": user_a},
    ]
    full_a, parsed_a = _call_llm_return_json_simple(model, tokenizer, messages_a,
                                                   max_new_tokens=max_new_tokens_a, retries=retries)

    if parsed_a is None:
        parsed_a = {
            "answer": "Uncertain",
            "thinking_steps": full_a or ""
        }

    parsed_a.setdefault("answer", "Uncertain")
    parsed_a.setdefault("thinking_steps", "")

    # --- Step B ---
    system_b = (
        "You are an expert AI pathology assistant.\n"
        "Task: Judge whether the current patch descriptions are sufficient to confidently support the answer.\n"
        "Output ONLY a JSON object like:\n"
        '{"sufficient": "Yes" or "No" }'
    )
    user_b = (
        f"Descriptions:\n{description}\n\n"
        f"Question: {question}{choices_text}\n\n"
        f"Previous answer and reasoning:\n{json.dumps(parsed_a, ensure_ascii=False)}\n\n"
        "Return JSON only."
    )
    messages_b = [{"role": "system", "content": system_b}, {"role": "user", "content": user_b}]
    full_b, parsed_b = _call_llm_return_json_simple(model, tokenizer, messages_b,
                                                   max_new_tokens=max_new_tokens_b, retries=retries)

    if parsed_b is None:
        parsed_b = {"sufficient": "Uncertain"}

    suff = parsed_b.get("sufficient", "").strip().lower()

    # --- If sufficient == "yes", return immediately ---
    if suff == "yes":
        return {
            "answer": parsed_a.get("answer"),
            "thinking_steps": parsed_a.get("thinking_steps"),
            "sufficient": parsed_b.get("sufficient", ""),
            "raw_texts": {
                "step_a_raw": full_a,
                "step_b_raw": full_b,
                "step_c_raw": None
            }
        }

    # --- Step C ---
    system_c = (
        "You are an expert AI pathology assistant.\n"
        "Task: If current data is insufficient, specify what visual evidence is missing, "
        "and whether zooming in could help obtain that evidence.\n"
        "Output ONLY a JSON object like:\n"
        "{\n"
        '  "missing_info": "short noun phrase",\n'
        '  "zoom_recommendation": "Yes" or "No",\n'
        '  "recommended_zoom_level": "None" or an integer like 10 or 20 or 40,\n'
        '  "zoom_reason": "brief reason why zooming helps"\n'
        "}"
    )

    user_c = (
        f"Descriptions:\n{description}\n\n"
        f"Question: {question}{choices_text}\n\n"
        f"Previous answer: {json.dumps(parsed_a, ensure_ascii=False)}\n"
        f"Sufficiency judgement: {json.dumps(parsed_b, ensure_ascii=False)}\n\n"
        "Now provide the JSON for missing info and zoom recommendation."
    )
    messages_c = [{"role": "system", "content": system_c}, {"role": "user", "content": user_c}]
    full_c, parsed_c = _call_llm_return_json_simple(model, tokenizer, messages_c,
                                                   max_new_tokens=max_new_tokens_c, retries=retries)

    if parsed_c is None:
        parsed_c = {
            "missing_info": "Uncertain",
            "zoom_recommendation": "Uncertain",
            "recommended_zoom_level": "Uncertain",
            "zoom_reason": full_c or ""
        }

    return {
        "answer": parsed_a.get("answer"),
        "thinking_steps": parsed_a.get("thinking_steps"),
        "sufficient": parsed_b.get("sufficient"),
        "missing_info": parsed_c.get("missing_info"),
        "zoom_recommendation": parsed_c.get("zoom_recommendation"),
        "zoom_level": parsed_c.get("recommended_zoom_level", 5),
        "zoom_reason": parsed_c.get("zoom_reason"),
        "raw_texts": {
            "step_a_raw": full_a,
            "step_b_raw": full_b,
            "step_c_raw": full_c
        }
    }
```

## 7. 最终答案：累计描述转为答案

来源：[models/inference.py:202](D:/PathCode/models/inference.py:202)，原文件第202—308行。

```python
def slide_llm_answer(
    model,
    tokenizer,
    descriptions_text,
    question,
    choices=None,
    magnification=None,
    case_name=None,
):
    """
    Slide-level LLM: Generates the final answer based on multiple patch descriptions.
    Optimized the prompt to focus the model on specific slide-level results rather than conceptual explanations.
    Automatically repairs JSON format errors in the model output.
    """
    system_prompt = (
        "You are an expert slide-level pathology assistant. "
        "You will be given a question and detailed patch-level descriptions of a pathology slide. "
        "Your task is to infer the specific **slide-level diagnostic result or quantitative value** "
        "based on the provided evidence — not to define or explain the medical term itself. "
        "The answer should directly reflect the information observable in the slide, such as biomarker expression level, "
        "presence or absence of features, or a numeric measurement.\n\n"
        "Rules:\n"
        "1. When the question asks about a biomarker (e.g., HER2, progesterone receptor, Ki-67), "
        "output the **observed status or score** (e.g., 'positive', 'negative', '2+', 'high expression'), not the definition.\n"
        "2. When the question asks about survival time or other quantitative results, "
        "output only the numerical value for the answer key in the response JSON.\n"
        "3. Always keep the 'answer' short — short phrase, or one number.\n"
        "4. Always include a brief reasoning in 'explanation', summarizing how the evidence supports the answer.\n"
        "5. **If choices are provided, your 'answer' must be exactly one of the given options. "
        "Never generate an answer outside the provided choices.**\n\n"
        "Respond strictly in JSON format with keys: 'answer' and 'explanation'."
    )

    if magnification is not None:
        system_prompt = f"[Slide-level Context | Magnification={magnification}x]\n" + system_prompt

    choices_text = f"\nChoices: {choices}" if choices else ""

    user_prompt = (
        f"Question: {question}{choices_text}\n\n"
        "Now, based on the following patch-level descriptions of the slide, "
        "determine the **slide-level result** that directly answers the question.\n\n"
        f"--- Patch Descriptions ---\n{descriptions_text}\n--- End of Descriptions ---\n\n"
        "Answer in JSON format:"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )

    with torch.no_grad():
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=1024,
            temperature=0.15,
            do_sample=False
        )

    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
    raw_answer = tokenizer.decode(output_ids, skip_special_tokens=True).strip()

    # ------------------- JSON Extraction and Repair Logic -------------------
    parsed = None
    json_candidates = re.findall(r"\{.*?\}", raw_answer, flags=re.DOTALL)

    for candidate in reversed(json_candidates):  # Parse the last one first
        try:
            parsed = json.loads(candidate)
            break
        except Exception:
            continue

    if parsed is None:
        print(f"JSON parsing failed ({case_name if case_name else 'unknown_case'}), raw output:\n{raw_answer}\n")

        # fallback: Extract the first word as the answer
        answer = {
            "answer": raw_answer.split()[0] if raw_answer else "Unknown",
            "explanation": "Model did not return valid JSON."
        }
    else:
        answer_text = parsed.get("answer", "").strip()
        if not answer_text:
            answer_text = "Unknown"

        explanation_text = parsed.get("explanation", "").strip()
        if not explanation_text:
            explanation_text = "No explanation provided."

        answer = {
            "answer": answer_text,
            "explanation": explanation_text
        }

    del model_inputs, generated_ids, output_ids
    torch.cuda.empty_cache()

    return answer
```

## 8. Perceptor：图像与任务转为描述

来源：[models/inference.py:310](D:/PathCode/models/inference.py:310)，原文件第310—392行。

```python
def patho_r1_describe(image, question=None,
                    patho_r1_processor=None, patho_r1_model=None,
                    coords=None, magnification=None, max_new_tokens=1024, choices=None, missing_info=None):

    meta_lines = []
    if coords is not None:
        meta_lines.append(f"Patch coordinates: ({coords[0]},{coords[1]})")
    if magnification is not None:
        meta_lines.append(f"Magnification: {magnification}x")
    meta_text = ""
    if meta_lines:
        meta_text = "[IMAGE META] " + " | ".join(meta_lines) + "\n\n"


    if question is None:
        prompt_body = "Please describe the pathology features in this image."
    else:
        prompt_body = (
            f"Question: {question}\n\n"
            "Answer the question and list the pathological features visible in the image that support your answer."
        )

    if choices is not None:
        prompt_body += f"\nChoices: {choices}"
    if missing_info is not None:
        prompt_body += f"\nMissing information: {missing_info}"

    full_text = meta_text + prompt_body

    # === Construct system prompt ===
    system_prompt = (
        "A conversation between a curious user and an AI medical assistant specialized in pathology image analysis. "
        "The assistant can interpret pathology images, describe observed features, and provide possible explanations based on medical knowledge, "
        "but will never give a definitive diagnosis or prescribe treatment. "
        "The assistant must always maintain a polite, clear, and professional tone. "
        "All answers should be supported by established, reliable medical sources. "
        "The assistant should carefully consider visual details in pathology images, such as cell morphology, staining patterns, and tissue architecture. "
        "If choices are given, the answer must be given from the choices."
        "If Missing information is given, you need to focus on this part of the image."
    )

    # === Construct message input ===
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": full_text},
            ],
        },
    ]

    # === Construct model input ===
    text = patho_r1_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = patho_r1_processor(
        text=[text],
        images=image_inputs,
        # videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(patho_r1_model.device)

    # === Inference generation ===
    with torch.no_grad():
        generated_ids = patho_r1_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    output_text = patho_r1_processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0].strip()

    return output_text
```

## 9. 描述摘要：控制累计证据的组织方式

来源：[models/inference.py:394](D:/PathCode/models/inference.py:394)，原文件第394—496行。

```python
def summarize_patches_in_chunks(
    model, tokenizer, descriptions_dict, patch_names,
    question_text=None, chunk_size=5, threshold=50, magnification=None
):
    """
    If the number of patches exceeds the threshold, summarize descriptions in chunks of `chunk_size`.
    The summary for each chunk will explicitly list the included patch coordinates and conduct a guided summary based on the question.
    """
    if len(patch_names) <= threshold:
        # Does not exceed threshold, concatenate original descriptions directly
        items = [(name, descriptions_dict[name]) for name in patch_names]
        return build_descriptions_with_meta(
            items, mag_level=magnification, include_header=True, include_coords=True
        )

    print(f"Patch count {len(patch_names)} exceeds {threshold}, performing chunked summarization...")
    summaries = []
    for i in range(0, len(patch_names), chunk_size):
        chunk_names = patch_names[i:i+chunk_size]
        items = [(name, descriptions_dict[name]) for name in chunk_names]

        # Extract coordinate list
        coords_list = []
        for name, _ in items:
            x, y = extract_coords_from_name(name)
            coords_list.append(f"({x},{y})" if x is not None and y is not None else "(unknown)")
        coords_str = ", ".join(coords_list)

        # Concatenate patch descriptions (with coordinates)
        chunk_text = build_descriptions_with_meta(
            items, mag_level=magnification, include_header=False, include_coords=True
        )

        # === Construct Prompt ===
        system_prompt = (
            "You are an expert pathology assistant. "
            "You will be given multiple patch-level descriptions of histopathology images. "
            "Your task is to summarize the key pathological features across these patches. "
            "The summary must be concise yet informative, highlighting significant morphological patterns. "
            "Additionally, focus on information that could help answer the following question "
            "about the slide, emphasizing details relevant to the diagnostic or interpretive context."
        )

        if magnification is not None:
            system_prompt = f"[Patch-level Summarization | Magnification={magnification}x]\n" + system_prompt

        user_prompt = (
            f"--- Patch Descriptions (with coordinates) ---\n{chunk_text}\n"
            "--- End of Descriptions ---\n\n"
        )

        if question_text:
            user_prompt += f"Related Question: {question_text}\n\n"

        user_prompt += (
            "Summarize the main pathological findings across these patches. "
            "Your summary should:\n"
            "- Capture key morphological features and cellular details.\n"
            "- If a question is provided, emphasize features that are relevant to answering it.\n"
            "- Do not provide a final answer or diagnosis.\n"
            "Output only the summary text, no JSON or extra formatting."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )

        with torch.no_grad():
            model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=512,
                temperature=0.2,
                do_sample=False
            )

        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        summary_text = tokenizer.decode(output_ids, skip_special_tokens=True).strip()

        # Release memory
        del model_inputs, generated_ids, output_ids
        torch.cuda.empty_cache()

        summaries.append(
            f"[Chunk {i//chunk_size+1} | Patches={coords_str}]\n{summary_text}"
        )

        print(f"Completed summary for chunk {i//chunk_size+1} (patch count: {len(chunk_names)})")

    # Concatenate summaries of all chunks
    combined_summary = (
        f"[Current Magnification: {magnification}x]\n\n" +
        "\n\n".join(summaries)
    )
    return combined_summary
```

## 10. 状态文本：倍率、坐标和描述

来源：[data_processing/utils.py:107](D:/PathCode/data_processing/utils.py:107)，原文件第107—131行。

```python
def build_descriptions_with_meta(items, mag_level=None, include_header=True, include_coords=True):
    """
    Combine multiple (patch_name, description) into text for LLM:
      - If include_header=True, inserts "[Current Magnification: {mag_level}x]" at the top
      - Each patch is annotated with coordinates (extracted from filename); if include_coords=False, coordinates are omitted
      - Output format example:
        [Current Magnification: 10x]
        [37856_28960 | Coord=(37856,28960)] description...
        [82912_86304 | Coord=(82912,86304)] description...
    """
    header = ""
    if include_header and mag_level is not None:
        header = f"[Current Magnification: {mag_level}x]\n\n"

    parts = []
    for name, desc in items:
        x, y = extract_coords_from_name(name)
        coord_str = f"({x},{y})" if (x is not None and y is not None) else "(unknown)"
        if include_coords:
            parts.append(f"[{name} | Coord={coord_str}] {desc}")
        else:
            parts.append(f"[{name}] {desc}")

    body = "\n\n".join(parts)
    return header + body
```

## 11. 原版zoom：已有图块的裁剪

来源：[data_processing/utils.py:146](D:/PathCode/data_processing/utils.py:146)，原文件第146—203行。

```python
def split_patch_for_zoom(patch_path, zoom_level):
    """
    Split a 5x patch image into sub-images at a higher magnification level,
    and return the sub-images along with their global coordinates.
    Supports patch_path as a file path or a PIL.Image.Image object.
    """

    if isinstance(patch_path, (str, os.PathLike)):
        img = Image.open(patch_path).convert("RGB")
        base_name = os.path.basename(patch_path)
        name_no_ext = os.path.splitext(base_name)[0]

    elif isinstance(patch_path, Image.Image):
        img = patch_path
        # If it is an image object, use default coordinates (0,0)
        name_no_ext = "0_0"
    else:
        raise TypeError(f"patch_path must be a file path or PIL.Image.Image object, received {type(patch_path)}")

    width, height = img.size

    # --- Check if zoom_level is valid ---
    if zoom_level % 5 != 0 or zoom_level < 5:
        raise ValueError(f"zoom_level must be >=5 and a multiple of 5, received {zoom_level}")

    # --- Try to parse global coordinates from filename ---
    try:
        base_x, base_y = map(int, name_no_ext.split("_"))
    except ValueError:
        base_x, base_y = 0, 0  # If parsing fails, default to starting from (0, 0)

    # --- Calculate split factor ---
    factor = zoom_level // 5
    if factor == 1:
        # If zoom_level is 5x, return the original image and its global coordinates directly
        return [(img, (base_x, base_y))]

    # --- Calculate sub-image size ---
    sub_w = width // factor
    sub_h = height // factor

    # --- Split image and calculate global coordinates ---
    patches = []
    for i in range(factor):
        for j in range(factor):
            left = j * sub_w
            upper = i * sub_h
            right = (j + 1) * sub_w if j < factor - 1 else width
            lower = (i + 1) * sub_h if i < factor - 1 else height
            patch = img.crop((left, upper, right, lower))

            # Calculate global coordinates of the sub-image
            global_x = base_x + left
            global_y = base_y + upper

            patches.append((patch, (global_x, global_y)))

    return patches
```

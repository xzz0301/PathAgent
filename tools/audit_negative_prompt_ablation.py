"""Fixed nine-ROI diagnostic audit; does not change production navigation."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--perceptor', required=True)
    parser.add_argument('--executor', required=True)
    a = parser.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    state = {'stage': 'preflight', 'started': time.time(), 'roi_completed': 0, 'text_completed': 0}
    save(a.output / 'status.json', state)
    try:
        import torch
        from PIL import Image
        from transformers import AutoProcessor, AutoTokenizer, AutoModelForCausalLM, Qwen2_5_VLForConditionalGeneration
        from models.inference import patho_r1_describe, evaluate_with_llm_chain, slide_llm_answer
        from data_processing.utils import build_descriptions_with_meta

        cases = {}
        source_hashes = {}
        for case in ['C6', 'C8', 'C12']:
            p = a.root / 'agent' / f'DX1_{case}_3032b9e5.json'
            source_hashes[str(p)] = digest(p)
            cases[case] = json.loads(p.read_text())
            assert len(cases[case]['trajectory']) == 3
            for row in cases[case]['trajectory']:
                assert digest(row['input_image_path']) == row['input_image_sha256']
                assert row['magnification'] == 10
        descriptions_path = a.root / 'inputs/descriptions.json'
        source_hashes[str(descriptions_path)] = digest(descriptions_path)
        generic = json.loads(descriptions_path.read_text())
        save(a.output / 'protocol.json', {
            'cases': list(cases), 'source_hashes': source_hashes,
            'scope': 'Fixed 9 ROIs from 3 previously false-positive cases. No navigation or new ROI selection.',
            'perceptor': a.perceptor, 'executor': a.executor, 'dtype': 'bfloat16',
            'seed_per_call': 128, 'perceptor_max_new_tokens': 1024,
            'prompt_control': 'Original function with original question/choices; seed reset for each call.',
            'prompt_morphology': 'Same function/system/image/meta; question=None, choices=None gives: Please describe the pathology features in this image.',
            'text_conditions': ['quilt', 'historical_pathor1', 'historical_mixed', 'morphology_pathor1', 'morphology_mixed'],
            'text_control': 'Historical text is frozen, not replaced by the diagnostic prompt replay.',
            'evaluation': 'Original sufficiency chain and final answer both run on fixed text. Final answer is forced for comparison even when sufficiency is No; no claimed navigation outcome.',
            'limitations': 'Selected negatives only, no sensitivity or overall accuracy claim. Generic prompt removes question and choices together. Weight vs prompt vs resolution causality not fully identified.',
            'script_sha256': digest(__file__),
        })

        def seed():
            torch.manual_seed(128)
            torch.cuda.manual_seed_all(128)

        state['stage'] = 'loading_perceptor'; save(a.output / 'status.json', state)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(a.perceptor, torch_dtype=torch.bfloat16, device_map='cuda:1')
        model.eval()
        processor = AutoProcessor.from_pretrained(a.perceptor)
        save(a.output / 'perceptor_generation_config.json', model.generation_config.to_dict())
        rows = []
        for case, baseline in cases.items():
            for row in baseline['trajectory']:
                item = {'case': case, 'step': row['step'], 'patch': row['patch_name'], 'image_sha256': row['input_image_sha256'],
                        'historical_description': row['description'], 'quilt': generic[baseline['long_id']][row['patch_name']]}
                image = Image.open(row['input_image_path']).convert('RGB')
                for arm in ['diagnostic_replay', 'morphology']:
                    seed(); audit = {}; start = time.time()
                    item[arm] = patho_r1_describe(image, question=baseline['question'] if arm == 'diagnostic_replay' else None,
                        choices=baseline['choices'] if arm == 'diagnostic_replay' else None,
                        patho_r1_processor=processor, patho_r1_model=model,
                        coords=(row['x'], row['y']), magnification=row['magnification'], max_new_tokens=1024, input_audit=audit)
                    item[arm + '_input'] = audit
                    item[arm + '_seconds'] = time.time() - start
                item['replay_exact_text_match'] = item['diagnostic_replay'] == row['description']
                rows.append(item); save(a.output / 'roi_results.json', rows)
                state.update(stage='perceptor_comparison', roi_completed=len(rows)); save(a.output / 'status.json', state)
                print(json.dumps({'case':case,'step':row['step'],'roi_completed':len(rows)}), flush=True)
        del model, processor
        import gc
        gc.collect(); torch.cuda.empty_cache()
        state['stage'] = 'loading_executor'; save(a.output / 'status.json', state)
        tokenizer = AutoTokenizer.from_pretrained(a.executor)
        model = AutoModelForCausalLM.from_pretrained(a.executor, torch_dtype=torch.bfloat16, device_map='cuda:0')
        model.eval()
        save(a.output / 'executor_generation_config.json', model.generation_config.to_dict())
        results = []
        for case, baseline in cases.items():
            source = [r for r in rows if r['case'] == case]
            for arm in ['quilt', 'historical_pathor1', 'historical_mixed', 'morphology_pathor1', 'morphology_mixed']:
                items = []
                for r in source:
                    value = {'quilt': r['quilt'], 'historical_pathor1': r['historical_description'],
                             'historical_mixed': (r['quilt']+' '+r['historical_description']).strip(),
                             'morphology_pathor1': r['morphology'],
                             'morphology_mixed': (r['quilt']+' '+r['morphology']).strip()}[arm]
                    items.append((r['patch'], value))
                text = build_descriptions_with_meta(items, mag_level=10, include_header=True, include_coords=True)
                seed(); decision = evaluate_with_llm_chain(model, tokenizer, text, baseline['question'], baseline['choices'])
                seed(); answer = slide_llm_answer(model, tokenizer, text, baseline['question'], baseline['choices'], magnification=10, case_name=case+'_'+arm)
                results.append({'case':case,'condition':arm,'input_text':text,'input_text_sha256':hashlib.sha256(text.encode()).hexdigest(),
                                'evaluation':decision,'forced_final_answer':answer})
                save(a.output / 'text_results.json', results)
                state.update(stage='text_ablation', text_completed=len(results)); save(a.output / 'status.json', state)
                print(json.dumps({'case':case,'arm':arm,'sufficient':decision.get('sufficient'),'answer':answer['answer']}), flush=True)
        unchanged = all(digest(path)==sha for path,sha in source_hashes.items())
        assert unchanged
        state.update(stage='complete', finished=time.time(), source_unchanged=unchanged)
        save(a.output / 'status.json', state)
    except Exception as exc:
        state.update(stage='failed', error=repr(exc), finished=time.time())
        save(a.output / 'status.json', state)
        raise


if __name__ == '__main__':
    main()

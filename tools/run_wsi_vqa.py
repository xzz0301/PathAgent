"""Durable, locked preparation and full-test evaluation of the paper-aligned profile."""
import argparse
import concurrent.futures
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from tools.prepare_wsi_vqa import save


def caption_worker(root, gpu, server):
    import torch
    torch.manual_seed(128); torch.cuda.manual_seed_all(128)
    spec = importlib.util.spec_from_file_location('quilt_generator', PROJECT/'data_preparation_script/description_generation.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    loader = module.load_pretrained_model
    cache = []

    def cached_loader(*args, **kwargs):
        if not cache:
            cache.append(loader(*args, **kwargs))
            cache[0][1].eval()
        return cache[0]

    module.load_pretrained_model = cached_loader
    folders = sorted((root/'inputs/patches').iterdir())
    worker_state = root/f'caption_worker_{gpu}.json'
    completed = 0
    for index, folder in enumerate(folders):
        if index % 3 != gpu:
            continue
        dest = root/'captions'/folder.name; dest.mkdir(parents=True, exist_ok=True)
        expected = {p.name for p in folder.glob('*.png')}
        result_file = dest/'descriptions.json'
        if result_file.exists():
            actual = json.loads(result_file.read_text()).get(folder.name, {})
            if set(actual) == expected and all(v.strip() for v in actual.values()):
                completed += 1
                continue
        slide_list = dest/'slides.txt'; slide_list.write_text(folder.name+'\n')
        save(worker_state, {'stage':'captioning','case':folder.name,'completed':completed,'pid':os.getpid()})
        torch.manual_seed(128); torch.cuda.manual_seed_all(128)
        settings = argparse.Namespace(model_path=str(server/'models_runtime/Quilt-Llava-v1.5-7b-local_20261031'),
            model_base=None, device='cuda', conv_mode='pathology_v0', temperature=.2, max_new_tokens=512,
            load_8bit=False, load_4bit=False, debug=False, image_aspect_ratio='pad',
            image_dir=str(folder.parent), output_json=str(result_file), slide_list=str(slide_list))
        module.main(settings)
        actual = json.loads(result_file.read_text())[folder.name]
        if set(actual) != expected or not all(v.strip() for v in actual.values()):
            raise ValueError('Incomplete generic captions: '+folder.name)
        model = cache[0][1]
        save(dest/'complete.json', {'patches':len(actual),'caption_sha256':hashlib.sha256(result_file.read_bytes()).hexdigest(),
                                  'model':settings.model_path,'dtype':str(model.dtype),'quantization':'none',
                                  'vision_loaded':model.get_vision_tower().is_loaded,'seed':128})
        completed += 1
    save(worker_state, {'stage':'complete','completed':completed,'pid':os.getpid()})


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--server', type=Path, required=True)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--caption-worker', type=int)
    a = p.parse_args()
    if a.caption_worker is not None:
        caption_worker(a.output,a.caption_worker,a.server)
        return
    a.output.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (a.output/'run.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    status_path = a.output/'status.json'
    state = {'stage':'starting','pid':os.getpid(),'started':time.time()}
    if status_path.exists() and json.loads(status_path.read_text()).get('stage') == 'complete':
        raise RuntimeError('Completed experiment must not be restarted')
    save(status_path,state)
    py = a.server/'envs/runtime/bin/python'
    quilt_py = a.server/'envs/quilt_20261031/bin/python'

    def run(command, filename, env=None):
        with (a.output/filename).open('a') as log:
            subprocess.run([str(x) for x in command],cwd=PROJECT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)

    try:
        state['stage']='preparing_full_candidates';save(status_path,state)
        prepared=a.output/'inputs/preparation_status.json'
        if not prepared.exists() or json.loads(prepared.read_text()).get('stage')!='complete':
            run([py,PROJECT/'tools/prepare_wsi_vqa.py','--dataset',a.dataset,'--output',a.output/'inputs'],'prepare.log')
        state['stage']='quilt_generic_descriptions';save(status_path,state)

        def worker(gpu):
            env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),
                 'PYTHONPATH':str(a.server/'code/quilt_7e70fc3_20261031'),
                 'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'}
            run([quilt_py,'-u',Path(__file__),'--server',a.server,'--dataset',a.dataset,'--output',a.output,
                 '--caption-worker',gpu],f'quilt_{gpu}.log',env)

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            futures=[pool.submit(worker,gpu) for gpu in range(3)]
            for future in futures:
                future.result()
        merged={}
        for file in sorted((a.output/'captions').glob('*/descriptions.json')):
            merged.update(json.loads(file.read_text()))
        manifest=json.loads((a.output/'inputs/wsi_manifest.json').read_text())
        assert set(merged)==set(manifest) and len(merged)==86
        save(a.output/'inputs/descriptions.json',merged)
        state['stage']='plip_features';save(status_path,state)
        if not (a.output/'features_complete.json').exists():
            run([py,PROJECT/'tools/prepare_camelyon_features.py','--input',a.output/'inputs','--project',PROJECT],'features.log')
            save(a.output/'features_complete.json', {'slides':86})
        config=json.loads((PROJECT/'configs/wsi_vqa_paper.json').read_text())
        for key in ['plip_lib_path','qwen_ckpt','plip_ckpt','patho_r1_ckpt']:
            config[key]=str(PROJECT/config[key])
        inp=a.output/'inputs'
        config.update(descriptions_file=str(inp/'descriptions.json'),questions_file=str(inp/'questions_blinded.json'),
                      feature_dir=str(inp/'features'),patch_root=str(inp/'patches'),save_dir=str(a.output/'agent'),
                      geometry_file=str(inp/'geometry.json'),wsi_manifest=str(inp/'wsi_manifest.json'))
        save(a.output/'run_config.json',config)
        state['stage']='full_test_inference';save(status_path,state)
        run([py,'-u',PROJECT/'pathagent.py','--config',a.output/'run_config.json'],'inference.log')
        state['stage']='metrics_and_selected_cases';save(status_path,state)
        run([py,PROJECT/'tools/evaluate_wsi_vqa.py','--root',a.output],'evaluation.log')
        state.update(stage='complete',finished=time.time());save(status_path,state)
    except Exception as exc:
        state.update(stage='failed',error=repr(exc),finished=time.time());save(status_path,state)
        raise


if __name__=='__main__':
    main()

"""Run frozen six-case preparation; independent Quilt workers use one GPU each."""
import concurrent.futures,json,os,subprocess,sys,time
from pathlib import Path

R=Path('/data/dataset/users/xuzhaozheng/pathagent')
PROJECT=R/'code/current';OUT=R/'artifacts/camelyon6_wsi_20261104';INPUT=OUT/'inputs'
def save(obj):
    p=OUT/'preparation_status.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(obj,indent=2));t.replace(p)
def run(command,log,env=None):
    with log.open('w') as f:subprocess.run(list(map(str,command)),stdout=f,stderr=subprocess.STDOUT,env=env,cwd=PROJECT,check=True)
def main():
    previous=OUT/'preparation_status.json'
    if previous.exists():
        prior=json.loads(previous.read_text())
        if prior['stage']!='failed':raise RuntimeError('Existing active or complete preparation; no duplicate launch')
        (OUT/f"preparation_failure_{int(time.time())}.json").write_text(json.dumps(prior,indent=2))
    status={'stage':'preparing','started':time.time(),'completed':[]};save(status)
    try:
        if not (INPUT/'candidates.json').exists():
            run([R/'envs/runtime/bin/python',PROJECT/'tools/prepare_camelyon_pilot.py','--wsi-root','/data/dataset/users/xuzhaozheng/datasets/CAMELYON16/images','--reference',OUT/'reference.csv','--output',INPUT,'--c1-existing',PROJECT/'demo/c1_independent/patches/DX1_C1'],OUT/'prepare.log')
        else:
            import hashlib
            for key,data in json.loads((INPUT/'candidates.json').read_text()).items():
                for row in data['patches']:assert hashlib.sha256((INPUT/'patches'/key/row['name']).read_bytes()).hexdigest()==row['sha256']
        status['stage']='quilt_descriptions';save(status)
        def worker(gpu,cases):
            for case in cases:
                env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'TOKENIZERS_PARALLELISM':'false',
                     'PYTHONPATH':str(R/'code/quilt_7e70fc3_20261031'),'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'}
                old=OUT/('quilt_'+case)
                if old.exists():
                    if (old/'complete.json').exists():continue
                    # Preserve failed setup output; never overwrite an active worker.
                    prior=json.loads((old/'status.json').read_text())
                    if prior['stage'] not in ('loading','failed'):raise RuntimeError('Unfinished nonfailed Quilt run: '+case)
                    if (old/'descriptions.json').exists():raise RuntimeError('Partial generated descriptions require explicit resume: '+case)
                    old.rename(OUT/(old.name+'_setup_failed_'+str(int(time.time()))))
                run([R/'envs/quilt_20261031/bin/python',PROJECT/'tools/quilt_acceptance.py','--project-root',PROJECT,'--patch-root',INPUT/'patches','--slide-id','DX1_'+case,'--model-path',R/'models_runtime/Quilt-Llava-v1.5-7b-local_20261031','--output-dir',OUT/('quilt_'+case),'--sample-count','0','--load-4bit','--max-new-tokens','512','--seed','128'],OUT/(case+'_quilt.log'),env)
                print('QUILT_COMPLETE '+case,flush=True)
            return cases
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            futures=[pool.submit(worker,g,cases) for g,cases in enumerate([['C2','C12'],['C6','C48'],['C8']])]
            for f in concurrent.futures.as_completed(futures):status['completed']+=f.result();save(status)
        merged=json.loads((R/'artifacts/quilt_c1_64_20261031/descriptions.json').read_text())
        for case in ['C2','C6','C8','C12','C48']:merged.update(json.loads((OUT/('quilt_'+case)/'descriptions.json').read_text()))
        for folder in (INPUT/'patches').iterdir():
            assert set(merged[folder.name])=={p.name for p in folder.glob('*.png')}
        (INPUT/'descriptions.json').write_text(json.dumps(merged,indent=2))
        status['stage']='plip_features';save(status)
        run([R/'envs/runtime/bin/python',PROJECT/'tools/prepare_camelyon_features.py','--input',INPUT,'--project',PROJECT],OUT/'features.log',env={**os.environ,'CUDA_VISIBLE_DEVICES':'2'})
        status.update(stage='complete',finished=time.time());save(status)
    except Exception as e:
        status.update(stage='failed',error=repr(e),finished=time.time());save(status);raise
if __name__=='__main__':main()

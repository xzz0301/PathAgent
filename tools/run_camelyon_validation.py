"""Run only the frozen six-case cohort after input preparation and WSI replay pass."""
import json,os,subprocess,time
from pathlib import Path
R=Path('/data/dataset/users/xuzhaozheng/pathagent');PROJECT=R/'code/current';OUT=R/'artifacts/camelyon6_wsi_20261104';PY=R/'envs/runtime/bin/python'
def save(obj):
    p=OUT/'validation_status.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(obj,indent=2));t.replace(p)
def main():
    if (OUT/'validation_status.json').exists():raise RuntimeError('Validation already started; inspect before resuming')
    status={'stage':'waiting_for_preparation','started':time.time()};save(status)
    try:
        while True:
            prep=json.loads((OUT/'preparation_status.json').read_text())
            if prep['stage']=='failed':raise RuntimeError('Preparation failed: '+prep.get('error',''))
            if prep['stage']=='complete':break
            time.sleep(20)
        replay=R/'artifacts/c1_wsi_replay_20261104/complete.json'
        if not replay.parent.exists():
            status['stage']='paired_wsi_replay';save(status)
            with (OUT/'replay.log').open('w') as log:
                subprocess.run([str(PY),'-u','tools/replay_wsi_observations.py',
                                '--baseline',str(R/'artifacts/c1_evidence_pilot_20261104/baseline/DX1_C1_3032b9e5.json'),
                                '--manifest',str(OUT/'inputs/wsi_manifest.json'),
                                '--checkpoint',str(R/'models/original_20261022/Patho-R1-7B'),
                                '--output',str(replay.parent)],cwd=PROJECT,stdout=log,stderr=subprocess.STDOUT,check=True)
        if not replay.exists():raise RuntimeError('Existing C1 replay is incomplete; inspect rather than duplicate it')
        if not json.loads(replay.read_text())['baseline_unchanged']:raise RuntimeError('Baseline provenance failure')
        config=json.loads((PROJECT/'configs/server_original.json').read_text())
        for key in ('plip_lib_path','qwen_ckpt','plip_ckpt','patho_r1_ckpt'):config[key]=str(PROJECT/config[key])
        inp=OUT/'inputs'
        config.update(descriptions_file=str(inp/'descriptions.json'),questions_file=str(inp/'questions_blinded.json'),feature_dir=str(inp/'features'),
                      patch_root=str(inp/'patches'),save_dir=str(OUT/'agent'),geometry_file=str(inp/'geometry.json'),wsi_manifest=str(inp/'wsi_manifest.json'),
                      image_backend='wsi',seed=128,initial_sample_count=3,max_attempts=5,min_observation_rounds=1,perceptor_max_new_tokens=1024,capture_observations=True)
        cfg=OUT/'run_config.json';cfg.write_text(json.dumps(config,indent=2))
        status['stage']='agent_inference';save(status)
        with (OUT/'agent.log').open('w') as log:
            subprocess.run([str(PY),'-u','pathagent.py','--config',str(cfg)],cwd=PROJECT,stdout=log,stderr=subprocess.STDOUT,check=True)
        status['stage']='evaluation_and_rendering';save(status)
        with (OUT/'evaluation.log').open('w') as log:
            subprocess.run([str(PY),'tools/evaluate_camelyon_pilot.py','--root',str(OUT)],cwd=PROJECT,stdout=log,stderr=subprocess.STDOUT,check=True)
        status.update(stage='complete',finished=time.time());save(status)
    except Exception as exc:
        status.update(stage='failed',error=repr(exc),finished=time.time());save(status);raise
if __name__=='__main__':main()

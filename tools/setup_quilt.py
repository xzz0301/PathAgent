"""Provision Quilt inference without changing the working PathAgent environment.

Uses the pinned, previously verified source archive and model snapshots. The new
venv shadows legacy API packages locally and reads the main runtime's CUDA stack
through a .pth file. It does not install Quilt's obsolete torch==2.0.1.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import time

REVISION = "7e70fc39f792ac55de010eb37bff0a6d6f491c13"
ARCHIVE_SHA256 = "7e594da24de8826c08e4838181e54ee0ec6a029b23b09be8ab6a015a9e5fd941"
LEGACY_PACKAGES = ["transformers==4.31.0", "tokenizers==0.13.3", "accelerate==0.21.0",
                   "peft==0.4.0", "sentencepiece==0.1.99", "shortuuid==1.0.13", "einops-exts==0.0.4"]

def run(args, **kwargs):
    print("RUN", " ".join(map(str, args)), flush=True)
    return subprocess.run(list(map(str, args)), check=True, **kwargs)

def capture(args, **kwargs):
    return subprocess.check_output(list(map(str, args)), text=True, **kwargs).strip()

def save(path, value):
    temp=path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding="utf-8")
    os.replace(temp,path)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--pathagent-source",type=Path,required=True)
    p.add_argument("--tag",default="20261031")
    p.add_argument("--index-url",default="https://mirrors.aliyun.com/pypi/simple")
    args=p.parse_args()
    if not args.tag.isalnum():raise ValueError("Use an alphanumeric deployment tag")
    root=args.root.resolve()
    main_python=root/"envs/runtime/bin/python"
    archive=root/"quilt_inference_source.tar.gz"
    envdir=root/("envs/quilt_"+args.tag)
    source=root/("code/quilt_7e70fc3_"+args.tag)
    overlay=root/("models_runtime/Quilt-Llava-v1.5-7b-local_"+args.tag)
    artifacts=root/("artifacts/quilt_setup_"+args.tag)
    artifacts.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=(artifacts/"setup.lock").open("a+")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    receipt=artifacts/"setup_complete.json"
    if receipt.exists():
        print(receipt.read_text());return
    # Existing incomplete deployments require inspection rather than silent replacement.
    if any(x.exists() for x in (envdir,source,overlay)):
        raise FileExistsError("Deployment paths exist; inspect preserved state and choose a new tag")
    before=capture([main_python,"-m","pip","freeze"])
    (artifacts/"main_freeze_before.txt").write_text(before+"\n")
    save(artifacts/"status.json",{"stage":"starting","pid":os.getpid(),"updated":time.time()})
    if sha(archive)!=ARCHIVE_SHA256:raise ValueError("Pinned Quilt source archive changed")
    source.mkdir(parents=True)
    with tarfile.open(archive,"r:gz") as bundle:
        for item in bundle.getmembers():
            target=(source/item.name).resolve()
            if not target.is_relative_to(source) or not (item.isfile() or item.isdir()):
                raise ValueError("Unexpected source archive member: "+item.name)
        bundle.extractall(source)
    manifest=json.loads((source/"source_revision.json").read_text())
    if manifest["revision"]!=REVISION:raise ValueError("Quilt source revision mismatch")
    verified=[]
    for item in manifest["files"]:
        f=source/item["path"]
        blob=f.read_bytes()
        digest=hashlib.sha1(b"blob "+str(len(blob)).encode()+b"\0"+blob).hexdigest()
        if digest!=item["sha"]:raise ValueError("Source blob mismatch: "+item["path"])
        verified.append(item["path"])
    # PathAgent's original README explicitly supplies this conversation module.
    conversation=args.pathagent_source/"data_preparation_script/conversation_pathology_v0.py"
    shutil.copy2(source/"llava/conversation.py",source/"llava/conversation.upstream.py")
    shutil.copy2(conversation,source/"llava/conversation.py")
    run([main_python,"-m","venv",envdir])
    python=envdir/"bin/python"
    own_site=Path(capture([python,"-c","import sysconfig; print(sysconfig.get_path('purelib'))"]))
    main_site=capture([main_python,"-c","import sysconfig; print(sysconfig.get_path('purelib'))"])
    (own_site/"pathagent_cuda_runtime.pth").write_text(main_site+"\n",encoding="utf-8")
    save(artifacts/"status.json",{"stage":"installing_legacy_api","pid":os.getpid(),"updated":time.time()})
    run([python,"-m","pip","install","--ignore-installed","--no-deps","--retries","3","--timeout","30",
         "--index-url",args.index_url,*LEGACY_PACKAGES])
    model=root/"models/Quilt-Llava-v1.5-7b"
    vision=root/"models/clip-vit-large-patch14-336"
    for folder in (model,vision):
        sm=json.loads((folder/"source_manifest.json").read_text())
        for item in sm["files"]:
            f=folder/item["rfilename"]
            if not f.is_file() or f.stat().st_size!=item["size"]:
                raise ValueError("Incomplete pinned model: "+str(f))
    overlay.mkdir(parents=True)
    for item in json.loads((model/"source_manifest.json").read_text())["files"]:
        name=item["rfilename"]
        if name=="config.json":continue
        target=overlay/name
        target.parent.mkdir(parents=True,exist_ok=True)
        target.symlink_to(model/name)
    config=json.loads((model/"config.json").read_text())
    config["mm_vision_tower"]=str(vision)
    save(overlay/"config.json",config)
    save(overlay/"local_config_change.json",{"original":str(model/"config.json"),"original_sha256":sha(model/"config.json"),"changed_field":"mm_vision_tower","value":str(vision),"weights_changed":False})
    process_env=dict(os.environ,PYTHONPATH=str(source),HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",HF_HUB_DISABLE_TELEMETRY="1")
    code="""import json, torch, transformers, tokenizers, accelerate, peft
from llava.model.builder import load_pretrained_model
from llava.conversation import conv_templates
assert 'pathology_v0' in conv_templates
assert transformers.__version__ == '4.31.0'
print(json.dumps({'torch':torch.__version__,'torch_file':torch.__file__,'transformers':transformers.__version__,'transformers_file':transformers.__file__,'tokenizers':tokenizers.__version__,'accelerate':accelerate.__version__,'peft':peft.__version__,'cuda':torch.version.cuda,'template':'pathology_v0'}))
"""
    imports=json.loads(capture([python,"-c",code],env=process_env))
    (artifacts/"quilt_freeze.txt").write_text(capture([python,"-m","pip","freeze"])+"\n")
    after=capture([main_python,"-m","pip","freeze"])
    (artifacts/"main_freeze_after.txt").write_text(after+"\n")
    if before!=after:raise RuntimeError("Main environment package list changed")
    # A simple wrapper makes the deployment reproducible from any cwd.
    import shlex
    wrapper=envdir/"run_quilt.sh"
    wrapper.write_text("#!/usr/bin/env bash\nset -euo pipefail\nexport PYTHONPATH="+shlex.quote(str(source))+"\nexport HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1\nexec "+shlex.quote(str(python))+" \"$@\"\n")
    wrapper.chmod(0o700)
    result={"stage":"imports_passed","pid":os.getpid(),"updated":time.time(),"environment":str(envdir),"source":str(source),"source_revision":REVISION,"source_archive_sha256":ARCHIVE_SHA256,"verified_source_files":len(verified),"conversation_sha256":sha(conversation),"model_overlay":str(overlay),"main_environment_unchanged":True,"legacy_packages":LEGACY_PACKAGES,"imports":imports,"inference_verified":False}
    save(receipt,result);save(artifacts/"status.json",result)
    print(json.dumps(result,indent=2),flush=True)

if __name__=="__main__":main()

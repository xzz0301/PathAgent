"""Run the project's description generator on real, explicitly selected patches.

Creates a fresh acceptance directory, uses pathology_v0 explicitly and records
model loading and source hashes. Never changes input patches or model weights.
"""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import time

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def save(path,data):
    temp=path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding="utf-8")
    os.replace(temp,path)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--project-root",type=Path,required=True)
    p.add_argument("--patch-root",type=Path,required=True)
    p.add_argument("--slide-id",required=True)
    p.add_argument("--model-path",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--sample-count",type=int,default=2,help="0 means all patches in the selected slide")
    p.add_argument("--max-new-tokens",type=int,default=512)
    p.add_argument("--seed",type=int,default=128)
    p.add_argument("--load-4bit",action="store_true")
    a=p.parse_args()
    if a.sample_count<0:raise ValueError("sample-count must be nonnegative")
    if Path(a.slide_id).name!=a.slide_id:raise ValueError("Expected a single slide folder name")
    paths=sorted(f for f in (a.patch_root/a.slide_id).iterdir() if f.suffix.lower() in {".png",".jpg",".jpeg"})
    if not paths:raise ValueError("No real input patches")
    if a.sample_count:
        count=min(a.sample_count,len(paths))
        selected=[paths[round(i*(len(paths)-1)/max(1,count-1))] for i in range(count)]
    else:selected=paths
    # Refuse reuse: a previously completed or partial run remains inspectable.
    a.output_dir.mkdir(parents=True,exist_ok=False)
    patch_dir=a.output_dir/"patches"/a.slide_id
    patch_dir.mkdir(parents=True)
    for path in selected:(patch_dir/path.name).symlink_to(path.resolve())
    slide_list=a.output_dir/"slides.txt"
    slide_list.write_text(a.slide_id+"\n",encoding="utf-8")
    original=a.project_root/"data_preparation_script/description_generation.py"
    manifest={"stage":"loading","started":time.time(),"pid":os.getpid(),"slide_id":a.slide_id,
              "available_patch_count":len(paths),"selected_patch_count":len(selected),
              "selection":"evenly spaced sorted filenames; all if sample-count=0",
              "patches":[{"name":f.name,"source":str(f.resolve()),"sha256":digest(f)} for f in selected],
              "script":str(original),"script_sha256":digest(original),"model_overlay":str(a.model_path),
              "seed":a.seed,"max_new_tokens":a.max_new_tokens,"temperature":0.2,"conversation":"pathology_v0",
              "requested_quantization":"nf4" if a.load_4bit else "none","cuda_visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES"),
              "packages":{n:importlib.metadata.version(n) for n in ["torch","torchvision","transformers","tokenizers","accelerate","peft","bitsandbytes"]}}
    save(a.output_dir/"status.json",manifest)
    import torch
    torch.manual_seed(a.seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(a.seed)
    spec=importlib.util.spec_from_file_location("pathagent_quilt_description_generation",original)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original_loader=module.load_pretrained_model
    def recording_loader(*args,**kwargs):
        tokenizer,model,processor,context=original_loader(*args,**kwargs)
        model.eval()
        tower=model.get_vision_tower()
        manifest["loaded_model"]={"class":type(model).__name__,"dtype":str(model.dtype),
            "is_loaded_in_4bit":bool(getattr(model,"is_loaded_in_4bit",False)),
            "hf_device_map":{k:str(v) for k,v in getattr(model,"hf_device_map",{}).items()},
            "vision_tower":tower.vision_tower_name,"vision_loaded":tower.is_loaded,
            "vision_dtype":str(tower.dtype),"image_processor":type(processor).__name__,"context_length":context,
            "gpu":torch.cuda.get_device_name(0),"allocated_bytes":torch.cuda.memory_allocated()}
        manifest["stage"]="generating"
        save(a.output_dir/"status.json",manifest)
        return tokenizer,model,processor,context
    module.load_pretrained_model=recording_loader
    args=argparse.Namespace(model_path=str(a.model_path),model_base=None,device="cuda",conv_mode="pathology_v0",
        temperature=0.2,max_new_tokens=a.max_new_tokens,load_8bit=False,load_4bit=a.load_4bit,debug=False,
        image_aspect_ratio="pad",image_dir=str(patch_dir.parent),output_json=str(a.output_dir/"descriptions.json"),slide_list=str(slide_list))
    try:
        module.main(args)
        results=json.loads((a.output_dir/"descriptions.json").read_text())
        actual=results.get(a.slide_id,{})
        if set(actual)!=set(f.name for f in selected):raise AssertionError("Output patch coverage mismatch")
        if not all(isinstance(s,str) and len(s.strip())>=20 for s in actual.values()):
            raise AssertionError("Empty or unexpectedly short generated description")
        if not manifest["loaded_model"]["vision_loaded"]:raise AssertionError("Vision tower not loaded")
        if manifest["loaded_model"]["is_loaded_in_4bit"]!=a.load_4bit:raise AssertionError("Quantization differs")
        manifest.update(stage="complete",finished=time.time(),output_count=len(actual),
                        output_sha256=digest(a.output_dir/"descriptions.json"),
                        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                        description_lengths={k:len(v) for k,v in actual.items()},
                        claim="Loading, real image description generation and file coverage verified; no diagnostic accuracy claim")
        save(a.output_dir/"complete.json",manifest)
        print(json.dumps({k:v for k,v in manifest.items() if k not in {"patches","description_lengths"}},indent=2),flush=True)
    except Exception as exc:
        manifest.update(stage="failed",finished=time.time(),error=repr(exc))
        save(a.output_dir/"failure.json",manifest)
        raise
    finally:save(a.output_dir/"status.json",manifest)

if __name__=="__main__":main()

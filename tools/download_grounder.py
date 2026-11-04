"""Fetch a separately versioned Qwen2.5-VL grounding candidate; no Agent replacement."""
import argparse
import concurrent.futures
import hashlib
import json
import shutil
import time
import urllib.parse
import urllib.request
from pathlib import Path


REPO="Qwen/Qwen2.5-VL-7B-Instruct"


def get_json(url):
    with urllib.request.urlopen(url,timeout=30) as response:
        return json.load(response)


def digest(path):
    value=hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda:source.read(8*1024*1024),b""):value.update(block)
    return value.hexdigest()


def prepare_manifest():
    ms=get_json(f"https://modelscope.cn/api/v1/models/{REPO}/repo/files?Revision=master&Recursive=true")
    hf=get_json(f"https://hf-mirror.com/api/models/{REPO}?blobs=true")
    hfiles={x["rfilename"]:x for x in hf["siblings"]}
    files=[]
    for item in ms["Data"]["Files"]:
        name=item["Path"]
        if item["Type"]!="blob" or "/" in name or not name.endswith((".safetensors",".json",".txt",".jinja")):
            continue
        if name.endswith(".safetensors"):
            candidate=hfiles.get(name,{})
            if candidate.get("lfs",{}).get("sha256")!=item["Sha256"]:
                raise ValueError("Publisher/mirror weight SHA256 disagreement: "+name)
        files.append({"name":name,"size":item["Size"],"sha256":item["Sha256"],"revision":item["Revision"]})
    if not any(x["name"].endswith(".safetensors") for x in files):raise ValueError("No verified model weights")
    return {"repository":REPO,"role":"independent evidence localizer candidate", "huggingface_revision":hf["sha"],
            "source":"Qwen publisher repository on ModelScope", "cross_check":"HF mirror LFS SHA256 equals publisher SHA256 for every weight shard",
            "files":files,"total_bytes":sum(x["size"] for x in files)}


def fetch(item,root):
    target=root/item["name"]
    if target.exists():
        if target.stat().st_size==item["size"] and digest(target)==item["sha256"]:
            print("VERIFIED_EXISTING "+item["name"],flush=True);return
        raise ValueError("Preserving differing existing model file: "+str(target))
    part=target.with_name(target.name+".incoming")
    query=urllib.parse.urlencode({"Revision":item["revision"],"FilePath":item["name"]})
    url=f"https://modelscope.cn/api/v1/models/{REPO}/repo?{query}"
    for attempt in range(3):
        offset=part.stat().st_size if part.exists() else 0
        try:
            if offset>item["size"]:raise ValueError("Oversized model partial")
            if offset<item["size"]:
                request=urllib.request.Request(url,headers={"Range":f"bytes={offset}-"} if offset else {})
                with urllib.request.urlopen(request,timeout=60) as response:
                    if offset and (response.status!=206 or not response.headers.get("Content-Range","").startswith(f"bytes {offset}-")):
                        raise ValueError("Server did not honor exact resume offset")
                    downloaded=offset;last=offset;started=time.monotonic()
                    with part.open("ab" if offset else "wb") as output:
                        for chunk in iter(lambda:response.read(4*1024*1024),b""):
                            output.write(chunk);downloaded+=len(chunk)
                            if downloaded>item["size"]:raise ValueError("Model response exceeds manifest size")
                            if downloaded-last>=128*1024*1024:
                                print(json.dumps({"file":item["name"],"MiB":round(downloaded/2**20),"total_MiB":round(item["size"]/2**20),"MiB_s":round((downloaded-offset)/2**20/(time.monotonic()-started),2)}),flush=True)
                                last=downloaded
            if part.stat().st_size!=item["size"] or digest(part)!=item["sha256"]:
                raise ValueError("Model size/SHA256 mismatch: "+item["name"])
            part.rename(target)
            print("TRANSFERRED_VERIFIED "+item["name"],flush=True);return
        except Exception as error:
            print(json.dumps({"file":item["name"],"attempt":attempt+1,"error":str(error)[:200]}),flush=True)
            if attempt==2:raise
            time.sleep(2**attempt)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    manifest_path=args.output/"GROUNDING_DOWNLOAD_MANIFEST.json"
    manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else prepare_manifest()
    if shutil.disk_usage(args.output).free<manifest["total_bytes"]+2*1024**3:
        raise ValueError("Insufficient free space for candidate weights")
    manifest_path.write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(json.dumps({"files":len(manifest["files"]),"bytes":manifest["total_bytes"],"revision":manifest["huggingface_revision"]}),flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda item:fetch(item,args.output),manifest["files"]))
    (args.output/"DOWNLOAD_COMPLETE.json").write_text(json.dumps({"verified":True,"files":len(manifest["files"]),"repository":REPO,"completed_unix":time.time()}),encoding="utf-8")
    print("GROUNDING_MODEL_DOWNLOAD_VERIFIED",flush=True)


if __name__=="__main__":main()

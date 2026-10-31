# -*- coding: utf-8 -*-
"""下载 PathAgent demo 所需的两个 LLM（走 hf-mirror）。"""
import os
import sys
import json
import time

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
# 禁用 xet 存储后端（cas-bridge.xethub.hf.co 在此网络不通），改走普通 HTTP 直链
os.environ["HF_HUB_DISABLE_XET"] = "1"

from huggingface_hub import snapshot_download

MODELS = [
    ("Qwen/Qwen3-4B", r"D:\PathCode\checkpoints\Qwen3-4B"),
    ("Qwen/Qwen2.5-VL-3B-Instruct", r"D:\PathCode\checkpoints\Qwen2.5-VL-3B-Instruct"),
]

def _complete(local):
    """校验 checkpoint 是否完整：index.json 里列出的每个 shard 都存在且 >1MB。"""
    if not os.path.isdir(local):
        return False
    idx = os.path.join(local, "model.safetensors.index.json")
    if os.path.isfile(idx):
        with open(idx, "r", encoding="utf-8") as f:
            data = json.load(f)
        shards = set(data.get("weight_map", {}).values())
        for s in shards:
            p = os.path.join(local, s)
            if not os.path.isfile(p) or os.path.getsize(p) < 1024 * 1024:
                return False
        return True
    # 单文件模型：直接看是否有一个够大的 .safetensors
    for f in os.listdir(local):
        if f.endswith(".safetensors"):
            if os.path.getsize(os.path.join(local, f)) > 1024 * 1024:
                return True
    return False

def main():
    for name, local in MODELS:
        if _complete(local):
            print(f"[skip] {name} 已完整存在", flush=True)
            continue
        print(f"[start] {name} -> {local}", flush=True)
        t0 = time.time()
        try:
            snapshot_download(name, local_dir=local, max_workers=4)
            print(f"[done] {name} 耗时 {time.time()-t0:.0f}s", flush=True)
        except Exception as e:
            print(f"[FAIL] {name}: {e}", flush=True)
    print("[ALL DONE]", flush=True)

if __name__ == "__main__":
    main()

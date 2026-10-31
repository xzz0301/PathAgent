# -*- coding: utf-8 -*-
"""
Stage-1 demo：为 SICAPv2 子集的每个 patch 生成 PLIP 512 维特征，存成 .npy。
  目录结构： {feature_dir}/{CASE_ID}/{x}_{y}.npy   （与 pathagent.py 的读取逻辑一致）
"""
import os
import sys
import numpy as np
from PIL import Image

PLIP_LIB = r"D:\PathCode\external\PLIP"
PLIP_CKPT = r"D:\PathCode\checkpoints\plip"
CASE_ID = "DX1_SICAP001"
DEMO = r"D:\PathCode\demo"
PATCH_ROOT = os.path.join(DEMO, "patches", CASE_ID)
FEATURE_DIR = os.path.join(DEMO, "features", CASE_ID)

sys.path.insert(0, PLIP_LIB)
from plip import PLIP  # noqa: E402


def main():
    os.makedirs(FEATURE_DIR, exist_ok=True)
    names = sorted(f for f in os.listdir(PATCH_ROOT) if f.lower().endswith(".png"))
    print(f"待编码 patch 数: {len(names)}")

    plip = PLIP(PLIP_CKPT)
    print("PLIP 已加载 (device=%s)" % plip.device)

    images = [Image.open(os.path.join(PATCH_ROOT, n)).convert("RGB") for n in names]
    embs = plip.encode_images(images, batch_size=8)
    print("特征 shape:", embs.shape, " 维数:", embs.shape[-1])

    for n, e in zip(names, embs):
        stem = os.path.splitext(n)[0]  # {x}_{y}
        np.save(os.path.join(FEATURE_DIR, f"{stem}.npy"), e)

    print(f"已写出 {len(names)} 个 .npy 到 {FEATURE_DIR}")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
Stage-1 demo 数据准备（SICAPv2 子集）：
  1) 从 SICAPv2.zip 提取单个病人的 tile（512x512 jpg）
  2) 按文件名里的 xini/yini 坐标重命名为 {x}_{y}.png（符合 PathAgent 数据契约）
  3) 生成 descriptions.json + questions.json
  （PLIP 特征由 compute_plip_features.py 单独生成）

仅用 1 个病人做最小 demo，不声称复现论文数据集。
"""
import os
import re
import json
import zipfile
from PIL import Image

ZIP = r"D:\PathCode\data\SICAPv2.zip"
PATIENT = "16B0001851"          # Gleason 4+5=9 (grade group 5)
CASE_ID = "DX1_SICAP001"        # 目录名需含 "DX1"（PathAgent load_all_vqa_pairs 的过滤条件）
MAX_TILES = 60                  # 演示用小样本，控制显存/速度

DEMO = r"D:\PathCode\demo"
PATCH_ROOT = os.path.join(DEMO, "patches")
FEATURE_DIR = os.path.join(DEMO, "features")
DESC_FILE = os.path.join(DEMO, "descriptions.json")
QUEST_FILE = os.path.join(DEMO, "questions.json")

COORD_RE = re.compile(r"_xini_(\d+)_yini_(\d+)\.(jpg|jpeg|png)$", re.IGNORECASE)

def main():
    os.makedirs(os.path.join(PATCH_ROOT, CASE_ID), exist_ok=True)

    with zipfile.ZipFile(ZIP) as zf:
        names = [n for n in zf.namelist()
                 if n.startswith(f"SICAPv2/images/{PATIENT}_") and n.lower().endswith(".jpg")]
    print(f"病人 {PATIENT} 共有 {len(names)} 个 tile")

    # 解析坐标并排序（按 y 再 x）
    parsed = []
    for n in names:
        m = COORD_RE.search(n)
        if not m:
            continue
        x, y = int(m.group(1)), int(m.group(2))
        parsed.append((n, x, y))
    parsed.sort(key=lambda t: (t[2], t[1]))  # 按 (y, x) 排序

    # 坐标去重检查（同病人不同 region 理论上不冲突，但保险起见）
    seen = {}
    dup = 0
    for n, x, y in parsed:
        key = (x, y)
        if key in seen:
            dup += 1
        seen[key] = n
    print(f"唯一坐标 {len(seen)} 个，冲突 {dup} 个")

    # 截取 MAX_TILES
    selected = parsed[:MAX_TILES]
    print(f"取前 {len(selected)} 个 tile 做 demo")

    descriptions = {}
    with zipfile.ZipFile(ZIP) as zf:
        for n, x, y in selected:
            patch_name = f"{x}_{y}.png"
            dst = os.path.join(PATCH_ROOT, CASE_ID, patch_name)
            with zf.open(n) as src, open(dst, "wb") as out:
                out.write(src.read())
            # 转成 PNG（PathAgent load_image 用 PIL，jpg 内容其实也能读，但统一成 png 更干净）
            img = Image.open(dst).convert("RGB")
            img.save(dst, "PNG")
            descriptions[patch_name] = ""  # 离线描述留空，靠 Patho-R1(替身) 在线生成
    print(f"已写出 {len(descriptions)} 个 PNG patch 到 {os.path.join(PATCH_ROOT, CASE_ID)}")

    # descriptions.json: {long_id: {patch_name: desc}}
    with open(DESC_FILE, "w", encoding="utf-8") as f:
        json.dump({CASE_ID: descriptions}, f, ensure_ascii=False, indent=2)

    # questions.json: WSI-VQA 格式 [{Id, Question, Choice, Answer}]
    questions = [{
        "Id": CASE_ID,   # 必须 == 目录名前 12 字符
        "Question": "What is the primary Gleason pattern of this prostate tissue slide?",
        "Choice": ["3", "4", "5"],
        "Answer": "4"     # 来自 wsi_labels.xlsx：Gleason_primary=4
    }]
    with open(QUEST_FILE, "w", encoding="utf-8") as f:
        json.dump(questions, f, ensure_ascii=False, indent=2)

    print(f"\ndescriptions.json -> {DESC_FILE}")
    print(f"questions.json    -> {QUEST_FILE}")
    print(f"下一步：运行 compute_plip_features.py 生成 PLIP 特征")

if __name__ == "__main__":
    main()

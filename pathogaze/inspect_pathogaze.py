# -*- coding: utf-8 -*-
"""勘验 PathoGaze1.0：确认是否含原始 WSI、公开数据能否支撑"低倍概览 + 精简诊断性眼动轨迹"。"""
import os
import json
import pandas as pd
from PIL import Image

P = r"D:\pathologyData\PathoGaze1.0"
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 60)

def show_csv(path, n=3, name=None):
    name = name or os.path.basename(path)
    print(f"\n{'='*70}\n### {name}  ({os.path.getsize(path)/1e6:.1f} MB)\n{'='*70}")
    df = pd.read_csv(path, nrows=200000)  # 只读前 20 万行提速
    print("形状(前20万行):", df.shape)
    print("列:", list(df.columns))
    print("--- 前 %d 行 ---" % n)
    print(df.head(n).to_string())
    return df

# 1) 实验1 的眼动/鼠标数据
gaze = show_csv(os.path.join(P, "P10S60T600", "GazeOnlyData.csv"))
print("\n唯一 experimentImageName 数:", gaze["experimentImageName"].nunique())
print("唯一 participantID:", sorted(gaze["participantID"].unique().tolist()))
print("EyeMovementType 取值:", gaze["EyeMovementType"].value_counts().to_dict())
for c in ["inImage", "inNavigator", "inMenu", "outOfScreen"]:
    if c in gaze.columns:
        print(f"{c} 取值:", gaze[c].value_counts(dropna=False).to_dict())
for c in ["zoomScale", "zoomMagnification"]:
    if c in gaze.columns:
        print(f"{c}: min={gaze[c].min()} max={gaze[c].max()} 唯一数={gaze[c].nunique()}")

show_csv(os.path.join(P, "P10S60T600", "Metadata.csv"), n=2)
show_csv(os.path.join(P, "ImageMetrics.csv"), n=3)
show_csv(os.path.join(P, "participantCorrespondence.csv"), n=12)

# 2) 掩码
for sub in ["TissueMasks", "TumorMasks"]:
    d = os.path.join(P, sub)
    fs = sorted(os.listdir(d))
    print(f"\n### {sub}: {len(fs)} 个 -> {fs}")
    im = Image.open(os.path.join(d, fs[0]))
    print(f"  样本 {fs[0]} 尺寸 {im.size} mode={im.mode}")

# 3) 是否存在第二个实验
print("\n### 第二个实验目录是否存在")
for cand in ["EMSVDP9D397T540", "P9D397T540"]:
    print(f"  {cand}: {os.path.isdir(os.path.join(P, cand))}")

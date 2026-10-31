# -*- coding: utf-8 -*-
"""算出"只为 PathoGaze 实验I 的 60 张切片下载 CAMELYON16 WSI 本体"需要多少体积。"""
import os
import re
import pandas as pd

C = r"D:\pathologyData\CAMELYON16"
P = r"D:\pathologyData\PathoGaze1.0"

# 1) 未下载清单：文件名 -> 字节数
sizes = {}
with open(os.path.join(C, "_images_not_downloaded.txt"), "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line or "\t" not in line:
            continue
        name, sz = line.rsplit("\t", 1)
        sizes[os.path.basename(name).strip()] = int(sz)
print("未下载清单条目数:", len(sizes))

# 2) 实验I 实际用到的 60 张（从眼动数据取，最可靠）
gaze = pd.read_csv(os.path.join(P, "P10S60T600", "GazeOnlyData.csv"),
                   usecols=["experimentImageName"], low_memory=False)
used = sorted(gaze["experimentImageName"].dropna().unique())
print("实验I 用到的图像数:", len(used))

# 3) 映射到 CAMELYON 名
mp = pd.read_csv(os.path.join(P, "ImageMapping.csv"))
m = dict(zip(mp["experimentImageName"], mp["CAMELYONImageName"]))
lab = dict(zip(mp["experimentImageName"], mp["lab"]))

rows = []
for u in used:
    cn = m.get(u)
    fn = f"{cn}.tif" if cn else None
    rows.append({"exp": u, "CAMELYON": cn, "file": fn,
                 "GB": (sizes.get(fn, 0) / 1e9) if fn else None,
                 "in_list": fn in sizes if fn else False,
                 "lab": lab.get(u)})
df = pd.DataFrame(rows)
print("\n=== 抽样 ===")
print(df.head(10).to_string(index=False))

missing = df[~df["in_list"]]
print(f"\n清单里找不到的: {len(missing)}")
if len(missing):
    print(missing.to_string(index=False))

tot = df["GB"].sum()
print(f"\n★ 这 {len(df)} 张 WSI 合计: {tot:.1f} GB (十进制) = {tot*1e9/2**30:.1f} GiB")
print(f"  单张平均: {tot/len(df):.2f} GB   最大: {df['GB'].max():.2f} GB   最小: {df['GB'].min():.2f} GB")
print(f"  按 lab 分布: {df['lab'].value_counts().to_dict()}")
print(f"  按 CAMELYON 前缀: {df['CAMELYON'].str.split('_').str[0].value_counts().to_dict()}")

# -*- coding: utf-8 -*-
"""核验 PathoGaze1.0 眼动坐标 → WSI 坐标系的映射是否成立（可行性命门）。
   并 prototype 一条"精简诊断性轨迹"。"""
import os
import numpy as np
import pandas as pd
from PIL import Image

Image.MAX_IMAGE_PIXELS = None   # 掩膜是全分辨率超大 TIFF，解除 PIL 解压炸弹上限

P = r"D:\pathologyData\PathoGaze1.0"
C = r"D:\pathologyData\CAMELYON16"
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 60)

# ---------- 1) 载入 ----------
gaze = pd.read_csv(os.path.join(P, "P10S60T600", "GazeOnlyData.csv"), low_memory=False)
im = pd.read_csv(os.path.join(P, "ImageMetrics.csv"))
mp = pd.read_csv(os.path.join(P, "ImageMapping.csv"))
print("GazeOnlyData 全量形状:", gaze.shape)

# ---------- 2) 只看落在图像内的 Fixation ----------
fx = gaze[(gaze["EyeMovementType"] == "Fixation") & (gaze["inImage"] == True)].copy()
print("inImage=True 的 Fixation 行数:", len(fx))
print("其中 ImageFixationPointX 非空:", fx["ImageFixationPointX(px)"].notna().sum())
print("其中 ImageFixationPointY 非空:", fx["ImageFixationPointY(px)"].notna().sum())

fx = fx.dropna(subset=["ImageFixationPointX(px)", "ImageFixationPointY(px)"])

# ---------- 3) 每张图的 WSI 尺寸 vs 坐标范围 ----------
dim = im.set_index("experimentImageName")[["CAMELYONImageName", "width(px)", "height(px)",
                                            "diagnosis", "tumorSize", "tumorDiameter(mm)"]]
rows = []
for img, g in fx.groupby("experimentImageName"):
    if img not in dim.index:
        continue
    W, H = dim.loc[img, "width(px)"], dim.loc[img, "height(px)"]
    if isinstance(W, pd.Series):   # 万一重复
        W, H = W.iloc[0], H.iloc[0]
    x, y = g["ImageFixationPointX(px)"], g["ImageFixationPointY(px)"]
    rows.append({
        "img": img, "CAMELYON": dim.loc[img, "CAMELYONImageName"] if not isinstance(dim.loc[img,"CAMELYONImageName"], pd.Series) else dim.loc[img,"CAMELYONImageName"].iloc[0],
        "W": W, "H": H,
        "n_fix_in_image": len(g),
        "xmin": x.min(), "xmax": x.max(), "ymin": y.min(), "ymax": y.max(),
        "x_ok": bool((x >= 0).all() and (x <= W).all()),
        "y_ok": bool((y >= 0).all() and (y <= H).all()),
    })
chk = pd.DataFrame(rows).sort_values("img")
print("\n=== 每张图的 WSI 尺寸 vs 落在图内的注视点坐标范围 ===")
print(chk.head(12).to_string(index=False))
print(f"\n所有图 x 在 [0,W] 内: {chk['x_ok'].all()}    y 在 [0,H] 内: {chk['y_ok'].all()}")
print("覆盖图像数:", len(chk), " 每图注视点数中位数:", chk['n_fix_in_image'].median())

# ---------- 4) 掩膜尺寸 vs WSI 尺寸（判断能否用掩膜当低倍底图）----------
print("\n=== 掩膜尺寸 vs WSI 尺寸（抽样）===")
for img in chk["img"].head(4):
    cn = chk.loc[chk["img"] == img, "CAMELYON"].iloc[0]
    W = chk.loc[chk["img"] == img, "W"].iloc[0]
    H = chk.loc[chk["img"] == img, "H"].iloc[0]
    found = None
    for pat in [f"{cn}_mask.tif", f"{cn}_tissue.tif"]:
        fp = os.path.join(C, "masks", pat)
        if not os.path.exists(fp):
            fp = os.path.join(C, "background_tissue", pat)
        if os.path.exists(fp):
            with Image.open(fp) as m:
                found = (pat, m.size, m.mode)
            break
    if found:
        mw, mh = found[1]
        print(f"  {img} ({cn}): WSI={int(W)}x{int(H)}  掩膜{found[0]}={mw}x{mh} m={found[2]}  "
              f"缩放比 W/mw={W/mw:.2f} H/mh={H/mh:.2f}")
    else:
        print(f"  {img} ({cn}): 未找到掩膜")

# ---------- 5) 一条"精简诊断性轨迹" prototype ----------
print("\n=== Prototype：精简轨迹（按注视时长阈值 + 时序）===")
cand = chk[chk["n_fix_in_image"] > 30].sort_values("img")
img = "C48" if "C48" in set(cand["img"]) else cand["img"].iloc[0]
g = fx[fx["experimentImageName"] == img].sort_values("relativeTimestamp(ms)")
W = dim.loc[img, "width(px)"]; H = dim.loc[img, "height(px)"]
if isinstance(W, pd.Series): W, H = W.iloc[0], H.iloc[0]
print(f"用例 {img}: WSI={int(W)}x{int(H)}  图内注视点={len(g)}  诊断={dim.loc[img,'diagnosis']}")
for thr in [0, 200, 300, 500, 800]:
    k = g[g["GazeEventDuration(ms)"] >= thr]
    print(f"  阈值 >={thr:4d}ms -> 保留 {len(k):4d} 个点 ({100*len(k)/len(g):5.1f}%)")
print("\n  前 10 个注视点（时序，含时长与倍率）:")
print(g.head(10)[["relativeTimestamp(ms)", "ImageFixationPointX(px)", "ImageFixationPointY(px)",
                  "GazeEventDuration(ms)", "zoomMagnification"]].to_string(index=False))

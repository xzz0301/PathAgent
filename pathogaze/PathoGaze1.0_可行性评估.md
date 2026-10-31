# PathoGaze1.0 可行性评估：能否做"低倍概览 + 精简诊断性眼动轨迹"

> 证据等级：`【一手】`= 我在本机数据上实测；`【二手】`= 来自项目内既有记录（`下载总结.md` / `README.pdf`）；`【待核】`= 尚未验证。
> 生成时间：本次会话。所有数字均来自实际读文件，非估算。

---

## 一、直接回答两个问题

### Q1：`D:\pathologyData\PathoGaze1.0` 里有没有原始 WSI 数据？

**没有。** `【一手】`

实测依据：
1. 全树搜 `.svs / .ndpi / .mrxs / .scn / .tif / .tiff / .dcm` → **0 个**。
2. 目录实有内容：24 个 `.mp4` 录屏（13.0 GB）、20 个掩码 `.png`、10 个 `.csv`、9 个眼动 `.tsv`、`AnnoSuite.zip`（标注工具 git 仓库，非图像）、`SharedDataProcessingCode/`、`Figures/`。
3. `README.pdf` 第 6 页自述：`CreateTissueAreaMasks.ipynb` "requires the **CAMELYON16 slide images**" —— 即图像本体本就不随包发布。
4. 而且 `D:\pathologyData\CAMELYON16\` 下**也没有** WSI 本体 —— 上一个会话**主动跳过**了（752.50 GB）`【二手】`，未下载清单在 `CAMELYON16/_images_not_downloaded.txt`（399 条）。

### Q2：PathoGaze1.0 的公开数据能否做成低倍概览图 + 精简诊断性轨迹？

**能，且关键环节已实证通过。** `【一手】`

---

## 二、为什么这条线能解决你原来的困境

你原来的问题：CoT 的数据是**已训练矫正过的**轨迹图文，找不到与之对应的**未矫正原始 WSI**，因此无法在同一张 WSI 上做"Agent 动态导航 vs 专家轨迹"的对照。

**PathoGaze1.0 这条线上不存在这个问题**，因为：

| 要素 | 情况 |
|---|---|
| 底层图像来源 | **CAMELYON16**（`ImageMapping.csv` 397 行映射：`CAMELYONImageName` ↔ `experimentImageName` ↔ `lab`）`【一手】` |
| 原始 WSI 是否公开 | **是**。CAMELYON16 为 **CC0 公有领域**，AWS S3 **免账号**可下 `【二手】` |
| 是否同一张切片 | **是**。眼动数据与 WSI 通过 `experimentImageName` 严格一一对应 |
| 需要下多少 | 实验 I 只用到 **60 张**，合计 **107.3 GB**（十进制）/ 99.9 GiB `【一手】` |
| 本机能否容纳 | D 盘可用 **279.3 GB** `【一手】` → 充裕 |

> 关键差别：CoT 的对照困境来自"专家标注轨迹没有对应的原始切片"；而 PathoGaze 的眼动记录**天然绑定在公开的 CAMELYON16 切片上**，切片可下、坐标可直接对齐。

---

## 三、实证依据（Q2 的支撑）

### 3.1 坐标系统一 —— 命门已通 `【一手】`

`P10S60T600/GazeOnlyData.csv`（201,679 行 × 24 列）中：

- 字段 **`ImageFixationPointX(px)` / `ImageFixationPointY(px)`** 是**图像空间（即 WSI 原始像素）**坐标，区别于屏幕空间的 `FixationPointX/Y`。
- 筛 `EyeMovementType == 'Fixation'` 且 `inImage == True` → **74,347 个注视点，坐标 100% 非空**。
- 用 `ImageMetrics.csv` 的 `width(px)/height(px)` 逐图校验：**60 张图，x∈[0,W]、y∈[0,H] 全部成立，无一例外**。
- 每图图内注视点数中位数 **1229**（最少 C19: 547，最多 C14: 2671）。

**结论：注视点坐标可直接落到 WSI 像素坐标系，无需任何标定。**

### 3.2 掩膜与 WSI 完全同尺寸 `【一手】`

抽样 4 张，`CAMELYON16/masks/{test_xxx}_mask.tif` 的尺寸与 `ImageMetrics.csv` 的 WSI 尺寸**缩放比恰为 1.00**：

| 图像 | WSI 尺寸 | 掩膜尺寸 | 缩放比 |
|---|---|---|---|
| C1 (test_001) | 86016×89600 | 86016×89600 | 1.00 |
| C10 (test_051) | 96768×211968 | 96768×211968 | 1.00 |
| C11 (test_052) | 97792×219648 | 97792×219648 | 1.00 |
| C12 (test_053) | 97792×218624 | 97792×218624 | 1.00 |

掩膜语义 `【二手】`（CAMELYON16 README）：**1 = normal tissue，2 = tumor**。
→ 可**直接判断每个注视点是否落在肿瘤区**，这是"诊断性"的客观锚点。

### 3.3 轨迹精简可量化 `【一手】`

用例 **C48**（WSI 97792×221184，`diagnosis=Malignant`，图内注视点 1178）：

| 注视时长阈值 | 保留点数 | 保留比例 |
|---:|---:|---:|
| ≥0 ms | 1178 | 100% |
| ≥200 ms | 650 | 55.2% |
| ≥300 ms | 323 | 27.4% |
| ≥500 ms | 69 | 5.9% |
| ≥800 ms | 11 | 0.9% |

→ "精简后的具有诊断性的轨迹"可以用**阈值 + 空间聚类 + 肿瘤区重叠**三步稳定产出。例如 500ms 阈值下 69 个点，聚成约 5–15 个 ROI，正好画"框→箭头→框"。

### 3.4 可区分"概览浏览"与"细节查看" `【一手】`

- `zoomMagnification`：范围 **0.68 – 153.96**（532 个不同值）
- `zoomScale`：**0.0065 – 1.0**（344 个不同值）
- 实测 C48 的**前 10 个注视点全部 `zoomMagnification = 1.0`**（概览阶段），之后才放大。

这正好对应论文的 `totalOrientingFixations`（开始鼠标导航前的定向注视）概念 —— **"先低倍扫全局、后高倍查细节"的过程可以被数据还原**，而这正是你想让 Agent 导航轨迹去对标的东西。

### 3.5 "诊断性"有真值可依 `【一手】`/`【二手】`

| 来源 | 内容 | 状态 |
|---|---|---|
| `CAMELYON16/masks/` | 399 个全分辨率掩膜（1=正常/2=肿瘤），8.77 GB | ✅ 已下载 |
| `CAMELYON16/annotations/` | 160 个 ASAP XML 肿瘤多边形 | ✅ 已下载 |
| `PathoGaze1.0/P10S60T600/Metadata.csv` | 600 行 = 60 图 × 10 人，含医生**自己画的 `annotations`(JSON)** 与 `diagnosis`/`correct` | ✅ 已下载 |
| `ImageMetrics.csv` | 397 行，含 `tumorArea` / `tumorDiameter(mm)` / `tumorSize` | ✅ 已下载 |

---

## 四、技术路线（两条，按"训练"的两种理解）

### 路线 A：规则化精简（不需要训练，立即可做）

```
GazeOnlyData.csv
  → 筛 inImage==True & Fixation & GazeEventDuration>=阈值
  → 按 relativeTimestamp 排序（得到时序）
  → 空间聚类（DBSCAN / 网格）→ 得到 ROI 序列
  → 用 CAMELYON16 mask 标注每个 ROI 是否含肿瘤（诊断性权重）
  → 结合 zoomMagnification 区分概览/细节层
  → 低倍底图（WSI thumbnail 或掩膜降采样）上画 框→箭头→框
```

产出：**每张 WSI 一张轨迹图 + 一份 ROI 序列表**。

### 路线 B：学习式精简（需训练，研究贡献）

把"哪些注视构成诊断性轨迹"当监督信号来学：
- 监督来源候选：① 注视点是否落在肿瘤掩膜内；② 是否与 `correct` 诊断相关；③ 医生自己在 `Metadata.annotations` 里画的框；
- 模型候选：行为序列模型（类似 CoT 论文的 Behavior Predictor）、或 ROI 选择策略网络；
- 价值：可以把学到的"诊断性 ROI 选择策略"与 **PathAgent 的 Navigator 策略**做直接对照 —— 这正是你要的"可解释性论证"。

> 两条路线可以串起来：A 先做出可视化与基线，B 再在其上做方法创新。

---

## 五、边界与风险（必须说清）

1. **低倍底图需要真 H&E**。已下载的掩膜只有轮廓（`mode=L`/`RGBA`），**不是染色图**。要出论文那种"低倍全景 + 轨迹"的图，需下 60 张 WSI（107.3 GB）。
   **过渡方案**：先用全分辨率掩膜当底图（几何 1.00 一致），把轨迹几何与精简算法跑通，WSI 下完再换真 H&E —— 代码零改动，只换底图。
2. **实验 II 的处理后数据缺失** `【二手】`：OSF 存档只有实验 I（`P10S60T600`），实验 II（`EMSVDP9D397T540`）的 CSV 不存在。若要用实验 II，需自己跑 `SharedDataProcessingCode/` 从 `RawData/` 重建。
3. **`TissueMasks`/`TumorMasks` 只有各 10 个**（而实验 I 用了 60 张）`【一手】` —— 这两个目录不完整，不能作为全覆盖的掩膜来源；**应改用 `CAMELYON16/masks/`（399 个，完整）**。
4. **论文表 6 示例数字与发布数据对不上** `【二手】` —— 引用论文数字时需注意，应以发布数据为准。
5. **`multiresolutionimageinterface`(mir) 装不上** `【二手】` —— `CreateTumorMasks.py` 开箱不可用；但**不需要它**，官方掩膜已下载。
6. **眼动数据含噪声**：存在 `outOfScreen`(196)、`inNavigator`(693)、`inMenu`(7360) 等非图像注视，必须先过滤 `inImage==True` 才可用。

---

## 六、可执行的第一步

**只下实验 I 需要的 60 张**（约 107.3 GB，CC0 免账号）：

```bash
# 逐张 HTTP（无需任何工具，浏览器也可）
# https://camelyon-dataset.s3.us-west-2.amazonaws.com/CAMELYON16/images/test_001.tif
# ... （60 个文件名可由 ImageMapping.csv 的 CAMELYONImageName 列筛出）

# 或 AWS CLI 批量（推荐，支持断点续传）
aws s3 cp --no-sign-request s3://camelyon-dataset/CAMELYON16/images/test_001.tif .\images\
```

---

## 七、待你确认的问题

1. 走**路线 A（规则化，快）**还是**路线 B（学习式，有创新点）**，还是 A 先行、B 跟进？
2. 低倍底图用**真 H&E（先下 107 GB）**，还是**先用掩膜过渡**把算法打通？
3. 轨迹图的"精简粒度"目标是什么观感 —— 每图 5–10 个框，还是 10–20 个框？
4. 对照设计：是"**Agent 导航轨迹 vs 单个专家轨迹**"，还是"**Agent vs 多位专家的共识/分歧区域**"？（后者更贴近 CoBRA 的方向）

---

## 附：本次核验用到的脚本

| 脚本 | 作用 |
|---|---|
| `D:\PathCode\pathogaze\inspect_pathogaze.py` | 目录勘察 + 各 CSV 字段/取值统计 |
| `D:\PathCode\pathogaze\verify_gaze_feasibility.py` | 眼动坐标 → WSI 坐标校验、掩膜尺寸比对、精简轨迹 prototype |
| `D:\PathCode\pathogaze\calc_wsi_download_size.py` | 算 60 张 WSI 的下载体积 |

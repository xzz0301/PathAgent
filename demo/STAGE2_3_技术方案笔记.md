# Stage 2 / Stage 3 技术方案笔记（供后续讨论，暂不执行）

> 状态：仅技术思路，未实现、未验证。每条"技术手段"标注 `【待核】` 表示尚未在本机验证可用。
> 目的：把"如何用可视化轨迹让 Agent 的导航过程可解释/可追溯"想清楚，等 Stage 1 跑通后再落地。

---

## 背景：为什么要画"浏览轨迹"

PathAgent / Pathology-CoT 这类 WSI Agent 的核心动作是**在整张病理切片上多步导航**（先看某块区域 → 判断证据不足 → 放大/换区域再看）。文字版 `process`（JSON 里的导航日志）对人不直观。把它画成 **低倍全景图 + 框1 → 箭头 → 框2 → …** 的轨迹图，是"证据链可视化"的第一步——让黑盒模型"看过了哪里、按什么顺序看"变成一张可审阅的图。

---

## Stage 2：处理 Pathology-CoT 的预览切片 + conversation.json，合成轨迹图

**输入（来自 Pathology-CoT 论文数据）：**
- 预览切片（preview slices）：低倍 WSI 概览图（或可直接降采样得到的低倍图）`【待核：需确认该数据实际字段名/格式】`
- `conversation.json`：Agent 浏览文本记录，含每次"看哪个区域"的坐标/倍率 `【待核：需确认坐标字段】`

**技术路线（分 4 步）：**

1. **解析导航序列** —— 读 `conversation.json`，抽出一条有序的 `[step_i → (region_i)]` 列表，region 含坐标 `(x, y)`、尺寸 `(w, h)`、倍率 `zoom`。用 `json` + 简单的字段映射即可，必要时写一个小 parser 记录每步的 patch id 与坐标。
   - 技术：Python `json` / `pandas`。

2. **得到低倍全景底图** —— 两条路：
   - (a) 有原始 WSI：用 **OpenSlide**（`openslide-python`）读金字塔 TIFF，取低倍 level 渲染成 RGB 底图（或 `tifffile` + 自定义降采样）。
   - (b) 只有 preview slices：直接用论文提供的低倍图做底图。
   - 技术：`openslide-python` / `tifffile` / `Pillow` `【待核：openslide 在 Windows 上的二进制 DLL 需单独装】`

3. **坐标映射** —— 把高倍 patch 坐标映射到底图像素坐标：`底图坐标 = 高倍坐标 / downsample_ratio`（注意 x/y 轴方向与原点、以及可能的旋转/偏移校正）。
   - 技术：纯数值换算，无外部依赖；关键是要拿到准确的 downsample 倍率。

4. **画轨迹** —— 在底图上叠加：
   - 每个 visited region 画一个**矩形框**（box）；
   - 框内标注**步骤序号** 1、2、3…（按访问顺序）；
   - 相邻框之间画**箭头**（1→2→3…）；
   - 用颜色/透明度区分：早期探索 vs 最终作答区域高亮。
   - 技术：**matplotlib**（最通用，箭头 `annotate`/`FancyArrowPatch`、矩形 `Rectangle`）+ 导出 PNG；或 **OpenCV**（`cv2.rectangle` + `cv2.arrowedLine`）更轻量。

**产出：** 每例一张 `trajectory.png`，与导航日志并列，便于审阅 Agent"看没看对地方"。

---

## Stage 3：改 PathAgent，让动态导航也输出轨迹图

**关键点：PathAgent 本身就是多步导航循环**（Navigator 用 PLIP 相似度挑 top-k patches → Perceptor 描述 top-5 → Executor 决定 predict/self-reflect/zoom/find-missing → 最多 T=5 步），因此每一步"看的是哪些 patch"天然就是一条轨迹，不需要新造数据，只需**在循环里埋点 + 记录坐标 + 画图**。

**技术路线：**

1. **在循环里埋点记录轨迹** —— 在 `pathagent.py` / `models/inference.py` 的每步导航处，把"本步选中的 patch 集合"（patch 名 `{x}_{y}.png` → 直接解析出 `(x, y)` 坐标）追加到一个 `trajectory` 列表，随最终 JSON 一起保存。
   - 技术：复用 `extract_coords_from_name`（已存在），改动很小（纯日志）。

2. **构造低倍底图** —— PathAgent 用的是预切 tile（CLAM 输出，无原始 WSI）。底图两条路：
   - (a) 若有原始 WSI：OpenSlide 降采样（同 Stage 2）；
   - (b) 只有 tiles：把 tiles 按坐标拼成一个 **网格蒙太奇（grid montage）**，作为"全景底图"——坐标天然对齐，无需 OpenSlide。
   - 技术：`Pillow`/`numpy` 拼图，或 `openslide-python` `【待核】`。

3. **映射 + 画图** —— 与 Stage 2 相同：每个 step 的 visited patch 画框 + 序号 + 箭头，最终作答区域高亮。
   - 技术：matplotlib / OpenCV。

4. **集成到输出** —— 在 `save_dir` 每例的 JSON 旁多存一张 `trajectory.png`，做到"答案 + 文字过程 + 轨迹图"三者可一起审阅。

---

## 两阶段共用的技术清单（一句话汇总）

| 环节 | 首选技术 | 备选 | 待核点 |
|---|---|---|---|
| 解析导航记录 | Python `json`/`pandas` | — | 数据实际字段名 |
| 低倍底图 | OpenSlide（`openslide-python`） | `tifffile` 降采样 / tile 网格拼图 | Windows 需 openslide 二进制 DLL |
| 坐标映射 | 纯数值换算 | — | 确切的 downsample 倍率与方向 |
| 画框/箭头/序号 | matplotlib | OpenCV / PIL | — |
| 埋点记录轨迹 | 复用 `extract_coords_from_name` | — | — |

## 待与用户确认的问题（先写下来，不擅自决定）

1. 轨迹图的**底图分辨率/倍率**用哪一档（1.25x / 2.5x / 更低）？
2. 轨迹**方向**：只画"看过哪些 patch"，还是也要体现 zoom in/out 的层级？
3. 最终输出是**每例一张图**，还是要合成**多例并排**的对比图（供 CoBRA 盲区检测用）？
4. Stage 2 的 Pathology-CoT 数据：预览切片与 conversation.json 的**确切字段**需拿到数据后再核对（现在是"待核"）。

# PathAgent 与专家同片轨迹 demo V2

当前版本：2026-10-09 双流程 V2。PathoGaze 论文没有证明 viewport 或 gaze 临床价值孰高，因此两条分析并列、独立调用，不拼成一条路径。

|样例/P1|视口唯一点 / 合格访问|眼动唯一点 / 合格访问|
|---|---|---|
|C1 / test_001|5 / 5|12 / 19|
|C48 / test_029|35 / 43|54 / 55|
|C6 / test_036|16 / 19|21 / 21|

视口 ThT=60ms，眼动 ThT=120ms，由 P1 六十试次和原始采样节奏校准。几何、有效区域、可靠质量与连续有效停留共同成立；无首尾、缩放或固定转角豁免。同区域同倍率一个真实代表点，回访引用原节点保存在 visits.csv。12 的单位与距离方向见设计。

完整技术设计：TECHNICAL_DESIGN.md。交付目录：`C:/Users/14910/Documents/ChatGPT/病理医学研究/outputs/PathAgent轨迹对照_双流程V2_20261009`。

## 专家运行

在项目根目录按以下顺序调用，依赖 D:/PathCode/.venv310 和本机 PathoGaze 原始数据。已保存原始配准输入，不需为本三例重新下载数据。

```powershell
& 'D:\PathCode\.venv310\Scripts\python.exe' -X utf8 trajectory_demo\run_dual_v2.py --prepare
& 'D:\PathCode\.venv310\Scripts\python.exe' -X utf8 trajectory_demo\run_dual_v2.py --modality viewport
& 'D:\PathCode\.venv310\Scripts\python.exe' -X utf8 trajectory_demo\run_dual_v2.py --modality gaze
& 'D:\PathCode\.venv310\Scripts\python.exe' -X utf8 trajectory_demo\render_dual_v2.py
& 'D:\PathCode\.venv310\Scripts\python.exe' -X utf8 trajectory_demo\verify_dual_v2.py
```

运行全套专家测试时先切换到 trajectory_demo，再执行 `D:/PathCode/.venv310/Scripts/python.exe -m unittest discover -s tests`，本轮35项通过。

输出为 `trajectory_demo/runs/dual_v2_20261009/{viewport,gaze}`，每例包含 nodes、visits、event_audit、edges、result、sensitivity。原始时钟配准输入保存在 runs/video_audit_20261009，仅其 full/audit/raw 文件作为 V2 输入，其旧 ThT150 精简与旧对照指标不再用于当前报告。

## Agent 运行与真实结果

`D:/PathCode/demo/run_c1_demo.ps1` 默认为最低1轮。C1 原始策略实际观察5块后预测 Benign，真值 Malignant。`-MinimumRounds 2` 是外部最低覆盖干预，实际追加3块后共8块，仍预测 Benign。本轮没有自然发生高分辨率 zoom。不能据单例推导临床效果。

独立64候选位于 D:/PathCode/demo/c1_independent，未使用专家或肿瘤真值。Agent中心为level-0原点+1024，而不是+256。512px读取下采样4，实际读取MPP约0.909，名义10×与专家相对zoom含义不同。

三张完整 WSI 已下载至 D:/pathologyData/CAMELYON16/images。PathoGaze 原始数据在 D:/pathologyData/PathoGaze1.0。

## 版本说明

V1 的150ms、自动保留首尾/倍率、固定转角、去重节点顺序指标和旧点数仅留历史。原 CSV-only CLI 供历史审计，V2 三例主入口是 run_dual_v2.py。V2只对P1三例完成原始质量重建，参数使用P1六十试次分布；第二实验远端公开但本轮未完整处理。

Agent 运行入口：D:/PathCode/demo/run_c1_demo.ps1。专家代码位于上述项目根目录的 trajectory_demo。

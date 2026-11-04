# 仓库范围

本仓库基线保存PathAgent源码、模型加载与权重准备、Quilt预处理、WSI读取、WSI-VQA复现与评价、运行证据记录及必要测试。

|内容|处理|
|---|---|
|PathoGaze、PEAN和专家眼动处理|移出Git跟踪，本地原文件保留|
|历史demo、SICAP/CAMELYON试验、临时模型定位试验|移出Git跟踪，本地保留|
|旧分析、迁移过程和个人学习笔记|移出Git跟踪，本地保留|
|`external/PLIP`|保留，是Navigator的固定源码依赖|
|`trajectory.py`、ROI展示工具|保留，是当前Agent运行证据记录与展示依赖|
|`configs/local_8gb.json`|保留，仅用于显式本地运行和原配置回归；论文复现使用服务器正式模型配置|
|`tools/prepare_plip_features.py`|保留，将原通用特征编码器改为准确命名；计算方法不变|

提交前在仓库根目录运行：

```bash
python tools/check_repository_scope.py
```

该检查针对Git索引中的文件，允许本地继续留存归档，但阻止它们随批量提交重新进入基线。需要新增文件时先检查它是否服务于当前基线，并显式更新允许范围。

本次采用正常清理提交。历史提交仍可追溯，没有重写Git历史。服务器正在运行的实验所需旧文件保留在服务器；清理不会删除运行数据或中断任务。后续`dev_trajectory`阶段按授权另行扩展功能与仓库范围。

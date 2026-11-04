# PathAgent论文对齐与WSI-VQA基线

本轮按用户选定的七项对齐。原项目快照commit `7ddc437`；开发分支 `codex/paper-aligned-wsi-vqa`。所有开发仍在唯一项目，不复制demo。

|改动|位置|依据/行为|
|---|---|---|
|独立方法配置|pathagent.py:18、configs/wsi_vqa_paper.json|public_code保留，paper_aligned显式选择|
|真实5倍、无64候选上限|tools/prepare_wsi_vqa.py:29|从扫描物镜换算512px@5x视野；全组织网格20%覆盖，无截取Top64|
|ceil10%/5%|pathagent.py:250、navigation.py:16/177|按论文比例向上取整；初选固定数禁止用于本配置|
|问题相关形态描述|models/inference.py:327|使用§3.1原句；Perceptor不传诊断选项，保留问题上下文|
|缺失信息高倍查询|navigation.py:22/148|每条计划的missing_info传PLIP；不得用原问题替代|
|逐区域倍率列表|models/inference.py:167、navigation.py:22|图3行为；只接受真实ROI ID，由程序映射实际坐标，拒绝未知ID和无效倍率|
|高倍后复核|navigation.py:154–167|下一轮评价累计证据；最多5次评价。采用图例口径，和算法1的break存在已声明差异|
|完整证据记录|navigation.py:116、tools/evaluate_wsi_vqa.py|每轮状态/原始判断/请求计划与实际执行分开；只展示三例|
|大数据预检查|models/input_validation.py|同一WSI只检查一次，避免735题重复打开相同切片|
|避免反复读描述总文件|pathagent.py|使用已加载的描述字典，数据语义不变|
|持久运行|tools/run_wsi_vqa.py|文件锁防重启；86片准备、三GPU非量化Quilt、PLIP特征、全测试推理与评价|
|统计|tools/evaluate_wsi_vqa.py|390闭题：兼容SequenceMatcher口径与严格口径并列，空/缺失算错；345开题：BLEU1–4/METEOR/ROUGE-L|

## 论文未公开项的明确选择

没有捏造Trident原实验参数：当前组织候选为饱和度Otsu+闭操作、512px@5x全片网格、组织占比20%，边缘增加覆盖原点。与论文精确预处理等价性尚未获得作者确认。逐区域计划与高倍复核由用户明确选择论文图3行为，因此本配置属于论文对齐重建，不冒称公开原源码完全相同。

Quilt使用原权重非量化FP16；Patho-R1/Qwen3使用原权重BF16。最多5轮自然早停，不强迫长轨迹。未加专家轨迹新功能，不换模型。

## 数据与展示

服务器 `/data/dataset/users/xuzhaozheng/wsi-vqa` 已核对86WSI/735唯一QA：390闭题、345开题；mapped metadata与原始QA逐项相同，86路径存在。推理文件Answer全部WITHHELD；reference仅评估器读取。

案例在推理前固定为TCGA-BH-A202、TCGA-D8-A27M、TCGA-LD-A74U各自的首个测试问题，未按预测结果挑选。只展示总统计与三个案例；机器记录保留所有问题以支持复核。失败仍留在固定分母。

## 验证与Git阶段

本地32项测试通过，包括多区域不同倍率、缺失信息查询、高倍重新评价、预算耗尽、原配置回归、几何与既有专家算法。服务器需核对同步SHA并运行相同测试后启动。

完整测试结果尚未产生。基线冻结标签与dev_trajectory分支在复现验收完成后创建，不能在等待时提前宣告冻结。实验未完成状态、耗时、失败原因记录于服务器artifacts/wsi_vqa_paper_20261104/status.json。

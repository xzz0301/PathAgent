# PathAgent 三份独立分析

1. [原版PathAgent代码与论文对应分析](D:/PathCode/docs/agent_analysis_20261019/01_原版PathAgent代码与论文对应分析.md)：只讲原版，重点是三角色、Agent状态和调用、论文对应、输入输出及外围预处理。
2. [本地改动代码分析](D:/PathCode/docs/agent_analysis_20261019/02_本地改动代码分析.md)：只讲本地扩展，涵盖轨迹、几何、运行适配、策略干预、数据来源、代码审查及专家导航改进。
3. [双4090运行原版PathAgent的资源与部署分析](D:/PathCode/docs/agent_analysis_20261019/03_双4090运行原版PathAgent的资源与部署分析.md)：按两张4090、每卡24GB分析，包含BF16分卡、显存预算、CPU卸载、量化和离线流程。

配套材料：

- [原版Agent核心源码摘录](D:/PathCode/docs/agent_analysis_20261019/原版Agent核心源码摘录.md)：真实原版源码，按Agent控制流和模型接口集中提炼。
- [函数定位索引](D:/PathCode/docs/agent_analysis_20261019/函数定位索引.md)：原版与本地扩展分别列出函数签名和源码行号。
- [原版与本地入口差异](D:/PathCode/docs/agent_analysis_20261019/pathagent_demo_vs_original.diff)、[本地与服务器入口差异](D:/PathCode/docs/agent_analysis_20261019/pathagent_server_vs_demo.diff)。
- [代码核对清单](D:/PathCode/docs/agent_analysis_20261019/代码核对清单.json)：提交、SHA256、模型参数、特征范数和已有结果核对。
- [证据生成脚本](D:/PathCode/docs/agent_analysis_20261019/build_analysis_evidence.py)：重新生成摘录、差异、函数索引和代码清单。

论文依据为arXiv:2511.17052v1；模型参数依据为本地2026-10-15仓库元数据快照及checkpoint配置。分析日期2026-10-19。此次新增分析材料，未修改推理源码或重跑模型；双4090运行区间为资源规划估算。

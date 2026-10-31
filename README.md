# [ECCV 2026] PathAgent: Toward Interpretable Analysis of Whole-slide Pathology Images via Large Language Model-based Agentic Reasoning 
[![Project Page](https://img.shields.io/badge/Project-Page-green)](https://github.com/G14nTDo4/PathAgent)
[![arXiv](https://img.shields.io/badge/arXiv-2511.17052-b31b1b.svg)](https://arxiv.org/abs/2511.17052)
[![Paper](https://img.shields.io/badge/Paper-PDF-red)](https://arxiv.org/pdf/2511.17052.pdf)

<div align="center">

</div>

Official code implementation for paper "PathAgent: Toward Interpretable Analysis of Whole-slide Pathology Images via Large Language Model-based Agentic Reasoning"

>Jingyun Chen, Linghan Cai, Zhikang Wang, Yi Huang, Songhan Jiang, Shenjin Huang, Hongpeng Wang, Yongbing Zhang


## 本地开发版本与上游差异（dev，2026-10-31）

本分支以 `8c2abee2cd18ca2e2c839742c9296bd8a41738c5` 为上游代码基线，
记录本地整合、可复现配置和轨迹观测改动。以下是本地新增或修改内容；
后面的论文介绍与原始使用流程保留供对照。本分支不能仅凭单例测试称为论文结果复现。

| 范围 | 本地改动 | 对实验的影响 |
| --- | --- | --- |
| `pathagent.py` | 合并为唯一 Agent 推理入口；导入模块时不解析 CLI、不加载权重；支持 JSON 配置与命令行覆盖 | 配置路径相对项目根目录解析，可明确选择运行方案 |
| `models/runtime.py`、`configs/` | 分离模型加载；服务器配置使用 Qwen3-4B、Patho-R1-7B、PLIP，语言模型 BF16、无量化；本地 8GB 配置显式使用 3B 替代模型与 NF4 | 本地替代模型结果与服务器原模型结果分别报告；记录实际设备、dtype、权重路径和显存 |
| `models/input_validation.py` | 检查模型文件、Patho-R1 架构、问题/图块/特征/几何；默认拒绝空通用描述；`--check-inputs` 仅检查文件 | `--allow-empty-descriptions` 仅为显式连通性测试开关 |
| `trajectory.py` | 记录导航选择、观察起止、逐轮判断、最终回答；导出事件 JSONL 与轨迹 CSV/JSON/PNG | 坐标、图块尺寸、倍率、耗时及描述可追溯；观察 ROI 不等于肿瘤标注 |
| 导航循环修复 | 精确切片 ID 优先、歧义拒绝；单候选相似度保持一维；保留补检历史；放大观察遵循 token 预算 | 默认仍为初选 10%、补检 5%、最多 5 轮；已修复 `torch.device` 元数据无法 JSON 序列化的问题 |
| 可控实验参数 | 增加随机种子、固定初选数、轮数/输出预算与 `min_observation_rounds` | 最小轮数默认 1；调高属于外部覆盖干预，会记录原判断与干预事件 |
| `demo/`、`pathogaze/`、`tools/` | 保存数据准备、PLIP 特征、轨迹检查、PathoGaze 探查和分卷权重校验工具 | 部分历史工具仍带本机路径；运行前按自己的数据位置配置 |
| `tests/` | 主循环分支、坐标、设备序列化和输入保护回归 | 使用临时合成输入和模型响应替身，不需要下载模型或真实切片 |
| `external/PLIP` | 以 Git submodule 固定第三方 PLIP 源码版本 | 权重仍需单独准备；不把第三方 Git 历史复制进主项目 |

**保留的上游逻辑与现有限制：** `models/inference.py` 中的推理提示词与模型调用函数
未作本地修改。当前 zoom 仍裁剪已有 patch，不会重新读取 WSI 的更高分辨率层；
运行元数据标为 `crop_existing_patch_not_higher_resolution_wsi`。
`docs/analysis_*`、`docs/agent_analysis_*` 和部分 demo 笔记是历史分析，
其中旧入口、旧参数与旧机器状态以日期为准，当前入口以本 README 为准。

### 获取与运行

```bash
git clone --branch dev --recurse-submodules https://github.com/xzz0301/PathAgent.git
cd PathAgent
# 已有工作副本：git submodule update --init --recursive
python -m pip install -r requirements.txt
python -m pip install matplotlib
```

先按 GPU 与驱动安装兼容的 PyTorch/torchvision；上述依赖属于主 Agent 环境。
Quilt-LLaVA 使用独立的兼容环境，不能将其旧版 Transformers 依赖装进主环境。
模型与数据不在 Git 中：将原始权重放入 `checkpoints/`（或在配置中填写实际路径），
准备配套图块、PLIP 特征、通用描述、问题及几何文件，再运行：

```bash
python pathagent.py --config configs/server_original.json \
  --descriptions_file /path/to/descriptions.json \
  --questions_file /path/to/questions.json \
  --feature_dir /path/to/features --patch_root /path/to/patches \
  --geometry_file /path/to/geometry.json --save_dir /path/to/new_run \
  --check-inputs
# 文件检查通过后，去掉 --check-inputs 执行推理。
python -m unittest discover -s tests -v
python -m unittest discover -s demo -p test_trajectory.py -v
```

`server_original.json` 默认 Executor / Perceptor / PLIP 分别使用 CUDA 0 / 1 / 2，
可以通过配置或 CLI 修改；`local_8gb.json` 是单独的本地替代方案。
每次实验保留 Git commit、配置、模型版本、输入清单及独立结果目录。

### 已有验证范围

- 2026-10-23：6000D 上原 Qwen3-4B、Patho-R1-7B、PLIP 的单例加载、导航与轨迹导出通过。
- 此前 C1 验证仅使用 64 个真实候选图块及允许空通用描述的 smoke 配置，不能代表完整预处理或正式评测。
- 2026-10-24：Quilt-LLaVA 与其 CLIP 视觉塔权重完成文件校验；截至本次基线提交，专用环境与真实图块描述仍待验收。
- Git 仅保存源码、配置和说明；权重、切片、生成图块、特征、实验结果、虚拟环境和凭据均不提交。

早期合并过程见 [合并与迁移说明](docs/consolidation_20261022/合并与服务器迁移说明.md)。
该文档记录 2026-10-22 当时的状态；其中模型检查 JSON 等机器记录留在本地，不随源码发布。

## Overview

PathAgent is the first training-free interactive agent specifically designed for whole-slide pathology image (WSI) analysis. By coordinating off-the-shelf pathology models through an agent, it yields traceable decisions and competitive accuracy, suggesting a pragmatic route of computational pathology.

The contributions of PathAgent can be summarized in three aspects:

1. Dynamic analytic Logic: We replace single-step reasoning with Multi-Step Reasoning in the Executor. This mechanism can construct analytic logic and dynamically provide guidelines to retrieve task-relevant information.
2. Adaptive Magnification: PathAgent can adaptively select an appropriate scale based on the analytic state, generating more refined visual evidence.
3. Enhanced Evidence Retrieval: We improve the accuracy of evidence capture by simplifying the query strategy of the Navigator.

![architecture](./assets/CaseStudy.png)
<p align="center"><i>Figure. Illustration of PathAgent's inference procedure.</i></p>

## 1. Installation 

```bash
git clone https://github.com/G14nTDo4/PathAgent.git
cd PathAgent
conda create -n pathagent python=3.9 -y
conda activate pathagent

# Install PyTorch (specify CUDA version)
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## 2. Data Preparation
The following steps take the WSI-VQA dataset as an example.

### 2.1 Patch Generation
Copy data_preparation_script/coordinate_generation.py to the [CLAM](https://github.com/mahmoodlab/CLAM) project directory, and use the following code to generate the patches.
```bash
cd CLAM_PROJECT_DIRECTORY
python coordinate_generation.py \
  --source DATA_DIRECTORY  \
  --save_dir RESULTS_DIRECTORY \
  --preset tcga.csv \
  --step_size 4096 \
  --patch_size 4096 \
  --patch \
  --seg \
```

The DATA_DIRECTORY is the storage directory for svs files.
```bash
DATA_DIRECTORY/
	├── slide_1.svs
	├── slide_2.svs
	└── ...
```

The above command will segment every slide in DATA_DIRECTORY and generate the following folder structure at the specified RESULTS_DIRECTORY:
```bash
RESULTS_DIRECTORY/
	├── masks
    		├── slide_1.png
    		├── slide_2.png
    		└── ...
	├── patches
    		├── slide_1.h5
    		├── slide_2.h5
    		└── ...
	├── stitches
    		├── slide_1.png
    		├── slide_2.png
    		└── ...
	└── process_list_autogen.csv
```

Using the h5 file in RESULTS_DIRECTORY, we can extract the patch from WSI.
```bash
python data_preparation_script/patch_generation.py \
  --h5_dir RESULTS_DIRECTORY/patches \
  --slide_dir DATA_DIRECTORY \
  --output_root RESULTS_DIRECTORY/patches_output \
  --patch_size 4096
```

The above command will generate the following folder structure at the specified RESULTS_DIRECTORY/patches_output:
```bash
RESULTS_DIRECTORY/patches_output/
	├── slide_1
    		├── 0_23904.png
    		├── 4096_28000.png
    		└── ...
	├── slide_2
    		├── 4128_7296.png
    		├── 4128_11392.png
    		└── ...
	└── ...
```

### 2.2 Description Generation
Before generating the description, we first need to modify the Quilt-LLaVA system prompt, replacing quilt-llava/llava/conversation.py with data_preparation_script/conversation_pathology_v0.py. Please note that the final file name is still conversation.py. The different name is only to distinguish the file from the file in the [Quilt-LLaVA](https://github.com/aldraus/quilt-llava) project.

Multiple processes can be executed simultaneously to speed up the description generation process, which is also very user-friendly for users with limited VRAM. Only the description files for each block need to be concatenated at the end. Here, we divide all WSIs in the test set into four equal parts. Readers can adjust the number of parts as needed.
```bash
python data_preparation_script/split_files.py \
  --image_dir RESULTS_DIRECTORY/patches_output \
  --save_dir RESULTS_DIRECTORY/split_name \
  --num_splits 4
```

The above command will generate the following folder structure at the specified RESULTS_DIRECTORY/split_name:
```bash
RESULTS_DIRECTORY/split_name/
	├── slides_part1.txt
	├── slides_part2.txt
	└── ...
```

Copy data_preparation_script/description_generation.py and data_preparation_script/multi_description_generation.sh to the Quilt-LLaVA project directory, and use the following code to generate the description.
```bash
cd Quilt_LLaVA_DIRECTORY
bash multi_description_generation.sh
```

The above command will generate the following folder structure at the specified RESULTS_DIRECTORY/desc:
```bash
RESULTS_DIRECTORY/desc/
	├── patches_descriptions1.json
	├── patches_descriptions2.json
	├── patches_descriptions3.json
	└── patches_descriptions4.json
```

The following command is used to merge all JSON files in the desc folder into a single JSON file.
```bash
python data_preparation_script/merge_json_results.py \
  --input_dir RESULTS_DIRECTORY/desc \
  --save_dir RESULTS_DIRECTORY/desc/patches_descriptions.json \
```

### 2.3 Patch Embeddings Generation
Copy data_preparation_script/img_emb_generation.py and data_preparation_script/multi_emb_generation.sh to the [PLIP](https://github.com/PathologyFoundation/plip) project directory, and use the following code to generate the Embeddings.
```bash
cd PLIP_PROJECT_DIRECTORY
bash multi_emb_generation.sh
```

The above command will generate the following folder structure at the specified RESULTS_DIRECTORY/img_features:
```bash
RESULTS_DIRECTORY/img_features/
	├── slide_1
    		├── 0_23904.npy
    		├── 4096_28000.npy
    		└── ...
	├── slide_2
    		├── 4128_7296.npy
    		├── 4128_11392.npy
    		└── ...
	└── ...
```

## 3. Inference

```bash
python pathagent.py \
    --plip_lib_path PLIP_PROJECT_DIRECTORY \
    --qwen_ckpt QWEN_CHECKPOINT_PATH \
    --plip_ckpt PLIP_CHECKPOINT_PATH \
    --patho_r1_ckpt PATHOR1_7B_CHECKPOINT_PATH \
    --descriptions_file RESULTS_DIRECTORY/desc/patches_descriptions.json \
    --questions_file WSI_VQA_PROJECT_DIRECTORY/dataset/WSI_captions/WsiVQA_test.json \
    --feature_dir RESULTS_DIRECTORY/img_features \
    --patch_root RESULTS_DIRECTORY/patches_output \
    --save_dir RESULTS_DIRECTORY/results/wsi-vqa \
    --dataset_name "wsi_vqa"
```

## 4. Evaluation

```bash
python eval/metics.py \
    --results_dir RESULTS_DIRECTORY/results/wsi-vqa
```


## Citation
```bibtex
@inproceedings{chen2026pathagent,
  title={Toward Interpretable Analysis of Whole-slide Pathology Images via Large Language Model-based Agentic Reasoning},
  author={Jingyun Chen and Linghan Cai and Zhikang Wang and Yi Huang and Songhan Jiang and Shenjin Huang and Hongpeng Wang and Yongbing Zhang},
  booktitle={European Conference on Computer Vision},
  year={2026},
  organization={Springer}
}
```

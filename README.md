# [ECCV 2026] PathAgent: Toward Interpretable Analysis of Whole-slide Pathology Images via Large Language Model-based Agentic Reasoning 
[![Project Page](https://img.shields.io/badge/Project-Page-green)](https://github.com/G14nTDo4/PathAgent)
[![arXiv](https://img.shields.io/badge/arXiv-2511.17052-b31b1b.svg)](https://arxiv.org/abs/2511.17052)
[![Paper](https://img.shields.io/badge/Paper-PDF-red)](https://arxiv.org/pdf/2511.17052.pdf)

<div align="center">

</div>

Research fork of the official code implementation for paper "PathAgent: Toward Interpretable Analysis of Whole-slide Pathology Images via Large Language Model-based Agentic Reasoning"

>Jingyun Chen, Linghan Cai, Zhikang Wang, Yi Huang, Songhan Jiang, Shenjin Huang, Hongpeng Wang, Yongbing Zhang


## 当前仓库：PathAgent与WSI-VQA复现

这是作者公开实现的研究分支，上游基线为`8c2abee2cd18ca2e2c839742c9296bd8a41738c5`。仓库保存Agent、模型准备、WSI-VQA评测及必要测试。论文对齐配置采用已记录的逐区域倍率计划与高倍后重新评价；具体差异、文件位置和验证情况见[改动说明](docs/paper_alignment_wsi_vqa_20261104.md)。

|模块|用途|
|---|---|
|`pathagent.py`、`navigation.py`|统一推理入口，公开代码配置与论文对齐配置|
|`models/`、`configs/`|正式模型加载、提示词、输入检查与运行配置|
|`data_preparation_script/`、`external/PLIP`|作者预处理脚本和Navigator依赖|
|`wsi_regions.py`、`trajectory.py`|真实WSI读区、几何验证与运行证据|
|`tools/prepare_wsi_vqa.py`、`prepare_plip_features.py`|5倍全组织候选及PLIP特征|
|`tools/run_wsi_vqa.py`、`evaluate_wsi_vqa.py`|全测试运行、统计与三个案例展示|
|`tests/`|Agent、几何、导航与评测回归|

权重、WSI、实验输出和本地归档不提交。仓库范围与提交前检查见[仓库范围](docs/repository_scope.md)。

```bash
git clone --branch codex/paper-aligned-wsi-vqa --recurse-submodules https://github.com/xzz0301/PathAgent.git
cd PathAgent
python -m pip install -r requirements.txt
python tools/check_repository_scope.py
python -m unittest tests.test_consolidated tests.test_paper_navigation tests.test_wsi_regions tests.test_wsi_vqa_metrics -v
```

Quilt使用独立兼容环境，见[部署说明](docs/quilt_20261031.md)。WSI预处理还需要OpenSlide及OpenCV；开放题指标需要`pycocoevalcap`和Java，可用`tools/setup_wsi_metrics.py`准备用户目录依赖。正式服务器运行配置为`configs/wsi_vqa_paper.json`；`local_8gb.json`是可选本地配置，不用于论文复现。

```bash
python tools/run_wsi_vqa.py --server /path/to/pathagent --dataset /path/to/wsi-vqa --output /path/to/new_run --gpus 0,1
```

`--gpus`只允许一或两张物理GPU，例如`--gpus 2`为单卡；所有准备和推理阶段均限制在所选设备。默认将PLIP与Qwen3放在同一卡。

85张有可靠倍率的切片、726题推理及三个预定案例已完成。原735题完整分母闭题兼容准确率46.15%，实际校准子集46.75%，未达到论文56.32%；缺倍率的9题保持缺失并列报告。完整口径、开放题指标、方法歧义与运行来源见[结果说明](docs/wsi_vqa_baseline_results.md)。历史本轮曾用三卡，后续资源限制不改写已完成实验记录。统计和案例验收后冻结基线、建立`dev_trajectory`。以下保留作者原始项目说明，原说明中的路径和命令需按实际环境配置。

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

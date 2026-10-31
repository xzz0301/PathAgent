# 双 RTX 4090 运行原版 PathAgent：资源与部署分析

## 1. 结论与运行范围

按服务器配置为 **2张RTX 4090，每张24GB显存** 分析，原版PathAgent的在线推理具有可行的部署方案：**Qwen3-4B和PLIP放在GPU 0，Patho-R1-7B放在GPU 1，两个大模型使用BF16，单病例、单图串行推理**。应先验证这一方案，再根据视觉输入的峰值占用决定是否量化。

两张卡的显存由程序分别管理，不能当作一个可直接分配48GB张量的显存池。RTX 4090没有NVLink；跨卡交换走PCIe。PathAgent的三个模型主要通过文本和CPU上的NumPy特征交换结果，适合按模型分卡，通常无需把一个模型拆到两张卡上。

原版入口两次使用`device_map="auto"`，没有固定模型归属。自动分配可能把两个模型的大部分权重放在同一张卡上，也可能将模型拆开或卸载到CPU。为了获得稳定的显存分工，需要调整模型加载配置；Navigator、Perceptor、Executor的算法和调用顺序可以保持原版。

本文分析的是**推理和原版离线数据准备**，不包括模型训练。数值分为可计算的权重下限和用于部署规划的运行预算；当前没有这台双4090服务器上的实测峰值与吞吐记录。

## 2. 原版到底需要运行哪些模型

| 阶段 | 模型/组件 | 用途 | 是否需要与在线模型同时常驻 |
|---|---|---|---|
| WSI准备 | CLAM/OpenSlide | 组织分割、坐标生成、图块读取 | 否 |
| 离线通用描述 | Quilt-LLaVA-v1.5-7B及CLIP视觉塔 | 为所有候选图块生成通用病理描述 | 否，生成JSON后退出 |
| 离线特征 | PLIP | 编码所有图块，保存归一化特征 | 否，生成npy后退出 |
| 在线Navigator | PLIP | 编码问题/缺失信息，必要时编码zoom子块 | 是 |
| 在线Executor | Qwen3-4B | A/B/C判断链、长描述摘要、最终回答 | 是 |
| 在线Perceptor | Patho-R1-7B | 每轮前5块及zoom子块的视觉描述 | 是 |

在线入口并不加载Quilt-LLaVA。把离线描述生成模型也算入在线常驻显存，会高估在线要求。完整复现仍需提供原版使用的通用描述和特征缓存；用空描述替代属于输入方案变化。

源码依据：[模型加载](D:/PathCode/pathagent.py:87)、[Perceptor](D:/PathCode/models/inference.py:310)、[原版数据准备流程](D:/PathCode/README.md:47)。

## 3. 权重显存：按实际参数量计算

本文统一采用GiB，即`1 GiB = 2^30 bytes`。纯BF16权重按每参数2 bytes计算；该数值不包含KV cache、视觉中间张量、CUDA上下文、算子工作区和缓存分配器保留空间。

| 在线模型 | 核对依据 | 参数量/权重 | 纯权重占用 |
|---|---|---|---:|
| Qwen3-4B | 模型API元数据、本地checkpoint配置 | 4,022,468,096参数，BF16 | 7.49 GiB |
| Patho-R1-7B | 模型API元数据，Qwen2.5-VL结构 | 8,292,166,656参数，BF16，含视觉组件 | 15.45 GiB |
| PLIP | CLIP ViT-B/32结构、FP32默认加载 | 约1.51亿参数 | 约0.56 GiB |
| 合计 | 三模型保持上述精度 | 权重下限 | 约23.50 GiB |

“7B”是Patho-R1的模型名称，完整多模态checkpoint实际约82.92亿参数，不能只按70亿参数计算。单张24GB卡同时承载三模型时，纯权重已接近容量，运行张量没有充分余量；双卡分开后则具备可用空间。

元数据采用2026-10-15保存的模型仓库快照，版本与参数摘录保存在[代码核对清单](D:/PathCode/docs/agent_analysis_20261019/代码核对清单.json)。权重字节数是参数计算值，checkpoint文件大小还包括文件头和其他存储内容。

## 4. 双卡分配与运行显存预算

### 4.1 首选分工

| 显卡 | 放置内容 | 权重下限 | 首次验证的运行预算 |
|---|---|---:|---|
| GPU 0 | Qwen3-4B BF16 + PLIP FP32 | 约8.05 GiB | 约10—16 GiB，取决于文本长度和算子 |
| GPU 1 | Patho-R1-7B BF16 | 约15.45 GiB | 约18—23 GiB，限单图、中等视觉token预算 |

这些区间用于安排负载，不是该服务器的测量结果。GPU 1是更紧的一侧。优先控制单张图进入模型的尺寸和视觉token，再检查生成长度。若要求保留超大图的原始视觉输入，GPU 1可能超过24GB，需要8-bit或CPU卸载。

PathAgent在Python中依次调用PLIP、Patho-R1和Qwen，三者并不需要在同一时刻执行前向。按角色分卡可保持模型常驻，又减少模型内跨卡传输。

### 4.2 文本KV cache不是固定开销

Qwen3-4B的本地配置为36层、8个KV head、head_dim=128。batch=1且KV使用BF16时：

```text
每token KV = 2(K/V) × 36层 × 8头 × 128维 × 2 bytes
            = 147,456 bytes = 144 KiB

8192个缓存token ≈ 1.125 GiB
16384个缓存token ≈ 2.25 GiB
```

这里的token长度包括输入和已经生成的输出。原版A/B/C和最终回答是独立generate调用；不能把它们的最大KV相加当作同时常驻，但每次调用的输入描述越长，该次峰值越高。

原版超过50块才做分块摘要，并未对每块描述长度和最终拼接总长度统一预算。因此后续较大病例需要记录token长度，必要时增加有上限的证据摘要。

## 5. 视觉输入是24GB部署的主要变量

原版`patho_r1_describe()`调用`process_vision_info()`，再把处理后的图像交给processor。图块文件的尺寸、视觉工具的缩放规则、processor的pixel预算共同决定模型实际看到多少像素。README切块示例为4096×4096，但不能直接把磁盘尺寸当作最终视觉token数。

Qwen2.5-VL通常按28×28像素对应约1个合并后的视觉token估算：

| 目标视觉token上限 | 对应max_pixels | 近似方形边长 |
|---:|---:|---:|
| 1024 | 802,816 | 896 |
| 2048 | 1,605,632 | 1267，实际需按28对齐 |
| 4096 | 3,211,264 | 1792 |

未经缩放的4096²图像约对应21,400个视觉token。实际占用还受文本和输出长度影响。`max_new_tokens=1024`只限制新生成的文本，不能限制图像编码和输入序列带来的显存峰值。

部署时先保留原版视觉处理配置测量代表性样本；若峰值过高，可在图像message中设置`max_pixels`，并同步processor设置。下面是预算方案的写法，需要放在`process_vision_info(messages)`之前：

```python
# 示例：视觉预算2048 tokens；在构造image消息时设置
image_item = {
    "type": "image",
    "image": image,
    "max_pixels": 2048 * 28 * 28,
}
```

仅在processor设置上限，而视觉工具先按其他规则扩大图像，仍可能造成不必要的CPU图像处理开销。应检查实际传入模型的`image_grid_thw`及输入token数。

缩小视觉输入会改变可见细节，属于资源预算设置。比较论文效果时固定图像处理配置；比较服务器能否运行时报告该配置对应的峰值。需要保留细胞细节时，可采用真正的多尺度局部读取，但这已是导航实现扩展。

## 6. 最小加载调整：保持原版Agent算法

复制原版入口为`pathagent_2gpu.py`后，可用下列片段替换原来的模型加载段。此处提供修改位置和代码，本文没有改写实际运行入口。

```python
torch.cuda.set_device(0)

tokenizer = AutoTokenizer.from_pretrained(args.qwen_ckpt)
model = AutoModelForCausalLM.from_pretrained(
    args.qwen_ckpt,
    torch_dtype=torch.bfloat16,
    device_map={"": 0},
    attn_implementation="sdpa",
).eval()

# PLIP原封装默认device="cuda"，此时对应逻辑GPU 0
plip = PLIP(args.plip_ckpt)

patho_r1_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    args.patho_r1_ckpt,
    torch_dtype=torch.bfloat16,
    device_map={"": 1},
    attn_implementation="sdpa",
).eval()
patho_r1_processor = AutoProcessor.from_pretrained(args.patho_r1_ckpt)
```

原版推理封装的`inputs.to(model.device)`在这种“一个完整模型放一张卡”的配置下会把输入放到对应设备。PLIP也通过自身`device`迁移输入。没有必要改动检索排名、充分性判断或zoom分支。

SDPA用于降低注意力计算的中间开销。FlashAttention 2可作为后续优化，需要额外安装匹配的CUDA扩展；首轮验证不依赖它。

Linux服务器启动示例；先完成上述入口复制与加载修改，再执行：

```bash
CUDA_VISIBLE_DEVICES=0,1 python pathagent_2gpu.py \
  --plip_lib_path /srv/PathAgent/external/PLIP \
  --qwen_ckpt /srv/models/Qwen3-4B \
  --plip_ckpt /srv/models/plip \
  --patho_r1_ckpt /srv/models/Patho-R1-7B \
  --descriptions_file /srv/data/descriptions.json \
  --questions_file /srv/data/questions.json \
  --feature_dir /srv/data/features \
  --patch_root /srv/data/patches \
  --save_dir /srv/results/pathagent_bf16 \
  --dataset_name wsi_vqa
```

`/srv/...`是目录安排示例，需换成服务器实际路径。`CUDA_VISIBLE_DEVICES`会重编号；若设成`2,3`，程序内部仍用逻辑0、1。

明确的`device_map={"": 1}`表示整个模型放到该卡，不会因为另传`max_memory`就自动限额或迁移。运行峰值预算要通过输入控制、量化或真正的自动分层卸载实现。

## 7. CPU能否代替GPU显存

可以通过CPU推理或权重卸载减少显存，但CPU内存不等价于高速GPU显存。CPU方案改变执行设备或在每次前向时传输权重，速度主要取决于CPU、主存带宽和PCIe。

### 7.1 先迁移PLIP

PLIP约占0.56GiB权重，在线多数时候只编码短查询。把它移到CPU是最轻量的降显存措施：

```python
plip = PLIP(args.plip_ckpt)
plip.model = plip.model.to("cpu")
plip.device = "cpu"
```

原封装构造时会先迁移到默认CUDA卡，所以这段代码仍有短暂GPU占用。长期使用应给PLIP构造函数增加`device`参数，直接在CPU加载。zoom时的图像特征编码也会在CPU执行。

### 7.2 大模型分层卸载

Accelerate可以用`device_map="auto"`、`max_memory`将部分层放在CPU；必要时以`offload_folder`保存磁盘卸载内容。为避免Qwen占用预算混淆，Patho-R1的卸载可在单独可见GPU的工作进程中安排，或为整个进程显式规划两模型的剩余预算。

```python
# 示例：进程只可见用于Patho-R1的那张GPU，逻辑编号为0
patho_r1_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    args.patho_r1_ckpt,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    max_memory={0: "18GiB", "cpu": "48GiB"},
    offload_folder="/srv/cache/patho_offload",
    offload_state_dict=True,
    attn_implementation="sdpa",
)
```

`max_memory`主要用于权重映射规划，不是对生成过程显存的硬限制；这里给GPU留下约6GiB用于运行张量。`offload_state_dict=True`用于缓解加载阶段主存峰值，本身不代表已启用运行时CPU卸载，实际设备分布应检查`hf_device_map`。

拆分视觉模型时要按模型的模块约束规划，输入放到正确的执行设备，不能继续把`model.device`当作所有模块位置的完整描述。单独工作进程还需要RPC或队列接口，属于比按模型分卡更大的工程调整。

生成文本时，每个新token都要经过模型层；反复传输CPU权重会明显增加时延。CPU卸载适合容量不足时保留模型精度，不适合作为双4090的第一配置。

### 7.3 RAM与磁盘

建议服务器配置至少64GB主存，较大数据准备任务采用128GB。除了模型加载与卸载，程序还会把描述JSON和当前病例特征加载到主存。512维FP32特征每百万块约1.91GiB，Python字典、键名和JSON字符串另有开销。

模型缓存和安装临时空间建议预留50—100GB SSD；WSI、图块、描述和特征按数据规模另算。不要把模型仓库的所有格式总下载大小等同于常驻显存。

## 8. 量化的收益与选择顺序

### 8.1 理论下限和部署规划值

| 模型 | BF16权重 | 8-bit全参数理论下限 | 4-bit全参数理论下限 | 常见实现的权重及量化元数据规划区间 |
|---|---:|---:|---:|---|
| Qwen3-4B | 7.49 GiB | 3.75 GiB | 1.87 GiB | 8-bit约4.5—6；4-bit约2.5—4 GiB |
| Patho-R1-7B | 15.45 GiB | 7.72 GiB | 3.86 GiB | 8-bit约9—12；4-bit约5—8 GiB |

bitsandbytes主要替换支持的Linear层，嵌入、归一化、部分视觉模块、量化尺度等并不会全部变成1 byte或0.5 byte/参数。右侧区间用于选择方案，实际值应按量化模块覆盖与checkpoint验证；它也不包含完整生成峰值。

### 8.2 推荐顺序

1. **两模型BF16分卡**：先保留原模型精度，测量实际视觉输入。
2. **Patho-R1使用8-bit，Qwen继续BF16**：优先缓解GPU 1压力，保留Executor的数值配置。
3. **Patho-R1使用NF4 4-bit**：适合更大视觉输入或其他并存负载，增加量化效果评估。
4. **Qwen也量化**：需要单卡整合或大幅增加文本上下文时再考虑。
5. **量化与CPU卸载组合**：在前述方案仍不能容纳目标输入时使用。

Patho-R1 8-bit加载的关键配置为：

```python
from transformers import BitsAndBytesConfig

quant_config = BitsAndBytesConfig(load_in_8bit=True)
patho_r1_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    args.patho_r1_ckpt,
    torch_dtype=torch.bfloat16,
    quantization_config=quant_config,
    device_map={"": 1},
    attn_implementation="sdpa",
).eval()
```

NF4方案可使用`load_in_4bit=True`、`bnb_4bit_quant_type="nf4"`、`bnb_4bit_use_double_quant=True`、`bnb_4bit_compute_dtype=torch.bfloat16`。量化减少权重占用，但通常不会同比减少KV cache和视觉前向中间张量。

8-bit与CPU卸载组合时，`llm_int8_enable_fp32_cpu_offload=True`让卸载到CPU的权重使用FP32存储，主存开销高于8-bit权重。4-bit不能直接套用这套CPU策略，应按具体Transformers/bitsandbytes版本和视觉模块逐项验证。

量化验证除最终答案外，还应检查`missing_info`、充分性、zoom建议和观察区域。中间决策改变会改变后续轨迹，不能只比较最终一项答案。

## 9. 原版离线预处理能否在双4090上完成

### 9.1 通用描述生成

Quilt-LLaVA-v1.5-7B的语言权重按半精度约13GiB量级，另有视觉塔和投影层；独立CLIP ViT-L/14视觉塔半精度约0.56GiB。batch=1、受控输入与生成长度时，一张24GB卡有部署空间。

其Llama结构为32层、32个KV head、head_dim=128，半精度KV约0.5MiB/token；4096个缓存token约2GiB。离线运行也要限制总序列长度，避免同时叠加多图或大batch。

原版`description_generation.py`已支持`--load-8bit/--load-4bit`，`multi_description_generation.sh`实际默认传`--load-4bit`。因此，按仓库脚本运行离线描述时已经采用4-bit，不需要另行开发量化入口；上面的半精度估算用于分析取消量化后的容量。

该shell脚本按四个GPU任务写示例。双卡应将数据分片与worker数调整为2，每卡一个worker。完成JSON后退出进程，再启动在线三模型，不把Quilt与Patho-R1堆在同一张卡常驻。

Quilt依赖旧版LLaVA代码，建议建立独立Python环境；在线Qwen3/Patho-R1使用另一个环境，通过JSON文件交换描述。

### 9.2 特征和WSI准备

PLIP的特征生成显存压力较低，双卡可以按病例分工。调整`multi_emb_generation.sh`的GPU列表与分片数即可。离线描述与特征不要求同时启动。

WSI切块主要消耗CPU、RAM、磁盘容量和读写带宽。两张4090不能解决大量小PNG的磁盘开销；生产流程可考虑分块存储或按需WSI读取，但这属于数据层改进。

## 10. 环境与验收安排

推荐Linux服务器、匹配的NVIDIA驱动和PyTorch CUDA wheel。原README写`torch==2.7.0`配`cu121`索引，安装前应按PyTorch对应版本的官方发行矩阵选择实际提供的CUDA包，避免直接复制不匹配的组合。在线环境以源码要求的Transformers、Accelerate和qwen-vl-utils为基线，量化时再核对bitsandbytes支持。

需要准备完整的Qwen3-4B、PLIP、Patho-R1-7B checkpoint；Patho-R1的模型访问授权和文件完整性是启动输入条件。模型存在本地后可以离线运行，Agent不依赖外部在线模型API。

建议依次验收：

| 步骤 | 验收内容 | 记录 |
|---|---|---|
| 1 | 三模型加载后设备分配正确 | 参数dtype、hf_device_map、两卡基础显存 |
| 2 | 代表性图块和大尺寸图块各调用Perceptor | 原图尺寸、实际视觉token、峰值、耗时 |
| 3 | Executor A/B/C和最终整合 | 输入token、生成token、JSON解析、峰值 |
| 4 | 一个完整病例，包括zoom路径 | 原版输出JSON、各分支是否执行、总耗时 |
| 5 | 多病例串行 | 是否逐病例回落至稳定显存、失败样本 |

可用`nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv`检查系统显存；在各阶段前调用`torch.cuda.reset_peak_memory_stats(i)`，阶段后先`synchronize(i)`再读取`max_memory_allocated(i)`和`max_memory_reserved(i)`。PyTorch统计不覆盖所有驱动及第三方分配，结合nvidia-smi判断余量。

若出现OOM，按“确认模型没有堆在同一卡→检查实际视觉token→限制输入/输出预算→PLIP移CPU→Patho-R1 8-bit→分层卸载”的顺序处理。`empty_cache()`只能释放未使用的缓存块，不会释放仍被模型引用的权重，不能替代上述措施。

不建议用DataParallel或DDP解决单病例容量问题：它们通常复制模型；将两张卡各跑一套完整Agent属于吞吐方案，也要求每张卡能容纳整套模型。当前目标更适合按模型分卡。

## 11. 推荐落地配置

首轮配置采用**GPU 0：Qwen3-4B BF16 + PLIP；GPU 1：Patho-R1-7B BF16；batch=1；病例串行；离线/在线分阶段运行；主存64GB起，128GB更适合大数据准备**。先测原版视觉输入，GPU 1峰值超预算时优先考虑Patho-R1 8-bit；需要改变图像pixel预算时，在实验配置中固定并记录。

双4090能够提供原版推理所需的模型容量。实际部署的主要控制点是模型归属、视觉输入峰值、文本长度和离线任务调度；CPU卸载与量化用于扩展余量。

参考资料：[Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B)、[Patho-R1-7B](https://huggingface.co/WenchuanZhang/Patho-R1-7B)、[PLIP](https://huggingface.co/vinid/plip)、[Accelerate大模型推理](https://huggingface.co/docs/accelerate/usage_guides/big_modeling)、[bitsandbytes量化](https://huggingface.co/docs/transformers/v4.51.3/quantization/bitsandbytes)、[Qwen2.5-VL图像预算](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)、[PyTorch版本安装](https://pytorch.org/get-started/previous-versions/)。

分析日期：2026-10-19。

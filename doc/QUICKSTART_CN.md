# Volcano 快速上手指南

本文档面向多模态大模型（MLLM）方向研究者，帮助快速配置环境、运行推理与（可选）训练。

---

## 1. Conda 环境配置

### 1.1 依赖来源

仓库通过 **pyproject.toml** 管理依赖，未提供 `environment.yml` 或 `requirements.txt`。核心依赖包括：PyTorch 2.0、Transformers、LLaVA 系组件、flash-attn（训练用）等。

### 1.2 创建环境（推荐步骤）

```bash
# 创建 Python 3.10 环境（与 README 一致）
conda create -n volcano python=3.10 -y
conda activate volcano

# 升级 pip（支持 PEP 660，便于可编辑安装）
pip install --upgrade pip

# 安装项目（会安装 torch, transformers, peft, gradio 等）
pip install -e .

# 若需训练，安装训练依赖与 flash-attn
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```

### 1.3 可能的问题与解决

| 问题 | 建议 |
|------|------|
| **flash-attn 编译失败** | 需 CUDA 与对应 PyTorch 版本匹配；推理可不装 flash-attn，仅训练需要。 |
| **torch 版本冲突** | pyproject.toml 固定 `torch==2.0.1`、`torchvision==0.15.2`，若需其他版本可修改后 `pip install -e .`。 |
| **transformers 版本** | 仓库要求 `transformers==4.31.0`，避免擅自升级以免与 LLaVA 接口不兼容。 |

### 1.4 仅推理时的最简安装

若只跑推理、不训练，可省略训练依赖与 flash-attn：

```bash
conda create -n volcano python=3.10 -y && conda activate volcano
pip install --upgrade pip && pip install -e .
```

---

## 2. 仓库结构概览

```
volcano/
├── pyproject.toml          # 依赖与包配置（无 requirements.txt 时以此为准）
├── README.md               # 论文与官方说明
├── QUICKSTART_CN.md        # 本快速上手指南
├── run_inference.py        # 单图单问简易推理脚本（见下）
├── scripts/
│   └── zero2.json         # DeepSpeed ZeRO-2 配置（训练用）
└── llava/
    ├── constants.py        # 常量（如 IMAGE_TOKEN_INDEX）
    ├── conversation.py    # 对话模板
    ├── model/
    │   ├── builder.py      # 模型加载 load_pretrained_model
    │   └── ...
    ├── eval/
    │   ├── volcano_mmhal_bench.py   # MMHal-Bench 评测
    │   ├── volcano_pope.py         # POPE 评测
    │   └── volcano_gavie.py       # GAVIE 评测
    ├── train/
    │   ├── train_mem.py   # 训练入口（flash attn）
    │   └── train.py       # 训练逻辑与 dataclass 参数
    └── visualize/         # 注意力可视化（论文 Figure 4/5）
```

---

## 3. 模型权重下载

Volcano 在 LLaVA-1.5 基础上微调，**推理时必须同时具备**：

1. **Volcano 权重**（仅含 mm_projector 等增量）：[7B](https://huggingface.co/kaist-ai/volcano-7b) / [13B](https://huggingface.co/kaist-ai/volcano-13b)
2. **LLaVA-1.5 基座**（完整 LLM + 视觉编码器）：[7B](https://huggingface.co/liuhaotian/llava-v1.5-7b) / [13B](https://huggingface.co/liuhaotian/llava-v1.5-13b)

使用 Hugging Face CLI 下载到本地（可选，不下载则首次运行会从 Hub 拉取）：

```bash
# 安装 huggingface_hub 后
huggingface-cli download kaist-ai/volcano-7b --local-dir ./checkpoints/volcano-7b
huggingface-cli download liuhaotian/llava-v1.5-7b --local-dir ./checkpoints/llava-v1.5-7b
```

推理时通过 `--model_path` 与 `--model_base` 指定路径或 Hub 名称即可（见下）。

---

## 4. 推理

### 4.1 单张图片 + 单条问题（最简）

使用仓库根目录下的 **run_inference.py**（见第 5 节），适合快速验证与调试：

```bash
# 使用 7B（Hub 名称，首次运行会自动下载）
python run_inference.py \
  --model_path kaist-ai/volcano-7b \
  --model_base liuhaotian/llava-v1.5-7b \
  --image_path /path/to/your/image.jpg \
  --question "图片里有什么？"

# 使用本地已下载权重
python run_inference.py \
  --model_path ./checkpoints/volcano-7b \
  --model_base ./checkpoints/llava-v1.5-7b \
  --image_path ./test.jpg \
  --question "What color is the object?"
```

**输入**：任意本地图片路径 + 一条自然语言问题。  
**输出**：终端打印初始回答、多轮反馈/修订（若有）、以及最终采纳的回答。

### 4.2 MMHal-Bench 评测

输入为 JSON 列表，每项需包含 `image_src`（图片 URL 或路径）、`question`、以及 `gold_answer` 或 `gt_answer`（用于日志/评估）。仓库内示例数据在 `llava/visualize/data/MMHal-bench.json`。

```bash
python -m llava.eval.volcano_mmhal_bench \
  --model_path kaist-ai/volcano-7b \
  --model_base liuhaotian/llava-v1.5-7b \
  --input llava/visualize/data/MMHal-bench.json \
  --output results_mmhal.json
```

输出为带 `model_answer` 等字段的 JSON，可用于计算 MMHal 指标。

### 4.3 POPE 评测

需要 **图片目录** + **问题文件**（每行一个 JSON，含 `image`、`text` 等字段，格式参考 [POPE](https://github.com/RUCAIBox/POPE)）：

```bash
python -m llava.eval.volcano_pope \
  --model-path kaist-ai/volcano-7b \
  --model-base liuhaotian/llava-v1.5-7b \
  --image-folder /path/to/coco/val2014 \
  --question-file /path/to/pope_questions.jsonl \
  --answers-file pope_answers.jsonl
```

### 4.4 GAVIE 评测

按 GAVIE 官方说明准备输入目录/文件后：

```bash
python -m llava.eval.volcano_gavie \
  --model-path kaist-ai/volcano-7b \
  --model-base liuhaotian/llava-v1.5-7b \
  --input /path/to/gavie_input \
  --output /path/to/gavie_output
```

---

## 5. 简易推理脚本说明（run_inference.py）

**run_inference.py** 实现「单图 + 单问」的 Volcano 流程：加载模型 → 编码图像 → 生成初始回答 → 最多 3 轮「反馈 → 修订 → 决定是否采纳」，并打印每轮结果。

- **依赖**：与仓库一致（`pip install -e .` 即可），无需 flash-attn。
- **参数**：
  - `--model_path`：Volcano 权重路径或 Hub 名
  - `--model_base`：LLaVA-1.5 基座路径或 Hub 名
  - `--image_path`：本地图片路径
  - `--question`：问题字符串
  - `--max_revision_rounds`：修订轮数，默认 3
- **输出**：初始回答、各轮 feedback/revision/decision、最终答案。

---

## 6. 模型与训练需求概览

### 6.1 模型架构

- **视觉编码器**：CLIP ViT-L/14@336（`openai/clip-vit-large-patch14-336`），与 LLaVA-1.5 一致。
- **语言模型**：Vicuna（LLaMA 的对话微调版），7B/13B。
- **多模态连接**：MLP projector（2 层 GELU），将视觉特征映射到 LLM 词嵌入空间。
- **Volcano 增量**：在 LLaVA-1.5 上微调同一模型，使其能按固定模板做「Critique → Revise → Decide」的自反馈修订。

### 6.2 是否需要预训练/微调

- **仅推理**：只需 Volcano 权重 + LLaVA-1.5 基座，无需自己训练。
- **复现论文训练**：需要 [Volcano 训练数据](https://huggingface.co/datasets/kaist-ai/volcano-train)，并运行 `llava/train/train_mem.py`（见下）。

### 6.3 训练配置与成本（参考 README）

README 中示例为 13B、单机多卡：

- **脚本**：`deepspeed llava/train/train_mem.py`，需传入 `--deepspeed scripts/zero2.json`（仓库已提供 `scripts/zero2.json`）。
- **显存**：13B 全量微调通常需 40GB+ 级多卡（如 2×A100 或 4×A100），7B 略低但仍建议 24GB+ 单卡或以上。
- **训练时间**：依赖数据量、卡数与 batch size，未在仓库中给出具体小时数；建议先用小数据/小步数试跑。
- **注意**：训练依赖 flash-attn 与 `pip install -e ".[train]"`；若使用 README 中的 `train_mem.py`，需能成功编译/安装 flash-attn。

### 6.4 性能（来自 README）

论文报告 Volcano 在 MMHal-Bench、POPE、GAVIE 上 SOTA，并在 MM-Vet、MMBench 上优于前作。具体数字见 README 中的表格。

---

## 7. 小结 Checklist

| 步骤 | 命令/说明 |
|------|-----------|
| 环境 | `conda create -n volcano python=3.10 -y` → `pip install -e .`（训练则加 `.[train]` 与 flash-attn） |
| 权重 | Volcano 7B/13B + LLaVA-1.5 7B/13B（Hub 或本地路径） |
| 单图推理 | `python run_inference.py --model_path ... --model_base ... --image_path ... --question "..."` |
| MMHal | `python -m llava.eval.volcano_mmhal_bench --input ... --output ...` |
| 训练 | 准备数据与 zero2.json，运行 `deepspeed llava/train/train_mem.py ...` |

若某些数据格式或评测脚本与论文/官方仓库有差异，以官方仓库与论文为准；本指南以当前仓库代码与 README 为准整理，便于你快速上手与排查问题。

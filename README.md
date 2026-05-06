# ArcFace Torch (Enhanced)

**English** | [中文](README_zh.md)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/killing-two-birds-with-one-stone-efficient/face-verification-on-ijb-c)](https://paperswithcode.com/sota/face-verification-on-ijb-c?p=killing-two-birds-with-one-stone-efficient)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/killing-two-birds-with-one-stone-efficient/face-verification-on-ijb-b)](https://paperswithcode.com/sota/face-verification-on-ijb-b?p=killing-two-birds-with-one-stone-efficient)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/killing-two-birds-with-one-stone-efficient/face-verification-on-agedb-30)](https://paperswithcode.com/sota/face-verification-on-agedb-30?p=killing-two-birds-with-one-stone-efficient)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/killing-two-birds-with-one-stone-efficient/face-verification-on-cfp-fp)](https://paperswithcode.com/sota/face-verification-on-cfp-fp?p=killing-two-birds-with-one-stone-efficient)

> An enhanced fork of [InsightFace ArcFace Torch](https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch). Building on the original codebase, this project adds four core improvements: **lightweight model design (mbf_v3)**, **structured pruning**, **knowledge distillation**, and **multi-dataset training** — forming a complete model compression pipeline for edge deployment.

## Roadmap

- [x] Lightweight models mbf_v3 / mbf_v3_se
- [x] Structured pruning
- [x] Knowledge distillation
- [x] Multi-dataset training
- [ ] Open-source more pre-trained lightweight models
- [ ] Release WebFace42M dataset and trained models
- [ ] Release multi-dataset trained models
- [ ] Release a model trained on a private million-ID dataset

## Key Improvements

### 1. Lightweight Model mbf_v3

The original MobileFaceNet series uses MobileNetV2-style inverted residual blocks (1x1 conv expand → 3x3 depthwise conv → 1x1 conv project + residual). `mbf_v3` achieves lightweighting by tuning two core parameters — `scale` (channel width multiplier) and `blocks` (number of residual blocks per stage) — while keeping the same architecture:

| Variant | scale | blocks | Channels per stage | Total blocks | Params | FLOPs | Role |
|---------|-------|--------|-------------------|--------------|--------|-------|------|
| `mbf` | 2 | (1,4,6,2) | 128→128→256→256 | 13 | 2.05M | 0.45G | Original ultra-light |
| `mbf_large` | 4 | (2,8,12,4) | 256→256→512→512 | 26 | 6.30M | 1.85G | Original large |
| **`mbf_v3`** | **3** | **(2,6,8,3)** | **192→192→384→384** | **19** | **3.71M** | **1.15G** | **New lightweight** |
| `mbf_v3_se` | 3 | (2,6,8,3) | 192→192→384→384 | 19+SE | 3.86M | 1.15G | New + attention |

`mbf_v3` vs `mbf_large`:
- **Channel width**: scale 4→3, **-25%**
- **Depth**: blocks 26→19, **-27%**
- **Parameters**: 6.3M→3.7M, **-41%**
- **FLOPs**: 1.85G→1.15G, **-38%**
- **Architecture preserved**: same MobileNetV2 inverted residual + depthwise separable design

`mbf_v3_se` adds SE (Squeeze-and-Excitation) attention to `mbf_v3`, costing only +4% parameters (3.71M→3.86M), used to boost accuracy during distillation.

### 2. Structured Pruning

Channel-level structured pruning on `mbf_large` using [Torch-Pruning](https://github.com/VainF/Torch-Pruning)'s `GroupNormPruner`:

- L2 norm as channel importance metric
- GDC head (`features`, `conv_sep`) protected from pruning
- < 1% accuracy drop at 20% pruning ratio

### 3. Knowledge Distillation

Embedding-level distillation from a large Teacher to a lightweight Student:

- Teacher: `mbf_large` (6.3M) → Student: `mbf_v3_se` (3.9M)
- Distillation loss: cosine similarity or L2 norm
- Combined loss: `L = α * L_CE + (1-α) * L_distill`, α=0.5

### 4. Multi-Dataset Training

The original ArcFace Torch only supports single-dataset training. `train_multi_data.py` implements N-dataset joint training:

- **N DataLoaders** iterate in sync, each dataset sampled independently
- **Shared backbone** with a single forward pass: `torch.cat` → `backbone` → `torch.split`
- **N PartialFC heads** — one classification head per dataset with its own class space
- **Weighted loss** `L = Σ(w_i * L_CE_i)` with adjustable per-dataset weights
- **Distillation support** — `train_multi_data_distill.py` adds knowledge distillation on top

```
DataLoader_A (glint360k)    DataLoader_B (faces_umd)     ← N datasets
        ↓                           ↓
    img = torch.cat([img_A, img_B])
        ↓
    backbone(img)                   ← Shared backbone, single forward
        ↓
    torch.split(embeddings)
        ↓                           ↓
  PartialFC_A                 PartialFC_B               ← N independent heads
  loss_A * w_A                loss_B * w_B
        ↓                           ↓
    total_loss = Σ(loss_i * w_i)
        ↓
    [optional] + distill_loss(teacher_emb, student_emb)  ← Add distillation
        ↓
    total_loss.backward()          ← Single backward pass
```

### Compression Pipeline

```
mbf_large (6.3M, 1.85G)              ← Original large model, serves as Teacher
    │
    ├─→ Structured Pruning (20%) ─→ ~2.6M    ← -32% params, < 1% accuracy loss
    │
    └─→ Knowledge Distillation ─→ mbf_v3_se (3.9M, 1.15G)  ← -38% params, trained with Teacher guidance
                        │
                        ├─→ Multi-dataset training   ← Joint training on multiple datasets
                        │
                        └─→ ONNX export  ← Deployment ready
```

## Project Structure

```
arcface_torch/
├── backbones/                 # Backbone definitions
│   ├── iresnet.py             #   iResNet series (original)
│   ├── mobilefacenet.py       #   MobileFaceNet series (original + new mbf_v3/mbf_v3_se)
│   ├── vit.py                 #   Vision Transformer (original)
│   └── modules/               #   Sub-modules (new SE attention)
├── configs/                   # Training configs (original + new)
├── eval/                      # Evaluation modules (original)
├── utils/                     # Utility functions (original)
├── scripts/                   # Training and data processing scripts
├── tools/                     # Inference, export and evaluation tools
├── docs/                      # Documentation
│
├── train_v2.py               # Standard training (original)
├── train_v2_distill.py       # Single-dataset distillation training (new)
├── train_v2_prune.py         # Pruning training (new)
├── train_multi_data.py       # Multi-dataset training (new)
├── train_multi_data_distill.py  # Multi-dataset + distillation training (new)
├── dataset.py                # Data loading (original + new next_item)
├── losses.py                 # ArcFace/CosFace loss (original)
├── losses_distill.py         # Distillation loss (new)
├── partial_fc_v2.py          # PartialFC (original)
└── lr_scheduler.py           # Learning rate scheduler (original)
```

## Requirements

- Python >= 3.8
- PyTorch >= 1.12.0
- CUDA >= 11.0

### Installation

```bash
pip install -r requirements.txt
pip install torch-pruning  # Required for pruning
```

## Quick Start

### Dataset Preparation

Training uses Glint360K (360K identities, 17.1M images) in MXNet RecordIO format (`train.rec`, `train.idx`).

Update dataset paths in the config file:

```python
config.rec = "/path/to/glint360k"
config.num_classes = 360232
config.num_image = 17091657
```

### Lightweight Experiment Pipeline

Complete lightweight experiment steps using Glint360K on 4 GPUs:

#### Step 1: ResNet-100 Baseline (Reference)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train_v2.py \
    --config configs/glint360k_r100.py
```

> ResNet-100 (63M, 250MB) serves as the accuracy ceiling reference.

#### Step 2: mbf_v3 Baseline

```bash
bash scripts/run_mbf_v3.sh
```

Actual command:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port 29522 --nproc_per_node=4 \
    train_v2_prune.py --config configs/glint360k_mbf_v3.py
```

#### Step 3: mbf_v3 + SE Attention

```bash
bash scripts/run_mbf_v3_se.sh
```

Actual command:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port 29523 --nproc_per_node=4 \
    train_v2_prune.py --config configs/glint360k_mbf_v3_se.py
```

#### Step 4: Knowledge Distillation

```bash
bash scripts/run_mbf_v3_se_distill.sh
```

Actual command:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port 29524 --nproc_per_node=4 \
    train_v2_distill.py --config configs/glint360k_mbf_v3_se_distill.py
```

#### Step 5: Multi-Dataset Training

```bash
bash scripts/run_multi_data.sh
```

Actual command:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port 12345 --nproc_per_node=4 \
    train_multi_data.py configs/glint360k_facesumd_mbf_v3_se_multi.py
```

Multi-dataset config example:

```python
config.rec = ["/path/to/glint360k", "/path/to/faces_umd"]
config.num_classes = [360232, 8277]
config.num_image = [17091657, 811440]
config.batch_size = [128, 128]           # Per-dataset batch_size
config.loss_w = [0.7, 0.3]              # Weighted loss coefficients
```

#### Step 6: Multi-Dataset + Distillation

```bash
bash scripts/run_multi_data_distill.sh
```

Actual command:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --master_port 12346 --nproc_per_node=4 \
    train_multi_data_distill.py configs/glint360k_facesumd_mbf_v3_se_distill_multi.py
```

Extra distillation config:

```python
config.distill = True
config.teacher_network = "r100"
config.teacher_checkpoint = "/path/to/r100/model.pt"
config.distill_alpha = 0.5              # CE weight, (1-alpha) = distillation weight
config.distill_loss_type = "cosine"     # Cosine similarity distillation
```

#### Step 7: Distillation + Channel Pruning

```bash
bash scripts/run_mbf_v3_se_distill_prune.sh
```

Actual command:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --master_port 29526 --nproc_per_node=4 \
    train_v2_distill.py --config configs/glint360k_mbf_v3_se_distill_prune.py \
    --prune_ratio 0.2
```

### Model Validation & Export

```bash
# Standard model validation
CUDA_VISIBLE_DEVICES=0 python tools/inference_val.py \
    --config configs/glint360k_mbf_v3_se.py \
    --network mbf_v3_se \
    --weight /path/to/model.pt

# Pruned model validation
CUDA_VISIBLE_DEVICES=0 python tools/inference_val_prune.py \
    --config configs/glint360k_mbf_v3_se.py \
    --network mbf_v3_se \
    --weight /path/to/pruned/model.pt \
    --prune_ratio 0.2

# Pruned model validation + ONNX export
CUDA_VISIBLE_DEVICES=0 python tools/inference_val_prune.py \
    --config configs/glint360k_mbf_v3_se.py \
    --network mbf_v3_se \
    --weight /path/to/pruned/model.pt \
    --prune_ratio 0.2 --onnx

# Standard ONNX export
python tools/torch2onnx.py \
    --input /path/to/model.pt \
    --output /path/to/model.onnx \
    --network mbf_v3_se

# Model complexity analysis
python tools/flops.py mbf_v3_se
```

## Supported Backbones

### Original Networks

| Network | Description |
|---------|-------------|
| `r18` / `r34` / `r50` / `r100` / `r200` | iResNet series |
| `mbf` | MobileFaceNet (2.05M, 0.45G) |
| `mbf_large` | MobileFaceNet Large (6.3M) |
| `vit_t` / `vit_s` / `vit_b` / `vit_l` / `vit_h` | Vision Transformer series |

### New Lightweight Networks

| Network | Description |
|---------|-------------|
| `mbf_v3` | Lightweight MobileFaceNet with tuned scale/blocks (3.7M) |
| `mbf_v3_se` | mbf_v3 + SE attention module (3.9M) |

## Supported Datasets

| Dataset | Identities | Images |
|---------|-----------|--------|
| MS1MV2 | 87K | 5.8M |
| MS1MV3 | 93K | 5.2M |
| Glint360K | 360K | 17.1M |
| WebFace4M | 200K | 4.2M |
| WebFace12M | 600K | 12M |
| WebFace42M | 2M | 42.5M |

Data format: MXNet RecordIO (`train.rec`, `train.idx`). See [prepare_custom_dataset.md](docs/prepare_custom_dataset.md) for custom dataset preparation.

## Training Configuration

All config files are in `configs/`, inheriting from `configs/base.py`:

```python
config.network = "mbf_v3_se"      # Backbone
config.embedding_size = 512        # Feature dimension
config.batch_size = 256            # Per-GPU batch size
config.lr = 0.1                    # Learning rate
config.optimizer = "sgd"           # Optimizer (sgd/adamw)
config.margin_list = (1.0, 0.0, 0.4)  # CosFace loss
config.sample_rate = 0.9           # PartialFC sampling rate
config.fp16 = True                 # Mixed precision training
config.gradient_acc = 2            # Gradient accumulation steps
config.num_epoch = 90              # Training epochs
config.rec = "/path/to/dataset"    # Dataset path
config.num_classes = 360232        # Number of classes
```

Extra config for distillation:

```python
config.distill = True
config.teacher_network = "mbf_large"
config.teacher_checkpoint = "/path/to/teacher/model.pt"
config.distill_alpha = 0.5         # CE vs distillation loss weight
config.distill_loss_type = "cosine" # cosine / l2
```

Extra config for multi-dataset training:

```python
config.rec = ["/path/to/dataset_A", "/path/to/dataset_B"]  # N dataset paths
config.num_classes = [360232, 8277]      # Per-dataset class count
config.num_image = [17091657, 811440]    # Per-dataset image count
config.batch_size = [128, 128]           # Per-dataset batch_size
config.loss_w = [0.7, 0.3]              # Weighted loss coefficients
```

---

## Experimental Results

> Training dataset: Glint360K (360K identities, 17.1M images). Evaluation: 7 benchmarks.

### Lightweight Pipeline Comparison

| Phase | Model | Params | Size | LFW | VGG2_FP | AgeDB_30 | CALFW | CFP_FF | CPLFW | CFP_FP |
|-------|-------|--------|------|-----|---------|----------|-------|--------|-------|--------|
| Reference | ResNet-100 | 63M | 250MB | 99.82% | 96.02% | 98.77% | 96.05% | 99.84% | 94.85% | 99.27% |
| Phase 1 | mbf_v3 | 3.7M | 15MB | 99.78% | 95.54% | 97.83% | 96.03% | 99.89% | 93.43% | 98.44% |
| Phase 2 | mbf_v3_se | 3.9M | 16MB | 99.83% | 95.76% | 98.05% | 96.10% | 99.91% | 93.52% | 98.71% |
| Phase 3 | Distill | 3.9M | 16MB | 99.83% | 95.64% | 97.93% | 96.13% | 99.87% | 93.53% | 98.64% |
| Phase 3+ | Distill+Prune | 2.6M | 11MB | 99.82% | 95.52% | 97.80% | 96.02% | 99.86% | 93.00% | 98.26% |

### Key Findings

1. **mbf_v3: -41% parameters** (6.3M→3.7M), LFW drops only 0.09% — scale/blocks tuning is effective
2. **SE module: +4% parameters**, average +0.13% across 7 benchmarks — high cost-effectiveness
3. **20% channel pruning: -32% size** (16MB→11MB), accuracy drops only 0.01%~0.53%
4. **Final model mbf_v3_se_distill_prune**: 11MB / LFW 99.82% vs ResNet-100 (250MB) — **96% size reduction**, only 0.01% lower on LFW

> Detailed report: [docs/lightweight_experiment_report.md](docs/lightweight_experiment_report.md)

### Multi-Dataset Training Experiments

> Training: Glint360K (360K IDs, 17.1M images) + Faces_UMD (8K IDs, 0.8M images).
> Evaluation: LFW, CFP-FP, AgeDB-30, 10 epochs, 4 GPUs.

| Experiment | Model | LFW | CFP-FP | AgeDB-30 | Description |
|------------|-------|-----|--------|----------|-------------|
| Multi-dataset baseline | mbf_v3_se | 99.77% | 97.64% | 97.22% | Weighted joint training on two datasets |
| Multi-dataset + distill | mbf_v3_se | 99.73% | 97.20% | **97.35%** | Combined with R100 Teacher distillation |

#### Key Findings

1. **Distillation excels on hard tasks**: AgeDB-30 (age variation) is the most challenging benchmark, where distillation reaches 97.35%, surpassing the baseline's 97.22%
2. **Training efficiency**: Distillation achieves comparable performance in fewer steps (162k vs 298k), though each step is ~2x slower due to Teacher forward pass
3. **Multi-dataset + distillation are composable**: The two techniques don't conflict and can be combined for improved generalization

> Detailed results: [docs/multi_dataset_experiment_results.md](docs/multi_dataset_experiment_results.md)

## References

```bibtex
@inproceedings{deng2019arcface,
  title={Arcface: Additive angular margin loss for deep face recognition},
  author={Deng, Jiankang and Guo, Jia and Xue, Niannan and Zafeiriou, Stefanos},
  booktitle={CVPR},
  year={2019}
}

@inproceedings{an2022partialfc,
  author={An, Xiang and Deng, Jiankang and Guo, Jia and Feng, Ziyong and Zhu, XuHan and Yang, Jing and Liu, Tongliang},
  title={Killing Two Birds With One Stone: Efficient and Robust Training of Face Recognition CNNs by Partial FC},
  booktitle={CVPR},
  year={2022},
}

@inproceedings{zhu2021webface260m,
  title={Webface260m: A benchmark unveiling the power of million-scale deep face recognition},
  author={Zhu, Zheng and Huang, Guan and Deng, Jiankang and Ye, Yun and Huang, Junjie and Chen, Xinze and Zhu, Jiagang and Yang, Tian and Lu, Jiwen and Du, Dalong and Zhou, Jie},
  booktitle={CVPR},
  year={2021}
}
```

## Acknowledgements

- [InsightFace](https://github.com/deepinsight/insightface) - Original ArcFace Torch project
- [cavaface.pytorch](https://github.com/cavalleria/cavaface.pytorch) - MobileFaceNet implementation reference
- [Torch-Pruning](https://github.com/VainF/Torch-Pruning) - Structured pruning library

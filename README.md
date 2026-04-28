# ArcFace Torch (Enhanced)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/killing-two-birds-with-one-stone-efficient/face-verification-on-ijb-c)](https://paperswithcode.com/sota/face-verification-on-ijb-c?p=killing-two-birds-with-one-stone-efficient)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/killing-two-birds-with-one-stone-efficient/face-verification-on-ijb-b)](https://paperswithcode.com/sota/face-verification-on-ijb-b?p=killing-two-birds-with-one-stone-efficient)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/killing-two-birds-with-one-stone-efficient/face-verification-on-agedb-30)](https://paperswithcode.com/sota/face-verification-on-agedb-30?p=killing-two-birds-with-one-stone-efficient)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/killing-two-birds-with-one-stone-efficient/face-verification-on-cfp-fp)](https://paperswithcode.com/sota/face-verification-on-cfp-fp?p=killing-two-birds-with-one-stone-efficient)

> 基于 [InsightFace ArcFace Torch](https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch) 的增强版本。在保留原版全部能力的基础上，新增了**轻量化模型设计 (mbf_v3)**、**结构化剪枝**、**知识蒸馏**和**多数据集训练**四项核心改进，构建了面向边缘部署的完整模型压缩流水线。

## 核心改进

本项目在原版 ArcFace Torch 基础上做了三项主要改进：

### 1. 轻量化模型 mbf_v3

原版的 MobileFaceNet 系列采用 MobileNetV2 风格的倒残差结构（1x1 卷积升维 → 3x3 深度卷积 → 1x1 卷积降维 + 残差连接）。`mbf_v3` 在此基础上，通过调整 `scale`（通道宽度倍率）和 `blocks`（各阶段残差块数量）两个核心参数，在保持相同架构的前提下实现模型轻量化：

| 变体 | scale | blocks | 各阶段通道数 | 残差块总数 | 参数量 | FLOPs | 定位 |
|------|-------|--------|-------------|-----------|--------|-------|------|
| `mbf` | 2 | (1,4,6,2) | 128→128→256→256 | 13 | 2.05M | 0.45G | 原版超轻量 |
| `mbf_large` | 4 | (2,8,12,4) | 256→256→512→512 | 26 | 6.30M | 1.85G | 原版大模型 |
| **`mbf_v3`** | **3** | **(2,6,8,3)** | **192→192→384→384** | **19** | **3.71M** | **1.15G** | **新增轻量化** |
| `mbf_v3_se` | 3 | (2,6,8,3) | 192→192→384→384 | 19+SE | 3.86M | 1.15G | 新增+注意力 |

`mbf_v3` 相比 `mbf_large`：
- **通道宽度**: scale 4→3，减少 25%
- **网络深度**: blocks 总数 26→19，减少 27%
- **参数量**: 6.3M→3.7M，减少 **41%**
- **FLOPs**: 1.85G→1.15G，减少 **38%**
- **架构不变**: 保持 MobileNetV2 倒残差 + 深度可分离卷积的设计

`mbf_v3_se` 在 `mbf_v3` 基础上引入 SE (Squeeze-and-Excitation) 注意力模块，参数仅增加 4% (3.71M→3.86M)，用于在蒸馏阶段提升精度。

### 2. 结构化剪枝

基于 [Torch-Pruning](https://github.com/VainF/Torch-Pruning) 的 `GroupNormPruner`，对 `mbf_large` 进行通道级结构化剪枝：

- 使用 L2 范数作为通道重要性指标
- 保护 GDC 头部（`features`、`conv_sep`）不被剪枝
- 20% 剪枝率下精度损失 < 1%

### 3. 知识蒸馏

从大模型 (Teacher) 向轻量化模型 (Student) 进行 Embedding 级蒸馏：

- Teacher: `mbf_large` (6.3M) → Student: `mbf_v3_se` (3.9M)
- 蒸馏损失: 余弦相似度 (cosine) 或 L2 范数
- 总损失: `L = α * L_CE + (1-α) * L_distill`，α=0.5

### 4. 多数据集训练

原版 ArcFace Torch 仅支持单数据集训练。`train_multi_data.py` 实现了 N 数据集联合训练：

- **N 个 DataLoader** 同步迭代，每个数据集独立采样
- **共享骨干网络** 单次前向传播，`torch.cat` → `backbone` → `torch.split`
- **N 个 PartialFC 头** 每个数据集独立的分类头，独立的类别空间
- **加权损失** `L = Σ(w_i * L_CE_i)`，可调节各数据集的贡献权重
- **支持蒸馏** `train_multi_data_distill.py` 在多数据集基础上叠加知识蒸馏

```
DataLoader_A (glint360k)    DataLoader_B (faces_umd)     ← N 个数据集
        ↓                           ↓
    img = torch.cat([img_A, img_B])
        ↓
    backbone(img)                   ← 共享骨干，单次前向
        ↓
    torch.split(embeddings)
        ↓                           ↓
  PartialFC_A                 PartialFC_B               ← N 个独立分类头
  loss_A * w_A                loss_B * w_B
        ↓                           ↓
    total_loss = Σ(loss_i * w_i)
        ↓
    [可选] + distill_loss(teacher_emb, student_emb)      ← 叠加蒸馏
        ↓
    total_loss.backward()          ← 单次反向传播
```

### 压缩流水线

```
mbf_large (6.3M, 1.85G)              ← 原版大模型，作为 Teacher
    │
    ├─→ 结构化剪枝 (20%) ─→ ~2.6M    ← 参数量 -32%，精度损失 < 1%
    │
    └─→ 知识蒸馏 ─→ mbf_v3_se (3.9M, 1.15G)  ← 参数量 -38%，由 Teacher 指导训练
                        │
                        ├─→ 多数据集训练   ← 多个数据集联合训练，提升泛化性
                        │
                        └─→ ONNX 导出  ← 部署就绪
```

## 项目结构

```
arcface_torch/
├── backbones/                 # 骨干网络定义
│   ├── iresnet.py             #   iResNet 系列 (原版)
│   ├── mobilefacenet.py       #   MobileFaceNet 系列 (原版 + 新增 mbf_v3/mbf_v3_se)
│   ├── vit.py                 #   Vision Transformer (原版)
│   └── modules/               #   子模块 (新增 SE 注意力)
├── configs/                   # 训练配置 (原版 + 新增)
├── eval/                      # 评估验证模块 (原版)
├── utils/                     # 工具函数 (原版)
├── scripts/                   # 训练和数据处理脚本
├── tools/                     # 推理、导出和评估工具
├── docs/                      # 文档 (原版)
│
├── train_v2.py               # 标准训练 (原版)
├── train_v2_distill.py       # 单数据集蒸馏训练 (新增)
├── train_v2_prune.py         # 剪枝训练 (新增)
├── train_multi_data.py       # 多数据集训练 (新增)
├── train_multi_data_distill.py  # 多数据集 + 蒸馏训练 (新增)
├── dataset.py                # 数据加载 (原版 + 新增 next_item)
├── losses.py                 # ArcFace/CosFace 损失 (原版)
├── losses_distill.py         # 蒸馏损失 (新增)
├── partial_fc_v2.py          # PartialFC (原版)
└── lr_scheduler.py           # 学习率调度器 (原版)
```

## 环境要求

- Python >= 3.8
- PyTorch >= 1.12.0
- CUDA >= 11.0

### 安装

```bash
pip install -r requirements.txt
pip install torch-pruning  # 剪枝功能依赖
```

## 快速开始

### 数据集准备

使用 Glint360K (360K 身份, 17.1M 图片) 训练，数据格式为 MXNet RecordIO (`train.rec`, `train.idx`)。

修改配置文件中的数据集路径：

```python
config.rec = "/path/to/glint360k"
config.num_classes = 360232
config.num_image = 17091657
```

### 轻量化实验流水线

以下为完整的轻量化实验步骤，基于 Glint360K 数据集，4 GPU 训练：

#### Step 1: ResNet-100 基线 (参考)

```bash
# 4 GPU 训练 ResNet-100 基线模型
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train_v2.py \
    --config configs/glint360k_r100.py
```

> ResNet-100 (63M, 250MB) 作为大模型精度天花板参考。

#### Step 2: mbf_v3 基线训练

```bash
# Phase 1: mbf_v3 轻量化基线 (3.7M, scale=3, blocks=(2,6,8,3))
bash scripts/run_mbf_v3.sh
```

实际执行命令：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port 29522 --nproc_per_node=4 \
    train_v2_prune.py --config configs/glint360k_mbf_v3.py
```

#### Step 3: mbf_v3 + SE 注意力

```bash
# Phase 2: mbf_v3_se 增加 SE 注意力模块 (3.9M)
bash scripts/run_mbf_v3_se.sh
```

实际执行命令：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port 29523 --nproc_per_node=4 \
    train_v2_prune.py --config configs/glint360k_mbf_v3_se.py
```

#### Step 4: 知识蒸馏

```bash
# Phase 3: Teacher(mbf_large) → Student(mbf_v3_se), cosine 蒸馏
bash scripts/run_mbf_v3_se_distill.sh
```

实际执行命令：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port 29524 --nproc_per_node=4 \
    train_v2_distill.py --config configs/glint360k_mbf_v3_se_distill.py
```

#### Step 5: 多数据集训练

```bash
# 多数据集联合训练 (glint360k + faces_umd)
bash scripts/run_multi_data.sh
```

实际执行命令：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port 12345 --nproc_per_node=4 \
    train_multi_data.py configs/glint360k_facesumd_r50_multi.py
```

多数据集配置示例：

```python
config.rec = ["/path/to/glint360k", "/path/to/faces_umd"]
config.num_classes = [360232, 8277]
config.num_image = [17091657, 811440]
config.batch_size = [128, 128]           # 每个数据集独立的 batch_size
config.loss_w = [0.7, 0.3]              # 加权损失权重
```

#### Step 6: 多数据集 + 蒸馏训练

```bash
# 多数据集 + 知识蒸馏 (Teacher: R100, Student: mbf_v3_se)
bash scripts/run_multi_data_distill.sh
```

实际执行命令：

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --master_port 12346 --nproc_per_node=4 \
    train_multi_data_distill.py configs/glint360k_facesumd_mbf_v3_se_distill_multi.py
```

蒸馏额外配置：

```python
config.distill = True
config.teacher_network = "r100"
config.teacher_checkpoint = "/path/to/r100/model.pt"
config.distill_alpha = 0.5              # CE 权重，(1-alpha) 为蒸馏权重
config.distill_loss_type = "cosine"     # cosine 相似度蒸馏
```

#### Step 7: 蒸馏 + 通道剪枝

```bash
# 蒸馏同时进行 20% 通道剪枝
bash scripts/run_mbf_v3_se_distill_prune.sh
```

实际执行命令：

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --master_port 29526 --nproc_per_node=4 \
    train_v2_distill.py --config configs/glint360k_mbf_v3_se_distill_prune.py \
    --prune_ratio 0.2
```

### 模型验证与导出

```bash
# 标准模型验证
CUDA_VISIBLE_DEVICES=0 python tools/inference_val.py \
    --config configs/glint360k_mbf_v3_se.py \
    --network mbf_v3_se \
    --weight /path/to/model.pt

# 剪枝模型验证
CUDA_VISIBLE_DEVICES=0 python tools/inference_val_prune.py \
    --config configs/glint360k_mbf_v3_se.py \
    --network mbf_v3_se \
    --weight /path/to/pruned/model.pt \
    --prune_ratio 0.2

# 剪枝模型验证并导出 ONNX
CUDA_VISIBLE_DEVICES=0 python tools/inference_val_prune.py \
    --config configs/glint360k_mbf_v3_se.py \
    --network mbf_v3_se \
    --weight /path/to/pruned/model.pt \
    --prune_ratio 0.2 --onnx

# 标准 ONNX 导出
python tools/torch2onnx.py \
    --input /path/to/model.pt \
    --output /path/to/model.onnx \
    --network mbf_v3_se

# 模型复杂度分析
python tools/flops.py mbf_v3_se
```

## 支持的骨干网络

### 原版网络

| 网络 | 说明 |
|------|------|
| `r18` / `r34` / `r50` / `r100` / `r200` | iResNet 系列 |
| `mbf` | MobileFaceNet (2.05M, 0.45G) |
| `mbf_large` | MobileFaceNet Large (6.3M) |
| `vit_t` / `vit_s` / `vit_b` / `vit_l` / `vit_h` | Vision Transformer 系列 |

### 新增轻量化网络

| 网络 | 说明 |
|------|------|
| `mbf_v3` | 基于原版 MobileFaceNet 调整 scale/blocks 的轻量化版本 (3.7M) |
| `mbf_v3_se` | mbf_v3 + SE 注意力模块 (3.9M) |

## 支持的数据集

| 数据集 | 身份数 | 图片数 |
|--------|--------|--------|
| MS1MV2 | 87K | 5.8M |
| MS1MV3 | 93K | 5.2M |
| Glint360K | 360K | 17.1M |
| WebFace4M | 200K | 4.2M |
| WebFace12M | 600K | 12M |
| WebFace42M | 2M | 42.5M |

数据格式: MXNet RecordIO (`train.rec`, `train.idx`)。参考 [prepare_custom_dataset.md](docs/prepare_custom_dataset.md) 准备自定义数据集。

## 训练配置

所有配置文件位于 `configs/` 目录，继承自 `configs/base.py`：

```python
config.network = "mbf_v3_se"      # 骨干网络
config.embedding_size = 512        # 特征维度
config.batch_size = 256            # 每 GPU 批大小
config.lr = 0.1                    # 学习率
config.optimizer = "sgd"           # 优化器 (sgd/adamw)
config.margin_list = (1.0, 0.0, 0.4)  # CosFace 损失
config.sample_rate = 0.9           # PartialFC 采样率
config.fp16 = True                 # 混合精度训练
config.gradient_acc = 2            # 梯度累积步数
config.num_epoch = 90              # 训练轮数
config.rec = "/path/to/dataset"    # 数据集路径
config.num_classes = 360232        # 类别数
```

蒸馏训练额外配置：

```python
config.distill = True
config.teacher_network = "mbf_large"
config.teacher_checkpoint = "/path/to/teacher/model.pt"
config.distill_alpha = 0.5         # CE 和蒸馏损失的权重
config.distill_loss_type = "cosine" # cosine / l2
```

多数据集训练额外配置：

```python
config.rec = ["/path/to/dataset_A", "/path/to/dataset_B"]  # N 个数据集路径
config.num_classes = [360232, 8277]      # 每个数据集的类别数
config.num_image = [17091657, 811440]    # 每个数据集的图片数
config.batch_size = [128, 128]           # 每个数据集的 batch_size
config.loss_w = [0.7, 0.3]              # 加权损失权重
```

---

## 实验结果

> 训练数据集: Glint360K (360K 身份, 17.1M 图片)，验证集: 7 个 benchmark。

### 轻量化流水线对比

| 阶段 | 模型 | 参数量 | 体积 | LFW | VGG2_FP | AgeDB_30 | CALFW | CFP_FF | CPLFW | CFP_FP |
|------|------|--------|------|-----|---------|----------|-------|--------|-------|--------|
| 参考基线 | ResNet-100 | 63M | 250MB | 99.82% | 96.02% | 98.77% | 96.05% | 99.84% | 94.85% | 99.27% |
| Phase 1 | mbf_v3 | 3.7M | 15MB | 99.78% | 95.54% | 97.83% | 96.03% | 99.89% | 93.43% | 98.44% |
| Phase 2 | mbf_v3_se | 3.9M | 16MB | 99.83% | 95.76% | 98.05% | 96.10% | 99.91% | 93.52% | 98.71% |
| Phase 3 | 蒸馏 | 3.9M | 16MB | 99.83% | 95.64% | 97.93% | 96.13% | 99.87% | 93.53% | 98.64% |
| Phase 3+ | 蒸馏+剪枝 | 2.6M | 11MB | 99.82% | 95.52% | 97.80% | 96.02% | 99.86% | 93.00% | 98.26% |

### 关键发现

1. **mbf_v3 参数量 -41%** (6.3M→3.7M)，LFW 仅降 0.09%，scale/blocks 调整策略有效
2. **SE 模块代价 +4% 参数**，7 个验证集平均 +0.13%，性价比高
3. **20% 通道剪枝体积 -32%** (16MB→11MB)，精度仅降 0.01%~0.53%
4. **最终模型 mbf_v3_se_distill_prune**: 11MB / LFW 99.82%，相比 ResNet-100 (250MB) 体积减少 **96%**，LFW 仅低 0.01%

> 详细实验报告见 [docs/lightweight_experiment_report.md](docs/lightweight_experiment_report.md)

### 多数据集训练实验

> 训练数据集: Glint360K (360K 身份, 17.1M 图片) + Faces_UMD (8K 身份, 0.8M 图片)，
> 验证集: LFW, CFP-FP, AgeDB-30，10 个 epoch，4 GPU 训练。

| 实验 | 模型 | LFW | CFP-FP | AgeDB-30 | 说明 |
|------|------|-----|--------|----------|------|
| 多数据集基线 | mbf_v3_se | 99.77% | 97.64% | 97.22% | 两个数据集加权联合训练 |
| 多数据集+蒸馏 | mbf_v3_se | 99.73% | 97.20% | **97.35%** | 叠加 R100 Teacher 蒸馏 |

#### 关键发现

1. **蒸馏在难任务上更有优势**: AgeDB-30 (年龄变化) 是最具挑战性的验证集，蒸馏达到 97.35%，超过基线的 97.22%
2. **训练效率**: 蒸馏在更少的 step (162k vs 298k) 就达到了可比的性能，尽管每步因 Teacher 前向传播慢约 2x
3. **多数据集+蒸馏可叠加**: 两种技术互不冲突，可组合使用以提升模型泛化能力

> 详细实验数据见 [docs/multi_dataset_experiment_results.md](docs/multi_dataset_experiment_results.md)

## 参考文献

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

## 致谢

- [InsightFace](https://github.com/deepinsight/insightface) - 原版 ArcFace Torch 项目
- [cavaface.pytorch](https://github.com/cavalleria/cavaface.pytorch) - MobileFaceNet 实现参考
- [Torch-Pruning](https://github.com/VainF/Torch-Pruning) - 结构化剪枝库

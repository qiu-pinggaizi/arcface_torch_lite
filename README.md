# ArcFace Torch (Enhanced)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/killing-two-birds-with-one-stone-efficient/face-verification-on-ijb-c)](https://paperswithcode.com/sota/face-verification-on-ijb-c?p=killing-two-birds-with-one-stone-efficient)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/killing-two-birds-with-one-stone-efficient/face-verification-on-ijb-b)](https://paperswithcode.com/sota/face-verification-on-ijb-b?p=killing-two-birds-with-one-stone-efficient)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/killing-two-birds-with-one-stone-efficient/face-verification-on-agedb-30)](https://paperswithcode.com/sota/face-verification-on-agedb-30?p=killing-two-birds-with-one-stone-efficient)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/killing-two-birds-with-one-stone-efficient/face-verification-on-cfp-fp)](https://paperswithcode.com/sota/face-verification-on-cfp-fp?p=killing-two-birds-with-one-stone-efficient)

> 基于 [InsightFace ArcFace Torch](https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch) 的增强版本。在保留原始全部能力的基础上，新增了**轻量化模型设计**、**知识蒸馏**和**结构化剪枝**的模型压缩流水线，面向边缘部署场景。

## 与原版的关系

本项目 fork 自 InsightFace 官方的 `arcface_torch`，保留了原版的全部功能：

- 分布式训练 (多机多卡)、混合精度 (FP16)、PartialFC 大规模分类
- iResNet / MobileFaceNet / ViT 等骨干网络
- MS1MV2 / MS1MV3 / Glint360K / WebFace42M 等数据集支持
- ONNX 导出、IJB-C 评估等工具链

在此基础上，本项目**新增**了以下能力：

| 新增能力 | 入口文件 | 说明 |
|----------|----------|------|
| 轻量化骨干网络 | `backbones/mobilefacenet.py` | mbf_v3 (3.7M) / mbf_v3_se (3.9M) 等轻量变体 |
| SE 注意力模块 | `backbones/modules/se_module.py` | Squeeze-and-Excitation 模块，可插入任意网络 |
| 知识蒸馏训练 | `train_v2_distill.py` | Teacher-Student 蒸馏，支持 cosine/L2 embedding 蒸馏 |
| 结构化剪枝 | `train_v2_prune.py` | 基于 GroupNormPruner 的通道剪枝 |
| 蒸馏损失函数 | `losses_distill.py` | Embedding 级蒸馏损失 (cosine similarity / L2) |

## 项目结构

```
arcface_torch/
├── backbones/                 # 骨干网络定义
│   ├── iresnet.py             #   iResNet 系列 (原版)
│   ├── mobilefacenet.py       #   MobileFaceNet 系列 (原版 + 新增轻量变体)
│   ├── vit.py                 #   Vision Transformer (原版)
│   └── modules/               #   子模块 (新增 SE 注意力等)
├── configs/                   # 训练配置 (原版 + 新增)
├── eval/                      # 评估验证模块 (原版)
├── utils/                     # 工具函数 (原版)
├── scripts/                   # 训练和数据处理脚本
├── tools/                     # 推理、导出和评估工具
├── docs/                      # 文档 (原版)
│
├── train_v2.py               # 标准训练 (原版)
├── train_v2_distill.py       # 知识蒸馏训练 (新增)
├── train_v2_prune.py         # 剪枝训练 (新增)
├── dataset.py                # 数据加载 (原版)
├── losses.py                 # ArcFace/CosFace 损失 (原版)
├── losses_distill.py         # 蒸馏损失 (新增)
├── partial_fc_v2.py          # PartialFC (原版)
└── lr_scheduler.py           # 学习率调度器 (原版)
```

## 轻量化模型压缩流水线

本项目的核心价值在于提供了一套**从大模型到边缘部署的完整压缩流水线**：

```
┌─────────────────────────────────────────────────────────┐
│  Stage 1: 基线训练                                        │
│  mbf_large (6.3M params, 1.86 GFLOPs)                    │
│  → Glint360K, 90 epochs, PartialFC                       │
├─────────────────────────────────────────────────────────┤
│  Stage 2: 结构化剪枝                                      │
│  GroupNormPruner, 25% channel pruning                    │
│  → 4.7M params, 精度损失 < 1%                             │
├─────────────────────────────────────────────────────────┤
│  Stage 3: 知识蒸馏                                        │
│  Teacher: mbf_large → Student: mbf_v3_se (3.9M)          │
│  → Embedding cosine distillation + CE loss               │
├─────────────────────────────────────────────────────────┤
│  Stage 4: ONNX 导出                                       │
│  → 部署就绪的 ONNX 模型                                   │
└─────────────────────────────────────────────────────────┘
```

### 轻量化模型对比

| 模型 | 参数量 | FLOPs | 相比 mbf_large | 说明 |
|------|--------|-------|----------------|------|
| `mbf_large` | 6.30M | 1.86G | baseline | 原版 MobileFaceNet Large |
| `mbf_v3` | 3.71M | ~1.1G | **-41%** | scale=3, blocks=(2,6,8,3) |
| `mbf_v3_se` | 3.86M | ~1.15G | **-39%** | mbf_v3 + SE 注意力 |

## 环境要求

- Python >= 3.8
- PyTorch >= 1.12.0
- CUDA >= 11.0

### 安装

```bash
pip install -r requirements.txt
# 剪枝功能需要 torch-pruning
pip install torch-pruning
```

## 快速开始

### 标准训练 (原版能力)

```bash
# 单 GPU
python train_v2.py configs/ms1mv3_r50_onegpu

# 8 GPU 分布式
torchrun --nproc_per_node=8 train_v2.py configs/ms1mv3_r50

# 多机分布式 (2 机 x 8 卡)
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr="ip1" --master_port=12581 \
    train_v2.py configs/wf42m_pfc02_16gpus_r100
```

### 轻量化模型训练 (新增)

```bash
# mbf_v3 基线训练
bash scripts/run_mbf_v3.sh

# mbf_v3 + SE 注意力
bash scripts/run_mbf_v3_se.sh
```

### 知识蒸馏 (新增)

```bash
# Teacher: mbf_large → Student: mbf_v3_se
bash scripts/run_mbf_v3_se_distill.sh

# 带剪枝的蒸馏
bash scripts/run_mbf_v3_se_distill_prune.sh
```

### 结构化剪枝 (新增)

```bash
# mbf_large 25% 通道剪枝
bash scripts/run.5max.sh
```

### 推理与导出

```bash
# 模型推理
python tools/inference.py --weight model.pt --network mbf_v3_se --img test.jpg

# 模型验证
python tools/inference_val.py --config configs/glint360k_mbf_v3_se.py --network mbf_v3_se --weight model.pt

# ONNX 导出
python tools/torch2onnx.py

# 模型复杂度分析
python tools/flops.py mbf_v3_se

# IJB-C 评估
python tools/eval_ijbc.py
```

## 支持的骨干网络

### 原版网络

| 网络 | 说明 |
|------|------|
| `r18` / `r34` / `r50` / `r100` / `r200` | iResNet 系列 |
| `mbf` | MobileFaceNet (0.45G FLOPs) |
| `mbf_large` | MobileFaceNet Large (6.3M) |
| `vit_t` / `vit_s` / `vit_b` / `vit_l` / `vit_h` | Vision Transformer 系列 |

### 新增轻量化网络

| 网络 | 说明 |
|------|------|
| `mbf_v3` | 轻量化 MobileFaceNet V3 (3.7M, scale=3) |
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

---

## 原版性能基准

> 以下数据来自 InsightFace 原版项目。

### Single-Host GPU

| Datasets       | Backbone            | **MFR-ALL** | IJB-C(1E-4) | IJB-C(1E-5) |
|:---------------|:--------------------|:------------|:------------|:------------|
| MS1MV2         | mobilefacenet-0.45G | 62.07       | 93.61       | 90.28       |
| MS1MV2         | r50                 | 75.13       | 95.97       | 94.07       |
| MS1MV2         | r100                | 78.12       | 96.37       | 94.27       |
| MS1MV3         | r50                 | 79.14       | 96.37       | 94.47       |
| MS1MV3         | r100                | 81.97       | 96.85       | 95.02       |
| Glint360K      | r50                 | 86.34       | 97.16       | 95.81       |
| Glint360K      | r100                | 89.52       | 97.55       | 96.38       |
| WF4M           | r100                | 89.87       | 97.19       | 95.48       |
| WF12M-PFC-0.2  | r100                | 94.75       | 97.60       | 95.90       |
| WF42M-PFC-0.2  | r100                | 96.27       | 97.70       | 96.31       |
| WF42M-PFC-0.3  | ViT-B-11G           | 97.16       | 97.91       | 97.05       |

### Multi-Host GPU

| Datasets         | Backbone(bs*gpus) | **MFR-ALL** | IJB-C(1E-4) | IJB-C(1E-5) |
|:-----------------|:------------------|:------------|:------------|:------------|
| WF42M-PFC-0.2    | r50(512*16)       | 93.96       | 97.46       | 96.12       |
| WF42M-PFC-0.2    | r100(256*16)      | 96.69       | 97.85       | 96.63       |

### ViT For Face Recognition

| Datasets      | Backbone(bs)  | FLOPs | **MFR-ALL** | IJB-C(1E-4) | IJB-C(1E-5) |
|:--------------|:--------------|:------|:------------|:------------|:------------|
| WF42M-PFC-0.3 | VIT-T(384*64) | 1.5   | 92.24       | 97.31       | 95.97       |
| WF42M-PFC-0.3 | VIT-S(384*64) | 5.7   | 95.87       | 97.73       | 96.57       |
| WF42M-PFC-0.3 | VIT-B(384*64) | 11.4  | 97.42       | 97.90       | 97.04       |
| WF42M-PFC-0.3 | VIT-L(384*64) | 25.3  | 97.85       | 98.00       | 97.23       |

### Speed Benchmark

> Training Speed (Samples/sec) on Tesla V100 32GB x 8

| Number of Identities | Data Parallel | Model Parallel | Partial FC 0.1 |
|:---------------------|:--------------|:---------------|:---------------|
| 125,000              | 4681          | 4824           | 5004           |
| 1,400,000            | 1672          | 3043           | 4738           |
| 5,500,000            | -             | 1389           | 3975           |
| 16,000,000           | -             | -              | 2679           |
| 29,000,000           | -             | -              | 1855           |

## 文档

- [安装指南](docs/install.md)
- [DALI 安装](docs/install_dali.md)
- [模型库](docs/modelzoo.md)
- [速度基准](docs/speed_benchmark.md)
- [自定义数据集](docs/prepare_custom_dataset.md)
- [WebFace42M 准备](docs/prepare_webface42m.md)
- [评估方法](docs/eval.md)

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

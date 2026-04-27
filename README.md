# ArcFace Torch - 人脸识别训练框架

基于 PyTorch 的分布式人脸识别训练框架，支持多种骨干网络、知识蒸馏、模型剪枝和量化训练。

## 项目概述

本项目是 [InsightFace ArcFace](https://github.com/deepinsight/insightface) 的改进版本，专注于轻量级人脸识别模型的训练和部署。主要特性包括：

- **多骨干网络支持**: iResNet (r18/r34/r50/r100/r200)、MobileFaceNet、ViT 及多种自定义 NAS 架构
- **大规模数据集**: 支持 MS1MV2、MS1MV3、Glint360K、WebFace4M/12M/42M 等数据集
- **分布式训练**: 支持多机多卡分布式训练，混合精度 (FP16) 训练
- **PartialFC**: 高效的大规模分类训练，支持千万级类别
- **模型压缩流水线**: 知识蒸馏 → 结构化剪枝 → QAT 量化 → ONNX 导出

## 项目结构

```
arcface_torch/
├── backbones/                 # 骨干网络定义
│   ├── iresnet.py            #   iResNet 系列 (r18-r200)
│   ├── iresnet2060.py        #   iResNet2060
│   ├── mobilefacenet.py      #   MobileFaceNet 系列 (mbf/mbf_large/mbf_v3/mbf_v3_se)
│   ├── vit.py                #   Vision Transformer 系列
│   ├── fused_*.py            #   NAS 搜索得到的融合架构
│   └── modules/              #   子模块 (SE 注意力模块等)
├── configs/                   # 训练配置文件
│   ├── base.py               #   基础配置
│   ├── glint360k_*.py        #   Glint360K 数据集配置
│   ├── ms1mv2_*.py           #   MS1MV2 数据集配置
│   ├── ms1mv3_*.py           #   MS1MV3 数据集配置
│   ├── wf4m_*.py             #   WebFace4M 数据集配置
│   ├── wf12m_*.py            #   WebFace12M 数据集配置
│   └── wf42m_*.py            #   WebFace42M 数据集配置
├── eval/                      # 评估验证模块
├── utils/                     # 工具函数
├── scripts/                   # 训练和数据处理脚本
├── tools/                     # 推理、导出和评估工具
├── experiments/               # NAS 实验归档
├── docs/                      # 文档
├── train_v2.py               # 标准训练入口
├── train_v2_distill.py       # 知识蒸馏训练
├── train_v2_prune.py         # 剪枝训练
├── train_v2_qat.py           # 量化感知训练 (QAT)
├── dataset.py                # 数据加载
├── losses.py                 # 损失函数 (ArcFace/CosFace/Combined)
├── losses_distill.py         # 蒸馏损失
├── partial_fc_v2.py          # PartialFC 分布式分类
└── lr_scheduler.py           # 学习率调度器
```

## 环境要求

- Python >= 3.8
- PyTorch >= 1.12.0
- CUDA >= 11.0

### 安装依赖

```bash
pip install -r requirement.txt
pip install torch-pruning  # 剪枝功能依赖
```

## 快速开始

### 1. 单 GPU 训练

```bash
python train_v2.py configs/ms1mv3_r50_onegpu
```

### 2. 多 GPU 分布式训练 (8 卡)

```bash
torchrun --nproc_per_node=8 train_v2.py configs/ms1mv3_r50
```

### 3. 多机分布式训练 (2 机 x 8 卡)

```bash
# Node 0
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr="ip1" --master_port=12581 \
    train_v2.py configs/wf42m_pfc02_16gpus_r100

# Node 1
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr="ip1" --master_port=12581 \
    train_v2.py configs/wf42m_pfc02_16gpus_r100
```

### 4. 剪枝训练

```bash
# 使用预训练模型进行结构化剪枝
bash scripts/run.5max.sh
```

### 5. 知识蒸馏训练

```bash
# Teacher: mbf_large → Student: mbf_v3_se
bash scripts/run_mbf_v3_se_distill.sh
```

### 6. 量化感知训练 (QAT)

```bash
bash scripts/run_mbf_v3_se_qat.sh
```

## 支持的骨干网络

| 网络 | 参数量 | 说明 |
|------|--------|------|
| `r18` | 11.2M | iResNet-18 |
| `r34` | 22.3M | iResNet-34 |
| `r50` | 43.6M | iResNet-50 |
| `r100` | 65.1M | iResNet-100 |
| `r200` | 124.5M | iResNet-200 |
| `mbf` | 0.45G FLOPs | MobileFaceNet |
| `mbf_large` | 6.3M | MobileFaceNet Large |
| `mbf_v3` | 3.7M | MobileFaceNet V3 (轻量化) |
| `mbf_v3_se` | 3.9M | MBF V3 + SE 注意力 |
| `vit_t` | 1.5G FLOPs | Vision Transformer Tiny |
| `vit_s` | 5.7G FLOPs | Vision Transformer Small |
| `vit_b` | 11.4G FLOPs | Vision Transformer Base |
| `vit_l` | 25.3G FLOPs | Vision Transformer Large |

## 支持的数据集

| 数据集 | 身份数 | 图片数 | 说明 |
|--------|--------|--------|------|
| MS1MV2 | 87K | 5.8M | MS-Celeb-1M 清洗版 |
| MS1MV3 | 93K | 5.2M | MS-Celeb-1M RetinaFace 清洗版 |
| Glint360K | 360K | 17.1M | 大规模人脸数据集 |
| WebFace4M | 200K | 4.2M | WebFace260M 子集 |
| WebFace12M | 600K | 12M | WebFace260M 子集 |
| WebFace42M | 2M | 42.5M | WebFace260M 完整清洗版 |

数据格式: MXNet RecordIO (`train.rec`, `train.idx`)

## 模型压缩流水线

本项目支持完整的模型压缩流水线：

```
训练基线模型 (mbf_large, 6.3M)
    ↓ 结构化剪枝 (25%)
剪枝模型 (4.7M)
    ↓ 知识蒸馏 (Teacher → Student)
轻量化模型 (mbf_v3_se, 3.9M)
    ↓ QAT INT8 量化
量化模型 (~0.9MB)
    ↓ ONNX 导出
部署模型
```

### 工具脚本

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

## 数据准备

### MXNet RecordIO 格式

```bash
# 将图片目录转换为 RecordIO 格式
bash scripts/run_make_rec.sh

# 使用 DALI 加速数据读取时，需要先打乱数据
python scripts/shuffle_rec.py ms1m-retinaface-t1
```

### 自定义数据集

参考 [prepare_custom_dataset.md](docs/prepare_custom_dataset.md) 准备自定义数据集。

## 训练配置

所有配置文件位于 `configs/` 目录，继承自 `configs/base.py`。主要配置项：

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

## 许可证

本项目基于原始 [InsightFace](https://github.com/deepinsight/insightface) 项目，仅供学术研究使用。

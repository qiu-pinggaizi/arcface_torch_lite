# 轻量化模型实验报告

> 基于 InsightFace ArcFace Torch，使用 Glint360K (360K 身份, 17.1M 图片) 数据集训练，7 个验证集评估。

## 实验总览

```
mbf_large (6.3M, 25MB)                    ← 原版大模型，精度上限参考
    │
    ├─ Phase 1: mbf_v3 (3.7M, 15MB)       ← scale/blocks 参数调整，-41% 参数量
    │
    ├─ Phase 2: mbf_v3_se (3.9M, 16MB)    ← +SE 注意力，平均 +0.13%
    │
    ├─ Phase 3: 蒸馏 (3.9M, 16MB)         ← Teacher: mbf_large → Student: mbf_v3_se
    │
    ├─ Phase 3+: 蒸馏+剪枝 (2.6M, 11MB)   ← 20% 通道剪枝，-32% 体积
    │
参考基线:
  ResNet-100 (63M, 250MB)                 ← 大模型精度天花板
```

---

## 一、基线模型性能

### 1.ResNet-100 基线 (63M, 250MB)

| 验证集   | Accuracy |
| -------- | -------- |
| LFW      | 99.82%   |
| VGG2_FP  | 96.02%   |
| AgeDB_30 | 98.77%   |
| CALFW    | 96.05%   |
| CFP_FF   | 99.84%   |
| CPLFW    | 94.85%   |
| CFP_FP   | 99.27%   |

> 大模型精度天花板，为轻量化方案提供精度上限参考。

---

## 二、轻量化实验 (Phase 1 ~ Phase 3+)

### Phase 1: mbf_v3 基线

**模型设计**: 在原版 MobileFaceNet (MobileNetV2 倒残差结构) 基础上调整参数:

- `scale`: 4 → 3 (通道宽度 -25%)
- `blocks`: (2,8,12,4) → (2,6,8,3) (残差块 -27%)
- 参数量: 6.3M → 3.7M (**-41%**)
- 架构不变: 1x1 Conv 升维 → 3x3 Depthwise Conv → 1x1 Conv 降维 + 残差连接

**训练配置**:

- 数据集: Glint360K (360K 身份, 17.1M 图片)
- 优化器: SGD, lr=0.1
- 精度: FP16 混合精度
- Epoch: 90
- GPU: 4x (batch_size=256×4)

**验证结果**:

| 验证集   | Accuracy |
| -------- | -------- |
| LFW      | 99.78%   |
| VGG2_FP  | 95.54%   |
| AgeDB_30 | 97.83%   |
| CALFW    | 96.03%   |
| CFP_FF   | 99.89%   |
| CPLFW    | 93.43%   |
| CFP_FP   | 98.44%   |

**模型文件**: `output/mbf_v3/model.pt` (15MB)

---

### Phase 2: mbf_v3 + SE 注意力

**模型设计**: 在 mbf_v3 每个 DepthWise 块后插入 SE (Squeeze-and-Excitation) 注意力模块:

- 参数量: 3.7M → 3.9M (+4%)
- SE reduction ratio: 16

**训练配置**: 同 Phase 1

**验证结果**:

| 验证集   | Accuracy | vs Phase 1       |
| -------- | -------- | ---------------- |
| LFW      | 99.83%   | +0.05%           |
| VGG2_FP  | 95.76%   | **+0.22%** |
| AgeDB_30 | 98.05%   | **+0.22%** |
| CALFW    | 96.10%   | +0.07%           |
| CFP_FF   | 99.91%   | +0.02%           |
| CPLFW    | 93.52%   | +0.09%           |
| CFP_FP   | 98.71%   | **+0.27%** |

**结论**: SE 模块在所有 7 个验证集上均带来提升，平均提升约 **+0.13%**，参数代价仅 +4%。

**模型文件**: `output/mbf_v3_se/model.pt` (16MB)

---

### Phase 3: 知识蒸馏

**训练方案**:

- Teacher: mbf_large (6.3M) → Student: mbf_v3_se (3.9M)
- 蒸馏损失: 余弦相似度 (cosine)
- 总损失: `L = α * L_CE + (1-α) * L_distill`, α=0.5
- 训练: 90 epoch, SGD, lr=0.1, FP16

**验证结果**:

| 验证集   | Accuracy | vs Phase 2 |
| -------- | -------- | ---------- |
| LFW      | 99.83%   | +0.00%     |
| VGG2_FP  | 95.64%   | -0.12%     |
| AgeDB_30 | 97.93%   | -0.12%     |
| CALFW    | 96.13%   | +0.03%     |
| CFP_FF   | 99.87%   | -0.04%     |
| CPLFW    | 93.53%   | +0.01%     |
| CFP_FP   | 98.64%   | -0.07%     |

**结论**: 蒸馏后整体与 Phase 2 持平，LFW/CALFW/CPLFW 微升，VGG2_FP/AgeDB_30 微降。蒸馏的主要价值在于为后续剪枝提供更好的初始化。

**模型文件**: `output/mbf_v3_se_distill/model.pt` (16MB)

---

### Phase 3+: 蒸馏 + 20% 通道剪枝

**训练方案**:

- 基础模型: Phase 3 蒸馏模型
- 剪枝方法: GroupNormPruner (Torch-Pruning), L2 范数重要性
- 剪枝率: 20% 通道
- 保护层: GDC 头部 (`features`, `conv_sep`)
- 剪枝后继续训练 90 epoch

**验证结果**:

| 验证集   | Accuracy | vs Phase 3 |
| -------- | -------- | ---------- |
| LFW      | 99.82%   | -0.01%     |
| VGG2_FP  | 95.52%   | -0.12%     |
| AgeDB_30 | 97.80%   | -0.13%     |
| CALFW    | 96.02%   | -0.11%     |
| CFP_FF   | 99.86%   | -0.01%     |
| CPLFW    | 93.00%   | -0.53%     |
| CFP_FP   | 98.26%   | -0.38%     |

**结论**: 20% 通道剪枝后模型体积减小 32% (15.75MB → 10.71MB)，各指标仅微降 0.01%~0.53%，剪枝策略有效。

**模型文件**: `output/mbf_v3_se_distill_prune/model.pt` (11MB)

---

## 三、最终对比总览

### 轻量化流水线对比

| 阶段     | 模型      | 体积 | LFW    | VGG2_FP | AgeDB_30 | CALFW  | CFP_FF | CPLFW  | CFP_FP |
| -------- | --------- | ---- | ------ | ------- | -------- | ------ | ------ | ------ | ------ |
| 基线     | mbf_large | 25MB | 99.87% | -       | 98.10%   | -      | -      | -      | 98.96% |
| Phase 1  | mbf_v3    | 15MB | 99.78% | 95.54%  | 97.83%   | 96.03% | 99.89% | 93.43% | 98.44% |
| Phase 2  | mbf_v3_se | 16MB | 99.83% | 95.76%  | 98.05%   | 96.10% | 99.91% | 93.52% | 98.71% |
| Phase 3  | 蒸馏      | 16MB | 99.83% | 95.64%  | 97.93%   | 96.13% | 99.87% | 93.53% | 98.64% |
| Phase 3+ | 蒸馏+剪枝 | 11MB | 99.82% | 95.52%  | 97.80%   | 96.02% | 99.86% | 93.00% | 98.26% |

### 全模型对比 (含参考基线)

| 模型                    | 体积  | LFW    | VGG2_FP | AgeDB_30 | CALFW  | CFP_FF | CPLFW  | CFP_FP |
| ----------------------- | ----- | ------ | ------- | -------- | ------ | ------ | ------ | ------ |
| ResNet-100              | 250MB | 99.82% | 96.02%  | 98.77%   | 96.05% | 99.84% | 94.85% | 99.27% |
| mbf_v3_se_distill       | 16MB  | 99.83% | 95.64%  | 97.93%   | 96.13% | 99.87% | 93.53% | 98.64% |
| mbf_v3_se_distill_prune | 11MB  | 99.82% | 95.52%  | 97.80%   | 96.02% | 99.86% | 93.00% | 98.26% |

---

## 四、实验步骤总结

### Step 1: mbf_v3 模型设计

- 基于原版 MobileFaceNet (MobileNetV2 倒残差架构)
- 调整 `scale=3`, `blocks=(2,6,8,3)`
- 代码: `backbones/mobilefacenet.py` → `get_mbf_v3()`

### Step 2: mbf_v3_se 增加 SE 注意力

- 在每个 DepthWise 块后插入 SEModule
- 代码: `backbones/modules/se_module.py`, `backbones/mobilefacenet.py` → `get_mbf_v3_se()`

### Step 3: 知识蒸馏训练

- Teacher: mbf_large → Student: mbf_v3_se
- 蒸馏损失: cosine similarity, α=0.5
- 代码: `train_v2_distill.py`, `losses_distill.py`

### Step 4: 结构化剪枝

- 基于 Torch-Pruning GroupNormPruner
- 20% 通道剪枝，保护 GDC 头部
- 代码: `train_v2_distill.py` (蒸馏+剪枝一体化训练), `tools/inference_val_prune.py` (剪枝验证/ONNX导出)

---

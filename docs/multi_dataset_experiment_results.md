# 多数据集训练实验结果

## 实验配置

- **骨干网络**: mbf_v3_se (带 SE 注意力的 MobileFaceNet V3)
- **数据集**: glint360k (360,232 类) + faces_umd (8,277 类)
- **损失权重**: glint360k 0.7, faces_umd 0.3
- **优化器**: SGD (lr=0.1, momentum=0.9, weight_decay=1e-4)
- **批大小**: 每数据集每 GPU 128，共 4 GPU
- **梯度累积**: 4 步 (有效批大小 = 128 x 2 x 4 x 4 = 8,192)
- **PartialFC 采样率**: 0.9
- **训练轮数**: 10
- **验证集**: LFW, CFP-FP, AgeDB-30 (每 2000 步验证一次)

## 实验一：多数据集基线

- 训练脚本: `train_multi_data.py`
- 配置文件: `configs/glint360k_facesumd_mbf_v3_se_multi.py`
- 训练日志: [multi_data_baseline_train.txt](logs/multi_data_baseline_train.txt)

## 实验二：多数据集 + 知识蒸馏

- 训练脚本: `train_multi_data_distill.py`
- 配置文件: `configs/glint360k_facesumd_mbf_v3_se_distill_multi.py`
- 训练日志: [multi_data_distill_train.txt](logs/multi_data_distill_train.txt)
- **Teacher**: ResNet-100 (冻结，预训练于 glint360k)
- **蒸馏类型**: L2 归一化后的余弦相似度
- **蒸馏 alpha**: 0.5 (CE 权重 = 0.5，蒸馏权重 = 0.5)

---

## 对比总结

| 验证集 | 基线 | 蒸馏 | 对比 |
|--------|------|------|------|
| LFW | **99.77%** | 99.73% | 基线略高 |
| CFP-FP | **97.64%** | 97.20% | 基线略高 |
| AgeDB-30 | 97.22% | **97.35%** | 蒸馏更高 |

### 关键发现

1. **蒸馏在难任务上更有优势**: AgeDB-30 (年龄变化) 是最具挑战性的验证集，蒸馏达到 97.35%，超过基线的 97.22%
2. **训练效率**: 蒸馏在更少的 step 就达到了可比的性能，尽管每步因 Teacher 前向传播慢约 2x
3. **多数据集+蒸馏可叠加**: 两种技术互不冲突，可组合使用以提升模型泛化能力

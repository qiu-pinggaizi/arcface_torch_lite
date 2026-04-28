"""
知识蒸馏损失函数
Embedding 级蒸馏：student 向 teacher 学习特征表示
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbeddingDistillationLoss(nn.Module):
    def __init__(self, alpha=0.5, loss_type="cosine"):
        """
        Args:
            alpha: CE loss 权重 (1-alpha 为 distill loss 权重)
            loss_type: "cosine" 或 "l2"
        """
        super().__init__()
        self.alpha = alpha
        self.loss_type = loss_type

    def forward(self, student_emb, teacher_emb, ce_loss):
        """
        Args:
            student_emb: student 模型的 embedding (batch, 512)
            teacher_emb: teacher 模型的 embedding (batch, 512)
            ce_loss: 分类损失
        Returns:
            loss: 总损失
            distill_loss_val: 蒸馏损失值 (用于日志)
        """
        # L2 归一化
        student_norm = F.normalize(student_emb, p=2, dim=1)
        teacher_norm = F.normalize(teacher_emb, p=2, dim=1)
        
        if self.loss_type == "cosine":
            cos_sim = F.cosine_similarity(student_norm, teacher_norm, dim=1)
            distill_loss = (1 - cos_sim).mean()
        elif self.loss_type == "l2":
            distill_loss = F.mse_loss(student_norm, teacher_norm)
        else:
            raise ValueError(f"Unsupported loss_type: {self.loss_type}")

        loss = self.alpha * ce_loss + (1 - self.alpha) * distill_loss
        return loss, distill_loss.item()

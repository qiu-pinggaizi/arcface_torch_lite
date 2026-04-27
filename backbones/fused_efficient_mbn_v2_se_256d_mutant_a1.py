"""
EfficientMobileNetV2SE_MutantA1 - 轻量级人脸识别模型
基于 fused_efficient_mobilenetv2_se_256d_mutant_a1/model.py 集成到 arcface_torch_5max 框架
输入: RGB 3通道图像 (B, 3, 112, 112)
输出: 512维 L2 归一化特征
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv1x1(in_ch, out_ch, stride=1):
    return nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False)


def conv3x3_dw(ch, stride=1):
    return nn.Conv2d(ch, ch, 3, stride, 1, groups=ch, bias=False)


class SEModule(nn.Module):
    """SE 注意力模块"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        reduced = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels, bias=False),
            nn.Hardsigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = F.adaptive_avg_pool2d(x, 1).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class InvertedResidualV2(nn.Module):
    def __init__(self, in_ch, out_ch, stride, expand_ratio, use_se=False, se_reduction=16):
        super().__init__()
        hidden = int(in_ch * expand_ratio)
        self.use_res = stride == 1 and in_ch == out_ch
        layers = []

        if expand_ratio != 1:
            layers += [conv1x1(in_ch, hidden), nn.BatchNorm2d(hidden), nn.ReLU6()]

        layers += [conv3x3_dw(hidden, stride), nn.BatchNorm2d(hidden), nn.ReLU6()]

        if use_se:
            layers.append(SEModule(hidden, reduction=se_reduction))

        layers += [conv1x1(hidden, out_ch), nn.BatchNorm2d(out_ch)]

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        if self.use_res:
            return out + x
        return out


class EfficientMobileNetV2SE_MutantA1(nn.Module):
    """
    4-stage, 13-layer (stem + 12 blocks) 轻量骨干
    - Stem: 32ch, stride 2 → 56x56
    - Stage1: 32ch × 1 block (stride1)
    - Stage2: 64ch × 3 blocks (1 downsample + 2 stride1)
    - Stage3: 128ch × 3 blocks (1 downsample + 2 stride1)
    - Stage4: 256ch × 2 blocks (1 downsample + 1 stride1)
    - Head: GDConv(7×7) + BN + Flatten + FC + BN + L2Norm → 512d
    """
    def __init__(self, embedding_size=512):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6()
        )

        self.stage1 = nn.Sequential(
            InvertedResidualV2(32, 32, stride=1, expand_ratio=2),
        )

        self.stage2 = nn.Sequential(
            InvertedResidualV2(32, 64, stride=2, expand_ratio=4),
            InvertedResidualV2(64, 64, stride=1, expand_ratio=2),
            InvertedResidualV2(64, 64, stride=1, expand_ratio=2, use_se=True, se_reduction=8),
        )

        self.stage3 = nn.Sequential(
            InvertedResidualV2(64, 128, stride=2, expand_ratio=4, use_se=True, se_reduction=8),
            InvertedResidualV2(128, 128, stride=1, expand_ratio=2),
            InvertedResidualV2(128, 128, stride=1, expand_ratio=2, use_se=True, se_reduction=8),
        )

        self.stage4 = nn.Sequential(
            InvertedResidualV2(128, 256, stride=2, expand_ratio=2, use_se=True, se_reduction=8),
            InvertedResidualV2(256, 256, stride=1, expand_ratio=2),
        )

        self.gdconv = nn.Sequential(
            nn.Conv2d(256, 256, 7, groups=256, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.Flatten()
        )

        self.embedding_fc = nn.Linear(256, embedding_size, bias=False)
        self.embedding_bn = nn.BatchNorm1d(embedding_size)
        self.embedding_size = embedding_size

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.gdconv(x)
        x = self.embedding_fc(x)
        x = self.embedding_bn(x)
        x = F.normalize(x, p=2, dim=1)
        return x


def get_efficient_mbn_v2_se_mutant_a1(fp16=False, num_features=512):
    return EfficientMobileNetV2SE_MutantA1(embedding_size=num_features)

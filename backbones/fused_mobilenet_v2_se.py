"""
FusedMobileNetV2SE - 轻量级人脸识别模型
基于 fused_mobilenet_v2_se/model.py 集成到 arcface_torch_5max 框架
输入: RGB 3通道图像 (B, 3, 112, 112)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class HardSwish(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3., inplace=self.inplace) / 6.


class SEModule(nn.Module):
    """SE 注意力模块"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        reduced = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class InvertedResidual(nn.Module):
    def __init__(self, in_ch, out_ch, stride, expand_ratio, use_se=True):
        super().__init__()
        hidden = int(in_ch * expand_ratio)
        self.use_res = stride == 1 and in_ch == out_ch
        layers = []

        if expand_ratio != 1:
            layers += [
                nn.Conv2d(in_ch, hidden, 1, bias=False),
                nn.BatchNorm2d(hidden),
                HardSwish()
            ]

        layers += [
            nn.Conv2d(hidden, hidden, 3, stride, 1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            HardSwish()
        ]

        # SE 模块
        if use_se:
            layers.append(SEModule(hidden, reduction=16))

        layers += [
            nn.Conv2d(hidden, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch)
        ]

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        if self.use_res:
            return out + x
        return out


class FusedMobileNetV2SE(nn.Module):
    def __init__(self, num_features=256, channels=(64, 128, 256), use_se=True):
        """
        Args:
            num_features: 输出特征维度 (默认 256)
            channels: 各阶段通道数 (64, 128, 256)
            use_se: 是否使用 SE 模块
        """
        super().__init__()

        c1, c2, c3 = channels

        # 初始卷积: 输入 RGB 3通道图像
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, c1, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            HardSwish()
        )

        # InvertedResidual 块
        self.blocks = nn.Sequential(
            InvertedResidual(c1, c1, stride=1, expand_ratio=4, use_se=use_se),
            InvertedResidual(c1, c1, stride=1, expand_ratio=4, use_se=use_se),
            InvertedResidual(c1, c2, stride=2, expand_ratio=4, use_se=use_se),
            InvertedResidual(c2, c2, stride=1, expand_ratio=4, use_se=use_se),
            InvertedResidual(c2, c3, stride=2, expand_ratio=4, use_se=use_se),
            InvertedResidual(c3, c3, stride=1, expand_ratio=4, use_se=use_se),
        )

        # 全局平均池化
        self.gap = nn.AdaptiveAvgPool2d(1)

        # 特征头: FC + LayerNorm
        self.fc = nn.Linear(c3, num_features)
        self.ln_fc = nn.LayerNorm(num_features)

    def forward(self, x):
        # x shape: (B, 3, 112, 112)
        x = self.conv1(x)       # (B, c1, 56, 56)
        x = self.blocks(x)      # (B, c3, 14, 14)
        x = self.gap(x)         # (B, c3, 1, 1)
        x = x.view(x.size(0), -1)  # (B, c3)
        x = self.fc(x)          # (B, num_features)
        x = self.ln_fc(x)       # (B, num_features)
        x = F.normalize(x, p=2, dim=1)  # L2 归一化
        return x


def get_fused_mobilent_v2_se(fp16=False, num_features=256, channels=(64, 128, 256), use_se=True):
    """
    FusedMobileNetV2SE 工厂函数

    Args:
        fp16: 是否使用 FP16 (框架兼容，本模型不直接处理)
        num_features: 输出特征维度
        channels: 各阶段通道数
        use_se: 是否使用 SE 模块
    """
    return FusedMobileNetV2SE(
        num_features=num_features,
        channels=channels,
        use_se=use_se
    )

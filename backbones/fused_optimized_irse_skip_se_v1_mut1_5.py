"""
FusedOptimizedIRSESkipSE_v1_mut1_5 - 轻量级人脸识别模型
基于 fused_optimized_irse_skip_se_v1_mut1_5/model.py 集成到 arcface_torch_5max 框架
输入: RGB 3通道图像 (B, 3, 112, 112)
输出: 512维 L2 归一化特征

架构: Stem(40, HardSwish) → ir1(40) → [ir2(72,SE) + ir3(72) skip] → [ir4(128,SE) + ir5(128) skip] → ir6(256) → GDConv(7×7) + FC → 512d
参数量: ~279K (原始), ~280K (集成后)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SEModule(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * torch.sigmoid(y)


class InvertedResidual(nn.Module):
    def __init__(self, in_ch, out_ch, stride, expand_ratio, use_se=False, se_reduction=16, activation='HardSwish'):
        super().__init__()
        hidden = int(round(in_ch * expand_ratio))
        self.use_res = stride == 1 and in_ch == out_ch
        self.use_se = use_se

        if activation == 'HardSwish':
            act_fn = nn.Hardswish(inplace=True)
        elif activation == 'ReLU6':
            act_fn = nn.ReLU6(inplace=True)
        else:
            act_fn = nn.ReLU(inplace=True)

        # Expansion (pointwise)
        if expand_ratio != 1:
            self.expand = nn.Sequential(
                nn.Conv2d(in_ch, hidden, 1, bias=False),
                nn.BatchNorm2d(hidden),
                act_fn
            )
        else:
            self.expand = None

        # Depthwise
        self.depthwise = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, stride, 1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            act_fn
        )

        # SE (applied to depthwise output, channels=hidden)
        self.se = SEModule(hidden, se_reduction) if use_se else nn.Identity()

        # Projection (pointwise)
        self.project = nn.Sequential(
            nn.Conv2d(hidden, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch)
        )

    def forward(self, x):
        out = self.expand(x) if self.expand is not None else x
        out = self.depthwise(out)
        out = self.se(out)
        out = self.project(out)
        if self.use_res:
            return out + x
        return out


class FusedOptimizedIRSESkipSE_v1_mut1_5(nn.Module):
    """
    6-block InvertedResidual_SE 骨干 + SkipConnect
    - Stem: 40ch, HardSwish, stride 2 → 56x56
    - Stage1: ir1(40→40, stride1, no SE)
    - Stage2: ir2(40→72, stride2, SE r=16) + ir3(72→72, stride1, no SE) + Skip
    - Stage3: ir4(72→128, stride2, SE r=8) + ir5(128→128, stride1, no SE) + Skip
    - Stage4: ir6(128→256, stride2, no SE)
    - Head: GDConv(7×7) + BN + Flatten + FC + BN + L2Norm → 512d
    """
    def __init__(self, embedding_size=512):
        super().__init__()
        # Stem (RGB 3ch input)
        self.conv_stem = nn.Conv2d(3, 40, 3, 2, 1, bias=False)
        self.bn_stem = nn.BatchNorm2d(40)
        self.act_stem = nn.Hardswish(inplace=True)

        # Backbone blocks
        self.ir1 = InvertedResidual(40, 40, stride=1, expand_ratio=2, use_se=False, activation='HardSwish')
        self.ir2 = InvertedResidual(40, 72, stride=2, expand_ratio=2, use_se=True, se_reduction=16, activation='HardSwish')
        self.ir3 = InvertedResidual(72, 72, stride=1, expand_ratio=2, use_se=False, activation='HardSwish')
        self.ir4 = InvertedResidual(72, 128, stride=2, expand_ratio=2, use_se=True, se_reduction=8, activation='HardSwish')
        self.ir5 = InvertedResidual(128, 128, stride=1, expand_ratio=2, use_se=False, activation='HardSwish')
        self.ir6 = InvertedResidual(128, 256, stride=2, expand_ratio=2, use_se=False, activation='HardSwish')

        # Head
        self.gdconv = nn.Conv2d(256, 256, 7, 1, 0, groups=256, bias=False)
        self.bn_head = nn.BatchNorm2d(256)
        self.flatten = nn.Flatten()

        # 512d projection
        self.embedding_fc = nn.Linear(256, embedding_size, bias=False)
        self.embedding_bn = nn.BatchNorm1d(embedding_size)
        self.embedding_size = embedding_size

    def forward(self, x):
        # Stem
        x = self.act_stem(self.bn_stem(self.conv_stem(x)))

        # Stage 1
        x = self.ir1(x)

        # Stage 2 with skip connection
        ir2_out = self.ir2(x)
        ir3_out = self.ir3(ir2_out)
        x = ir2_out + ir3_out  # Skip connection

        # Stage 3 with skip connection
        ir4_out = self.ir4(x)
        ir5_out = self.ir5(ir4_out)
        x = ir4_out + ir5_out  # Skip connection

        # Stage 4
        x = self.ir6(x)

        # Head
        x = self.gdconv(x)
        x = self.bn_head(x)
        x = self.flatten(x)

        # 512d projection
        x = self.embedding_fc(x)
        x = self.embedding_bn(x)
        x = F.normalize(x, p=2, dim=1)

        return x


def get_irse_skip_se_v1_mut1_5(fp16=False, num_features=512):
    return FusedOptimizedIRSESkipSE_v1_mut1_5(embedding_size=num_features)

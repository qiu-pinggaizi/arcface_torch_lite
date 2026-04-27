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

class InvertedResidual_SE(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio, use_se=False):
        super().__init__()
        hidden_channels = int(in_channels * expand_ratio)
        self.use_res = stride == 1 and in_channels == out_channels
        
        layers = []
        if expand_ratio != 1:
            layers += [
                nn.Conv2d(in_channels, hidden_channels, 1, bias=False),
                nn.BatchNorm2d(hidden_channels),
                nn.Hardswish(inplace=True)
            ]
        
        layers += [
            nn.Conv2d(hidden_channels, hidden_channels, 3, stride, 1, groups=hidden_channels, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.Hardswish(inplace=True)
        ]
        
        layers += [
            nn.Conv2d(hidden_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        ]
        
        self.conv = nn.Sequential(*layers)
        self.se = SEModule(hidden_channels) if use_se else nn.Identity()
        
    def forward(self, x):
        out = self.conv(x)
        if self.use_res:
            return out + x
        return out

class GeneratedModel(nn.Module):
    def __init__(self, feat_dim=256):
        super().__init__()
        # Stem layer
        self.stem_conv = nn.Conv2d(1, 32, 3, 2, 1, bias=False)
        self.stem_bn = nn.BatchNorm2d(32)
        self.stem_act = nn.Hardswish(inplace=True)
        
        # InvertedResidual_SE blocks
        self.ir1 = InvertedResidual_SE(32, 32, 1, 4, use_se=False)
        self.ir2 = InvertedResidual_SE(32, 32, 1, 4, use_se=False)
        self.ir3 = InvertedResidual_SE(32, 64, 2, 4, use_se=False)
        self.ir4 = InvertedResidual_SE(64, 64, 1, 4, use_se=False)
        self.ir5 = InvertedResidual_SE(64, 128, 2, 4, use_se=True)
        self.ir6 = InvertedResidual_SE(128, 128, 1, 4, use_se=True)
        self.ir7 = InvertedResidual_SE(128, 256, 2, 4, use_se=True)
        self.ir8 = InvertedResidual_SE(256, 256, 1, 4, use_se=True)
        
        # GDConv head
        self.gdconv = nn.Conv2d(256, 256, 7, 1, 0, groups=256, bias=False)
        self.bn_head = nn.BatchNorm2d(256)
        
    def forward(self, x):
        # Stem
        x = self.stem_conv(x)
        x = self.stem_bn(x)
        x = self.stem_act(x)
        
        # InvertedResidual_SE blocks
        x = self.ir1(x)
        x = self.ir2(x)
        x = self.ir3(x)
        x = self.ir4(x)
        x = self.ir5(x)
        x = self.ir6(x)
        x = self.ir7(x)
        x = self.ir8(x)
        
        # GDConv head
        x = self.gdconv(x)
        x = self.bn_head(x)
        
        # Flatten and L2 normalize
        x = x.view(x.size(0), -1)
        x = F.normalize(x, p=2, dim=1)
        
        return x
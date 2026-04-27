import torch
import torch.nn as nn
import torch.nn.functional as F

class HardSwish(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.inplace = inplace
        
    def forward(self, x):
        return x * F.relu6(x + 3., inplace=self.inplace) / 6.

class InvertedResidual(nn.Module):
    def __init__(self, in_ch, out_ch, stride, expand_ratio):
        super().__init__()
        hidden = int(in_ch * expand_ratio)
        self.use_res = stride == 1 and in_ch == out_ch
        layers = []
        if expand_ratio != 1:
            layers += [nn.Conv2d(in_ch, hidden, 1, bias=False), nn.BatchNorm2d(hidden), HardSwish()]
        layers += [nn.Conv2d(hidden, hidden, 3, stride, 1, groups=hidden, bias=False), nn.BatchNorm2d(hidden), HardSwish()]
        layers += [nn.Conv2d(hidden, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch)]
        self.conv = nn.Sequential(*layers)
        
    def forward(self, x):
        out = self.conv(x)
        if self.use_res:
            return out + x
        return out

class GeneratedModel(nn.Module):
    def __init__(self, feat_dim=256):
        super().__init__()
        
        # Initial convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            HardSwish()
        )
        
        # InvertedResidual blocks
        self.blocks = nn.Sequential(
            InvertedResidual(64, 64, 1, 4),
            InvertedResidual(64, 64, 1, 4),
            InvertedResidual(64, 128, 2, 4),
            InvertedResidual(128, 128, 1, 4),
            InvertedResidual(128, 256, 2, 4),
            InvertedResidual(256, 256, 1, 4)
        )
        
        # Global average pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Head: FC + LayerNorm (替代BN1d以避免batch_size=1时的错误)
        self.fc = nn.Linear(256, feat_dim)
        self.ln_fc = nn.LayerNorm(feat_dim)
        
    def forward(self, x):
        # x shape: (B, 1, 112, 112)
        x = self.conv1(x)  # (B, 64, 56, 56)
        x = self.blocks(x)  # (B, 256, 14, 14)
        x = self.gap(x)  # (B, 256, 1, 1)
        x = x.view(x.size(0), -1)  # (B, 256)
        x = self.fc(x)  # (B, feat_dim)
        x = self.ln_fc(x)  # (B, feat_dim)
        x = F.normalize(x, p=2, dim=1)  # L2 normalized
        return x
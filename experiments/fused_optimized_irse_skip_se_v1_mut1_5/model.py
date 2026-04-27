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
        layers = []
        
        if activation == 'HardSwish':
            act_fn = nn.Hardswish(inplace=True)
        elif activation == 'ReLU6':
            act_fn = nn.ReLU6(inplace=True)
        else:
            act_fn = nn.ReLU(inplace=True)
            
        if expand_ratio != 1:
            layers += [nn.Conv2d(in_ch, hidden, 1, bias=False), nn.BatchNorm2d(hidden), act_fn]
        
        layers += [nn.Conv2d(hidden, hidden, 3, stride, 1, groups=hidden, bias=False), nn.BatchNorm2d(hidden), act_fn]
        layers += [nn.Conv2d(hidden, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch)]
        
        self.conv = nn.Sequential(*layers)
        self.se = SEModule(hidden, se_reduction) if use_se else nn.Identity()
        
    def forward(self, x):
        out = self.conv(x)
        if self.use_res:
            return out + x
        return out

class GeneratedModel(nn.Module):
    def __init__(self, feat_dim=256):
        super().__init__()
        # Stem
        self.conv_stem = nn.Conv2d(1, 40, 3, 2, 1, bias=False)
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
        x = F.normalize(x, p=2, dim=1)
        
        return x
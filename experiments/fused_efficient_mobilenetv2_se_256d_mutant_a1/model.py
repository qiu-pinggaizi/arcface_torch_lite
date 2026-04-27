import torch
import torch.nn as nn
import torch.nn.functional as F

class InvertedResidual_SE(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio, use_se=False, se_reduction=8, activation='ReLU6'):
        super().__init__()
        self.stride = stride
        self.use_se = use_se
        hidden_dim = int(round(in_channels * expand_ratio))
        self.use_res_connect = self.stride == 1 and in_channels == out_channels
        
        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(nn.Conv2d(in_channels, hidden_dim, 1, 1, 0, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            if activation == 'ReLU6':
                layers.append(nn.ReLU6(inplace=True))
            elif activation == 'HardSwish':
                layers.append(nn.Hardswish(inplace=True))
            else:
                layers.append(nn.ReLU(inplace=True))
        
        # dw
        layers.append(nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False))
        layers.append(nn.BatchNorm2d(hidden_dim))
        if activation == 'ReLU6':
            layers.append(nn.ReLU6(inplace=True))
        elif activation == 'HardSwish':
            layers.append(nn.Hardswish(inplace=True))
        else:
            layers.append(nn.ReLU(inplace=True))
        
        # pw-linear
        layers.append(nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        
        self.conv = nn.Sequential(*layers)
        
        if self.use_se:
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(out_channels, out_channels // se_reduction, 1, 1, 0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels // se_reduction, out_channels, 1, 1, 0, bias=True),
                nn.Sigmoid()
            )
    
    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class GeneratedModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Stem
        self.conv_stem = nn.Conv2d(1, 32, 3, 2, 1, bias=False)
        self.bn_stem = nn.BatchNorm2d(32)
        self.act_stem = nn.ReLU6(inplace=True)
        
        # Stage 1
        self.ir_stage1_1 = InvertedResidual_SE(32, 32, 1, 2, use_se=False, activation='ReLU6')
        
        # Stage 2
        self.ir_stage2_ds = InvertedResidual_SE(32, 64, 2, 4, use_se=False, activation='ReLU6')
        self.ir_stage2_1 = InvertedResidual_SE(64, 64, 1, 2, use_se=False, activation='ReLU6')
        self.ir_stage2_2 = InvertedResidual_SE(64, 64, 1, 2, use_se=True, se_reduction=8, activation='ReLU6')
        
        # Stage 3
        self.ir_stage3_ds = InvertedResidual_SE(64, 128, 2, 4, use_se=True, se_reduction=8, activation='ReLU6')
        self.ir_stage3_1 = InvertedResidual_SE(128, 128, 1, 2, use_se=False, activation='ReLU6')
        self.ir_stage3_2 = InvertedResidual_SE(128, 128, 1, 2, use_se=True, se_reduction=8, activation='ReLU6')
        
        # Stage 4
        self.ir_stage4_ds = InvertedResidual_SE(128, 256, 2, 2, use_se=True, se_reduction=8, activation='ReLU6')
        self.ir_stage4_1 = InvertedResidual_SE(256, 256, 1, 2, use_se=False, activation='ReLU6')
        
        # Head
        self.gdconv = nn.Conv2d(256, 256, 7, groups=256, padding=0, bias=False)
        self.bn_head = nn.BatchNorm2d(256)
        
    def forward(self, x):
        # Stem
        x = self.conv_stem(x)
        x = self.bn_stem(x)
        x = self.act_stem(x)
        
        # Stage 1
        x = self.ir_stage1_1(x)
        
        # Stage 2
        x = self.ir_stage2_ds(x)
        x = self.ir_stage2_1(x)
        x = self.ir_stage2_2(x)
        
        # Stage 3
        x = self.ir_stage3_ds(x)
        x = self.ir_stage3_1(x)
        x = self.ir_stage3_2(x)
        
        # Stage 4
        x = self.ir_stage4_ds(x)
        x = self.ir_stage4_1(x)
        
        # Head
        x = self.gdconv(x)
        x = self.bn_head(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = F.normalize(x, p=2, dim=1)  # L2 normalization
        
        return x

# Test the model
if __name__ == "__main__":
    model = GeneratedModel()
    input_tensor = torch.randn(1, 1, 112, 112)
    output = model(input_tensor)
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output norm: {torch.norm(output, p=2, dim=1)}")  # Should be 1.0
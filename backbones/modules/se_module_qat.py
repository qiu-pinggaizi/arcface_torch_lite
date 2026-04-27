"""
QAT-compatible Squeeze-and-Excitation (SE) Module
Uses FloatFunctional for quantization-aware multiply
"""
import torch
import torch.nn as nn


class SEModuleQAT(nn.Module):
    """QAT-compatible Squeeze-and-Excitation block

    Args:
        channels: Number of input channels
        reduction: Channel reduction ratio (default: 16)
    """
    def __init__(self, channels, reduction=16):
        super(SEModuleQAT, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
        # QAT-compatible multiply
        self.mul_op = nn.quantized.FloatFunctional()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return self.mul_op.mul(x, y.expand_as(x))

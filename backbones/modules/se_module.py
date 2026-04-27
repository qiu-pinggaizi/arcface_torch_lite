"""
Squeeze-and-Excitation (SE) Module
Paper: Squeeze-and-Excitation Networks (https://arxiv.org/abs/1709.01507)
"""
import torch
import torch.nn as nn


class SEModule(nn.Module):
    """Squeeze-and-Excitation block

    Args:
        channels: Number of input channels
        reduction: Channel reduction ratio (default: 16)
    """
    def __init__(self, channels, reduction=16):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

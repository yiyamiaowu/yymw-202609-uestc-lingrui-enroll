import torch
import torch.nn as nn
import torch.nn.functional as F
from models.baseline_net import ResidualBlock

class Branch(nn.Module):
    def __init__(self, channels=64, num_blocks=8):
        super().__init__()
        # 8层ResBlock
        self.body = nn.Sequential(*[ResidualBlock(channels) for _ in range(num_blocks)])

    def forward(self, x):
        return x + self.body(x)     # 与处理前的图片相加
        
        
class DerainNet(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.head = nn.Conv2d(3, channels, 3, 1, 1)
        # branch创建分支
        self.branch_full = Branch(channels)         
        self.branch_half = Branch(channels)
        self.branch_quarter = Branch(channels)

        self.integrate = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 3, 3, 1, 1)
        )

    def forward(self, x):
        feat = self.head(x)
        H, W = feat.shape[-2:]

        # 全分辨率
        full = self.branch_full(feat)

        # 1/2分辨率
        feat = F.max_pool2d(feat, 2)
        half = self.branch_half(feat)
        half = F.interpolate(half, size=(H, W), mode="bilinear", align_corners=False)

        # 1/4分辨率
        feat = F.max_pool2d(feat, 2)
        quarter = self.branch_quarter(feat)
        quarter = F.interpolate(quarter, size=(H, W), mode="bilinear", align_corners=False)

        rain = self.integrate(torch.concat([full, half, quarter], dim=1))

        return x - rain   


        
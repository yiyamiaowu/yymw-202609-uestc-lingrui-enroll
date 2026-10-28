import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self,channels = 64):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels,channels,kernel_size=3,
                stride=1, padding=1,bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels,channels,kernel_size=3,
                stride=1, padding=1,bias=False),
            nn.BatchNorm2d(channels),
        )
        self.ReLU = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.ReLU(self.body(x) + x)


class BaselineNet(nn.Module):
    def __init__(self, in_out_channel = 3, channels = 64, num_blocks = 8):
        super().__init__()
        self.head = nn.Conv2d(in_out_channel,channels,
            kernel_size=3,stride=1, padding=1,bias=False)
        self.block = nn.Sequential(
            *[ResidualBlock(channels) for _ in range(num_blocks)]   # *解包列表
        )
        self.tail = nn.Conv2d(channels,in_out_channel,
            kernel_size=3,stride=1, padding=1,bias=False)
    
    def forward(self, x):
        feat = self.head(x)
        feat = self.block(feat)
        rain = self.tail(feat)
        clean = x - rain
        return clean

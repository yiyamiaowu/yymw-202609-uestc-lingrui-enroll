import torch
import torch.nn as nn
import torch.nn.functional as F



# 整个SPANet就参考了 https://arxiv.org/abs/1904.01538 这篇文章以及其对应的Github仓库

# 标准残差网络
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        else:
            self.shortcut = nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.body(x) + self.shortcut(x))

# 轻量RB，提取特征
class LightweightEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.out_channels = 256

        self.head = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.rb1 = ResidualBlock(64, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.rb2 = ResidualBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.rb3 = ResidualBlock(128, 256)

    def forward(self, x):
        x = self.head(x)        # (B, 64, H, W)
        x = self.rb1(x)
        x = self.pool1(x)       # (B, 64, H/2, W/2)

        x = self.rb2(x)
        x = self.pool2(x)       # (B, 128, H/4, W/4)

        x = self.rb3(x)         # (B, 256, H/4, W/4)
        return x

# 局部注意力
class directionalAttention(nn.Module):
    def __init__(self, channels, T=2, alpha=0.2):
        super().__init__()
        self.T = T
        self.alpha_raw = nn.Parameter(torch.tensor([0.2, 0.2, 0.2, 0.2]))   # 可学习累积率

        # 四方向1×1卷积
        self.weight_left = nn.Conv2d(channels, channels, 1, groups=channels, bias=False)
        self.weight_right = nn.Conv2d(channels, channels, 1, groups=channels, bias=False)
        self.weight_up = nn.Conv2d(channels, channels, 1, groups=channels, bias=False)
        self.weight_down = nn.Conv2d(channels, channels, 1, groups=channels, bias=False)

        # 四方向特征融合
        self.integrate = nn.Conv2d(channels * 4, channels, 1)


    def forward(self,x):
        B, C, H, W = x.shape

        alpha = torch.sigmoid(self.alpha_raw)
        a_l, a_r, a_u, a_d = alpha[0], alpha[1], alpha[2], alpha[3]


        # 四方向初始贡献
        contrib_l = self.weight_left(x)
        contrib_r = self.weight_right(x)
        contrib_u = self.weight_up(x)
        contrib_d = self.weight_down(x)


        # 初始值+relu
        h_l = torch.relu(x)
        h_r = torch.relu(x)
        h_u = torch.relu(x)
        h_d = torch.relu(x)


        # 循环展开T轮，沿对应方向逐步累积信息
        for _ in range(self.T):
            # 左 -> 右：out-of-place
            cols = [h_l[..., 0]]
            for i in range(1, W):
                cols.append(torch.relu(a_l * contrib_l[..., i] + (1 - a_l) * cols[-1]))
            h_l = torch.stack(cols, dim=-1)                 # (B, C, H, W)

            # 右 -> 左：out-of-place
            cols = [h_r[..., W - 1]]
            for i in range(W - 2, -1, -1):
                cols.append(torch.relu(a_r * contrib_r[..., i] + (1 - a_r) * cols[-1]))
            h_r = torch.stack(cols[::-1], dim=-1)           # 反转，保持列顺序

            # 上 -> 下：out-of-place
            rows = [h_u[..., 0, :]]
            for i in range(1, H):
                rows.append(torch.relu(a_u * contrib_u[..., i, :] + (1 - a_u) * rows[-1]))
            h_u = torch.stack(rows, dim=-2)                 # (B, C, H, W)

            # 下 -> 上：out-of-place
            rows = [h_d[..., H - 1, :]]
            for i in range(H - 2, -1, -1):
                rows.append(torch.relu(a_d * contrib_d[..., i, :] + (1 - a_d) * rows[-1]))
            h_d = torch.stack(rows[::-1], dim=-2)

        # 四方向融合
        out = self.integrate(torch.cat([h_l, h_r, h_u, h_d], dim=1))
        return out

# 显式监督注意力图
class AttentionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()

    def forward(self, attention, rainy, clean):
        M = torch.clamp((rainy - clean).abs(), 0.0, 1.0)
        A = attention.mean(dim=1, keepdim=True) if attention.shape[1] > 1 else attention
        if M.shape[1] > 1:
            M = M.mean(dim=1, keepdim=True)
        if A.shape[-2:] != M.shape[-2:]:
            A = F.interpolate(A, size=M.shape[-2:], mode="bilinear", align_corners=False)
        A = torch.clamp(A, 1e-6, 1 - 1e-6)    # clamp将A的值限制在(1e-6, 1 - 1e-6),避免BCE出现log(0)
        return self.bce(A, M)


# 全局注意力图
class GlobalAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.direction = directionalAttention(channels)

        # 全局通道注意力
        self.globe = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, 1),
            nn.Sigmoid()
        )

        # 融合生成空间注意力图
        self.integrate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid(),
        )

        self.residual = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels // 4, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, 1),
        )

    def forward(self,x):
        direction_feat = self.direction(x)
        globe_weight = self.globe(x)
        globe_feat = x * globe_weight
        attention_feat = self.integrate(torch.cat([direction_feat, globe_feat], dim=1))
        result = x * attention_feat
        out = result + self.residual(x)
        return out, attention_feat

class SPANet(nn.Module):
    def __init__(self , num_sam = 4):
        super().__init__()

        # 特征提取
        self.feature_catcher = LightweightEncoder()
        channels = self.feature_catcher.out_channels   # 256

        # 多个空间注意力模块
        self.attention = nn.Sequential(
            *[GlobalAttention(channels) 
            for _ in range(num_sam)]
            )

        # 重建头
        self.decoder = nn.Sequential(
            nn.Conv2d(channels, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 3, 1, 1),
        )

        self.attention_loss = AttentionLoss()

    def forward(self, x):
        H, W = x.shape[-2:]
        feat = self.feature_catcher(x)          # (B, 256, H/4, W/4)
        
        # 注意力图
        attention_maps = []
        for sam in self.attention:
            feat, attn = sam(feat)
            attention_maps.append(attn)

        feat_rain = self.decoder(feat)          # (B, 3, H/4, W/4)

        # 上采样恢复分辨率
        rain = F.interpolate(feat_rain, size=(H, W),    # (B, 3, H, W)
         mode="bilinear", align_corners = False)

        clean = x - rain
        return clean, attention_maps

    # 求注意力loss
    def compute_attention_loss(self, attention_maps, rainy, clean):
        if attention_maps is None:
            return None
        loss = 0.0
        for attn in attention_maps:
            attn_up = F.interpolate(
                attn, size=rainy.shape[-2:], 
                mode="bilinear", align_corners=False)
            loss = loss + self.attention_loss(attn_up, rainy, clean)
        return loss / len(attention_maps)



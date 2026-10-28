import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
from torchvision.models import vgg19, VGG19_Weights

# 加载vgg
def load_vgg():
    vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features[:35].eval()
    for p in vgg.parameters():
        p.requires_grad = False
    return vgg

# 余弦相似度
class EasyContrastiveLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg = load_vgg()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    def _feat_vec(self, x):
        x_n = self.normalize(x)
        feat = self.vgg(x_n)
        vec = self.pool(feat).flatten(1)
        return F.normalize(vec, dim=1)

    def forward(self, pred, rainy, clean):
        vec_pred  = self._feat_vec(pred)
        vec_rainy = self._feat_vec(rainy)
        vec_clean = self._feat_vec(clean)

        sim_pos = F.cosine_similarity(vec_pred, vec_clean, dim=1)
        sim_neg = F.cosine_similarity(vec_pred, vec_rainy, dim=1)

        # 最大化正样本(clean)相似度，最小化负样本(rainy)相似度
        loss = (1 - sim_pos).mean() + sim_neg.mean()
        return loss

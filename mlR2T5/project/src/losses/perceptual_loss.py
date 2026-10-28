import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import vgg19, VGG19_Weights


def load_vgg():
    vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features[:35].eval()
    for p in vgg.parameters():
        p.requires_grad = False
    return vgg


class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg = load_vgg()
        self.l1 = nn.L1Loss()
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    def forward(self, generated, ground_truth):
        generated_n = self.normalize(generated)
        ground_truth_n = self.normalize(ground_truth)
        feat_generated = self.vgg(generated_n)
        feat_ground_truth = self.vgg(ground_truth_n)
        return self.l1(feat_generated, feat_ground_truth)
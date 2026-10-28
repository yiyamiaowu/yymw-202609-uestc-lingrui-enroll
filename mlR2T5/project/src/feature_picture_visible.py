import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from torchvision.models import vgg19, VGG19_Weights

# 加载VGG19前35层
vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features[:35].eval()
for p in vgg.parameters():
    p.requires_grad = False

# 读取图像并预处理
img = Image.open("project/data/test/input/1.png").convert("RGB")

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(               
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

x = transform(img).unsqueeze(0)
# 提取特征
with torch.no_grad():
    feat = vgg(x)



# 可视化
feat_vis = feat[0, :16].detach().cpu()
# 创建子图网格
fig, axes = plt.subplots(4, 4, figsize=(8, 8))  #fig:整个画布对象  axes:4×4的数组
for i, ax in enumerate(axes.flat):
    ax.imshow(feat_vis[i], cmap="viridis")   # colormap把数值映射到有颜色
    ax.axis("off")
plt.tight_layout()   # 自动调整子图间距
plt.show()
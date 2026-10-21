import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.models import vgg16, VGG16_Weights
from PIL import Image
from PIL import ImageEnhance

# 全局配置

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("使用设备:", device)

content_path = "project/data/content.jpg"
style_path_1 = "project/data/style_1.jpg"
style_path_2 = "project/data/style_2.jpg"
save_path  = "project/data/output.jpg"

picture_size = 512        # 生成图尺寸
epoches = 150       # 迭代次数

weight_content = 2.0
weight_style_1 = 1e5
weight_style_2 = 1e5
weight_tv = 1e-5

# 内容层/风格层
content_layer = ["conv4_2"]
style_layer = ["conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1"]

# 归一化参数平均值与方差（VGG16预训练时所使用的，让图片可以被vgg看懂）
channel_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)    #变形状与[3,H,W]广播计算
channel_std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)



#加载图片并预处理

def load_image(path,size=picture_size,mean = channel_mean,std = channel_std,lighten=False):
    img = Image.open(path)
    if lighten:
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(2.0)
    w, h = img.size   #加了括号就报错了
    scale = size / min(w, h)
    img = img.resize((int(w*scale),int(h*scale)),Image.LANCZOS)#LANCZOS重采样，返回高质量图片
    # 归一化
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean.squeeze().tolist(),      #squeeze删除所有长度为1的维度，变回一维列表，匹配pytorch官方文档中的sequence要求
                             std=std.squeeze().tolist()),
    ])
    img = transform(img).unsqueeze(0).to(device)  #unsqueece,在最前面新增长度为一的维度[1, 3, H, W]，成为batch，让模型读，并移动到GPU
    return img

# 反归一化，变回正常图片
def unnormalize(tensor,mean = channel_mean,std = channel_std):
    t = tensor.clone().detach().squeeze(0).to(torch.device("cpu"))   # detach去掉梯度，去掉batch那一维，移回cpu（pillow只能处理cpu）
    t = t*std + mean
    t = torch.clamp(t, 0, 1)
    return t

# 预训练 VGG16（只保留卷积层）
vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features.to(device).eval()

# 冻结参数
for p in vgg.parameters():
    p.requires_grad = False

# 从VGG中抽取指定层的输出作为特征图返回
def get_features(x,model,layers):
    features = {}
    last = 0
    for name, idx in sorted(layers.items(), key=lambda kv: kv[1]):   #按照key-value中的第二个元素进行排序
        x = model[last:idx+1](x)    
        features[name] = x
        last = idx + 1
    return features

# 计算gram矩阵
def gram_matrix(tensor):
    b, c, h, w = tensor.size()
    feat = tensor.view(b,c,h*w)
    gram = torch.bmm(feat, feat.transpose(1, 2))  
    return gram/(c*h*w)

# vgg_features是nn.Sequential(神经层对应的函数)，用enumerate找出每个conv名字与对应的位置
def find_layer_index(vgg_features,target_names):
    index_map={}
    block_sign,conv_sign = 1, 1
    for i,layer in enumerate(vgg_features):
        if isinstance(layer, nn.Conv2d):
            name = f"conv{block_sign}_{conv_sign}"
            if name in target_names:
                index_map[name] = i
            conv_sign += 1
        elif isinstance(layer, nn.MaxPool2d):
            block_sign += 1
            conv_sign = 1
    return index_map

all_layer_names = content_layer + style_layer
layer_index = find_layer_index(vgg,all_layer_names)
print("层索引:",layer_index)

# 只保留到最深层需要的层，节省计算
vgg = vgg[:max(layer_index.values())+1]

# 损失函数
mse = nn.MSELoss()

# 内容损失
def content_loss(gen_feat, content_feat):
    loss = mse(gen_feat,content_feat)
    return loss

# 风格损失
def style_loss(gen_feats, style_feats):
    loss = 0.0
    for name in style_feats:
        g = gram_matrix(gen_feats[name])
        s = gram_matrix(style_feats[name])
        loss += mse(g, s)
    return loss

# 总变分损失
def totalVariation_loss(x):
    diff_h = x[:, :, 1:, :] - x[:, :, :-1, :]      # 上下差
    diff_w = x[:, :, :, 1:] - x[:, :, :, :-1]      # 左右差
    loss = torch.sum(diff_h ** 2) + torch.sum(diff_w ** 2)
    return loss

# 初始化图片
content_img = load_image(content_path,size=picture_size,lighten = False)
style_img_1   = load_image(style_path_1,size=picture_size)
style_img_2   = load_image(style_path_2,size=picture_size)
gen_img = content_img.clone().requires_grad_(True)

# 预计算内容层和风格层的特征图，并detach
with torch.no_grad():
    content_target = get_features(content_img, vgg, layer_index)["conv4_2"].detach()    #从返回的字典里取出conv4_2层的特征图
    style_feats_1 = get_features(style_img_1, vgg, layer_index)
    style_feats_2 = get_features(style_img_2, vgg, layer_index)
    style_targets_1 = {n: style_feats_1[n].detach() for n in style_layer}
    style_targets_2 = {n: style_feats_2[n].detach() for n in style_layer}

# 优化器（L-BFGS）
optimizer = optim.LBFGS([gen_img], max_iter=20, history_size=10,line_search_fn="strong_wolfe")

# 训练循环
def major_cycle():
    optimizer.zero_grad()  #清空优化器内存

    # 前向
    gen_feats = get_features(gen_img, vgg, layer_index)

    # 损失计算
    co_loss = content_loss(gen_feats["conv4_2"], content_target)
    st_loss_1 = style_loss(gen_feats, style_targets_1)
    st_loss_2 = style_loss(gen_feats, style_targets_2)
    tv_loss = totalVariation_loss(gen_img)
    to_loss = weight_content * co_loss + weight_style_1 * st_loss_1 + weight_style_2 * st_loss_2 + weight_tv * tv_loss
    to_loss.backward()

    # 记录
    major_cycle.step += 1
    if major_cycle.step % 50 == 0:
        print(f"step        |{major_cycle.step:4d}\n"
              f"content_loss|{co_loss.item():.4f}\n"
              f"style_loss  |{st_loss_1.item():.6f}\n"
              f"style_loss  |{st_loss_2.item():.6f}\n"
              f"tv_loss     |{tv_loss.item():.4f}\n"
              f"total_loss  |{to_loss.item():.4f}\n")

    return to_loss

major_cycle.step = 0

# 主流程
for i in range(epoches):
    optimizer.step(major_cycle)
    # 保存中间结果
    if (i + 1) % 50 == 0:
        with torch.no_grad():
            mid = unnormalize(gen_img)
            save_img = transforms.ToPILImage()(mid)
            save_img.save(f"project/data/output_step_{i+1}.jpg")

# 输出
with torch.no_grad():
    final_img = unnormalize(gen_img)                 
    final_pil = transforms.ToPILImage()(final_img)
    final_pil.save(save_path)


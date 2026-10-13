import os
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import vgg16, VGG16_Weights

# 全局配置
seed        = 42
img_size    = 224
batch_size  = 16
num_classes = 5
num_epochs  = 50
lr          = 1e-4
num_workers = 2

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

split_dir = "project/data_split/flower"
save_path = "project/docs/best_model_vgg.pth"




# transforms数据增强+dataloader

def get_transforms():
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomRotation(30),
        transforms.RandomAffine(degrees=0,
                                translate=(0.15, 0.15),
                                shear=15,
                                scale=(0.8, 1.2)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return train_tf, eval_tf


def build_dataloaders(root, batch_size=batch_size, num_workers=num_workers):
    train_tf, eval_tf = get_transforms()

    train_ds = datasets.ImageFolder(os.path.join(root, "train"), transform=train_tf)
    val_ds   = datasets.ImageFolder(os.path.join(root, "val"),   transform=eval_tf)
    test_ds  = datasets.ImageFolder(os.path.join(root, "test"),  transform=eval_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size,shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,shuffle=False, num_workers=num_workers)

    print(f"训练集:{len(train_ds)}张")
    print(f"验证集:{len(val_ds)}张")
    print(f"测试集:{len(test_ds)}张")
    return train_loader, val_loader, test_loader






# 3. 单次训练函数

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct    = 0
    total      = 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += imgs.size(0)

    return total_loss / total, correct / total


# 评估函数
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct    = 0
    total      = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += imgs.size(0)

    return total_loss / total, correct / total



# 绘图函数

def plot(history, save_path="project/docs/training_curve_vgg.png"):
    epochs = range(1, len(history["train_loss"]) + 1)
    # 创建左右子图
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    # loss曲线
    axes[0].plot(epochs, history["train_loss"], label="Train Loss")
    axes[0].plot(epochs, history["val_loss"], label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Curve")
    axes[0].legend()
    # 准确率曲线
    axes[1].plot(epochs, history["train_acc"], label="Train Acc")
    axes[1].plot(epochs, history["val_acc"], label="Val Acc")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy Curve")
    axes[1].legend()

    plt.tight_layout()    # 自适应图片布局
    plt.savefig(save_path, dpi=150)
    plt.show()

def Mymodel(num_classes=num_classes):
    model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
# 冻结卷积层
    for p in model.features.parameters():
        p.requires_grad = False

    in_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(in_features, num_classes)
    return model

# 主体函数
def main():
    # 数据
    train_loader, val_loader, test_loader = build_dataloaders(split_dir)

    # 模型 损失 优化器 学习率调节器
    model = Mymodel(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)   

    # 初始参数
    history = {"train_loss": [], "train_acc": [],
               "val_loss":   [], "val_acc":   []}
    best_val_acc = 0.0
    best_val_loss = float("inf")
    min_delta = 0.001
    counter = 0
    patience = 7

    # 训练循环
    for epoch in range(1, num_epochs + 1):

        train_loss,train_acc = train_one_epoch(model, train_loader,criterion, optimizer, device)
        val_loss,val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # 保存表现最好的模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
        
        # 早停
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f'Early stopping at epoch {epoch}')
                break

        print(f"Epoch {epoch:2d}","|",
      f"train_loss={train_loss:.4f}","|",f"train_acc={train_acc:.4f}","|",
      f"val_loss={val_loss:.4f}","|",f"val_acc={val_acc:.4f}")


    # 评估
    model.load_state_dict(torch.load(save_path, map_location=device))
    print("*" * 30,"评估环节","*" * 30)
    results = {}
    for name, loader in (("Train", train_loader),("Val",   val_loader),("Test",  test_loader)):
        loss, acc = evaluate(model, loader, criterion, device)
        results[name] = acc
        print(f"{name}:Loss = {loss} | Accuracy = {acc}")

    # 画图
    plot(history)

    return model, history, results



if __name__ == "__main__":
    # GPU训练
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用设备:", device)
    print("数据目录:", split_dir, "存在:", os.path.isdir(split_dir))
    model, history, results = main()
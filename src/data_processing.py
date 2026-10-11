import os
import random
import numpy as np
from PIL import Image, ImageFile, ImageEnhance, ImageOps
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True

# 配置
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

ori_dir   = "project/data/flower"
clean_dir = "project/data_clean/flower"
slipt_dir = "project/data_split/flower"

classes = ["daisy", "dandelion", "rose", "sunflower", "tulip"]
img_size = 128



# 创建处理后的文件夹

    

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# 删去上轮遗留数据
def remove_dir(path):
    if not os.path.exists(path):
        return
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))
    os.rmdir(path)

# 复制到新文件夹中
def copy_file(src, dst):
    with open(src, "rb") as fs, open(dst, "wb") as fd:
        fd.write(fs.read())

# 复制文件夹，防止损坏源文件
def copy_tree(src, dst):
    ensure_dir(dst)
    # 逐个读取文件
    for root, dirs, files in os.walk(src):
        rel = os.path.relpath(root, src)
        target = dst if rel == "." else os.path.join(dst, rel)
        ensure_dir(target)
        for f in files:
            copy_file(os.path.join(root, f), os.path.join(target, f))

# 验证文件是否可以打开
def image_valid(path):
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            img.load()
        return True
    except Exception:
        return False


# 清理损坏图片
def clean(src_root, dst_root):
    remove_dir(dst_root)
    ensure_dir(dst_root)

    for split in ("train", "test"):
        for i in classes:
            src_classes = os.path.join(src_root, split, i)
            dst_classes = os.path.join(dst_root, split, i)
            ensure_dir(dst_classes)

            for j in os.listdir(src_classes):
                fp = os.path.join(src_classes, j)
                if not os.path.isfile(fp):
                    continue
                if not j.lower().endswith(".jpg"):
                    continue
                if not image_valid(fp):
                    continue
                copy_file(fp, os.path.join(dst_classes, j))

# 划分验证集与训练集，测试集原样拷贝
def split(src_dir, dst_dir, val_rate=0.2):
    remove_dir(dst_dir)
    ensure_dir(dst_dir)

    for i in classes:
        src_classes = os.path.join(src_dir,"train",i)
        files = [f for f in os.listdir(src_classes)]
        random.shuffle(files)
        n_val = int(len(files) * val_rate)
        veri_files   = files[:n_val]
        train_files = files[n_val:]

        for sets, flist in (("train", train_files), ("val", veri_files)):
            out_dir = os.path.join(dst_dir, sets, i)
            ensure_dir(out_dir)
            for f in flist:
                copy_file(os.path.join(src_classes, f), os.path.join(out_dir, f))

    copy_tree(os.path.join(src_dir, "test"),os.path.join(dst_dir, "test"))


# 执行
if __name__ == "__main__":
    clean(ori_dir, clean_dir)
    split(clean_dir, slipt_dir, val_rate=0.2)


            
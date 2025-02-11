# utils.py
import os
from PIL import Image
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

def get_max_image_size(data_dir):
    """
    获取指定文件夹中最大图像的宽度和高度

    参数:
    - data_dir: 图像数据集的根目录

    返回:
    - (max_width, max_height): 数据集中最大图像的宽度和高度
    """
    max_width = 0
    max_height = 0

    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(("png", "jpg", "jpeg")):
                img_path = os.path.join(root, file)
                with Image.open(img_path) as img:
                    width, height = img.size
                    max_width = max(max_width, width)
                    max_height = max(max_height, height)

    return 128, 128  # return max_width, max_height

def create_transform(max_width, max_height):
    """
    创建图像预处理转换
    """
    return transforms.Compose([
        transforms.Resize((max_width, max_height)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

def split_data(dataset, test_size=0.2):
    """
    划分数据集为训练集和测试集
    """
    targets = dataset.targets
    train_indices, test_indices = train_test_split(
        list(range(len(dataset))), test_size=test_size, stratify=targets
    )
    train_subset = Subset(dataset, train_indices)
    test_subset = Subset(dataset, test_indices)
    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=32, shuffle=False)
    return train_loader, test_loader

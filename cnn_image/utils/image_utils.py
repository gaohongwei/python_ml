# utils.py
import os
from PIL import Image
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

def get_max_image_size_and_channels(image_dir):
    """
    获取指定文件夹中最大图像的宽度和高度，并确定数据集中最大输入图像的通道数（in_channels）,返回:
    - (max_width, max_height, max_in_channels): 数据集中最大图像的宽度、高度，以及最大通道数
    """
    max_width = 0
    max_height = 0
    max_in_channels = 0  # 用于存储数据集中最大图像的通道数

    for root, _, files in os.walk(image_dir):
        for file in files:
            if file.lower().endswith(("png", "jpg", "jpeg")):
                img_path = os.path.join(root, file)
                with Image.open(img_path) as img:
                    width, height = img.size
                    max_width = max(max_width, width)
                    max_height = max(max_height, height)
                    
                    # 获取当前图像的通道数
                    current_in_channels = len(img.getbands())  # 获取图像的通道数
                    max_in_channels = max(max_in_channels, current_in_channels)  # 更新最大通道数

    # 如果没有图像或未能确定通道数，设定默认值
    if max_in_channels == 0:
        raise ValueError("数据集中的图像无法确定通道数。请确保数据集中包含有效的图像。")
    
    # return max_width, max_height, max_in_channels
    return 128,128,3

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

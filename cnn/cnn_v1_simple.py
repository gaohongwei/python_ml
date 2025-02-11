import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Subset
from PIL import Image

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

    # 遍历文件夹中的所有图像文件
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(("png", "jpg", "jpeg")):
                img_path = os.path.join(root, file)
                with Image.open(img_path) as img:
                    width, height = img.size
                    max_width = max(max_width, width)
                    max_height = max(max_height, height)

    return 128, 128
    # return max_width, max_height


def create_transform(max_width, max_height):
    """
    创建图像预处理转换
    """
    return transforms.Compose(
        [
            transforms.Resize((max_width, max_height)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )


def split_data(dataset, test_size=0.2):

    targets = dataset.targets

    # Perform stratified splitting
    train_indices, test_indices = train_test_split(
        list(range(len(dataset))), test_size=test_size, stratify=targets
    )

    # Create subsets based on indices
    train_subset = Subset(dataset, train_indices)
    test_subset = Subset(dataset, test_indices)

    # Create DataLoader objects for both training and testing
    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=32, shuffle=False)

    return train_loader, test_loader


def build_model(num_classes):
    """
    构建简单的 CNN 模型
    """

    class SimpleCNN(nn.Module):
        def __init__(self, num_classes):
            super(SimpleCNN, self).__init__()
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
            self.relu = nn.ReLU()
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.fc1 = nn.Linear(64 * 32 * 32, 128)
            self.fc2 = nn.Linear(128, num_classes)

        def forward(self, x):
            x = self.pool(self.relu(self.conv1(x)))
            x = self.pool(self.relu(self.conv2(x)))
            x = x.view(x.size(0), -1)
            x = self.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    return SimpleCNN(num_classes)


def train_model(model, train_loader, criterion, optimizer, device, epochs=5):
    """
    训练模型
    """
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")
    print("训练完成！")


def evaluate_model(model, test_loader, device):
    """
    在测试集上评估模型的准确率
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy


def main():
    # 设定数据集路径
    data_dir = "/opt/ml/uploads/fashion_data300"

    max_width, max_height = get_max_image_size(data_dir)
    print(f"数据集中最大图像的宽度: {max_width}, 高度: {max_height}")

    # 创建图像预处理转换
    transform = create_transform(max_width, max_height)
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    train_loader, test_loader = split_data(dataset, test_size=0.2)

    # 构建模型
    num_classes = len(dataset.classes)
    model = build_model(num_classes).to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    train_model(model, train_loader, criterion, optimizer, device)

    # 评估模型
    accuracy = evaluate_model(model, test_loader, device)
    print(f"在测试集上的准确率: {accuracy:.2f}%")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main()

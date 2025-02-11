import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
from image_utils import get_max_image_size, create_transform, split_data  
from cnn_models import SimpleCNN


def build_model(num_classes):
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

def run_training(data_dir, epochs=5):
    """
    主训练过程的封装函数
    """
    max_width, max_height = get_max_image_size(data_dir)
    print(f"数据集中最大图像的宽度: {max_width}, 高度: {max_height}")

    # 创建图像预处理转换
    transform = create_transform(max_width, max_height)
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    train_loader, test_loader = split_data(dataset, test_size=0.2)

    # 构建模型
    num_classes = len(dataset.classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes).to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    train_model(model, train_loader, criterion, optimizer, device, epochs)

    # 评估模型
    accuracy = evaluate_model(model, test_loader, device)
    print(f"在测试集上的准确率: {accuracy:.2f}%")

def main():
    data_dir = "../data/fashion_data300"  # specify the data directory
    run_training(data_dir, epochs=5)  # call the training function

if __name__ == "__main__":
    main()

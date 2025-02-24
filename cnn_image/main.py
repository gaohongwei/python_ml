import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
from utils.image_utils import get_max_image_size_and_channels, create_transform, split_data  
from models.models_factory import load_model
from image_train import train_loop,evaluate_model

def run_train(data_dir, max_epochs=5):
    """
    主训练过程的封装函数
    """
    max_width, max_height,max_channels = get_max_image_size_and_channels(data_dir)
    print(f"数据集中最大图像的宽度: {max_width}, 高度: {max_height}")

    # 创建图像预处理转换
    transform = create_transform(max_width, max_height)
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    train_loader, test_loader = split_data(dataset, test_size=0.2)

    # 构建模型
    for model_name in ["resnet18","resnet50","vgg16","vgg19"]:
        num_classes = len(dataset.classes)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_model(model_name=model_name,num_channels=max_channels,num_classes=num_classes).to(device)

        # 训练模型
        train_loop(model, device=device, train_loader=train_loader, max_epochs=max_epochs)

        # 评估模型
        accuracy = evaluate_model(model, test_loader, device)
        print(f"在测试集上的准确率: {accuracy:.2f}%, model_name={model_name}")

def main():
    data_dir = "../data/fashion_data300"  # specify the data directory
    run_train(data_dir, max_epochs=3)  # call the training function

if __name__ == "__main__":
    main()

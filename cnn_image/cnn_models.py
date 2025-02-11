import torch
import torch.nn as nn

def build_model(num_channels,num_classes):
    # TODO: get hyperparams and feed into SimpleCNN/nn.Module
    return SimpleCNN(num_channels,num_classes)

class SimpleCNN(nn.Module):
    def __init__(self, num_channels, num_classes):
        super(SimpleCNN, self).__init__()

        # 第一个卷积层
        # 输入通道数为 num_channels, 3（RGB 图像）
        # 输出通道数为 32
        # 卷积核大小为 3x3
        # padding=1 保证输出大小与输入相同
        self.conv1 = nn.Conv2d(num_channels, 32, kernel_size=3, padding=1)

        # ReLU 激活函数，增加非线性
        self.relu = nn.ReLU()

        # 2x2 最大池化层，减小特征图的空间维度
        self.pool = nn.MaxPool2d(2, 2)

        # 第二个卷积层
        # 输入通道数为 32
        # 输出通道数为 64
        # 卷积核大小为 3x3
        # padding=1 保证输出大小与输入相同
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        # 第一个全连接层
        # 将卷积输出的特征图展平成一维
        # 输入大小是 64*32*32
        # 输出大小是 128
        self.fc1 = nn.Linear(64 * 32 * 32, 128)

        # 第二个全连接层
        # 输出类别数 num_classes
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # 前向传播：依次通过卷积层、ReLU 激活函数、池化层、卷积层、展平、全连接层
        
        # 第一个卷积层 + ReLU + 池化
        x = self.pool(self.relu(self.conv1(x)))

        # 第二个卷积层 + ReLU + 池化
        x = self.pool(self.relu(self.conv2(x)))

        # 展平操作，将二维特征图展平为一维
        x = x.view(-1, 64 * 32 * 32)

        # 第一个全连接层 + ReLU
        x = self.relu(self.fc1(x))

        # 第二个全连接层（输出类别数）
        x = self.fc2(x)

        return x




import os
import torch
import torchvision.models as models
from models.cnn_models import build_cnn_model

MODEL_DICT = {
    'resnet18': models.resnet18,
    'resnet50': models.resnet50,
    'vgg16': models.vgg16,
    'vgg19': models.vgg19
}


def load_model(model_name: str, num_channels: int = 3, num_classes: int = 10, weights: bool = True, model_dir: str = "./"):
    """加载或创建模型"""
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{model_name}.pth")
    model_name_lower = model_name.lower()
    
    if model_name_lower == "cnn":
        if os.path.exists(model_path):
            print(f"加载自定义CNN模型权重: {model_path}")
            model = build_cnn_model(num_channels, num_classes)
            model.load_state_dict(torch.load(model_path))
        else:
            print("创建新的自定义CNN模型")
            model = build_cnn_model(num_channels, num_classes)
            torch.save(model.state_dict(), model_path)
        return model


    if model_name_lower not in MODEL_DICT:
        raise ValueError(f"不支持的模型: {model_name}. 请选择 'resnet18', 'resnet50', 'vgg16', 'vgg19'.")

    if os.path.exists(model_path):
        print(f"加载模型权重: {model_path}")
        model = MODEL_DICT[model_name_lower](weights=True)
        model.load_state_dict(torch.load(model_path))
    else:
        print(f"下载 {model_name} 预训练模型")
        model = MODEL_DICT[model_name_lower](weights=weights)
        torch.save(model.state_dict(), model_path)

    return model

def load_model2(model_name: str, num_channels: int = 3, num_classes: int = 10, weights: bool = True, model_dir: str = "./"):
    """
    下载模型并保存到本地，之后加载已保存的模型，或者创建自定义CNN模型。
    
    参数：
        model_name (str): 要加载的模型名称（'resnet18'、'vgg16'、'resnet50'、'vgg19'，或 'cnn'）。
        num_channels (int): 输入图像的通道数，默认3（RGB图像）。
        num_classes (int): 模型的分类数。
        weights (bool): 是否加载预训练权重（默认是True）。
        model_dir (str): 模型保存的目录。
        
    返回：
        torch.nn.Module: 加载并准备好的模型。
    """
    # 如果模型名称是 'cnn'，创建一个自定义CNN模型
    if model_name.lower() == "cnn":
        print(f"创建自定义CNN模型，通道数: {num_channels}, 类别数: {num_classes}")
        return build_cnn_model(num_channels, num_classes)

    # 确保模型保存目录存在
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # 构建模型保存路径
    model_path = os.path.join(model_dir, f"{model_name}.pth")

    # 检查模型是否已经存在
    if os.path.exists(model_path):
        print(f"从本地加载模型权重: {model_path}")
        model = torch.load(model_path)
    else:
        print(f"正在下载并保存模型权重: {model_name}...")

        # 根据模型名称选择合适的模型
        if model_name.lower() == 'resnet18':
            model = models.resnet18(weights=weights)
        elif model_name.lower() == 'vgg16':
            model = models.vgg16(weights=weights)
        elif model_name.lower() == 'resnet50':
            model = models.resnet50(weights=weights)
        elif model_name.lower() == 'vgg19':
            model = models.vgg19(weights=weights)
        else:
            raise ValueError(f"模型 '{model_name}' 不支持，选择 'resnet18', 'vgg16', 'resnet50' 或 'vgg19'。")
        
        # 保存模型权重到本地
        torch.save(model.state_dict(), model_path)
        print(f"模型权重已保存到: {model_path}")
    
    return model

# 示例使用
# 创建一个自定义CNN模型
cnn = load_model(model_name="cnn", num_channels=3, num_classes=10)

# 加载预训练的ResNet18模型
pretrained_resnet = load_model(model_name="resnet18", weights=True)
import os
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision import models

def get_preprocessing_pipeline(resize_size=224, is_rgb=True, normalize_mean=None, normalize_std=None):
    """
    根据输入的参数构建图像预处理流水线
    - resize_size：调整图像的大小
    - is_rgb：是否转换为RGB模式
    - normalize_mean：归一化的均值
    - normalize_std：归一化的标准差
    """
    transform_list = []
    
    # Resize 图像大小
    transform_list.append(transforms.Resize((resize_size, resize_size)))
    
    # 如果是 RGB 格式并且图像是灰度图，进行转换
    if is_rgb:
        transform_list.append(transforms.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img))
    
    # 将图像转换为 tensor
    transform_list.append(transforms.ToTensor())
    
    # 如果需要归一化
    if normalize_mean is not None and normalize_std is not None:
        transform_list.append(transforms.Normalize(mean=normalize_mean, std=normalize_std))

    # 返回预处理流水线
    return transforms.Compose(transform_list)

def load_and_preprocess_image(image_path, resize_size=224, is_rgb=True, normalize_mean=None, normalize_std=None):
    """
    加载并预处理图像，返回 4D 张量（含 batch 维度）
    """
    image = Image.open(image_path)
    pipeline = get_preprocessing_pipeline(resize_size, is_rgb, normalize_mean, normalize_std)
    img_tensor = pipeline(image).unsqueeze(0)  # 增加 batch 维度
    return img_tensor

def predict_on_tensor(img_tensor, model):
    """
    对已经预处理好的图像张量进行预测，并返回预测的类别和概率。
    - img_tensor：已经预处理的图像张量
    - model：预训练的模型
    """
    model.eval()  # 设置为评估模式
    with torch.no_grad():
        output = model(img_tensor)  # 输入模型进行预测
        probabilities = torch.nn.functional.softmax(output, dim=1)  # 获取类别概率
        predicted_class = torch.argmax(probabilities, dim=1).item()  # 获取预测的类别
    return predicted_class, probabilities.squeeze().tolist()  # 返回类别和概率

def predict_from_file(file_path, model, resize_size=224, is_rgb=True, normalize_mean=None, normalize_std=None):
    """
    对指定的单个文件进行预测，返回预测类别及其概率。
    """
    # 1. 加载并预处理图像
    img_tensor = load_and_preprocess_image(file_path, resize_size, is_rgb, normalize_mean, normalize_std)

    # 2. 调用predict_on_tensor进行预测
    return predict_on_tensor(img_tensor, model)

def predict_from_directory(directory_path, model, resize_size=224, is_rgb=True, normalize_mean=None, normalize_std=None):
    """
    遍历目录下的每个子文件夹（作为标签），对每个图像进行预测
    并返回包含文件路径、父目录标签、预测类别及概率的结果列表
    """
    predictions = []
    
    # 遍历目录下的每个子文件夹
    for label_folder in os.listdir(directory_path):
        label_path = os.path.join(directory_path, label_folder)
        if os.path.isdir(label_path):
            # 遍历子文件夹内的每个图像文件
            for filename in os.listdir(label_path):
                file_path = os.path.join(label_path, filename)
                if os.path.isfile(file_path) and file_path.lower().endswith(('png', 'jpg', 'jpeg')):
                    # 获取预测结果
                    predicted_class, prediction_scores = predict_from_file(file_path, model, resize_size, is_rgb, normalize_mean, normalize_std)
                    predictions.append({
                        "file": file_path,
                        "label": label_folder,  # 父目录名称作为标签
                        "prediction": predicted_class,
                        "scores": prediction_scores
                    })
    return predictions


# ===============================
# 示例：加载预训练模型并进行预测
# ===============================
if __name__ == "__main__":
    # 1. 加载预训练的 ResNet50 模型
    model = models.resnet50(pretrained=True)

    # 2. 设置预测参数
    directory_path = "your_directory"  # 请替换为实际目录路径
    resize_size = 224  # 可根据需要调整
    is_rgb = True  # 如果为 RGB 图像，设置为 True
    normalize_mean = [0.485, 0.456, 0.406]  # ImageNet 的均值
    normalize_std = [0.229, 0.224, 0.225]  # ImageNet 的标准差

    # 3. 调用预测函数
    predictions = predict_from_directory(directory_path, model, resize_size=resize_size, is_rgb=is_rgb, normalize_mean=normalize_mean, normalize_std=normalize_std)

    # 打印预测结果
    for prediction in predictions:
        print(f"File: {prediction['file']}, Label: {prediction['label']}, Prediction: {prediction['prediction']}, Scores: {prediction['scores']}")
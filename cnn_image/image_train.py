import torch
import torch.nn as nn
import torch.optim as optim

from utils.system_utils import get_system_and_usage_data,get_model_size_gb,get_memory_usage

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def train_loop(model, device, train_loader, max_epochs):
    model.train()
    # 定义损失函数和优化器
    compute_loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(max_epochs):
        running_loss = 0.0
        batch_count = 0
        image_count = 0

        optimizer.zero_grad()
        for inputs, labels in train_loader:
            batch_count += 1
            inputs, labels = inputs.to(device), labels.to(device)
            image_count += inputs.size(0)

            # optimizer.zero_grad()
            outputs = model(inputs)
            loss = compute_loss(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            system_data=get_system_and_usage_data()
            memory_usage_gb=system_data.get('current_usage').get('memory_usage_gb') 
            model_size = get_memory_usage()  
            print(model_size)
   
            logger.info(
                f"epoch={epoch}, batches={batch_count}, image_count={image_count}, memory_usage_gb={memory_usage_gb},running_loss={running_loss}"
            )
            
            
def evaluate_model(model, test_loader, device):
    """
    在测试集上评估模型的准确率
    # TODO: calculate balanced_accuracy, f1 etc 
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

    # 
    accuracy = 100 * correct / total
    return accuracy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import random   # for random.shuffle
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix    # for confusion matrix
import pandas as pd
import seaborn as sn
import time
import copy   # for deep copy   # for deep copy


# 1. Load data
# 2. Preprocess data
# 3. Create CNN model
# 4. Train the model
# 5. Test the model
# 6. Save the model
# 7. Load the model
# 8. Make prediction
# 9. Evaluate model

# 1. Load data
# Load data
# Load data
# Load data
# Load data

nn.Conv2d(3, 6, 5)   # 3 input channel, 6 output channel, 5x5 kernel
nn.MaxPool2d(2, 2)    # 2x2 kernel, stride 2
nn.Conv2d(6, 16, 5)   # 6 input channel, 16 output channel, 5x5 kernel
nn.MaxPool2d(2, 2)    # 2x2 kernel, stride 2
nn.Linear(16*5*5, 120)   # 16*5*5 input, 120 output
nn.Linear(120, 84)   # 120 input, 84 output
nn.Linear(84, 10)    # 84 input, 10 output
nn.ReLU()   # activation function
nn.Sigmoid()   # activation function
nn.CrossEntropyLoss()   # loss function
nn.MSELoss()   # loss function


# 2. Preprocess data
# Preprocess data
# Preprocess data
torch.save(model.state_dict(), 'model.pth')
model.load_state_dict(torch.load('model.pth'))
model.eval()

# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        # 定义第一个卷积层
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        # 定义池化层
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # 定义第二个卷积层
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        # 定义第一个全连接层
        self.fc1 = nn.Linear(64 * 100 * 100, 512)
        # 定义第二个全连接层
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # 前向传播：卷积 -> 激活 -> 池化
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # 展平张量
        x = x.view(-1, 64 * 100 * 100)
        # 前向传播：全连接层 -> 激活
        x = F.relu(self.fc1(x))
        # 前向传播：全连接层
        x = self.fc2(x)
        return x  # 返回输出

    def save_model(self, path):
        torch.save(self.state_dict(), path)
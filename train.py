# train.py

import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from model import SimpleCNN
from dataset import load_data
import time

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

def train_model(
    data_dir,
    num_classes,
    num_epochs=25,
    batch_size=32,
    learning_rate=0.001,
    test_split=0.2,
):
    # 加载训练和测试数据
    train_loader, test_loader = load_data(data_dir, batch_size, test_split)

    # 初始化模型
    model = SimpleCNN(num_classes)

    # 定义损失函数
    criterion = torch.nn.CrossEntropyLoss()

    # 选择设备
    model.to(device)
    criterion.to(device)

    # 定义优化器
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 定义学习率调度器
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # 训练模型
    for epoch in range(num_epochs):
        print(f"Starting epoch {epoch + 1}/{num_epochs}")  # 添加输出
        start_time = time.time()  # 记录开始时间
        model.train()  # 设置模型为训练模式
        running_loss = 0.0

        # 遍历训练数据
        batch_start_time = time.time()  # 记录批次开始时间
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)  # 切换设备
            labels = labels.to(device)  # 切换设备
            optimizer.zero_grad()  # 清空梯度
            outputs = model(inputs)  # 前向传播
            loss = criterion(outputs, labels)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数
            running_loss += loss.item() * inputs.size(0)  # 累加损失

            if i % 10 == 9:  # 每10个批次输出一次进度
                batch_time = time.time() - batch_start_time  # 计算批次耗时
                print(f"Batch {i + 1}/{len(train_loader)}, Loss: {loss.item():.4f}, Time: {batch_time:.2f}s")
                batch_start_time = time.time()  # 重置批次开始时间

        scheduler.step()  # 更新学习率
        epoch_loss = running_loss / len(train_loader.dataset)  # 计算平均损失
        epoch_time = time.time() - start_time  # 计算耗时
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Time: {epoch_time:.2f}s")  # 打印损失和耗时

    print("Training complete")  # 添加输出
    return model  # 返回训练好的模型

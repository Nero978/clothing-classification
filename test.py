import torch
from dataset import load_data
from device import device


def test_model(model, data_dir, batch_size=32, test_split=0.2):
    # 加载测试数据
    _, test_loader = load_data(data_dir, batch_size, test_split)

    # 设置模型为评估模式
    model.eval()

    # 将模型移动到设备
    model.to(device)

    correct = 0
    total = 0

    # 禁用梯度计算
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)  # 将数据移动到设备
            outputs = model(inputs)  # 前向传播
            _, predicted = torch.max(outputs.data, 1)  # 获取预测结果
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if i % 10 == 9:  # 每10个批次输出一次进度
                print(
                    f"Batch {i + 1}/{len(test_loader)}, Accuracy: {correct / total:.4f}"
                )

    # 计算准确率
    accuracy = correct / total
    print(f"Final Accuracy: {accuracy:.4f}")

# 测试模型
if __name__ == "__main__":
    # 模型路径
    model_path = "model_xxxx.pth"
    # 数据集路径
    data_dir = "./clothing-dataset-full"
    # 测试集比例
    test_split = 0.2
    # 批量大小
    batch_size = 32

    model = torch.load(model_path, map_location=device)
    test_model(model, data_dir, batch_size, test_split)
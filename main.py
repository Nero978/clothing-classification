# main.py

from train import train_model
from test import test_model
from datetime import datetime

if __name__ == "__main__":
    # 数据集路径
    data_dir = "./clothing-dataset-full"
    # 测试集比例
    test_split = 0.2
    # 类别数量
    num_classes = 20
    # 训练轮数
    num_epochs = 25
    # 批量大小
    batch_size = 32
    # 学习率
    learning_rate = 0.001
    # 模型保存路径
    model_name = "model"

    # 训练模型
    model = train_model(
        data_dir, num_classes, num_epochs, batch_size, learning_rate, test_split
    )
    # 测试模型
    test_model(model, data_dir, batch_size, test_split)

    # 询问用户是否保存模型
    save_model = input("Save model? (y/n): ")
    if save_model == "y":
        date_time = datetime.now().strftime("%Y%m%d%H%M%S")
        model_path = f"{model_name}_{date_time}.pth"
        model.save_model(model_path)

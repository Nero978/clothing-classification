# Clothing Classification

这是一个用于服装分类的深度学习项目，使用 PyTorch 实现。

## 项目结构

```
clothing-classification/
│
├── dataset.py          # 数据集加载和处理
├── device.py           # 设备选择
├── main.py             # 主程序入口
├── map.py              # 标签映射
├── model.py            # 模型定义
├── test.py             # 测试模型
├── train.py            # 训练模型
└── README.md           # 项目说明
```

## 环境依赖

- Python 3.7+
- PyTorch 1.7+
- torchvision
- pandas
- Pillow

可以使用以下命令安装依赖：

```bash
pip install -r requirements.txt
```

## 数据集

本项目使用了一个服装分类数据集：[Clothing dataset (full, high resolution)](https://www.kaggle.com/datasets/agrigorev/clothing-dataset-full)

请将数据集放置在 `./clothing-dataset-full` 目录下，数据集应包含以下文件：

- `images.csv`：包含图像文件名和标签的 CSV 文件
- `images_original/`：包含图像文件的目录

## 训练模型

可以使用以下命令训练模型：

```bash
python main.py
```

训练过程中会定期保存 checkpoint 文件 `checkpoint.pth.tar`，以便在中断后继续训练。

## 测试模型

训练完成后，模型会自动进行测试，并输出测试集的准确率。

## 模型保存

训练完成后，模型会自动保存到当前目录，文件名格式为 `model_YYYYMMDDHHMMSS.pth`。

## 断点续训

如果训练过程中断，可以通过加载 checkpoint 文件继续训练。checkpoint 文件默认保存路径为 `checkpoint.pth.tar`。

## 性能优化

### CUDA 加速

如果有可用的 CUDA 设备，项目将自动使用 CUDA 加速。

如果 CUDA 不可用，可能是安装了 CPU 版本的 Pytorch。请考虑按照 [指南](https://pytorch.org/get-started/locally/) 重新安装可用于 CUDA 加速的 PyTorch。

### MPS

如果有多个 GPU，将自动使用 MPS（Multi-Process Service）来提高性能。

### 数据集预处理

前往 `dataset.py` 文件，修改 `num_workers`、`prefech_factor` 变量来优化数据集加载性能。具体见代码注释。
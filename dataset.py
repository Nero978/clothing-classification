# dataset.py

import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms

class ClothingDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        # 读取CSV文件，包含图像文件名和标签     
        self.annotations = pd.read_csv(csv_file)
        # 图像文件所在的根目录
        self.root_dir = root_dir
        # 图像变换操作
        self.transform = transform

    def __len__(self):
        # 返回数据集的大小
        return len(self.annotations)

    def __getitem__(self, idx):
        # 获取图像文件的路径
        img_name = os.path.join(self.root_dir, self.annotations.iloc[idx, 0])
        # 添加文件扩展名
        img_name = img_name + '.jpg'
        # 打开图像文件
        image = Image.open(img_name).convert("RGB")
        # 获取图像对应的标签
        label = self.annotations.iloc[idx, 2]
        
        # 定义填充操作
        padding = transforms.Pad((0, 0, max(0, 400 - image.size[0]), max(0, 400 - image.size[1])), fill=(255, 255, 255))
        # 定义调整大小操作
        resize = transforms.Resize((400, 400))
        
        # 应用填充和调整大小操作
        image = resize(padding(image))
        
        # 如果定义了变换操作，则应用于图像
        if self.transform:
            image = self.transform(image)

        # 返回图像和标签
        return image, label

def load_data(data_dir, batch_size=32, test_split=0.2):
    # 定义CSV文件和图像文件所在的目录
    csv_file = os.path.join(data_dir, 'images.csv')
    root_dir = os.path.join(data_dir, 'images_compressed')
    
    # 定义图像变换操作
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # 创建数据集对象
    dataset = ClothingDataset(csv_file=csv_file, root_dir=root_dir, transform=transform)
    # 计算训练集和测试集的大小
    train_size = int((1-test_split) * len(dataset))
    test_size = len(dataset) - train_size
    # 划分训练集和测试集
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 返回训练集和测试集的数据加载器
    return train_loader, test_loader
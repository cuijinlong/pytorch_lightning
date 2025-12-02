import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from pytorch_lightning import LightningDataModule
import yaml
from pathlib import Path


class MNISTDataset(Dataset):
    """MNIST数据集类"""

    def __init__(self, data_dir, split='train', transform=None):
        """
        初始化MNIST数据集

        Args:
            data_dir: 数据目录
            split: 数据集分割 ('train', 'val', 'test')
            transform: 数据变换
        """
        self.data_dir = data_dir
        self.split = split
        self.transform = transform

        # 创建数据目录
        os.makedirs(data_dir, exist_ok=True)

        # 下载或加载MNIST数据集
        self.dataset = torchvision.datasets.MNIST(
            root=data_dir,
            train=(split == 'train'),
            download=True,
            transform=self.transform
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]

        # MNIST是单通道，但YOLO需要3通道
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)

        return image, label


class MNISTDataModule(LightningDataModule):
    """MNIST数据模块"""

    def __init__(self, config):
        """
        初始化数据模块

        Args:
            config: 数据配置字典
        """
        super().__init__()
        self.config = config
        self.data_dir = Path(config.get('data', './data/mnist'))
        self.batch_size = config.get('batch_size', 64)
        self.num_workers = config.get('workers', 4)
        self.img_size = config.get('img_size', 224)

        # 数据增强配置
        self.augment = config.get('augment', True)
        self.hflip = config.get('fliplr', 0.5)

        # 创建数据变换
        self.train_transform = self._create_transform(train=True)
        self.val_transform = self._create_transform(train=False)

        # 数据集
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def _create_transform(self, train=True):
        """创建数据变换管道"""
        transform_list = []

        # 训练时的数据增强
        if train and self.augment:
            transform_list.extend([
                transforms.RandomRotation(10),
                transforms.RandomAffine(
                    degrees=0,
                    translate=(0.1, 0.1),
                    scale=(0.9, 1.1)
                ),
            ])

            if self.hflip > 0:
                transform_list.append(transforms.RandomHorizontalFlip(p=self.hflip))

        # 调整大小和转换为张量
        transform_list.extend([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  # MNIST的均值和标准差
        ])

        return transforms.Compose(transform_list)

    def setup(self, stage=None):
        """设置数据集"""
        # 训练集
        if stage == 'fit' or stage is None:
            self.train_dataset = MNISTDataset(
                data_dir=self.data_dir,
                split='train',
                transform=self.train_transform
            )

            # 从训练集中划分验证集（80%训练，20%验证）
            train_size = int(0.8 * len(self.train_dataset))
            val_size = len(self.train_dataset) - train_size

            self.train_dataset, self.val_dataset = torch.utils.data.random_split(
                self.train_dataset, [train_size, val_size]
            )

            # 更新验证集的变换
            self.val_dataset.dataset.transform = self.val_transform

        # 测试集
        if stage == 'test' or stage is None:
            self.test_dataset = MNISTDataset(
                data_dir=self.data_dir,
                split='test',
                transform=self.val_transform
            )

    def train_dataloader(self):
        """训练数据加载器"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )

    def val_dataloader(self):
        """验证数据加载器"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def test_dataloader(self):
        """测试数据加载器"""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def predict_dataloader(self):
        """预测数据加载器"""
        return self.test_dataloader()

    def get_class_names(self):
        """获取类别名称"""
        return [str(i) for i in range(10)]


def prepare_mnist_dataset(config_path):
    """
    准备MNIST数据集

    Args:
        config_path: 配置文件路径

    Returns:
        MNISTDataModule实例
    """
    # 加载配置
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 创建数据模块
    datamodule = MNISTDataModule(config)

    # 设置数据模块
    datamodule.setup()

    print(f"训练集大小: {len(datamodule.train_dataset)}")
    print(f"验证集大小: {len(datamodule.val_dataset)}")
    if datamodule.test_dataset:
        print(f"测试集大小: {len(datamodule.test_dataset)}")

    return datamodule


if __name__ == "__main__":
    # 测试数据模块
    config = {
        'data': './data/mnist',
        'batch_size': 32,
        'workers': 2,
        'img_size': 224,
        'augment': True
    }

    datamodule = MNISTDataModule(config)
    datamodule.setup()

    # 测试一个批次
    train_loader = datamodule.train_dataloader()
    for batch_idx, (images, labels) in enumerate(train_loader):
        print(f"Batch {batch_idx}:")
        print(f"  图像形状: {images.shape}")  # [batch_size, 3, 224, 224]
        print(f"  标签形状: {labels.shape}")  # [batch_size]
        print(f"  类别数量: 10")

        # 显示第一个图像的信息
        print(f"  第一个图像 - 最小像素值: {images[0].min():.4f}")
        print(f"  第一个图像 - 最大像素值: {images[0].max():.4f}")
        print(f"  第一个图像 - 均值: {images[0].mean():.4f}")
        print(f"  第一个图像 - 标签: {labels[0].item()}")

        break
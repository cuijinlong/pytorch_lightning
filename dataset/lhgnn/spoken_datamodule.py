# dataset/lhgnn/spoken_datamodule.py
from typing import Any, Dict, Optional, Tuple
import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
import json
import os
import glob
from torch.utils.data.distributed import DistributedSampler
from dataset.lhgnn.spoken_dataset import SpokenDataset

class SpokenDataModule(LightningDataModule):

    def __init__(self,
                 data_dir: str = "/opt/datasets/spoken/optional/wav",
                 batch_size: int = 8,
                 num_workers: int = 1,
                 pin_memory: bool = True,
                 persistent_workers: bool = True,
                 sr: int = 16000,
                 fmin: int = 20,
                 fmax: int = 8000,
                 num_mels: int = 128,
                 window_type: str = "hanning",
                 target_len: int = 1024,
                 freqm: int = 128,
                 timem: int = 1024,
                 mixup: float = 0.5,
                 norm_mean: float = -4.5,
                 norm_std: float = 4.5,
                 num_devices: int = 1,
                 train_val_split: float = 0.8
                 ) -> None:

        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.sr = sr
        self.fmin = fmin
        self.fmax = fmax
        self.num_mels = num_mels
        self.window_type = window_type
        self.target_len = target_len
        self.freqm = freqm
        self.timem = timem
        self.mixup = mixup
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.num_devices = num_devices
        self.train_val_split = train_val_split

        self.audio_conf = {
            'sr': sr, 'fmin': fmin, 'fmax': fmax,
            'num_mels': num_mels, 'window_type': window_type,
            'target_len': target_len, 'freqm': freqm, 'timem': timem,
            'norm_mean': norm_mean, 'norm_std': norm_std, 'mixup': mixup
        }

    def setup(self, stage: Optional[str] = None) -> None:
        """创建训练、验证和测试数据集"""

        # 获取所有wav文件
        all_files = glob.glob(os.path.join(self.data_dir, "*.wav"))

        # 分离训练/验证集和测试集
        train_val_files = []
        test_files = []

        for file_path in all_files:
            filename = os.path.basename(file_path)
            try:
                # 解析文件名: digit_person_style.wav
                parts = filename.replace('.wav', '').split('_')
                if len(parts) >= 3:
                    digit = parts[0]
                    style = parts[-1]

                    # 说话方式0-4作为测试集
                    if style.isdigit() and 0 <= int(style) <= 4:
                        test_files.append({
                            'wav': file_path,
                            'labels': digit  # 标签是数字
                        })
                    else:
                        train_val_files.append({
                            'wav': file_path,
                            'labels': digit
                        })
            except:
                continue

        # 划分训练集和验证集
        train_size = int(len(train_val_files) * self.train_val_split)
        val_size = len(train_val_files) - train_size

        train_files = train_val_files[:train_size]
        val_files = train_val_files[train_size:]

        print(f"训练集大小: {len(train_files)}")
        print(f"验证集大小: {len(val_files)}")
        print(f"测试集大小: {len(test_files)}")

        # 创建数据集
        self.train_dataset = SpokenDataset(train_files, self.audio_conf, mode='train')
        self.val_dataset = SpokenDataset(val_files, self.audio_conf, mode='eval')
        self.test_dataset = SpokenDataset(test_files, self.audio_conf, mode='eval')

        # 获取类别数量
        self.num_classes = self.train_dataset.num_classes

    def train_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            shuffle=True
        )

    def val_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            persistent_workers=self.persistent_workers
        )

    def test_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            persistent_workers=self.persistent_workers
        )


# 使用示例
if __name__ == "__main__":
    # 创建数据模块
    datamodule = SpokenDataModule(
        data_dir="/opt/datasets/spoken/optional/wav",
        batch_size=32,
        num_workers=1
    )

    # 设置数据
    datamodule.setup()

    # 获取数据加载器
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    test_loader = datamodule.test_dataloader()

    # 测试一个批次
    for batch in train_loader:
        features, labels = batch
        print(f"特征形状: {features.shape}")  # [batch_size, target_len, num_mels]
        print(f"标签形状: {labels.shape}")  # [batch_size, num_classes]
        print(f"类别数量: {datamodule.num_classes}")
        break
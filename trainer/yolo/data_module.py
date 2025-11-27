# /opt/pytorch_lightning/trainer/yolo/data_module.py
import pytorch_lightning as pl
from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.utils import IterableSimpleNamespace
from pathlib import Path
import yaml
import os
from ultralytics.cfg import DEFAULT_CFG_DICT

cfg = IterableSimpleNamespace(**DEFAULT_CFG_DICT)


class YoloDataModule(pl.LightningDataModule):
    def __init__(self, config):
        """
        初始化YOLO数据模块

        Args:
            config: 配置字典，包含数据路径、批大小等参数
        """
        super().__init__()
        self.config = config
        data_path = Path(config['data']['path'])
        self.data_yaml = data_path / 'coco128.yaml'
        self.datasets = {}
        self._is_setup = False
        self.data_dict = None

        os.environ['YOLO_DATA_CFG'] = str(self.data_yaml)

    def _resolve_paths(self, data_dict):
        """解析数据路径，确保使用绝对路径"""
        base_path = Path(self.data_yaml).parent

        # 解析训练路径
        train_path = data_dict.get('train', '')
        if train_path and not Path(train_path).is_absolute():
            train_path = base_path / train_path
        data_dict['train'] = str(train_path) if train_path else ''

        # 解析验证路径
        val_path = data_dict.get('val', '')
        if val_path and not Path(val_path).is_absolute():
            val_path = base_path / val_path
        data_dict['val'] = str(val_path) if val_path else ''

        # 解析测试路径
        test_path = data_dict.get('test', '')
        if test_path and not Path(test_path).is_absolute():
            test_path = base_path / test_path
        data_dict['test'] = str(test_path) if test_path else ''

        return data_dict

    def setup(self, stage=None):
        """数据准备阶段，可用于数据预处理"""
        if self._is_setup:
            return

        print("开始设置数据模块...")



        try:
            with open(self.data_yaml, 'r') as f:
                self.data_dict = yaml.safe_load(f)
            print(f"成功加载数据配置文件: {self.data_yaml}")
            print(f"数据配置内容: {self.data_dict}")

            # 解析路径为绝对路径
            self.data_dict = self._resolve_paths(self.data_dict)
            print(f"解析后的数据路径 - 训练: {self.data_dict.get('train', '')}")
            print(f"解析后的数据路径 - 验证: {self.data_dict.get('val', '')}")

        except Exception as e:
            print(f"加载数据配置文件失败: {e}")
            raise

        # 检查路径是否存在
        train_path = Path(self.data_dict.get('train', ''))
        if not train_path.exists():
            print(f"警告: 训练路径不存在: {train_path}")
            # 尝试自动查找训练数据
            possible_paths = [
                Path(self.config['data']['path']) / 'images' / 'train2017',
                Path(self.config['data']['path']) / 'train2017',
                Path(self.config['data']['path']) / 'images' / 'train',
                Path(self.config['data']['path']) / 'train'
            ]
            for path in possible_paths:
                if path.exists():
                    self.data_dict['train'] = str(path)
                    self.data_dict['val'] = str(path)  # 如果没有验证集，使用训练集
                    print(f"自动找到训练数据路径: {path}")
                    break
            else:
                raise FileNotFoundError(f"无法找到训练数据，请检查数据路径: {self.config['data']['path']}")

        cfg.data = str(self.data_yaml)
        cfg.imgsz = self.config['training'].get('image_size', 512)
        cfg.batch = self.config['training']['batch_size']
        cfg.workers = self.config['data']['num_workers']

        # 构建训练和验证数据集
        try:
            print("开始构建训练数据集...")
            self.datasets['train'] = build_yolo_dataset(
                cfg=cfg,
                img_path=self.data_dict.get('train', ''),
                batch=cfg.batch,
                data=self.data_dict,
                mode='train',
                rect=False,
                stride=32
            )

            print("开始构建验证数据集...")
            # 如果没有单独的验证集，使用训练集的一部分作为验证
            val_path = self.data_dict.get('val', '')
            if not val_path or val_path == self.data_dict.get('train', ''):
                print("警告: 使用训练集作为验证集，这可能导致评估不准确")
                val_path = self.data_dict.get('train', '')

            self.datasets['val'] = build_yolo_dataset(
                cfg=cfg,
                img_path=val_path,
                batch=cfg.batch,
                data=self.data_dict,
                mode='val',
                rect=False,
                stride=32
            )

            print("数据集构建成功")
            print(f"训练集大小: {len(self.datasets['train'])}")
            print(f"验证集大小: {len(self.datasets['val'])}")
            self._is_setup = True

        except Exception as e:
            print(f"构建数据集失败: {e}")
            if not hasattr(self, 'datasets'):
                self.datasets = {}
            raise

    def train_dataloader(self):
        """构建训练数据加载器"""
        if not self._is_setup:
            self.setup()
        return self._build_dataloader('train')

    def val_dataloader(self):
        """构建验证数据加载器"""
        if not self._is_setup:
            self.setup()
        return self._build_dataloader('val')

    def _build_dataloader(self, mode='train'):
        """构建数据加载器"""
        try:
            if not hasattr(self, 'datasets'):
                self.datasets = {}

            if mode not in self.datasets:
                raise ValueError(f"数据集 {mode} 未在setup中构建")

            shuffle = (mode == 'train')
            dataloader = build_dataloader(
                dataset=self.datasets[mode],
                batch=self.config['training']['batch_size'],
                workers=self.config['data']['num_workers'],
                shuffle=shuffle,
                rank=-1,
                drop_last=(mode == 'train'),
                pin_memory=True
            )

            print(f"成功创建 {mode} 数据加载器")
            return dataloader

        except Exception as e:
            print(f"创建 {mode} 数据加载器失败: {e}")
            raise

    def teardown(self, stage=None):
        """数据清理（可选）"""
        if hasattr(self, 'datasets'):
            self.datasets.clear()
        self._is_setup = False
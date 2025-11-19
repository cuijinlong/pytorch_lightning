# D:\workspace_py\pytorch_lightning\trainer\lightning\data_module.py
import pytorch_lightning as pl
from dataset.single_modal_data_loader import SingleModalDataLoader

class ResNetDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.data_loader = SingleModalDataLoader(config)
        self._train_loader = None
        self._val_loader = None
        self._test_loader = None
        self._is_initialized = False

    def _ensure_initialized(self):
        """确保数据加载器已经初始化"""
        if not self._is_initialized:
            self.setup()
            self._is_initialized = True

    def setup(self, stage=None):
        if self._train_loader is None:
            self._train_loader, self._val_loader, self._test_loader = self.data_loader.create_data_loaders()
            self._is_initialized = True

    def train_dataloader(self):
        self._ensure_initialized()
        return self._train_loader

    def val_dataloader(self):
        self._ensure_initialized()
        return self._val_loader if self._val_loader else self._train_loader

    def test_dataloader(self):
        self._ensure_initialized()
        return self._test_loader if self._test_loader else self._val_loader

    def predict_dataloader(self):
        self._ensure_initialized()
        return self.test_dataloader()

    @property
    def num_classes(self):
        self._ensure_initialized()
        return self.data_loader.get_num_classes()

    @property
    def label_dict(self):
        self._ensure_initialized()
        return self.data_loader.get_label_dict()
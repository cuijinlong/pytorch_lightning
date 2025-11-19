# D:\workspace_py\pytorch_lightning\trainer\lightning\data_module.py
import pytorch_lightning as pl
from dataset.single_modal_data_loader import SingleModalDataLoader

class ResNetDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.data_loader = SingleModalDataLoader(config)

    def setup(self, stage=None):
        # 不需要手动调用，Lightning 会自动在合适的时候调用
        self.train_loader, self.val_loader, self.test_loader = self.data_loader.create_data_loaders()

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader

    def predict_dataloader(self):
        return self.test_loader

    @property
    def num_classes(self):
        return self.data_loader.get_num_classes()

    @property
    def label_dict(self):
        return self.data_loader.get_label_dict()
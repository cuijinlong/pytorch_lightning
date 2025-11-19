# D:\workspace_py\pytorch_lightning\trainer\lightning\resnet_lightning.py
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchvision import models
from torchmetrics import Accuracy
import torch.optim as optim
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

class ResNetLightning(pl.LightningModule):
    def __init__(self, config, num_classes, label_dict):
        super().__init__()
        self.config = config
        self.num_classes = num_classes
        self.label_dict = label_dict
        self.save_hyperparameters()

        # 模型架构
        model_name = config.get('model_name', 'resnet18')
        pretrained = config.get('pretrained', True)

        self.model = getattr(models, model_name)(pretrained=pretrained)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

        # 损失函数
        self.criterion = nn.CrossEntropyLoss()

        # 评估指标
        self.train_accuracy = Accuracy(num_classes=num_classes)
        self.val_accuracy = Accuracy(num_classes=num_classes)
        self.test_accuracy = Accuracy(num_classes=num_classes)

        # 用于记录预测结果
        self.test_predictions = []
        self.test_labels = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch['image'], batch['label']
        outputs = self(images)
        loss = self.criterion(outputs, labels)

        # 记录指标
        self.train_accuracy(outputs, labels)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', self.train_accuracy, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch['image'], batch['label']
        outputs = self(images)
        loss = self.criterion(outputs, labels)

        self.val_accuracy(outputs, labels)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_acc', self.val_accuracy, on_epoch=True, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        images, labels = batch['image'], batch['label']
        outputs = self(images)
        loss = self.criterion(outputs, labels)

        self.test_accuracy(outputs, labels)
        self.log('test_loss', loss, on_epoch=True)
        self.log('test_acc', self.test_accuracy, on_epoch=True)

        # 收集预测结果用于后续分析
        probabilities = torch.softmax(outputs, dim=1)
        predicted_probs, predicted_indices = torch.max(probabilities, 1)

        for i in range(len(labels)):
            self.test_predictions.append({
                'predicted_class': predicted_indices[i].item(),
                'confidence': predicted_probs[i].item(),
                'true_class': labels[i].item()
            })

        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        images = batch['image']
        outputs = self(images)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_probs, predicted_indices = torch.max(probabilities, 1)

        return {
            'predictions': predicted_indices,
            'probabilities': probabilities,
            'confidences': predicted_probs
        }

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.config.get('learning_rate', 1e-4),
            weight_decay=self.config.get('weight_decay', 1e-4)
        )

        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.config.get('scheduler_step', 10),
            gamma=self.config.get('scheduler_gamma', 0.1)
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch'
            }
        }

    def on_train_epoch_end(self):
        # 每个训练epoch结束时可以添加自定义逻辑
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('learning_rate', current_lr, on_epoch=True)
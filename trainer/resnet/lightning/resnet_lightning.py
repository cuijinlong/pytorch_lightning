# D:\workspace_py\pytorch_lightning\trainer\lightning\resnet_lightning.py
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchvision import models
from torchmetrics import Accuracy
import torch.optim as optim

class ResNetLightning(pl.LightningModule):
    # KeyPoint2：pl.LightningModule替代了传统的nn.Module + 训练逻辑
    def __init__(self, config, num_classes, label_dict):
        super().__init__()
        self.config = config
        self.num_classes = num_classes
        self.label_dict = label_dict
        self.save_hyperparameters() # KeyPoint3：了手动保存配置参数的代码

        # 模型架构
        model_name = config.get('model_name', 'resnet18')
        pretrained = config.get('pretrained', True)

        self.model = getattr(models, model_name)(pretrained=pretrained)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

        # KeyPoint4：直接在类中定义损失函数和评估指标
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

    """
      1、training_step 替代了训练循环中的单个batch处理逻辑
      2、self.log() 替代了手动打印和记录训练指标
      3、自动处理 zero_grad(), backward(), step() 等操作
      def train_epoch(model, dataloader, optimizer, criterion, device):
          model.train()
          for batch_idx, (images, labels) in enumerate(dataloader):
            # 前向传播
            # 反向传播
            # 手动记录日志
    """
    def training_step(self, batch, batch_idx):
        # KeyPoint4：前向传播过程
        images, labels = batch['image'], batch['label']
        outputs = self(images)

        # KeyPoint5：反向传播过程
        loss = self.criterion(outputs, labels)

        # KeyPoint6：手动记录日志
        self.train_accuracy(outputs, labels)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', self.train_accuracy, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    """
      1、validation_step / test_step 替代了手动的验证/测试循环
      2、自动处理 model.eval() 和 torch.no_grad()
      3、自动计算和聚合指标
      def validate(model, dataloader, criterion, device):
          model.eval()
          with torch.no_grad():
            for images, labels in dataloader:
    """
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

    """
      1、集中管理所有优化相关配置
      2、自动处理学习率调度器的调用时机
    """
    def configure_optimizers(self):
        # KeyPoint7：手动创建优化器和调度器
        # optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.config.get('learning_rate', 1e-4),
            weight_decay=self.config.get('weight_decay', 1e-4)
        )
        # KeyPoint8：调度器
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.config.get('scheduler_step', 10),
            gamma=self.config.get('scheduler_gamma', 0.1)
        )

        # KeyPoint9：在训练循环中手动调用
        # for epoch in range(epochs):
        #    # 训练...
        #    scheduler.step()

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
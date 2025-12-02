# trainer/yolo_mnist_classification/models/classification_module.py
from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torchmetrics import Accuracy, Precision, Recall, F1Score, ConfusionMatrix
from torchmetrics.classification import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassF1Score
import numpy as np
from ultralytics import YOLO
import logging


class YOLOClassificationModule(LightningModule):
    """
    YOLO分类模块
    使用Ultralytics YOLO进行图像分类
    """

    def __init__(
            self,
            model_config: Dict[str, Any],
            optimizer_config: Dict[str, Any],
            scheduler_config: Optional[Dict[str, Any]] = None,
            num_classes: int = 10,
            pretrained: bool = True,
            freeze_backbone: bool = False,
            label_smoothing: float = 0.1,
            learning_rate: float = 0.001
    ):
        """
        初始化YOLO分类模块

        Args:
            model_config: 模型配置
            optimizer_config: 优化器配置
            scheduler_config: 学习率调度器配置
            num_classes: 类别数
            pretrained: 是否使用预训练权重
            freeze_backbone: 是否冻结主干网络
            label_smoothing: 标签平滑系数
            learning_rate: 学习率
        """
        super().__init__()
        self.save_hyperparameters()

        # 模型配置
        self.model_config = model_config
        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.freeze_backbone = freeze_backbone
        self.label_smoothing = label_smoothing
        self.learning_rate = learning_rate

        # 创建YOLO分类模型
        self.model = self._create_model()

        # 损失函数
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        # 训练指标
        self.train_accuracy = MulticlassAccuracy(num_classes=num_classes, average='macro')
        self.train_precision = MulticlassPrecision(num_classes=num_classes, average='macro')
        self.train_recall = MulticlassRecall(num_classes=num_classes, average='macro')
        self.train_f1 = MulticlassF1Score(num_classes=num_classes, average='macro')
        self.train_loss = nn.ModuleList()  # 用于存储每个batch的损失

        # 验证指标
        self.val_accuracy = MulticlassAccuracy(num_classes=num_classes, average='macro')
        self.val_precision = MulticlassPrecision(num_classes=num_classes, average='macro')
        self.val_recall = MulticlassRecall(num_classes=num_classes, average='macro')
        self.val_f1 = MulticlassF1Score(num_classes=num_classes, average='macro')
        self.val_loss = nn.ModuleList()

        # 测试指标
        self.test_accuracy = MulticlassAccuracy(num_classes=num_classes, average='macro')
        self.test_precision = MulticlassPrecision(num_classes=num_classes, average='macro')
        self.test_recall = MulticlassRecall(num_classes=num_classes, average='macro')
        self.test_f1 = MulticlassF1Score(num_classes=num_classes, average='macro')
        self.test_confusion = ConfusionMatrix(task='multiclass', num_classes=num_classes)
        self.test_loss = nn.ModuleList()

        # 最佳指标跟踪
        self.best_val_accuracy = 0.0
        self.best_val_loss = float('inf')

        # 存储预测结果用于分析
        self.val_predictions = []
        self.val_targets = []
        self.test_predictions = []
        self.test_targets = []

        # 日志记录

    def _create_model(self):
        """
        创建YOLO分类模型

        Returns:
            YOLO分类模型
        """
        model = YOLO(f'/Users/cuijinlong/Documents/workspace_py/pytorch_lightning/trainer/yolo_mnist_classification/configs/model/yolo11-cls.yaml')

        # 修改分类头以适应MNIST（10个类别）
        if hasattr(model.model, 'model'):
            # 获取模型结构
            model_structure = model.model.model

            # 查找分类头
            for name, module in model_structure.named_modules():
                if 'classify' in name or isinstance(module, nn.Linear):
                    # 修改分类头的输出维度
                    if isinstance(module, nn.Linear):
                        in_features = module.in_features
                        module.out_features = self.num_classes
                        # 重新初始化权重
                        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                        if module.bias is not None:
                            nn.init.constant_(module.bias, 0)

        # 加载预训练权重
        if self.pretrained and self.model_config.get('pretrained', True):
            try:
                model.load_pretrained_weights()
                logging.info("加载预训练权重成功")
            except:
                logging.warning("加载预训练权重失败，使用随机初始化")

        # 冻结主干网络
        if self.freeze_backbone:
            self._freeze_backbone(model)

        return model

    def _freeze_backbone(self, model):
        """冻结主干网络"""
        # 冻结前N层
        freeze_layers = self.model_config.get('freeze', [])

        if not freeze_layers:
            # 默认冻结除分类头外的所有层
            for name, param in model.named_parameters():
                if 'classify' not in name and 'head' not in name:
                    param.requires_grad = False
        else:
            # 冻结指定层
            for idx, param in enumerate(model.parameters()):
                if idx in freeze_layers:
                    param.requires_grad = False

        logging.info(f"冻结了 {sum(1 for p in model.parameters() if not p.requires_grad)} 个参数")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入图像 [B, C, H, W]

        Returns:
            分类logits [B, num_classes]
        """
        # YOLO模型需要将输入标准化到[0, 1]
        x = x.float() / 255.0 if x.max() > 1.0 else x

        # 前向传播
        outputs = self.model(x)

        # 提取分类logits
        if isinstance(outputs, tuple):
            # 如果是多输出，取最后一个（通常是分类头）
            logits = outputs[-1]
        elif hasattr(outputs, 'probs'):
            # 如果有probs属性，取probs
            logits = outputs.probs
        else:
            # 否则直接使用输出
            logits = outputs

        # 确保输出形状正确
        if logits.dim() > 2:
            logits = logits.view(logits.size(0), -1)

        if logits.size(1) != self.num_classes:
            # 如果维度不匹配，使用线性层进行映射
            if not hasattr(self, 'adaptor'):
                self.adaptor = nn.Linear(logits.size(1), self.num_classes).to(logits.device)
            logits = self.adaptor(logits)

        return logits

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        训练步骤

        Args:
            batch: (images, labels)
            batch_idx: 批次索引

        Returns:
            损失字典
        """
        images, labels = batch

        # 前向传播
        logits = self(images)

        # 计算损失
        loss = self.criterion(logits, labels)

        # 计算预测
        preds = torch.argmax(logits, dim=1)

        # 更新训练指标
        self.train_accuracy.update(preds, labels)
        self.train_precision.update(preds, labels)
        self.train_recall.update(preds, labels)
        self.train_f1.update(preds, labels)

        # 记录损失
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/accuracy', self.train_accuracy, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/lr', self.trainer.optimizers[0].param_groups[0]['lr'], on_step=True, on_epoch=False)

        return {'loss': loss, 'preds': preds, 'targets': labels}

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        验证步骤

        Args:
            batch: (images, labels)
            batch_idx: 批次索引

        Returns:
            结果字典
        """
        images, labels = batch

        # 前向传播
        with torch.no_grad():
            logits = self(images)

        # 计算损失
        loss = self.criterion(logits, labels)

        # 计算预测
        preds = torch.argmax(logits, dim=1)
        probs = F.softmax(logits, dim=1)

        # 更新验证指标
        self.val_accuracy.update(preds, labels)
        self.val_precision.update(preds, labels)
        self.val_recall.update(preds, labels)
        self.val_f1.update(preds, labels)

        # 存储预测结果
        self.val_predictions.append(probs.cpu())
        self.val_targets.append(labels.cpu())

        # 记录损失
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True)

        return {'loss': loss, 'preds': preds, 'targets': labels}

    def on_validation_epoch_end(self) -> None:
        """
        验证epoch结束回调
        """
        # 计算验证指标
        val_acc = self.val_accuracy.compute()
        val_precision = self.val_precision.compute()
        val_recall = self.val_recall.compute()
        val_f1 = self.val_f1.compute()

        # 记录验证指标
        self.log('val/accuracy', val_acc, prog_bar=True)
        self.log('val/precision', val_precision)
        self.log('val/recall', val_recall)
        self.log('val/f1', val_f1)

        # 更新最佳模型
        if val_acc > self.best_val_accuracy:
            self.best_val_accuracy = val_acc
            self.log('val/best_accuracy', self.best_val_accuracy, prog_bar=True)

        # 重置指标
        self.val_accuracy.reset()
        self.val_precision.reset()
        self.val_recall.reset()
        self.val_f1.reset()

        # 清空存储
        self.val_predictions.clear()
        self.val_targets.clear()

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        测试步骤

        Args:
            batch: (images, labels)
            batch_idx: 批次索引

        Returns:
            结果字典
        """
        images, labels = batch

        # 前向传播
        with torch.no_grad():
            logits = self(images)

        # 计算损失
        loss = self.criterion(logits, labels)

        # 计算预测
        preds = torch.argmax(logits, dim=1)
        probs = F.softmax(logits, dim=1)

        # 更新测试指标
        self.test_accuracy.update(preds, labels)
        self.test_precision.update(preds, labels)
        self.test_recall.update(preds, labels)
        self.test_f1.update(preds, labels)
        self.test_confusion.update(preds, labels)

        # 存储预测结果
        self.test_predictions.append(probs.cpu())
        self.test_targets.append(labels.cpu())

        # 记录损失
        self.log('test/loss', loss, on_step=False, on_epoch=True)

        return {'loss': loss, 'preds': preds, 'targets': labels}

    def on_test_epoch_end(self) -> None:
        """
        测试epoch结束回调
        """
        # 计算测试指标
        test_acc = self.test_accuracy.compute()
        test_precision = self.test_precision.compute()
        test_recall = self.test_recall.compute()
        test_f1 = self.test_f1.compute()

        # 计算混淆矩阵
        confusion_matrix = self.test_confusion.compute()

        # 记录测试指标
        self.log('test/accuracy', test_acc, prog_bar=True)
        self.log('test/precision', test_precision)
        self.log('test/recall', test_recall)
        self.log('test/f1', test_f1)

        # 打印混淆矩阵（前5x5）
        logging.info(f"测试准确率: {test_acc:.4f}")
        logging.info(f"混淆矩阵 (前5x5):\n{confusion_matrix[:5, :5]}")

        # 重置指标
        self.test_accuracy.reset()
        self.test_precision.reset()
        self.test_recall.reset()
        self.test_f1.reset()
        self.test_confusion.reset()

        # 清空存储
        self.test_predictions.clear()
        self.test_targets.clear()

    def predict_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """
        预测步骤

        Args:
            batch: 输入图像 [B, C, H, W]
            batch_idx: 批次索引

        Returns:
            预测概率 [B, num_classes]
        """
        # 前向传播
        with torch.no_grad():
            logits = self(batch)
            probs = F.softmax(logits, dim=1)

        return probs

    def configure_optimizers(self):
        """
        配置优化器和学习率调度器

        Returns:
            优化器配置
        """
        # 获取优化器参数
        optimizer_type = self.optimizer_config.get('type', 'AdamW')
        lr = self.optimizer_config.get('lr', self.learning_rate)
        weight_decay = self.optimizer_config.get('weight_decay', 0.0005)

        # 创建优化器
        if optimizer_type == 'AdamW':
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_type == 'Adam':
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_type == 'SGD':
            momentum = self.optimizer_config.get('momentum', 0.9)
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"不支持的优化器类型: {optimizer_type}")

        # 配置学习率调度器
        if self.scheduler_config:
            scheduler_type = self.scheduler_config.get('type', 'CosineAnnealingLR')

            if scheduler_type == 'CosineAnnealingLR':
                T_max = self.scheduler_config.get('T_max', self.trainer.max_epochs)
                eta_min = self.scheduler_config.get('eta_min', 1e-6)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=T_max,
                    eta_min=eta_min
                )
            elif scheduler_type == 'ReduceLROnPlateau':
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode='min',
                    factor=0.1,
                    patience=5,
                    verbose=True
                )
            elif scheduler_type == 'MultiStepLR':
                milestones = self.scheduler_config.get('milestones', [30, 60, 90])
                gamma = self.scheduler_config.get('gamma', 0.1)
                scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    optimizer,
                    milestones=milestones,
                    gamma=gamma
                )
            else:
                raise ValueError(f"不支持的学习率调度器类型: {scheduler_type}")

            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'val/loss',
                    'interval': 'epoch',
                    'frequency': 1
                }
            }

        return optimizer

    def get_predictions(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取所有预测结果

        Returns:
            (predictions, targets)
        """
        if self.test_predictions:
            predictions = torch.cat(self.test_predictions, dim=0)
            targets = torch.cat(self.test_targets, dim=0)
            return predictions, targets
        elif self.val_predictions:
            predictions = torch.cat(self.val_predictions, dim=0)
            targets = torch.cat(self.val_targets, dim=0)
            return predictions, targets
        else:
            return torch.tensor([]), torch.tensor([])


def create_classification_module(config: Dict[str, Any]) -> YOLOClassificationModule:
    """
    创建分类模块

    Args:
        config: 配置字典

    Returns:
        YOLOClassificationModule实例
    """
    # 提取配置
    model_config = config.get('model', {})
    optimizer_config = config.get('optimizer', {})
    scheduler_config = config.get('scheduler', {})

    # 创建模块
    module = YOLOClassificationModule(
        model_config=model_config,
        optimizer_config=optimizer_config,
        scheduler_config=scheduler_config,
        num_classes=model_config.get('nc', 10),
        pretrained=model_config.get('pretrained', True),
        freeze_backbone=model_config.get('freeze_backbone', False),
        label_smoothing=model_config.get('loss', {}).get('label_smoothing', 0.1),
        learning_rate=optimizer_config.get('lr', 0.001)
    )

    return module
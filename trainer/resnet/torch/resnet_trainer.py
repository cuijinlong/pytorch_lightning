import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import time
import os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from dataset.single_modal_data_loader import SingleModalDataLoader

class ResNetTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 初始化TensorBoard
        self.writer = SummaryWriter(log_dir=config.get('log_dir', 'runs/experiment'))

        # 初始化数据加载器
        self.data_loader = SingleModalDataLoader(config)
        self.train_loader, self.val_loader, self.test_loader = self.data_loader.create_data_loaders()

        # 初始化模型
        self.model = self._create_model()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.get('learning_rate', 1e-4),
            weight_decay=config.get('weight_decay', 1e-4)
        )

        # 学习率调度器
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=config.get('scheduler_step', 10),
            gamma=config.get('scheduler_gamma', 0.1)
        )

        # 训练记录
        self.train_losses, self.val_losses = [], []
        self.train_accuracies, self.val_accuracies = [], []

        # 记录模型图和一些初始信息到TensorBoard
        self._log_model_graph()

        print(f"使用设备: {self.device}")
        print(f"类别数量: {self.data_loader.get_num_classes()}")
        print(f"标签字典: {self.data_loader.get_label_dict()}")
        print(f"TensorBoard日志目录: {self.writer.log_dir}")

    def _create_model(self):
        """创建ResNet模型"""
        model_name = self.config.get('model_name', 'resnet18')
        num_classes = self.data_loader.get_num_classes()
        pretrained = self.config.get('pretrained', True)

        model = getattr(models, model_name)(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model.to(self.device)

    def _log_model_graph(self):
        """记录模型图到TensorBoard"""
        try:
            # 获取一个样本batch来记录模型图
            sample_batch = next(iter(self.train_loader))
            sample_images = sample_batch['image'].to(self.device)
            self.writer.add_graph(self.model, sample_images)
            print("模型图已记录到TensorBoard")
        except Exception as e:
            print(f"记录模型图时出错: {e}")

    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        running_loss, correct, total = 0.0, 0, 0
        pbar = tqdm(self.train_loader, desc=f"训练 Epoch {epoch}")
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)

            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            # 反向传播
            loss.backward()
            self.optimizer.step()

            # 统计
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # 记录batch级别的损失和准确率
            global_step = epoch * len(self.train_loader) + batch_idx
            self.writer.add_scalar('Batch/Train_Loss', loss.item(), global_step)
            batch_acc = 100. * (predicted == labels).sum().item() / labels.size(0)
            self.writer.add_scalar('Batch/Train_Accuracy', batch_acc, global_step)

            # 更新进度条
            current_loss = running_loss / (batch_idx + 1)
            current_acc = 100. * correct / total
            pbar.set_postfix({
                'Loss': f'{current_loss:.4f}',
                'Acc': f'{current_acc:.2f}%'
            })

        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total

        # 记录epoch级别的训练指标
        self.writer.add_scalar('Epoch/Train_Loss', epoch_loss, epoch)
        self.writer.add_scalar('Epoch/Train_Accuracy', epoch_acc, epoch)
        self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)

        return epoch_loss, epoch_acc

    def validate_epoch(self, epoch):
        """验证一个epoch"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        all_predictions = []
        all_labels = []

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'验证 Epoch {epoch}')
            for batch_idx, batch in enumerate(pbar):
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # 收集预测和标签用于后续分析
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                current_loss = running_loss / (batch_idx + 1)
                current_acc = 100. * correct / total
                pbar.set_postfix({
                    'Loss': f'{current_loss:.4f}',
                    'Acc': f'{current_acc:.2f}%'
                })

        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100. * correct / total

        # 记录epoch级别的验证指标
        self.writer.add_scalar('Epoch/Val_Loss', epoch_loss, epoch)
        self.writer.add_scalar('Epoch/Val_Accuracy', epoch_acc, epoch)

        return epoch_loss, epoch_acc

    def train(self, epochs):
        """完整的训练流程"""
        print("开始训练...")
        best_val_acc = 0.0
        checkpoint_dir = self.config.get('checkpoint_dir', 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)

        start_time = time.time()

        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print("-" * 50)

            # 训练
            train_loss, train_acc = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)

            # 验证
            if self.val_loader is not None:
                val_loss, val_acc = self.validate_epoch(epoch)
                self.val_losses.append(val_loss)
                self.val_accuracies.append(val_acc)

                # 学习率调度
                self.scheduler.step()

                # 保存最佳模型
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'val_acc': val_acc,
                        'val_loss': val_loss,
                        'label_dict': self.data_loader.get_label_dict(),
                        'config': self.config
                    }, best_model_path)
                    print(f"保存最佳模型，验证准确率: {val_acc:.2f}%")

            print(f"训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.2f}%")
            if self.val_loader is not None:
                print(f"验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.2f}%")

        # 保存最终模型
        final_model_path = os.path.join(checkpoint_dir, 'final_model.pth')
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'label_dict': self.data_loader.get_label_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies
        }, final_model_path)

        training_time = time.time() - start_time
        print(f"\n训练完成! 总耗时: {training_time // 60:.0f}分 {training_time % 60:.0f}秒")
        if self.val_loader is not None:
            print(f"最佳验证准确率: {best_val_acc:.2f}%")

        # 关闭TensorBoard writer
        self.writer.close()

    def test(self, model_path=None):
        """测试模型"""
        if model_path:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"加载模型: {model_path}")

        if self.test_loader is None:
            print("没有测试集!")
            return

        self.model.eval()
        correct = 0
        total = 0

        # 记录混淆矩阵所需的数据
        all_predictions = []
        all_labels = []

        print("\n开始测试...")
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.test_loader, desc='测试')):
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)

                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # 收集预测和标签
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        test_acc = 100. * correct / total
        print(f"测试准确率: {test_acc:.2f}%")

        # 记录测试准确率到TensorBoard
        self.writer.add_scalar('Test/Accuracy', test_acc)

        # 记录一些测试样本图像和预测结果
        self._log_test_samples()

        return test_acc

    def _log_test_samples(self):
        """记录一些测试样本到TensorBoard"""
        if self.test_loader is None:
            return

        # 获取一个测试batch
        test_batch = next(iter(self.test_loader))
        images = test_batch['image'][:8]  # 取前8个样本
        labels = test_batch['label'][:8]

        # 预测
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(images.to(self.device))
            _, predictions = torch.max(outputs, 1)

        # 反标准化图像（如果需要）
        # 注意：这里假设图像已经标准化，如果需要反标准化，请根据实际预处理操作调整

        # 记录图像到TensorBoard
        self.writer.add_images('Test/Sample_Images', images, 0)

        # 记录预测结果
        label_dict = self.data_loader.get_label_dict()
        for i in range(min(8, len(images))):
            true_label = label_dict.get(labels[i].item(), f"Class {labels[i].item()}")
            pred_label = label_dict.get(predictions[i].item(), f"Class {predictions[i].item()}")
            self.writer.add_text(f'Test/Sample_{i}',
                                 f'True: {true_label}, Pred: {pred_label}', 0)

    def close(self):
        """关闭TensorBoard writer"""
        self.writer.close()


# ======================================================================
#                             主程序
# ======================================================================
if __name__ == '__main__':
    base_dir = '/opt/datasets/pathmnist/output_dataset_basic'
    # 配置参数
    config = {
        'image_base_dir': f"{base_dir}",
        'train_csv': f"{base_dir}/train_metadata.csv",
        'val_csv': f"{base_dir}/val_metadata.csv",
        'test_csv': f"{base_dir}/test_metadata.csv",
        'image_col': "image_path",
        'label_col': "label",
        'batch_size': 32,
        'num_workers': 4,
        'image_size': (224, 224),  # ResNet的标准输入尺寸

        # 训练参数
        'model_name': 'resnet18',  # 可选: resnet18, resnet34, resnet50, resnet101
        'pretrained': True,  # 使用预训练权重
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        'scheduler_step': 10,
        'scheduler_gamma': 0.1,
        'checkpoint_dir': 'checkpoints',
        'log_dir': 'runs/resnet_experiment'  # TensorBoard日志目录
    }

    # 创建训练器
    trainer = ResNetTrainer(config)

    try:
        # 开始训练
        trainer.train(epochs=20)

        # 测试最佳模型
        trainer.test('checkpoints/best_model.pth')

    finally:
        # 确保关闭TensorBoard writer
        trainer.close()

    print(f"\n训练完成！使用以下命令查看TensorBoard:")
    print(f"tensorboard --logdir={config['log_dir']}")
    print("然后在浏览器中打开 http://localhost:6006")
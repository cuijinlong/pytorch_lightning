# trainer/lhgnn/models/tagging_module.py
from typing import Any, Dict, Tuple
import numpy as np
import torch
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification import Accuracy, Precision, Recall, F1Score
from trainer.lhgnn.models.utils.stats import calculate_stats
import torch.distributed as dist
from timm.scheduler import CosineLRScheduler, StepLRScheduler
from torchmetrics import AveragePrecision
import logging


class TaggingModule(LightningModule):

    def __init__(
            self,
            net: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler,
            compile: bool,
            loss: str,
            opt_warmup: bool,
            learning_rate: float,
            lr_rate: list,
            lr_scheduler_epoch: list,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.net = net
        self.optimizer = optimizer
        self.warmup = opt_warmup
        self.scheduler = scheduler
        self.compile = compile
        self.loss = loss
        self.lr_scheduler_epoch = lr_scheduler_epoch
        self.lr_rate = lr_rate

        # 损失函数
        if self.loss == 'bce':
            self.criterion = torch.nn.BCELoss()
        elif self.loss == 'cross_entropy':
            self.criterion = torch.nn.CrossEntropyLoss()
        elif self.loss == 'bcelogit':
            self.criterion = torch.nn.BCEWithLogitsLoss()

        # 损失指标
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # 最佳指标跟踪
        self.val_mAP_best = MaxMetric()

        # 预测和目标存储
        self.val_predictions = []
        self.val_targets = []
        self.test_predictions = []
        self.test_targets = []

        # 类别数和任务类型
        num_classes = net.num_classes

        # 根据损失函数确定任务类型
        if self.loss in ['bce', 'bcelogit']:
            task_type = 'multilabel'
            # 对于多标签分类，目标应该是0或1
            self._convert_targets = lambda x: x.long() if x.dtype != torch.long else x
        else:  # cross_entropy
            task_type = 'multiclass'
            # 对于多分类，目标应该是类别索引
            self._convert_targets = lambda x: x.long() if x.dtype != torch.long else x

        # mAP 指标
        self.ap = AveragePrecision(num_classes=num_classes, average='macro')
        self.ap_test = AveragePrecision(num_classes=num_classes, average='macro')

        # 训练指标
        self.train_accuracy = Accuracy(num_classes=num_classes, average='macro')
        self.train_precision = Precision(num_classes=num_classes, average='macro')
        self.train_recall = Recall(num_classes=num_classes, average='macro')
        self.train_f1 = F1Score(num_classes=num_classes, average='macro')

        # 验证指标
        self.val_accuracy = Accuracy(num_classes=num_classes, average='macro')
        self.val_precision = Precision(num_classes=num_classes, average='macro')
        self.val_recall = Recall(num_classes=num_classes, average='macro')
        self.val_f1 = F1Score(num_classes=num_classes, average='macro')

        # 测试指标
        self.test_accuracy = Accuracy(num_classes=num_classes, average='macro')
        self.test_precision = Precision(num_classes=num_classes, average='macro')
        self.test_recall = Recall(num_classes=num_classes, average='macro')
        self.test_f1 = F1Score(num_classes=num_classes, average='macro')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`."""
        return self.net(x)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        self.val_loss.reset()
        self.val_accuracy.reset()
        self.val_precision.reset()
        self.val_recall.reset()
        self.val_f1.reset()
        self.val_mAP_best.reset()

    def model_step(
            self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data."""
        x, y = batch
        preds = self.forward(x)
        loss = self.criterion(preds, y)
        return loss, preds, y

    def on_train_batch_start(self, batch, batch_idx):
        global_step = self.trainer.global_step
        optimizer = self.optimizers()
        if global_step <= 1000 and global_step % 50 == 0:
            warm_lr = (global_step / 1000) * self.hparams.optimizer.keywords['lr']
            for param_group in optimizer.param_groups:
                param_group['lr'] = warm_lr
            self.log('lr', warm_lr, on_step=True, on_epoch=False, logger=True)
        current_lr = next(iter(optimizer.param_groups))['lr']
        self.log('cur-lr', current_lr, on_step=False, on_epoch=True, logger=True)

    def training_step(
            self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set."""
        if batch_idx == 0:
            print(f'hyperparameters: {self.hparams}')

        loss, preds, y = self.model_step(batch)

        # 转换目标为整数类型
        targets_int = self._convert_targets(y)

        # 更新训练指标
        self.train_loss(loss)
        self.train_accuracy(preds, targets_int)
        self.train_precision(preds, targets_int)
        self.train_recall(preds, targets_int)
        self.train_f1(preds, targets_int)

        # 记录训练指标
        self.log("train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train/accuracy", self.train_accuracy, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train/precision", self.train_precision, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train/recall", self.train_recall, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train/f1", self.train_f1, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        return loss

    def on_train_epoch_end(self) -> None:
        """Lightning hook that is called when a training epoch ends."""
        pass

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set."""
        loss, preds, targets = self.model_step(batch)

        # 转换目标为整数类型
        targets_int = self._convert_targets(targets)

        # 更新验证指标
        self.val_loss(loss)
        self.val_accuracy(preds, targets_int)
        self.val_precision(preds, targets_int)
        self.val_recall(preds, targets_int)
        self.val_f1(preds, targets_int)
        self.ap.update(preds, targets_int)

        # 存储预测和目标（用于后续计算）
        self.val_predictions.append(preds)
        self.val_targets.append(targets)

        # 记录验证损失
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def on_validation_epoch_end(self) -> None:
        """Lightning hook that is called when a validation epoch ends."""
        # 记录所有验证指标
        self.log("val/accuracy", self.val_accuracy.compute(), on_step=False, on_epoch=True, prog_bar=True,
                 sync_dist=True)
        self.log("val/precision", self.val_precision.compute(), on_step=False, on_epoch=True, prog_bar=True,
                 sync_dist=True)
        self.log("val/recall", self.val_recall.compute(), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/f1", self.val_f1.compute(), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/mAP", self.ap.compute().mean(), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        # 更新最佳mAP
        current_mAP = self.ap.compute().mean()
        self.val_mAP_best(current_mAP)
        self.log("val/mAP_best", self.val_mAP_best.compute(), on_step=False, on_epoch=True, prog_bar=True,
                 sync_dist=True)

        # 重置指标
        self.val_accuracy.reset()
        self.val_precision.reset()
        self.val_recall.reset()
        self.val_f1.reset()
        self.ap.reset()

        # 清空列表
        self.val_predictions.clear()
        self.val_targets.clear()

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set."""
        loss, preds, targets = self.model_step(batch)

        # 转换目标为整数类型
        targets_int = self._convert_targets(targets)

        # 更新测试指标
        self.test_loss(loss)
        self.test_accuracy(preds, targets_int)
        self.test_precision(preds, targets_int)
        self.test_recall(preds, targets_int)
        self.test_f1(preds, targets_int)
        self.ap_test.update(preds, targets_int)

        # 存储预测和目标用于后续计算
        self.test_predictions.append(preds)
        self.test_targets.append(targets)

        # 记录测试损失
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        test_preds = torch.cat(self.test_predictions, dim=0)
        test_targets = torch.cat(self.test_targets, dim=0)

        # 转换目标为整数类型（用于自定义计算）
        test_targets_int = self._convert_targets(test_targets)

        # 分布式训练处理
        if torch.cuda.device_count() > 1 and dist.is_initialized():
            gather_pred = [torch.zeros_like(test_preds) for _ in range(dist.get_world_size())]
            gather_target = [torch.zeros_like(test_targets_int) for _ in range(dist.get_world_size())]
            dist.barrier()
            dist.all_gather(gather_pred, test_preds)
            dist.all_gather(gather_target, test_targets_int)

            if dist.get_rank() == 0:
                gather_pred = torch.cat(gather_pred, dim=0).cpu().detach().numpy()
                gather_target = torch.cat(gather_target, dim=0).cpu().detach().numpy()
                stats = calculate_stats(gather_pred, gather_target)
                mAP = np.mean([stat['AP'] for stat in stats])

                logging.info(f'logging on rank {dist.get_rank()}')
                logging.info(f'test_mAP: {mAP}')
                self.log("test/mAP_custom", mAP, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        else:
            # 单GPU情况
            stats = calculate_stats(test_preds.cpu().detach().numpy(), test_targets_int.cpu().detach().numpy())
            mAP = np.mean([stat['AP'] for stat in stats])
            self.log("test/mAP_custom", mAP, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        # 记录所有测试指标（使用torchmetrics计算的）
        self.log("test/accuracy", self.test_accuracy.compute(), on_step=False, on_epoch=True, prog_bar=True,
                 sync_dist=True)
        self.log("test/precision", self.test_precision.compute(), on_step=False, on_epoch=True, prog_bar=True,
                 sync_dist=True)
        self.log("test/recall", self.test_recall.compute(), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("test/f1", self.test_f1.compute(), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("test/mAP", self.ap_test.compute().mean(), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        # 重置指标
        self.test_accuracy.reset()
        self.test_precision.reset()
        self.test_recall.reset()
        self.test_f1.reset()
        self.ap_test.reset()

        # 清空列表
        self.test_predictions.clear()
        self.test_targets.clear()

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit, validate, test, or predict."""
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization."""
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
            
            
                 
                 
              

            #  if epoch < 3:
            #      # warm up lr
            #      lr_scale = self.lr_rate[epoch]
            #      print(f'warmup lr_scale:{lr_scale}')
            #  else:
            #      # warmup schedule
            #     lr_pos = int(-1 - bisect.bisect_left(self.lr_scheduler_epoch, epoch))
            #     if lr_pos < -3:
            #         lr_scale = max(self.lr_rate[0] * (0.98 ** epoch), 0.03)
            #         print(f'nonwarmup first lr_scale:{lr_scale}')
            #     else:
            #         lr_scale = self.lr_rate[lr_pos]
                    
            #  return lr_scale
            #scheduler = CosineLRScheduler(optimizer,t_initial=50, warmup_t=1, warmup_lr_init=1e-6,lr_min=1e-7)
            #scheduler = StepLRScheduler(optimizer, decay_t=5, warmup_t=2, warmup_lr_init=5e-5, decay_rate=0.5)
            #scheduler = torch.optim.lr_scheduler.LambdaLR( optimizer,  lr_lambda=lr_foo)
            
    
    
    
    

    

        
        # optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        

        # if self.hparams.scheduler is not None:
        #     scheduler = torch.optim.lr_scheduler.LambdaLR(
        #      optimizer,
        #      lr_lambda=lr_foo)
        
        #     #scheduler = self.hparams.scheduler(optimizer=optimizer)
        #     return {
        #         "optimizer": optimizer,
        #         "lr_scheduler": {
        #             "scheduler": scheduler,
        #             "monitor": "val/loss",
        #             "interval": "epoch",
        #             "frequency": 1,
        #         },
        #     }
        # return {"optimizer": optimizer}
        # if self.hparams.scheduler is not None:
            
        #     def lr_foo(epoch):
             
        #      if epoch < 1:
        #          # warm up lr
        #          lr_scale = self.lr_rate[epoch]
        #      else:
        #          # warmup schedule
        #         lr_pos = int(-1 - bisect.bisect_left(self.milestones, epoch))
        #         if lr_pos < -3:
        #             lr_scale = max(self.lr_rate[0] * (0.98 ** epoch), 0.03)
        #         else:
        #             lr_scale = self.lr_rate[lr_pos]
        #             lr_scale = 0.95 ** epoch
        #      return lr_scale
        #     scheduler = torch.optim.lr_scheduler.LambdaLR(
        #     optimizer,
        #     lr_lambda=lr_foo)
         
        #     return {
        #           "optimizer": optimizer,
        #           "lr_scheduler": {
        #               "scheduler": scheduler,
        #               "monitor": "val/loss",
        #               "interval": "epoch",
        #               "frequency": 1,
        #           },}
        # return {"optimizer": optimizer}
         
        #     scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-6, last_epoch=-1)
        #     return {
        #         "optimizer": optimizer,
        #         "lr_scheduler": {
        #             "scheduler": scheduler,
        #             "monitor": "val/loss",
        #             "interval": "epoch",
        #             "frequency": 1,
        #         },
        #     }
        # return {"optimizer": optimizer}
    
        # def lr_foo(epoch):
        #     if epoch < 1:
        #         # warm up lr
        #         lr_scale = self.lr_rate[epoch]
        #     else:
        #         # warmup schedule
        #         #lr_pos = int(-1 - bisect.bisect_left(self.milestones, epoch))
        #         #if lr_pos < -3:
        #         #    lr_scale = max(self.lr_rate[0] * (0.98 ** epoch), 0.03)
        #         #else:
        #         #    lr_scale = self.lr_rate[lr_pos]
        #         lr_scale = 0.95 ** epoch
        #     return lr_scale
        # scheduler = torch.optim.lr_scheduler.LambdaLR(
        #     optimizer,
        #     lr_lambda=lr_foo
        # )
        # return {
        #          "optimizer": optimizer,
        #          "lr_scheduler": {
        #              "scheduler": scheduler,
        #              "monitor": "val/loss",
        #              "interval": "epoch",
        #              "frequency": 1,
        #          },
        # }
        #  if self.hparams.scheduler is None:
        #     print("No scheduler")
        #  #print(optimizer)
        #  if self.hparams.scheduler is not None:
             
             #scheduler = self.hparams.scheduler(optimizer=optimizer)
             
            # }
         #return {"optimizer": optimizer}
         #return torch.optim.Adam(self.net.parameters(), lr=5e-4)
    

    # def optimizer_step(self,
    #                     epoch,
    #                     batch_idx,
    #                     optimizer,
    #                     optimizer_closure,
    #   ):
        
        
    #      if self.trainer.global_step <= 1000 and self.trainer.global_step % 50 == 0 and self.warmup == True:
            
    #          warm_lr = (self.trainer.global_step / 1000) * optimizer.param_groups[0]['lr']
    #          for pg in optimizer.param_groups:
    #              pg['lr'] = warm_lr
        
    #      optimizer.step(closure=optimizer_closure)
    
            


        

        
   
    



    





    







    

    

    

    

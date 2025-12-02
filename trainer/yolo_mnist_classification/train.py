# trainer/yolo_mnist_classification/train.py
#!/usr/bin/env python3
"""
YOLO MNIST分类训练脚本
"""

import os
import sys
import torch
import yaml
import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import rootutils
from models.datamodules.mnist_datamodule import MNISTDataModule, prepare_mnist_dataset
from models.classification_module import create_classification_module
from models.utils.rich_utils import print_config_tree, enforce_tags
from models.utils.logging_utils import log_hyperparameters
from models.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True,cwd=False)
os.environ['HYDRA_FULL_ERROR'] = '1'

@hydra.main(version_base="1.3", config_path="./configs", config_name="train")
def train(cfg: DictConfig):
    """
    训练函数

    Args:
        cfg: Hydra配置对象
    """
    # 应用额外配置
    apply_extras(cfg)

    # 打印配置树
    if cfg.get("print_config", True):
        print_config_tree(cfg, resolve=True)

    # 设置随机种子
    pl.seed_everything(cfg.seed, workers=True)

    # 创建数据模块
    log.info("创建数据模块...")
    datamodule = MNISTDataModule(cfg)

    # 创建模型模块
    log.info("创建模型模块...")
    model_module = create_classification_module(cfg)

    # 创建回调函数
    callbacks = create_callbacks(cfg)

    # 创建日志记录器
    loggers = create_loggers(cfg)

    # 创建训练器
    log.info("创建训练器...")
    trainer = pl.Trainer(
        max_epochs=cfg.epochs,
        accelerator=cfg.device if cfg.device != "cpu" else "auto",
        devices=1 if cfg.device == "cpu" else "auto",
        logger=loggers,
        callbacks=callbacks,
        deterministic=True,
        enable_progress_bar=True,
        log_every_n_steps=10,
        check_val_every_n_epoch=1,
        num_sanity_val_steps=0
    )

    # 记录超参数
    object_dict = {
        "cfg": cfg,
        "model": model_module,
        "trainer": trainer,
    }
    if loggers:
        log.info("记录超参数...")
        log_hyperparameters(object_dict)

    # 训练模型
    if cfg.train:
        log.info("开始训练...")
        trainer.fit(model_module, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    # 评估模型
    if cfg.eval:
        log.info("开始评估...")
        trainer.test(model_module, datamodule=datamodule, ckpt_path="best")

    # 保存最终模型
    if cfg.train and trainer.is_global_zero:
        save_final_model(model_module, cfg)


def apply_extras(cfg: DictConfig):
    """
    应用额外配置

    Args:
        cfg: 配置对象
    """
    # 确保输出目录存在
    os.makedirs(cfg.paths.output_dir, exist_ok=True)

    # 设置标签
    if cfg.get("tags"):
        log.info(f"实验标签: {cfg.tags}")

    # 设置设备
    if cfg.device == "cuda" and not torch.cuda.is_available():
        log.warning("CUDA不可用，使用CPU")
        cfg.device = "cpu"


def create_callbacks(cfg: DictConfig):
    """
    创建回调函数

    Args:
        cfg: 配置对象

    Returns:
        回调函数列表
    """
    callbacks = []

    # 模型检查点回调
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(cfg.paths.output_dir, "checkpoints"),
        filename="epoch_{epoch:03d}",
        monitor="val/accuracy",
        mode="max",
        save_top_k=3,
        save_last=True,
        verbose=True
    )
    callbacks.append(checkpoint_callback)

    # 早停回调
    early_stop_callback = EarlyStopping(
        monitor="val/loss",
        patience=10,
        mode="min",
        verbose=True
    )
    callbacks.append(early_stop_callback)

    # 学习率监控回调
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    callbacks.append(lr_monitor)

    return callbacks


def create_loggers(cfg: DictConfig):
    """
    创建日志记录器

    Args:
        cfg: 配置对象

    Returns:
        日志记录器列表
    """
    loggers = []

    # TensorBoard日志记录器
    if cfg.logger == "tensorboard" or cfg.logger == "both":
        tb_logger = TensorBoardLogger(
            save_dir=cfg.paths.log_dir,
            name=cfg.task_name,
            version="",
            default_hp_metric=False
        )
        loggers.append(tb_logger)

    # CSV日志记录器
    if cfg.logger == "csv" or cfg.logger == "both":
        csv_logger = CSVLogger(
            save_dir=cfg.paths.log_dir,
            name=cfg.task_name,
            version=""
        )
        loggers.append(csv_logger)

    return loggers


def save_final_model(model_module, cfg: DictConfig):
    """
    保存最终模型

    Args:
        model_module: 模型模块
        cfg: 配置对象
    """
    # 保存PyTorch Lightning检查点
    checkpoint_path = os.path.join(cfg.paths.output_dir, "final_model.ckpt")
    torch.save(model_module.state_dict(), checkpoint_path)
    log.info(f"保存模型检查点到: {checkpoint_path}")

    # 保存YOLO格式模型
    yolo_model_path = os.path.join(cfg.paths.output_dir, "best.pt")
    try:
        model_module.model.save(yolo_model_path)
        log.info(f"保存YOLO模型到: {yolo_model_path}")
    except Exception as e:
        log.warning(f"保存YOLO模型失败: {e}")


if __name__ == "__main__":
    train()
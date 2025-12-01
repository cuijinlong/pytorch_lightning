# test.py
from typing import Any, Dict, Optional, Tuple
import hydra
import lightning as L
import torch
import logging
from pytorch_lightning import LightningDataModule, LightningModule, Trainer, Callback
from pytorch_lightning.loggers import Logger
from omegaconf import DictConfig
import rootutils
import os
import numpy as np

# 设置项目根目录
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=False)
os.environ['HYDRA_FULL_ERROR'] = '1'

from trainer.lhgnn.models.utils import (
    RankedLogger,
    extras,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)


def load_pretrained_weights(model: LightningModule, pretrain_path: str) -> None:
    """加载预训练权重"""
    log.info(f"Loading pretrained weights from: {pretrain_path}")

    if not os.path.exists(pretrain_path):
        log.error(f"Pretrain path does not exist: {pretrain_path}")
        return

    try:
        checkpoint = torch.load(pretrain_path, map_location="cpu")

        # 处理不同的检查点格式
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        # 移除可能的模块前缀
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("model.") or k.startswith("module."):
                new_state_dict[k.replace("model.", "").replace("module.", "")] = v
            else:
                new_state_dict[k] = v

        # 加载权重，允许部分匹配
        missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)

        if missing_keys:
            log.warning(f"Missing keys in pretrained weights: {missing_keys}")
        if unexpected_keys:
            log.warning(f"Unexpected keys in pretrained weights: {unexpected_keys}")

        log.info("Pretrained weights loaded successfully")

    except Exception as e:
        log.error(f"Failed to load pretrained weights: {e}")


def create_weighted_average_model(cfg: DictConfig, model: LightningModule, ckpt_dir: str) -> LightningModule:
    """创建加权平均模型"""
    log.info("Creating weighted average model")
    model_ckpt = []

    # 收集所有检查点
    for ckpt_file in os.listdir(ckpt_dir):
        if ckpt_file.endswith('.ckpt') and ckpt_file != 'wa.pth.tar':
            ckpt_path = os.path.join(ckpt_dir, ckpt_file)
            try:
                checkpoint = torch.load(ckpt_path, map_location="cpu")
                if "state_dict" in checkpoint:
                    model_ckpt.append(checkpoint["state_dict"])
                    log.info(f"Loaded checkpoint: {ckpt_file}")
            except Exception as e:
                log.warning(f"Failed to load checkpoint {ckpt_file}: {e}")

    if not model_ckpt:
        log.error("No valid checkpoints found for weighted average")
        return model

    # 重新实例化模型用于加权平均
    model_wa: LightningModule = hydra.utils.instantiate(cfg.model)
    own_state = model_wa.state_dict()

    # 计算加权平均
    log.info(f"Averaging {len(model_ckpt)} checkpoints")
    for name, params in own_state.items():
        if name in model_ckpt[0]:
            own_state[name] = torch.zeros_like(params)
            model_ckpt_key = torch.stack([d[name].float() for d in model_ckpt], dim=0)
            own_state[name].copy_(torch.mean(model_ckpt_key, dim=0))

    model_wa.load_state_dict(own_state)
    log.info("Weighted average model created successfully")

    return model_wa


@task_wrapper
def test(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """专门用于模型测试的函数

    :param cfg: Hydra 配置对象
    :return: 测试指标字典和对象字典
    """
    # 设置随机种子
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    # 实例化数据模块
    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    # 实例化模型
    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    # 加载预训练权重（如果提供）
    if cfg.get("pretrain_path"):
        load_pretrained_weights(model, cfg.pretrain_path)

    # 实例化回调函数
    log.info("Instantiating callbacks...")
    callbacks = instantiate_callbacks(cfg.get("callbacks"))

    # 实例化日志记录器
    log.info("Instantiating loggers...")
    logger = instantiate_loggers(cfg.get("logger"))

    # 创建训练器实例 - 专门用于测试
    log.info(f"Instantiating trainer for testing...")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=logger
    )

    # 记录所有实例化对象
    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "logger": logger,
        "callbacks": callbacks,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    # 模型测试逻辑
    test_results = None

    # 方式1：使用加权平均模型进行测试
    if cfg.get("wa"):
        log.info("Testing with weighted average model")

        # 获取检查点目录
        ckpt_dir = cfg.callbacks.model_checkpoint.dirpath
        if not os.path.exists(ckpt_dir):
            log.error(f"Checkpoint directory does not exist: {ckpt_dir}")
            return {}, object_dict

        model_wa = create_weighted_average_model(cfg, model, ckpt_dir)
        test_results = trainer.test(model=model_wa, datamodule=datamodule)
        log.info(f"Test results from weighted average model: {test_results}")

    # 方式2：使用指定的检查点进行测试
    elif cfg.get("ckpt_path"):
        ckpt_path = cfg.ckpt_path
        log.info(f"Testing with specified checkpoint: {ckpt_path}")

        if os.path.exists(ckpt_path):
            test_results = trainer.test(
                model=model,
                datamodule=datamodule,
                ckpt_path=ckpt_path
            )
            log.info(f"Test results with specified checkpoint: {test_results}")
        else:
            log.error(f"Checkpoint path does not exist: {ckpt_path}")
            return {}, object_dict

    # 方式3：使用预训练权重进行测试
    elif cfg.get("pretrain_path"):
        log.info("Testing with pretrained weights")
        test_results = trainer.test(model=model, datamodule=datamodule)
        log.info(f"Test results with pretrained weights: {test_results}")

    # 方式4：使用当前模型权重进行测试
    else:
        log.info("Testing with current model weights")
        test_results = trainer.test(model=model, datamodule=datamodule)
        log.info(f"Test results with current weights: {test_results}")

    log.info("Testing completed!")

    # 返回测试结果和对象字典
    metric_dict = test_results[0] if test_results and len(test_results) > 0 else {}
    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="./configs", config_name="test.yaml")
def main(cfg: DictConfig) -> None:
    """测试主入口点

    :param cfg: Hydra 配置对象
    """
    # 应用额外工具
    extras(cfg)

    log.info("Starting model testing process")

    # 执行测试
    test_metrics, object_dict = test(cfg)

    # 打印测试结果
    if test_metrics:
        log.info("=== Final Test Results ===")
        for metric_name, metric_value in test_metrics.items():
            if isinstance(metric_value, (int, float, np.number)):
                log.info(f"{metric_name}: {metric_value:.4f}")
            else:
                log.info(f"{metric_name}: {metric_value}")

        # 特别关注常见指标
        for key in ['mAP', 'map', 'accuracy', 'Accuracy', 'loss', 'Loss']:
            if key in test_metrics:
                log.info(f"🎯 Test {key}: {test_metrics[key]:.4f}")
                break
    else:
        log.warning("No test results obtained")


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
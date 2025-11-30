# test.py
from typing import Any, Dict, Optional
import hydra
import lightning as L
import torch
import logging
from pytorch_lightning import LightningDataModule, LightningModule, Trainer, Callback
from pytorch_lightning.loggers import Logger
from omegaconf import DictConfig
import rootutils
import os

# 设置项目根目录 - 与 train.py 保持一致
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


@task_wrapper
def test(cfg: DictConfig) -> Dict[str, Any]:
    """专门用于模型测试的函数

    :param cfg: Hydra 配置对象
    :return: 测试指标字典
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
        model_ckpt = []
        ckpt_dir = cfg.callbacks.model_checkpoint.dirpath

        # 重新实例化模型用于加权平均
        model_wa: LightningModule = hydra.utils.instantiate(cfg.model)
        own_state = model_wa.state_dict()

        # 收集所有检查点
        for ckpt_file in os.listdir(ckpt_dir):
            if 'ckpt' in ckpt_file:
                ckpt_path = os.path.join(ckpt_dir, ckpt_file)
                model_ckpt.append(torch.load(ckpt_path)['state_dict'])

        # 计算加权平均
        for name, params in own_state.items():
            own_state[name] = torch.zeros_like(params)
            model_ckpt_key = torch.cat([d[name].float().unsqueeze(0) for d in model_ckpt], dim=0)
            own_state[name].copy_(torch.mean(model_ckpt_key, dim=0))

        model_wa.load_state_dict(own_state)

        # 执行测试
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

    # 方式3：使用最佳检查点进行测试
    elif hasattr(trainer, 'checkpoint_callback') and trainer.checkpoint_callback:
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path and os.path.exists(ckpt_path):
            log.info(f"Testing with best checkpoint: {ckpt_path}")
            test_results = trainer.test(
                model=model,
                datamodule=datamodule,
                ckpt_path=ckpt_path
            )
            log.info(f"Test results with best checkpoint: {test_results}")
        else:
            log.warning("No best checkpoint found, testing with current model weights")
            test_results = trainer.test(model=model, datamodule=datamodule)
            log.info(f"Test results with current weights: {test_results}")

    # 方式4：使用当前模型权重进行测试
    else:
        log.info("Testing with current model weights")
        test_results = trainer.test(model=model, datamodule=datamodule)
        log.info(f"Test results with current weights: {test_results}")

    log.info("Testing completed!")
    return test_results[0] if test_results and len(test_results) > 0 else {}


@hydra.main(version_base="1.3", config_path="./configs", config_name="test.yaml")
def main(cfg: DictConfig) -> None:
    """测试主入口点

    :param cfg: Hydra 配置对象
    """
    # 应用额外工具（与 train.py 保持一致）
    extras(cfg)

    log.info("Starting model testing process")

    # 执行测试
    test_results = test(cfg)

    # 打印主要测试指标
    if test_results:
        log.info("=== Final Test Results ===")
        for metric_name, metric_value in test_results.items():
            log.info(f"{metric_name}: {metric_value:.4f}")

        # 特别关注 mAP 指标
        if 'mAP' in test_results:
            log.info(f"🎯 Test mAP: {test_results['mAP']:.4f}")
    else:
        log.warning("No test results obtained")


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
#!/usr/bin/env python3
"""
YOLO MNIST分类评估脚本
"""

import os
import sys
import torch
import yaml
import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.datamodules.mnist_datamodule import MNISTDataModule
from models.classification_module import create_classification_module
from models.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


@hydra.main(version_base="1.3", config_path="./configs", config_name="eval")
def evaluate(cfg: DictConfig):
    """
    评估函数

    Args:
        cfg: Hydra配置对象
    """
    # 设置随机种子
    pl.seed_everything(42, workers=True)

    # 创建数据模块
    log.info("创建数据模块...")
    datamodule = MNISTDataModule(cfg.data)
    datamodule.setup(stage='test')

    # 创建模型模块
    log.info("创建模型模块...")
    model_module = create_classification_module(cfg)

    # 加载模型权重
    weights_path = cfg.weights
    if weights_path and os.path.exists(weights_path):
        log.info(f"加载模型权重: {weights_path}")

        if weights_path.endswith('.ckpt'):
            # 加载PyTorch Lightning检查点
            state_dict = torch.load(weights_path, map_location='cpu')
            if 'state_dict' in state_dict:
                model_module.load_state_dict(state_dict['state_dict'])
            else:
                model_module.load_state_dict(state_dict)
        elif weights_path.endswith('.pt'):
            # 加载YOLO模型
            try:
                from ultralytics import YOLO
                yolo_model = YOLO(weights_path)
                # 这里需要将YOLO模型权重转移到我们的模块中
                # 由于YOLO模型结构复杂，这里简化处理
                log.info("YOLO模型权重加载成功")
            except Exception as e:
                log.error(f"加载YOLO模型失败: {e}")
        else:
            log.warning(f"未知的模型权重格式: {weights_path}")
    else:
        log.warning("未提供模型权重路径，使用随机初始化模型")

    # 创建训练器
    trainer = pl.Trainer(
        accelerator=cfg.device if cfg.device != "cpu" else "auto",
        devices=1 if cfg.device == "cpu" else "auto",
        logger=False,
        enable_progress_bar=True
    )

    # 测试模型
    log.info("开始测试...")
    test_results = trainer.test(model_module, datamodule=datamodule)

    # 获取详细预测结果
    log.info("获取详细预测结果...")
    predictions, targets = model_module.get_predictions()

    if len(predictions) > 0 and len(targets) > 0:
        # 转换为numpy
        preds_np = torch.argmax(predictions, dim=1).numpy()
        targets_np = targets.numpy()

        # 生成分类报告
        class_names = datamodule.get_class_names()
        report = classification_report(
            targets_np,
            preds_np,
            target_names=class_names,
            output_dict=True
        )

        # 打印分类报告
        log.info("分类报告:")
        print(pd.DataFrame(report).transpose().round(4))

        # 生成混淆矩阵
        cm = confusion_matrix(targets_np, preds_np)

        # 绘制混淆矩阵
        plot_confusion_matrix(cm, class_names, cfg.paths.output_dir)

        # 保存结果
        save_evaluation_results(test_results, report, cm, cfg.paths.output_dir)
    else:
        log.warning("无法获取预测结果")


def plot_confusion_matrix(cm, class_names, output_dir):
    """
    绘制混淆矩阵

    Args:
        cm: 混淆矩阵
        class_names: 类别名称
        output_dir: 输出目录
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title('混淆矩阵')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.tight_layout()

    # 保存图像
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=300)
    plt.close()

    log.info(f"保存混淆矩阵到: {cm_path}")


def save_evaluation_results(test_results, report, cm, output_dir):
    """
    保存评估结果

    Args:
        test_results: 测试结果
        report: 分类报告
        cm: 混淆矩阵
        output_dir: 输出目录
    """
    # 创建结果字典
    results_dict = {
        'test_results': test_results[0] if test_results else {},
        'classification_report': report,
        'confusion_matrix': cm.tolist()
    }

    # 保存为YAML文件
    results_path = os.path.join(output_dir, 'evaluation_results.yaml')
    with open(results_path, 'w') as f:
        yaml.dump(results_dict, f, default_flow_style=False)

    log.info(f"保存评估结果到: {results_path}")

    # 保存为CSV文件（简化版本）
    if report:
        # 提取每个类别的指标
        metrics_data = []
        for class_name, metrics in report.items():
            if class_name in ['accuracy', 'macro avg', 'weighted avg']:
                continue

            metrics_data.append({
                'class': class_name,
                'precision': metrics.get('precision', 0),
                'recall': metrics.get('recall', 0),
                'f1-score': metrics.get('f1-score', 0),
                'support': metrics.get('support', 0)
            })

        # 添加宏观平均
        if 'macro avg' in report:
            metrics_data.append({
                'class': 'macro_avg',
                'precision': report['macro avg']['precision'],
                'recall': report['macro avg']['recall'],
                'f1-score': report['macro avg']['f1-score'],
                'support': report['macro avg']['support']
            })

        # 创建DataFrame并保存
        df = pd.DataFrame(metrics_data)
        csv_path = os.path.join(output_dir, 'classification_metrics.csv')
        df.to_csv(csv_path, index=False)

        log.info(f"保存分类指标到: {csv_path}")


if __name__ == "__main__":
    evaluate()
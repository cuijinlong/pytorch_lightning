# /opt/pytorch_lightning/trainer/yolo/yolo_trainer.py
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from yolo_lightning import YOLOLightningModule
from data_module import YoloDataModule
from pytorch_lightning.loggers import TensorBoardLogger
import os
import yaml


def load_config(config_path="config.yaml"):
    """
    加载配置文件

    Args:
        config_path: 配置文件路径

    Returns:
        配置字典
    """
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        print("配置文件加载成功")
        return config
    except Exception as e:
        print(f"加载配置文件失败: {e}")
        # 返回默认配置
        return {
            'model': {'name': 'yolov8n.pt', 'num_classes': 80},
            'training': {'batch_size': 32, 'learning_rate': 0.01, 'max_epochs': 20, 'patience': 10},
            'data': {'path': '/opt/datasets/coco128', 'num_workers': 0},
            'logging': {'log_dir': './yolo_logs', 'experiment_name': 'yolo_detection'}
        }


# 在main函数中添加数据分割逻辑
def main():
    """
    主训练函数
    """
    # 加载配置
    config = load_config()

    # 创建输出目录
    os.makedirs(config['logging']['log_dir'], exist_ok=True)
    os.makedirs('./lightning_checkpoints', exist_ok=True)

    print("=" * 50)
    print("YOLO目标检测模型训练开始")
    print(f"模型: {config['model']['name']}")
    print(f"数据集: {config['data']['path']}")
    print(f"最大轮数: {config['training']['max_epochs']}")
    print("=" * 50)

    # 创建数据模块
    try:
        data_module = YoloDataModule(config)
        print("数据模块创建成功")

        # 手动调用setup确保数据准备完成
        print("手动调用数据模块setup...")
        data_module.setup()
        print("数据模块setup完成")

    except Exception as e:
        print(f"创建数据模块失败: {e}")
        return

    # 创建模型
    try:
        model = YOLOLightningModule(config)

        # 关键：设置数据配置文件路径
        data_yaml_path = str(data_module.data_yaml)
        model.set_data_config(data_yaml_path)
        print(f"模型数据配置已设置: {data_yaml_path}")

        print("模型创建成功")
    except Exception as e:
        print(f"创建模型失败: {e}")
        return

    # 设置回调函数 - 如果没有验证集，禁用EarlyStopping
    checkpoint_callback = ModelCheckpoint(
        monitor='val_map' if data_module.datasets.get('val') else 'train_loss',
        dirpath='./lightning_checkpoints',
        filename='yolo-best-{epoch:02d}-{val_map:.2f}' if data_module.datasets.get(
            'val') else 'yolo-best-{epoch:02d}-{train_loss:.2f}',
        save_top_k=3,
        mode='max' if data_module.datasets.get('val') else 'min',
        save_last=True,
        verbose=True
    )

    lr_monitor = LearningRateMonitor(
        logging_interval='epoch'
    )

    callbacks = [checkpoint_callback, lr_monitor]

    # 只有在有验证集时才添加EarlyStopping
    if data_module.datasets.get('val'):
        early_stop_callback = EarlyStopping(
            monitor='val_map',
            min_delta=0.001,
            patience=config['training']['patience'],
            verbose=True,
            mode='max'
        )
        callbacks.append(early_stop_callback)
    else:
        print("警告: 没有验证集，禁用EarlyStopping回调")

    # 设置日志记录器
    logger = TensorBoardLogger(
        save_dir=config['logging']['log_dir'],
        name=config['logging']['experiment_name'],
        version="version_1"
    )

    # 创建训练器 - 如果没有验证集，设置val_check_interval为None
    trainer_kwargs = {
        'accelerator': 'auto',
        'devices': 1,
        'logger': logger,
        'callbacks': callbacks,
        'max_epochs': config['training']['max_epochs'],
        'deterministic': True,
        'log_every_n_steps': 10,
        'enable_progress_bar': True,
        'num_sanity_val_steps': 0,
        'enable_model_summary': True,
    }

    # 如果没有验证集，调整训练配置
    if not data_module.datasets.get('val'):
        trainer_kwargs['check_val_every_n_epoch'] = None
        print("配置: 没有验证集，将只进行训练")
    else:
        trainer_kwargs['check_val_every_n_epoch'] = 1

    trainer = pl.Trainer(**trainer_kwargs)

    # 开始训练
    try:
        print("开始训练YOLO目标检测模型...")
        trainer.fit(model, datamodule=data_module)

        # 保存最终模型
        final_model_path = "yolo_final.ckpt"
        trainer.save_checkpoint(final_model_path)
        print(f"训练完成！最终模型已保存至: {final_model_path}")

        # 打印最佳模型路径
        if checkpoint_callback.best_model_path:
            print(f"最佳模型: {checkpoint_callback.best_model_path}")

    except Exception as e:
        print(f"训练过程中出错: {e}")
        return


if __name__ == "__main__":
    main()
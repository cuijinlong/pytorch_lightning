# /opt/pytorch_lightning/trainer/train_lightning.py
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from resnet_lightning import ResNetLightning
from data_module import ResNetDataModule


def main(resume_from_checkpoint=None):
    # 配置参数
    base_dir = '/opt/datasets/pathmnist/output_dataset_basic'
    config = {
        'image_base_dir': f"{base_dir}",
        'train_csv': f"{base_dir}/train_metadata.csv",
        'val_csv': f"{base_dir}/val_metadata.csv",
        'test_csv': f"{base_dir}/test_metadata.csv",
        'image_col': "image_path",
        'label_col': "label",
        'batch_size': 64,
        'num_workers': 0,
        'image_size': (224, 224),
        'max_epochs': 20,

        # 训练参数
        'model_name': 'resnet18',
        'pretrained': True,
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        'scheduler_step': 10,
        'scheduler_gamma': 0.1,
    }

    # 初始化数据模块
    data_module = ResNetDataModule(config)

    # 如果有检查点，从检查点加载模型
    if resume_from_checkpoint:
        print(f"从检查点恢复训练: {resume_from_checkpoint}")
        model = ResNetLightning.load_from_checkpoint(
            resume_from_checkpoint,
            config=config,
            num_classes=data_module.num_classes,
            label_dict=data_module.label_dict
        )
    else:
        print("开始新的训练")
        model = ResNetLightning(
            config=config,
            num_classes=data_module.num_classes,
            label_dict=data_module.label_dict
        )

    # 回调函数
    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc',
        dirpath='lightning_checkpoints',
        filename='resnet-{epoch:02d}-{val_acc:.2f}',
        save_top_k=3,
        mode='max',
        save_last=True
    )

    early_stop_callback = EarlyStopping(
        monitor='val_acc',
        min_delta=0.00,
        patience=10,
        verbose=False,
        mode='max'
    )

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # 日志记录器
    logger = TensorBoardLogger(
        save_dir='lightning_logs',
        name='resnet_experiment'
    )

    # 训练器
    trainer = pl.Trainer(
        max_epochs=config['max_epochs'],
        accelerator='auto',
        devices=1,
        logger=logger,
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
        deterministic=True,
        log_every_n_steps=10,
        check_val_every_n_epoch=1,
        enable_progress_bar=True,
        num_sanity_val_steps=0,
        enable_model_summary=False,
    )

    # 训练模型
    print("开始训练...")
    trainer.fit(model, datamodule=data_module, ckpt_path=resume_from_checkpoint)

    # 测试最佳模型
    print("开始测试...")
    best_model_path = checkpoint_callback.best_model_path
    if best_model_path:
        model = ResNetLightning.load_from_checkpoint(
            best_model_path,
            config=config,
            num_classes=data_module.num_classes,
            label_dict=data_module.label_dict
        )
        trainer.test(model, datamodule=data_module)
    else:
        trainer.test(model, datamodule=data_module)

    print(f"\n训练完成！使用以下命令查看TensorBoard:")
    print(f"tensorboard --logdir=lightning_logs/")


if __name__ == '__main__':
    # 使用方法1：从头开始训练
    # main()

    # 使用方法2：从特定检查点继续训练
    main(resume_from_checkpoint='lightning_checkpoints/resnet-epoch=00-val_acc=0.99.ckpt')

    # 使用方法3：从最后一个检查点继续训练
    # main(resume_from_checkpoint='lightning_checkpoints/last.ckpt')
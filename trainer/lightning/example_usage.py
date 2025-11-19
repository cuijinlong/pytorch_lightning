# example_usage.py
from train_lightning import main as train_main
from predict_lightning import LightningPredictor

# 训练模型
# train_main()

# 使用训练好的模型进行预测
checkpoint_path = "lightning_checkpoints/resnet-epoch=15-val_acc=0.95.ckpt"
predictor = LightningPredictor(checkpoint_path)

# 单张图像预测
result = predictor.predict_image("test_image.jpg")
print(result)

# 批量预测
results = predictor.predict_batch("test_images/")
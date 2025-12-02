# Ultralytics YOLO 11 MNIST 图像分类

使用 Ultralytics YOLO 11 进行 MNIST 手写数字分类的完整示例。

## 项目结构
yolo_mnist_classification/
├── configs/ # 配置文件
├── data/ # 数据目录
├── models/ # 模型定义
├── train.py # 训练脚本
├── evaluate.py # 评估脚本
├── predict.py # 预测脚本
└── README.md # 说明文档



## 安装依赖

```bash
# 安装依赖
pip install torch torchvision torchaudio
pip install ultralytics
pip install pytorch-lightning
pip install hydra-core
pip install omegaconf
pip install rich
pip install seaborn matplotlib
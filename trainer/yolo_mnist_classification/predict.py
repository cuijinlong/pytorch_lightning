#!/usr/bin/env python3
"""
YOLO MNIST分类预测脚本
"""

import os
import sys
import torch
import yaml
import hydra
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.classification_module import create_classification_module
from models.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


@hydra.main(version_base="1.3", config_path="./configs", config_name="predict")
def predict(cfg: DictConfig):
    """
    预测函数

    Args:
        cfg: Hydra配置对象
    """
    # 设置设备
    device = torch.device(cfg.device if torch.cuda.is_available() and cfg.device != "cpu" else "cpu")
    log.info(f"使用设备: {device}")

    # 创建模型模块
    log.info("创建模型模块...")
    model_module = create_classification_module(cfg).to(device)
    model_module.eval()

    # 加载模型权重
    weights_path = cfg.weights
    if weights_path and os.path.exists(weights_path):
        log.info(f"加载模型权重: {weights_path}")

        if weights_path.endswith('.ckpt'):
            # 加载PyTorch Lightning检查点
            state_dict = torch.load(weights_path, map_location=device)
            if 'state_dict' in state_dict:
                model_module.load_state_dict(state_dict['state_dict'])
            else:
                model_module.load_state_dict(state_dict)
        else:
            log.warning(f"仅支持.ckpt格式的权重文件: {weights_path}")
    else:
        log.error("未找到模型权重文件")
        return

    # 创建数据变换
    transform = create_transform(cfg)

    # 获取预测源
    source = cfg.source
    if not source:
        log.error("未指定预测源")
        return

    # 执行预测
    if os.path.isdir(source):
        predict_directory(source, model_module, transform, device, cfg)
    elif os.path.isfile(source):
        predict_single_image(source, model_module, transform, device, cfg)
    else:
        log.error(f"预测源不存在: {source}")


def create_transform(cfg: DictConfig):
    """
    创建数据变换

    Args:
        cfg: 配置对象

    Returns:
        数据变换
    """
    img_size = cfg.data.get('img_size', 224)

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST的均值和标准差
    ])

    return transform


def predict_single_image(image_path, model_module, transform, device, cfg):
    """
    预测单张图像

    Args:
        image_path: 图像路径
        model_module: 模型模块
        transform: 数据变换
        device: 设备
        cfg: 配置对象
    """
    try:
        # 加载图像
        image = Image.open(image_path).convert('L')  # 转换为灰度图

        # 应用变换
        image_tensor = transform(image)

        # 添加批次维度并复制为3通道
        image_tensor = image_tensor.unsqueeze(0)  # [1, 1, H, W]
        image_tensor = image_tensor.repeat(1, 3, 1, 1)  # [1, 3, H, W]

        # 移动到设备
        image_tensor = image_tensor.to(device)

        # 预测
        with torch.no_grad():
            logits = model_module(image_tensor)
            probs = torch.softmax(logits, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0, pred_class].item()

        # 获取类别名称
        class_names = [str(i) for i in range(10)]  # MNIST类别
        pred_label = class_names[pred_class]

        # 打印结果
        log.info(f"图像: {image_path}")
        log.info(f"预测类别: {pred_label} (置信度: {confidence:.4f})")

        # 显示前3个预测结果
        top_k = 3
        top_probs, top_indices = torch.topk(probs[0], k=top_k)

        log.info(f"Top-{top_k} 预测:")
        for i in range(top_k):
            idx = top_indices[i].item()
            prob = top_probs[i].item()
            log.info(f"  {class_names[idx]}: {prob:.4f}")

        # 保存结果
        if cfg.save_txt:
            save_prediction_result(image_path, pred_label, confidence, cfg)

        return pred_label, confidence

    except Exception as e:
        log.error(f"预测图像失败 {image_path}: {e}")
        return None, 0.0


def predict_directory(directory_path, model_module, transform, device, cfg):
    """
    预测目录中的所有图像

    Args:
        directory_path: 目录路径
        model_module: 模型模块
        transform: 数据变换
        device: 设备
        cfg: 配置对象
    """
    # 支持的图像格式
    supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']

    # 查找所有图像文件
    image_files = []
    for ext in supported_formats:
        image_files.extend(Path(directory_path).glob(f'*{ext}'))
        image_files.extend(Path(directory_path).glob(f'*{ext.upper()}'))

    if not image_files:
        log.warning(f"目录中没有找到支持的图像文件: {directory_path}")
        return

    log.info(f"找到 {len(image_files)} 个图像文件")

    # 批量预测
    batch_size = cfg.batch_size
    all_results = []

    for i in range(0, len(image_files), batch_size):
        batch_files = image_files[i:i + batch_size]
        batch_images = []
        batch_paths = []

        # 加载批次图像
        for img_path in batch_files:
            try:
                image = Image.open(img_path).convert('L')
                image_tensor = transform(image)
                image_tensor = image_tensor.repeat(3, 1, 1)  # [3, H, W]
                batch_images.append(image_tensor)
                batch_paths.append(str(img_path))
            except Exception as e:
                log.error(f"加载图像失败 {img_path}: {e}")
                continue

        if not batch_images:
            continue

        # 堆叠批次
        batch_tensor = torch.stack(batch_images, dim=0).to(device)

        # 预测批次
        with torch.no_grad():
            logits = model_module(batch_tensor)
            probs = torch.softmax(logits, dim=1)
            pred_classes = torch.argmax(probs, dim=1)
            confidences = probs[torch.arange(len(batch_images)), pred_classes]

        # 处理结果
        for j, img_path in enumerate(batch_paths):
            pred_class = pred_classes[j].item()
            confidence = confidences[j].item()

            # 获取类别名称
            class_names = [str(i) for i in range(10)]
            pred_label = class_names[pred_class]

            # 存储结果
            result = {
                'image_path': img_path,
                'prediction': pred_label,
                'confidence': confidence,
                'pred_class': pred_class
            }
            all_results.append(result)

            # 打印结果
            if cfg.show:
                log.info(f"{Path(img_path).name}: {pred_label} ({confidence:.4f})")

    # 保存所有结果
    if cfg.save_txt and all_results:
        save_batch_results(all_results, cfg)

    # 统计结果
    if all_results:
        log.info("预测完成!")
        log.info(f"总共处理 {len(all_results)} 张图像")

        # 计算平均置信度
        avg_confidence = np.mean([r['confidence'] for r in all_results])
        log.info(f"平均置信度: {avg_confidence:.4f}")

        # 统计类别分布
        class_counts = {}
        for r in all_results:
            pred = r['prediction']
            class_counts[pred] = class_counts.get(pred, 0) + 1

        log.info("预测类别分布:")
        for class_name in sorted(class_counts.keys()):
            count = class_counts[class_name]
            percentage = count / len(all_results) * 100
            log.info(f"  {class_name}: {count} ({percentage:.1f}%)")


def save_prediction_result(image_path, prediction, confidence, cfg):
    """
    保存单张图像的预测结果

    Args:
        image_path: 图像路径
        prediction: 预测结果
        confidence: 置信度
        cfg: 配置对象
    """
    # 创建保存目录
    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 保存结果到文本文件
    result_file = save_dir / "predictions.txt"

    with open(result_file, 'a') as f:
        f.write(f"{image_path},{prediction},{confidence:.4f}\n")


def save_batch_results(results, cfg):
    """
    保存批量预测结果

    Args:
        results: 结果列表
        cfg: 配置对象
    """
    # 创建保存目录
    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 保存结果到CSV文件
    import pandas as pd

    df = pd.DataFrame(results)
    csv_path = save_dir / "predictions.csv"
    df.to_csv(csv_path, index=False)

    log.info(f"保存预测结果到: {csv_path}")

    # 如果需要保存置信度，单独保存
    if cfg.save_conf:
        conf_path = save_dir / "confidences.txt"
        with open(conf_path, 'w') as f:
            for result in results:
                f.write(f"{result['image_path']}: {result['confidence']:.4f}\n")


if __name__ == "__main__":
    predict()
import numpy as np
import requests
from pathlib import Path
from PIL import Image
import pandas as pd
import random


def download(data_dir="pathmnist"):
    """
    下载PathMNIST数据集并转换为图像格式

    Args:
        data_dir: 数据保存目录
    """
    data_path = Path(data_dir)
    data_path.mkdir(exist_ok=True)

    # 数据集URL
    url = "https://zenodo.org/record/4269852/files/pathmnist.npz?download=1"
    npz_path = data_path / "pathmnist.npz"

    # 下载文件
    if not npz_path.exists():
        print("正在下载PathMNIST数据集...")
        response = requests.get(url, stream=True)
        response.raise_for_status()

        with open(npz_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("下载完成!")
    else:
        print("数据集文件已存在，跳过下载")

    return npz_path


def convert_to_images(npz_path, output_dir="optional_image"):
    """
    将PathMNIST数据集转换为图像文件

    Args:
        npz_path: .npz文件路径
        output_dir: 图像输出目录
    """
    # 加载数据
    data = np.load(npz_path)

    # 获取训练、验证、测试集
    train_images = data['train_images']
    train_labels = data['train_labels']
    val_images = data['val_images']
    val_labels = data['val_labels']
    test_images = data['test_images']
    test_labels = data['test_labels']

    print(f"训练集: {train_images.shape}, 标签: {train_labels.shape}")
    print(f"验证集: {val_images.shape}, 标签: {val_labels.shape}")
    print(f"测试集: {test_images.shape}, 标签: {test_labels.shape}")

    # 类别名称 (根据PathMNIST文档)
    class_names = [
        'adipose', 'background', 'debris', 'lymphocytes', 'mucus',
        'smooth_muscle', 'normal_colon_mucosa', 'cancer_stroma',
        'colorectal_adenocarcinoma'
    ]

    # 创建输出目录
    output_path = Path(output_dir)

    if output_path.exists():
        return output_path, class_names

    output_path.mkdir(exist_ok=True)

    # 为每个类别创建文件夹
    for i, class_name in enumerate(class_names):
        class_dir = output_path / class_name  # 使用数字作为文件夹名
        class_dir.mkdir(exist_ok=True)

    # 合并所有数据用于创建源数据集
    all_images = np.concatenate([train_images, val_images, test_images])
    all_labels = np.concatenate([train_labels, val_labels, test_labels])

    print(f"合并后总数据量: {all_images.shape}, 标签: {all_labels.shape}")

    # 保存图像到对应的类别文件夹
    image_count = 0
    for i, (image, label) in enumerate(zip(all_images, all_labels)):
        # 将numpy数组转换为PIL图像
        img = Image.fromarray(image)

        # 确定类别文件夹
        class_idx = int(label)
        class_dir = output_path / str(class_idx)

        # 保存图像
        filename = f"pathmnist_{i:06d}.png"
        img_path = class_dir / filename
        img.save(img_path)

        image_count += 1

        if (i + 1) % 10000 == 0:
            print(f"已处理 {i + 1} 张图像...")

    print(f"图像转换完成! 共保存 {image_count} 张图像")

    # 显示类别分布
    print("\n类别分布:")
    unique, counts = np.unique(all_labels, return_counts=True)
    for class_idx, count in zip(unique, counts):
        class_name = class_names[int(class_idx)]
        print(f"  类别 {class_idx} ({class_name}): {count} 张图像")

    return output_path, class_names


def create_sample_excel_for(image_dir, class_names, excel_path="data.xlsx"):
    """
    为PathMNIST数据集创建模拟的医疗Excel数据

    Args:
        image_dir: 图像目录
        class_names: 类别名称列表
        excel_path: Excel文件保存路径
    """
    # 收集所有图像文件
    all_images = []
    for class_dir in Path(image_dir).iterdir():
        if class_dir.is_dir():
            for img_file in class_dir.glob("*.png"):
                all_images.append({
                    'class': class_dir.name,
                    'filename': img_file.name
                })

    # 为每张图像生成模拟的医疗数据
    data = []
    for img_info in all_images:
        class_idx = int(img_info['class'])

        # 生成模拟的医疗数据
        row = {
            'image_name': img_info['filename'],
            'diagnosis': class_names[class_idx],  # 使用类别名称作为诊断
            'patient_age': random.randint(25, 85),
            'patient_gender': random.choice(['M', 'F']),
            'wbc_count': round(random.uniform(4.0, 15.0), 1),  # 白细胞计数
            'crp_level': round(random.uniform(0.5, 50.0), 1),  # C反应蛋白
            'temperature': round(random.uniform(36.5, 39.5), 1),  # 体温
            'blood_pressure_systolic': random.randint(100, 160),  # 收缩压
            'blood_pressure_diastolic': random.randint(60, 100),  # 舒张压
            'biopsy_result': random.choice(['Positive', 'Negative', 'Inconclusive']),
            'tumor_size': round(random.uniform(0.5, 8.0), 2),  # 肿瘤尺寸(cm)
        }
        data.append(row)

    # 创建DataFrame并保存为Excel
    df = pd.DataFrame(data)
    df.to_excel(excel_path, index=False)

    print(f"模拟医疗Excel数据生成完成，包含 {len(df)} 行数据")
    print(f"Excel文件保存为: {excel_path}")

    # 显示数据样例
    print("\nExcel数据样例:")
    print(df.head())

    return excel_path


class Config:
    """数据集配置类"""

    def __init__(self):
        # 基础路径配置 - 请根据实际情况修改这些路径
        self.base_data_dir = '/opt/datasets/pathmnist'
        self.output_dir = f"{self.base_data_dir}/optional_image"
        self.excel_path = f"{self.base_data_dir}/optional_image/data.xlsx"

        # 数据集分割比例
        self.train_ratio = 0.7
        self.val_ratio = 0.15
        self.test_ratio = 0.15

        # Excel配置
        self.image_col = "image_name"
        self.label_col = "diagnosis"  # 可选，如果使用Excel中的标签


if __name__ == "__main__":
    # 步骤1: 下载并转换数据集
    print("\n1. 下载并转换PathMNIST数据集...")
    config = Config()
    npz_path = download(data_dir=config.base_data_dir)
    image_dir, class_names = convert_to_images(npz_path, output_dir=config.output_dir)
    # 步骤2: 创建模拟Excel数据
    excel_path = create_sample_excel_for(image_dir,
                                         class_names,
                                         excel_path=config.excel_path)

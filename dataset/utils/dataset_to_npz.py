import numpy as np
import pandas as pd
from PIL import Image
import os
from pathlib import Path

class NPZDatasetConverter:
    """
    将分割后的数据集转换为 NPZ 格式
    """

    def __init__(self, image_size=(224, 224)):
        self.image_size = image_size

    def convert_split_to_npz(self, csv_path, image_root, output_npz_path,
                             label_mapping=None, normalize=True):
        """
        将单个分割集（train/val/test）转换为 NPZ 文件

        Args:
            csv_path: CSV 元数据文件路径
            image_root: 图像根目录
            output_npz_path: 输出的 NPZ 文件路径
            label_mapping: 标签映射字典（将字符串标签映射为整数）
            normalize: 是否归一化图像像素值
        """
        # 读取 CSV 文件
        df = pd.read_csv(csv_path)

        images = []
        labels = []
        filenames = []
        biomarkers = []

        # 提取生化指标列（如果有）
        bio_columns = [col for col in df.columns
                       if col not in ['image_path', 'filename', 'label']]

        print(f"处理 {len(df)} 个样本...")
        print(f"生化指标列: {bio_columns}")

        for idx, row in df.iterrows():
            try:
                # 构建完整的图像路径
                img_path = Path(image_root) / row['image_path']

                if not img_path.exists():
                    print(f"警告: 图像文件不存在: {img_path}")
                    continue

                # 加载并预处理图像
                image = self._load_and_preprocess_image(img_path, normalize)
                images.append(image)

                # 处理标签
                label_str = row['label']
                if label_mapping:
                    label = label_mapping.get(label_str, -1)
                else:
                    # 如果没有提供映射，自动创建数字标签
                    label = hash(label_str) % 1000  # 简单哈希作为临时标签

                labels.append(label)
                filenames.append(row['filename'])

                # 提取生化指标
                if bio_columns:
                    bio_values = [row[col] for col in bio_columns
                                  if pd.notna(row[col])]
                    biomarkers.append(bio_values)

            except Exception as e:
                print(f"处理图像 {row['filename']} 时出错: {e}")
                continue

        # 转换为 numpy 数组
        images_array = np.array(images)
        labels_array = np.array(labels)
        filenames_array = np.array(filenames)

        # 构建数据字典
        data_dict = {
            'images': images_array,
            'labels': labels_array,
            'filenames': filenames_array,
        }

        # 添加生化指标（如果有）
        if biomarkers and len(biomarkers) == len(images):
            biomarkers_array = np.array(biomarkers)
            data_dict['biomarkers'] = biomarkers_array
            data_dict['bio_columns'] = np.array(bio_columns)

        # 添加标签映射信息
        if label_mapping:
            data_dict['label_mapping'] = np.array(list(label_mapping.items()))

        # 保存为 NPZ 文件
        np.savez_compressed(output_npz_path, **data_dict)

        print(f"NPZ 文件已保存: {output_npz_path}")
        print(f"图像形状: {images_array.shape}")
        print(f"标签形状: {labels_array.shape}")
        if 'biomarkers' in data_dict:
            print(f"生化指标形状: {data_dict['biomarkers'].shape}")

    def _load_and_preprocess_image(self, image_path, normalize=True):
        """加载并预处理图像"""
        # 打开图像
        with Image.open(image_path) as img:
            # 转换为 RGB（如果是灰度图）
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # 调整大小
            img = img.resize(self.image_size, Image.Resampling.LANCZOS)

            # 转换为 numpy 数组
            img_array = np.array(img)

            # 归一化到 [0, 1]
            if normalize:
                img_array = img_array.astype(np.float32) / 255.0

            return img_array

    def create_label_mapping(self, labels):
        """创建标签映射字典"""
        unique_labels = sorted(set(labels))
        return {label: idx for idx, label in enumerate(unique_labels)}


def convert_dataset_to_npz(config):
    """
    将整个数据集转换为 NPZ 格式
    """
    converter = NPZDatasetConverter(image_size=(224, 224))

    # 基础输出目录
    base_output_dir = Path(config.output_dir).parent / "npz_datasets"
    base_output_dir.mkdir(parents=True, exist_ok=True)

    # 为两种数据集类型创建 NPZ
    dataset_types = ['_basic', '_multimodal']

    for dataset_type in dataset_types:
        dataset_dir = f"{config.output_dir}{dataset_type}"

        if not Path(dataset_dir).exists():
            print(f"数据集目录不存在: {dataset_dir}")
            continue

        print(f"\n处理数据集: {dataset_type}")

        # 收集所有标签来创建统一的映射
        all_labels = []
        for split in ['train', 'val', 'test']:
            csv_path = Path(dataset_dir) / f"{split}_metadata.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                all_labels.extend(df['label'].unique())

        # 创建标签映射
        label_mapping = converter.create_label_mapping(all_labels)
        print(f"标签映射: {label_mapping}")

        # 为每个分割集创建 NPZ
        for split in ['train', 'val', 'test']:
            csv_path = Path(dataset_dir) / f"{split}_metadata.csv"
            output_npz_path = base_output_dir / f"{split}{dataset_type}.npz"

            if csv_path.exists():
                print(f"\n转换 {split} 分割集...")
                converter.convert_split_to_npz(
                    csv_path=csv_path,
                    image_root=dataset_dir,
                    output_npz_path=output_npz_path,
                    label_mapping=label_mapping,
                    normalize=True
                )
            else:
                print(f"CSV 文件不存在: {csv_path}")


def load_npz_dataset(npz_path):
    """
    加载 NPZ 数据集并返回数据字典
    """
    data = np.load(npz_path, allow_pickle=True)

    result = {}
    for key in data.files:
        result[key] = data[key]

    data.close()
    return result


def demonstrate_npz_usage():
    """
    演示如何使用 NPZ 数据集
    """
    # 示例：加载训练集
    train_data = load_npz_dataset("npz_datasets/train_multimodal.npz")

    print("NPZ 数据集内容:")
    for key, value in train_data.items():
        if key in ['images', 'labels', 'biomarkers']:
            print(f"{key}: {value.shape} {value.dtype}")
        else:
            print(f"{key}: {value}")

    # 示例：访问数据
    images = train_data['images']
    labels = train_data['labels']

    if 'biomarkers' in train_data:
        biomarkers = train_data['biomarkers']
        bio_columns = train_data['bio_columns']
        print(f"\n生化指标列: {bio_columns}")
        print(f"第一个样本的生化指标: {biomarkers[0]}")

    print(f"\n第一个样本:")
    print(f"  图像形状: {images[0].shape}")
    print(f"  标签: {labels[0]}")
    print(f"  文件名: {train_data['filenames'][0]}")


# 修改主函数以包含 NPZ 转换
def main_with_npz():
    """
    包含 NPZ 转换的主函数
    """
    print("开始医疗数据集分割和 NPZ 转换...")

    # 初始化配置
    config = DatasetConfig()

    # 原有的数据集分割
    main()  # 调用原有的主函数

    # 新增 NPZ 转换
    print("\n" + "=" * 50)
    print("开始 NPZ 格式转换")
    print("=" * 50)

    convert_dataset_to_npz(config)

    # 演示 NPZ 使用
    print("\n" + "=" * 50)
    print("NPZ 数据集使用演示")
    print("=" * 50)

    demonstrate_npz_usage()


if __name__ == "__main__":
    # 运行包含 NPZ 转换的完整流程
    main_with_npz()
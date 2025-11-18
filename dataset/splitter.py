import os
import shutil
import random
import pandas as pd
from pathlib import Path
from datetime import datetime
import warnings
from PIL import Image, ImageDraw, ImageFont
import numpy as np


class DatasetSplitter:
    """
    医疗影像数据集分割工具类
    支持生成包含分类标签和生化指标的多模态数据集
    """

    def __init__(self, random_seed=42):
        """
        初始化

        Args:
            random_seed: 随机种子，确保结果可重现
        """
        self.random_seed = random_seed
        random.seed(random_seed)

    def split_dataset(self, source_dir, output_dir,
                      train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
                      extensions=None, copy_files=True,
                      excel_path=None, image_col='image_name', label_col=None):
        """
        分割数据集并生成CSV元数据文件

        Args:
            source_dir: 原始数据目录，包含按类别分组的子文件夹
            output_dir: 输出目录，将创建train/val/test子目录
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例
            extensions: 支持的图片扩展名，默认为常见图片格式
            copy_files: 是否复制文件（True）还是移动文件（False）
            excel_path: Excel文件路径，包含生化指标数据（可选）
            image_col: Excel中图片文件名的列名
            label_col: Excel中标签列的列名（如果与文件夹分类不一致时使用）

        Returns:
            dict: 包含分割统计信息的字典
        """

        # 检查比例总和是否为1
        total_ratio = train_ratio + val_ratio + test_ratio
        if abs(total_ratio - 1.0) > 1e-5:
            raise ValueError(f"比例总和应为1.0，当前为{total_ratio}")

        # 设置默认图片扩展名
        if extensions is None:
            extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.dcm'}

        # 创建输出目录结构
        output_path = Path(output_dir)
        train_dir = output_path / 'train'
        val_dir = output_path / 'val'
        test_dir = output_path / 'test'

        for dir_path in [train_dir, val_dir, test_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # 获取所有类别
        source_path = Path(source_dir)
        classes = [d.name for d in source_path.iterdir()
                   if d.is_dir() and not d.name.startswith('.')]

        if not classes:
            raise ValueError(f"在 {source_dir} 中未找到任何类别文件夹")

        print(f"找到 {len(classes)} 个类别: {classes}")

        # 读取Excel数据（如果提供）
        excel_data = None
        bio_columns = []  # 存储生化指标列名
        if excel_path:
            if not Path(excel_path).exists():
                warnings.warn(f"Excel文件 {excel_path} 不存在，将仅使用文件夹分类信息")
            else:
                try:
                    excel_data = pd.read_excel(excel_path)
                    print(f"成功读取Excel文件，包含 {len(excel_data)} 行数据")
                    print(f"Excel列名: {list(excel_data.columns)}")

                    # 获取生化指标列名（排除图片名和标签列）
                    bio_columns = [col for col in excel_data.columns
                                   if col not in [image_col, label_col] and label_col is not None]
                    if label_col is None:
                        bio_columns = [col for col in excel_data.columns if col != image_col]

                    print(f"生化指标列: {bio_columns}")
                except Exception as e:
                    warnings.warn(f"读取Excel文件失败: {e}，将仅使用文件夹分类信息")
                    excel_data = None

        # 统计数据
        stats = {
            'total_images': 0,
            'class_distribution': {},
            'split_distribution': {'train': 0, 'val': 0, 'test': 0},
            'excel_metadata_used': excel_data is not None,
            'bio_columns_count': len(bio_columns)
        }

        # 存储所有文件的分割信息用于生成CSV
        all_files_info = {
            'train': [],
            'val': [],
            'test': []
        }

        # 为每个类别创建输出目录并分割数据
        for class_name in classes:
            class_path = source_path / class_name

            # 在输出目录中创建类别子文件夹
            for dir_path in [train_dir, val_dir, test_dir]:
                (dir_path / class_name).mkdir(parents=True, exist_ok=True)

            # 获取该类别的所有图片文件
            image_files = []
            for ext in extensions:
                image_files.extend(list(class_path.glob(f'*{ext}')))
                image_files.extend(list(class_path.glob(f'*{ext.upper()}')))

            # 过滤出文件（排除目录）
            image_files = [f for f in image_files if f.is_file()]

            if not image_files:
                print(f"警告: 在 {class_name} 中未找到图片文件")
                continue

            stats['class_distribution'][class_name] = len(image_files)
            stats['total_images'] += len(image_files)

            # 随机打乱文件列表
            random.shuffle(image_files)

            # 计算分割点
            n_total = len(image_files)
            n_train = int(n_total * train_ratio)
            n_val = int(n_total * val_ratio)

            # 分割数据集
            train_files = image_files[:n_train]
            val_files = image_files[n_train:n_train + n_val]
            test_files = image_files[n_train + n_val:]

            # 处理训练集文件
            train_info = self._process_files(train_files, train_dir / class_name,
                                             class_name, copy_files, excel_data,
                                             image_col, label_col, bio_columns)
            all_files_info['train'].extend(train_info)

            # 处理验证集文件
            val_info = self._process_files(val_files, val_dir / class_name,
                                           class_name, copy_files, excel_data,
                                           image_col, label_col, bio_columns)
            all_files_info['val'].extend(val_info)

            # 处理测试集文件
            test_info = self._process_files(test_files, test_dir / class_name,
                                            class_name, copy_files, excel_data,
                                            image_col, label_col, bio_columns)
            all_files_info['test'].extend(test_info)

            # 更新统计
            stats['split_distribution']['train'] += len(train_files)
            stats['split_distribution']['val'] += len(val_files)
            stats['split_distribution']['test'] += len(test_files)

            print(f"类别 {class_name}: {len(image_files)} 张图片 -> "
                  f"训练: {len(train_files)}, 验证: {len(val_files)}, 测试: {len(test_files)}")

        # 生成CSV文件
        self._generate_csv_files(all_files_info, output_path, excel_data is not None)

        return stats

    def _process_files(self, file_list, target_dir, class_name, copy_files,
                       excel_data, image_col, label_col, bio_columns):
        """
        处理文件并返回文件信息

        Args:
            file_list: 文件列表
            target_dir: 目标目录
            class_name: 类别名称
            copy_files: 是否复制文件
            excel_data: Excel数据
            image_col: 图片列名
            label_col: 标签列名
            bio_columns: 生化指标列名列表

        Returns:
            list: 文件信息列表
        """
        files_info = []

        for file_path in file_list:
            target_path = target_dir / file_path.name

            # 处理文件名冲突
            counter = 1
            while target_path.exists():
                stem = file_path.stem
                suffix = file_path.suffix
                target_path = target_dir / f"{stem}_{counter}{suffix}"
                counter += 1

            if copy_files:
                shutil.copy2(file_path, target_path)
            else:
                shutil.move(str(file_path), str(target_path))

            # 构建文件信息
            file_info = {
                'image_path': str(target_path.relative_to(target_dir.parent.parent)),  # 相对路径
                'filename': target_path.name,
                'label': class_name
            }

            # 如果提供了Excel数据，添加生化指标
            if excel_data is not None:
                # 在Excel中查找匹配的行
                matched_rows = excel_data[excel_data[image_col] == file_path.name]

                if not matched_rows.empty:
                    # 取第一行匹配的数据
                    row = matched_rows.iloc[0]

                    # 如果指定了标签列且与文件夹分类不同，使用Excel中的标签
                    if label_col and label_col in excel_data.columns:
                        file_info['label'] = row[label_col]

                    # 添加所有生化指标列
                    for col in bio_columns:
                        if col in excel_data.columns:
                            file_info[col] = row[col]
                else:
                    # 如果没有匹配，为所有生化指标列赋值为None
                    for col in bio_columns:
                        file_info[col] = None
                    file_info['excel_data_missing'] = True

            files_info.append(file_info)

        return files_info

    def _generate_csv_files(self, all_files_info, output_path, has_excel_data):
        """
        为每个分割生成CSV文件

        Args:
            all_files_info: 所有文件信息
            output_path: 输出路径
            has_excel_data: 是否使用了Excel数据
        """
        for split_name, files_info in all_files_info.items():
            if files_info:  # 确保有数据
                df = pd.DataFrame(files_info)

                # 重新排列列，将重要列放在前面
                preferred_order = ['image_path', 'filename', 'label']
                other_cols = [col for col in df.columns if col not in preferred_order]
                df = df[preferred_order + other_cols]

                csv_path = output_path / f'{split_name}_metadata.csv'
                df.to_csv(csv_path, index=False, encoding='utf-8')
                print(f"生成 {split_name} CSV文件: {csv_path}")
                print(f"CSV文件包含 {len(df.columns)} 列: {list(df.columns)}")

    def create_dataset_info(self, output_dir, stats):
        """
        创建数据集信息文件

        Args:
            output_dir: 输出目录
            stats: 统计信息字典
        """
        info_file = Path(output_dir) / 'dataset_info.txt'

        with open(info_file, 'w', encoding='utf-8') as f:
            f.write("医疗影像数据集信息\n")
            f.write("=" * 50 + "\n\n")

            f.write(f"总图片数量: {stats['total_images']}\n")
            f.write(f"使用Excel元数据: {'是' if stats['excel_metadata_used'] else '否'}\n")
            f.write(f"生化指标列数: {stats['bio_columns_count']}\n\n")

            f.write("类别分布:\n")
            for class_name, count in stats['class_distribution'].items():
                f.write(f"  {class_name}: {count} 张图片\n")

            f.write(f"\n数据集分割:\n")
            f.write(f"  训练集: {stats['split_distribution']['train']} 张图片\n")
            f.write(f"  验证集: {stats['split_distribution']['val']} 张图片\n")
            f.write(f"  测试集: {stats['split_distribution']['test']} 张图片\n")

            f.write(f"\n生成的文件:\n")
            f.write(f"  train_metadata.csv - 训练集元数据\n")
            f.write(f"  val_metadata.csv - 验证集元数据\n")
            f.write(f"  test_metadata.csv - 测试集元数据\n")

            f.write(f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")


def demo_without_excel(config):
    """
    演示无Excel数据的情况

    Args:
        config: 数据集配置对象
    """
    print("\n" + "=" * 50)
    print("演示: 无Excel数据的情况")
    print("=" * 50)

    # 创建分割器实例
    splitter = DatasetSplitter(random_seed=42)

    # 分割数据集（不使用Excel）
    stats = splitter.split_dataset(
        source_dir=config.source_image_dir,
        output_dir=f"{config.output_dir}_basic",
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        copy_files=True
    )

    # 创建数据集信息文件
    splitter.create_dataset_info(f"{config.output_dir}_basic", stats)

    print(f"\n无Excel数据情况处理完成!")
    print(f"输出目录: {config.output_dir}_basic")

    # 显示生成的CSV文件内容
    train_csv_path = f"{config.output_dir}_basic/train_metadata.csv"
    if os.path.exists(train_csv_path):
        train_csv = pd.read_csv(train_csv_path)
        print(f"\n训练集CSV前3行:")
        print(train_csv.head(3))
        print(f"CSV列数: {len(train_csv.columns)}")
        print(f"列名: {list(train_csv.columns)}")

    return stats


def demo_with_excel(config):
    """
    演示有Excel数据的情况

    Args:
        config: 数据集配置对象
    """
    print("\n" + "=" * 50)
    print("演示: 有Excel数据的情况")
    print("=" * 50)

    # 创建分割器实例
    splitter = DatasetSplitter(random_seed=42)

    # 分割数据集（使用Excel）
    stats = splitter.split_dataset(
        source_dir=config.source_image_dir,
        output_dir=f"{config.output_dir}_multimodal",
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        copy_files=True,
        excel_path=config.excel_path,
        image_col=config.image_col,
        label_col=config.label_col  # 使用Excel中的诊断标签，而不是文件夹名称
    )

    # 创建数据集信息文件
    splitter.create_dataset_info(f"{config.output_dir}_multimodal", stats)

    print(f"\n有Excel数据情况处理完成!")
    print(f"输出目录: {config.output_dir}_multimodal")

    # 显示生成的CSV文件内容
    train_csv_path = f"{config.output_dir}_multimodal/train_metadata.csv"
    if os.path.exists(train_csv_path):
        train_csv = pd.read_csv(train_csv_path)
        print(f"\n训练集CSV前3行:")
        print(train_csv.head(3))
        print(f"CSV列数: {len(train_csv.columns)}")
        print(f"列名: {list(train_csv.columns)}")

    return stats


class DatasetConfig:
    """数据集配置类"""

    def __init__(self):
        # 基础路径配置 - 请根据实际情况修改这些路径
        self.base_data_dir = "/opt/datasets/pathmnist"
        self.source_image_dir = f"{self.base_data_dir}/optional_image"
        self.output_dir = f"{self.base_data_dir}/output_dataset"
        self.excel_path = f"{self.base_data_dir}/optional_image/data.xlsx"

        # 数据集分割比例
        self.train_ratio = 0.7
        self.val_ratio = 0.15
        self.test_ratio = 0.15

        # Excel配置
        self.image_col = "image_name"
        self.label_col = "diagnosis"  # 可选，如果使用Excel中的标签

def main():
    """
    主函数：完整演示流程
    """
    print("开始医疗数据集分割演示...")

    # 初始化配置
    config = DatasetConfig()

    print("当前配置:")
    print(f"  源图像目录: {config.source_image_dir}")
    print(f"  Excel文件: {config.excel_path}")
    print(f"  输出目录: {config.output_dir}")
    print(f"  分割比例: 训练{config.train_ratio}, 验证{config.val_ratio}, 测试{config.test_ratio}")

    # 检查源目录是否存在
    if not os.path.exists(config.source_image_dir):
        print(f"错误: 源目录不存在: {config.source_image_dir}")
        return

    # 演示无Excel数据的情况
    stats_basic = demo_without_excel(config)

    # 检查Excel文件是否存在
    # if os.path.exists(config.excel_path):
    #     # 演示有Excel数据的情况
    #     stats_multimodal = demo_with_excel(config)
    #
    #     # 比较两种情况的输出
    #     print("\n" + "=" * 50)
    #     print("结果比较")
    #     print("=" * 50)
    #
    #     print(f"无Excel数据情况:")
    #     print(f"  - 总图片数: {stats_basic['total_images']}")
    #     print(f"  - 使用Excel元数据: {stats_basic['excel_metadata_used']}")
    #     print(f"  - 生化指标列数: {stats_basic['bio_columns_count']}")
    #
    #     print(f"\n有Excel数据情况:")
    #     print(f"  - 总图片数: {stats_multimodal['total_images']}")
    #     print(f"  - 使用Excel元数据: {stats_multimodal['excel_metadata_used']}")
    #     print(f"  - 生化指标列数: {stats_multimodal['bio_columns_count']}")
    #
    # else:
    #     print(f"\n警告: Excel文件不存在: {config.excel_path}")
    #     print("跳过有Excel数据情况的演示")
    #
    # print(f"\n演示完成!")
    # print(f"生成的文件:")
    # print(f"  - 无Excel输出: {config.output_dir}_basic/")
    # if os.path.exists(config.excel_path):
    #     print(f"  - 有Excel输出: {config.output_dir}_multimodal/")


if __name__ == "__main__":
    main()
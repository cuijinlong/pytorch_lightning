import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2

class ImageAugmentation:
    """
    针对不同图像尺寸优化的影像数据增强管道
    """
    @staticmethod
    def get_small_image_transforms(image_size=(28, 28), mode='train'):
        """
        小尺寸图像增强 (<= 64x64)
        适用于: 28x28, 32x32, 64x64 等小图像
        """
        if mode == 'train':
            return A.Compose([
                # 保持原尺寸或轻微放大
                A.Resize(image_size[0], image_size[1]),

                # 轻微几何变换 - 小图像对变换敏感
                A.HorizontalFlip(p=0.3), # 水平翻转 (概率: 小图30%，中图50%，大图50%)
                A.VerticalFlip(p=0.3), # 垂直翻转 (概率: 小图30%，中图30%，大图50%)
                A.RandomRotate90(p=0.3), # 随机90度旋转 (概率: 小图30%，中图50%，大图50%)

                # 轻微颜色变换：轻微亮度对比度调整 (范围±0.1，概率40%)
                A.RandomBrightnessContrast(
                    brightness_limit=0.1,  # 减小范围
                    contrast_limit=0.1,  # 减小范围
                    p=0.4
                ),

                # 轻微高斯噪声 (方差5-15，概率20%)
                A.GaussNoise(var_limit=10.0, p=0.2),

                # 标准化 - 使用通用参数
                A.Normalize(
                    mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5]
                ),
                ToTensorV2()
            ])
        else:
            # 验证/测试模式 - 最小预处理
            return A.Compose([
                A.Resize(image_size[0], image_size[1]),
                A.Normalize(
                    mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5]
                ),
                ToTensorV2()
            ])

    @staticmethod
    def get_medium_image_transforms(image_size=(224, 224), mode='train'):
        """
        中等尺寸图像增强 (128x128 - 384x384)
        适用于: 128x128, 224x224, 256x256, 384x384
        """
        if mode == 'train':
            return A.Compose([
                # 基础几何变换
                A.Resize(image_size[0], image_size[1]),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.RandomRotate90(p=0.5),

                # 中等几何变换: 平移缩放旋转 (平移±5%，缩放±10%，旋转±10°，概率40%)
                A.ShiftScaleRotate(
                    shift_limit=0.05,  # 减小平移范围
                    scale_limit=0.1,
                    rotate_limit=10,  # 减小旋转角度
                    p=0.4,
                    border_mode=cv2.BORDER_CONSTANT
                ),

                # 弹性变换（轻微）: 轻微弹性变换 (概率20%)
                A.ElasticTransform(
                    alpha=0.5,  # 减小强度
                    sigma=25,
                    alpha_affine=25,
                    p=0.2
                ),

                # 颜色变换: 亮度对比度调整 (范围±0.15，概率50%)
                A.RandomBrightnessContrast(
                    brightness_limit=0.15,
                    contrast_limit=0.15,
                    p=0.5
                ),
                # 色调饱和度调整 (色调±5，饱和度±15，明度±10，概率40%)
                A.HueSaturationValue(
                    hue_shift_limit=5,  # 减小色调变化
                    sat_shift_limit=15,
                    val_shift_limit=10,
                    p=0.4
                ),

                # 噪声和模糊：高斯噪声 (方差10-30，概率30%)
                A.GaussNoise(var_limit=(10.0, 30.0), p=0.3),
                # 高斯模糊 (模糊限制3，概率20%)
                A.GaussianBlur(blur_limit=3, p=0.2),

                # 标准化 - 使用ImageNet统计量
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ])
        else:
            return A.Compose([
                A.Resize(image_size[0], image_size[1]),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ])

    @staticmethod
    def get_large_image_transforms(image_size=(512, 512), mode='train'):
        """
        大尺寸图像增强 (>= 512x512)
        适用于: 512x512, 768x768, 1024x1024 等大图像
        """
        if mode == 'train':
            return A.Compose([
                # 基础几何变换
                A.Resize(image_size[0], image_size[1]),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),

                # 较强的几何变换 强几何变换 (平移±10%，缩放±15%，旋转±15°，概率50%)
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.15,
                    rotate_limit=15,
                    p=0.5,
                    border_mode=cv2.BORDER_CONSTANT
                ),

                # 弹性变换和网格畸变 弹性变换 (概率30%)
                A.ElasticTransform(
                    alpha=1,
                    sigma=50,
                    alpha_affine=50,
                    p=0.3
                ),
                # 网格畸变 (概率20%)
                A.GridDistortion(p=0.2),

                # 透视变换: 透视变换 (概率30%)
                A.Perspective(scale=(0.05, 0.1), p=0.3),

                # 多种颜色变换: 亮度对比度、色调饱和度、CLAHE、Gamma校正
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.5
                ),
                # 增强的色调饱和度调整 (色调±10，饱和度±25，明度±15，概率40%)
                A.HueSaturationValue(
                    hue_shift_limit=10,
                    sat_shift_limit=20,
                    val_shift_limit=10,
                    p=0.5
                ),
                A.CLAHE(p=0.3),
                A.RandomGamma(p=0.3),

                # 噪声和模糊
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                A.GaussianBlur(blur_limit=5, p=0.3),
                A.MedianBlur(blur_limit=5, p=0.2),
                A.MotionBlur(blur_limit=5, p=0.2),

                # 图像质量变化
                A.ImageCompression(quality_lower=75, quality_upper=100, p=0.3),

                # 随机遮挡： (最多8个洞，概率30%)
                A.CoarseDropout(
                    max_holes=8,
                    max_height=image_size[0] // 16,
                    max_width=image_size[1] // 16,
                    min_holes=1,
                    fill_value=0,
                    p=0.3
                ),

                # 标准化
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ])
        else:
            return A.Compose([
                A.Resize(image_size[0], image_size[1]),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ])

    @staticmethod
    def get_specific_transforms(image_size, mode='train', modality='general'):
        """
        针对特定医疗影像模态的增强策略

        Args:
            image_size: 目标图像尺寸
            mode: train/val/test
            modality: 影像类型 - 'xray', 'ct', 'mri', 'pathology', 'dermatology'
        """
        base_config = {
            'xray': {
                'color_augmentation': False,  # X光片通常是灰度，减少颜色增强
                'elastic_transform': False,  # 减少弹性变换
                'strong_geometric': True,  # 保持几何变换
                'noise_augmentation': True,  # 添加噪声模拟真实X光
            },
            'ct': {
                'color_augmentation': False,  # CT通常是灰度
                'window_level_augmentation': True,  # CT窗宽窗位调整
                'strong_geometric': True,
                'elastic_transform': False,
            },
            'mri': {
                'color_augmentation': False,
                'strong_geometric': True,
                'bias_field_augmentation': True,  # MRI偏置场模拟
                'motion_artifacts': True,  # 运动伪影
            },
            'pathology': {
                'color_augmentation': True,  # 病理切片需要颜色增强
                'stain_augmentation': True,  # 染色变化
                'strong_geometric': True,
                'elastic_transform': True,  # 组织变形
            },
            'dermatology': {
                'color_augmentation': True,  # 皮肤图像需要颜色增强
                'strong_geometric': True,
                'lighting_variation': True,  # 光照变化
                'occlusion_augmentation': True,  # 遮挡增强
            }
        }

        config = base_config.get(modality, base_config['general'])

        if mode != 'train':
            # 验证和测试使用最小增强
            return A.Compose([
                A.Resize(image_size[0], image_size[1]),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ])

        # 根据图像尺寸选择基础变换
        if image_size[0] <= 64:
            transforms = ImageAugmentation.get_small_image_transforms(image_size, 'train')
        elif image_size[0] <= 384:
            transforms = ImageAugmentation.get_medium_image_transforms(image_size, 'train')
        else:
            transforms = ImageAugmentation.get_large_image_transforms(image_size, 'train')

        # 根据模态调整变换
        additional_transforms = []

        if modality == 'xray' and config['noise_augmentation']:
            additional_transforms.extend([
                A.GaussNoise(var_limit=(20.0, 80.0), p=0.4),
            ])

        if modality == 'pathology' and config['color_augmentation']:
            additional_transforms.extend([
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=25, val_shift_limit=15, p=0.4),
            ])

        if modality == 'dermatology' and config['lighting_variation']:
            additional_transforms.extend([
                A.RandomShadow(shadow_roi=(0, 0, 1, 1), num_shadows_lower=1, num_shadows_upper=2,
                               shadow_dimension=5, p=0.3),
            ])

        # 合并变换
        if additional_transforms:
            # 创建一个新的组合，包含基础变换和额外变换
            all_transforms = transforms.transforms + additional_transforms + [
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ]
            return A.Compose(all_transforms)
        else:
            return transforms

    @staticmethod
    def get_auto_transforms(image_size, mode='train', modality='general'):
        """
        自动根据图像尺寸选择最佳增强策略

        Args:
            image_size: 目标图像尺寸 (height, width)
            mode: train/val/test
            modality: 医疗影像模态
        """
        # 确定图像尺寸类别
        max_dim = max(image_size)

        if max_dim <= 64:
            print(f"检测到小尺寸图像 {image_size}，使用小图像增强策略")
            return ImageAugmentation.get_small_image_transforms(image_size, mode)
        elif max_dim <= 384:
            print(f"检测到中等尺寸图像 {image_size}，使用中等图像增强策略")
            return ImageAugmentation.get_medium_image_transforms(image_size, mode)
        else:
            print(f"检测到大尺寸图像 {image_size}，使用大图像增强策略")
            return ImageAugmentation.get_large_image_transforms(image_size, mode)

# 使用示例
def demo_augmentation():
    """演示优化后的增强策略"""

    # 小图像示例 (28x28)
    small_transform = ImageAugmentation.get_auto_transforms(
        image_size=(28, 28),
        mode='train'
    )
    print("小图像增强配置已创建")

    # 中等图像示例 (224x224)
    medium_transform = ImageAugmentation.get_auto_transforms(
        image_size=(224, 224),
        mode='train'
    )
    print("中等图像增强配置已创建")

    # 大图像示例 (512x512)
    large_transform = ImageAugmentation.get_auto_transforms(
        image_size=(512, 512),
        mode='train'
    )
    print("大图像增强配置已创建")

    # 医疗特定增强示例
    pathology_transform = ImageAugmentation.get_specific_transforms(
        image_size=(224, 224),
        mode='train',
        modality='pathology'
    )
    print("病理图像增强配置已创建")

    return {
        'small': small_transform,
        'medium': medium_transform,
        'large': large_transform,
        'pathology': pathology_transform
    }


class TestImageAugmentation:
    """测试图像增强管道的类"""

    def __init__(self, output_dir="augmentation_results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def load_and_preprocess_image(self, image_path, target_size=None):
        """
        加载并预处理图像

        Args:
            image_path: 图像文件路径
            target_size: 目标尺寸 (height, width)，如果为None则保持原尺寸
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图像文件不存在: {image_path}")

        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图像文件: {image_path}")

        print(f"成功加载图像: {image_path}")
        print(f"原始图像尺寸: {image.shape}")

        # 调整尺寸（如果需要）
        if target_size is not None:
            image = cv2.resize(image, (target_size[1], target_size[0]))
            print(f"调整后尺寸: {image.shape}")

        return image

    def visualize_augmentations(self, image, transforms, num_samples=5, title="Augmentation"):
        """
        可视化增强效果

        Args:
            image: 原始图像
            transforms: 增强管道
            num_samples: 生成样本数量
            title: 图表标题
        """
        # 转换BGR到RGB用于显示
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 创建子图
        fig, axes = plt.subplots(2, num_samples, figsize=(15, 6))
        fig.suptitle(title, fontsize=16)

        # 显示原始图像
        axes[0, 0].imshow(image_rgb)
        axes[0, 0].set_title('Original')
        axes[0, 0].axis('off')

        # 显示增强后的图像
        for i in range(num_samples):
            try:
                augmented = transforms(image=image)['image']

                # 如果是tensor，转换为numpy
                if hasattr(augmented, 'numpy'):
                    augmented = augmented.permute(1, 2, 0).numpy()
                    # 反标准化 - 根据实际使用的标准化参数调整
                    if hasattr(transforms, 'transforms'):
                        # 检查是否使用了标准化
                        normalize_found = False
                        for t in transforms.transforms:
                            if 'Normalize' in str(type(t)):
                                normalize_found = True
                                break

                        if normalize_found:
                            augmented = augmented * 0.5 + 0.5  # 假设使用了mean=0.5, std=0.5

                    augmented = np.clip(augmented * 255, 0, 255).astype(np.uint8)

                # 在第一行显示原始图像（除了第一个位置）
                if i > 0:
                    axes[0, i].imshow(image_rgb)
                    axes[0, i].set_title(f'Original {i + 1}')
                    axes[0, i].axis('off')

                # 在第二行显示增强图像
                axes[1, i].imshow(augmented)
                axes[1, i].set_title(f'Augmented {i + 1}')
                axes[1, i].axis('off')

            except Exception as e:
                print(f"第 {i + 1} 次增强失败: {e}")
                # 显示错误信息
                if i > 0:
                    axes[0, i].imshow(image_rgb)
                    axes[0, i].set_title(f'Original {i + 1}')
                    axes[0, i].axis('off')

                axes[1, i].text(0.5, 0.5, f'Error:\n{str(e)}',
                                ha='center', va='center', transform=axes[1, i].transAxes)
                axes[1, i].set_title(f'Failed {i + 1}')
                axes[1, i].axis('off')

        plt.tight_layout()

        # 保存图像
        save_path = os.path.join(self.output_dir, f"{title.replace(' ', '_').lower()}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"增强结果已保存: {save_path}")
        return save_path

    def test_single_image_augmentation(self, image_path, target_size=None):
        """
        测试单张图像的所有增强策略

        Args:
            image_path: 输入图像路径
            target_size: 目标尺寸 (height, width)
        """
        print(f"测试单张图像增强: {image_path}")

        # 加载图像
        image = self.load_and_preprocess_image(image_path, target_size)
        original_size = image.shape[:2]  # (height, width)

        results = {}

        # 测试1: 基于尺寸的自动增强
        print("\n1. 测试基于尺寸的自动增强...")
        auto_transform = ImageAugmentation.get_auto_transforms(
            image_size=original_size, mode='train'
        )
        results['auto'] = self.visualize_augmentations(
            image, auto_transform,
            title=f"Auto Augmentation - {original_size}"
        )

        # 测试2: 小图像增强（如果适用）
        if max(original_size) <= 64:
            print("\n2. 测试小图像增强...")
            small_transform = ImageAugmentation.get_small_image_transforms(
                image_size=original_size, mode='train'
            )
            results['small'] = self.visualize_augmentations(
                image, small_transform,
                title="Small Image Augmentation"
            )

        # 测试3: 中等图像增强（如果适用）
        elif max(original_size) <= 384:
            print("\n2. 测试中等图像增强...")
            medium_transform = ImageAugmentation.get_medium_image_transforms(
                image_size=original_size, mode='train'
            )
            results['medium'] = self.visualize_augmentations(
                image, medium_transform,
                title="Medium Image Augmentation"
            )

        # 测试4: 大图像增强（如果适用）
        else:
            print("\n2. 测试大图像增强...")
            large_transform = ImageAugmentation.get_large_image_transforms(
                image_size=original_size, mode='train'
            )
            results['large'] = self.visualize_augmentations(
                image, large_transform,
                title="Large Image Augmentation"
            )

        # 测试5: 不同模式对比
        print("\n3. 测试不同模式对比...")
        train_transform = ImageAugmentation.get_auto_transforms(
            image_size=original_size, mode='train'
        )
        val_transform = ImageAugmentation.get_auto_transforms(
            image_size=original_size, mode='val'
        )

        # 创建对比图
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        # 原始图像
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        axes[0].imshow(image_rgb)
        axes[0].set_title('Original')
        axes[0].axis('off')

        # 训练模式
        train_result = train_transform(image=image)['image']
        if hasattr(train_result, 'numpy'):
            train_vis = train_result.permute(1, 2, 0).numpy()
            train_vis = train_vis * 0.5 + 0.5
            train_vis = np.clip(train_vis * 255, 0, 255).astype(np.uint8)
        axes[1].imshow(train_vis)
        axes[1].set_title('Train Mode')
        axes[1].axis('off')

        # 验证模式
        val_result = val_transform(image=image)['image']
        if hasattr(val_result, 'numpy'):
            val_vis = val_result.permute(1, 2, 0).numpy()
            val_vis = val_vis * 0.5 + 0.5
            val_vis = np.clip(val_vis * 255, 0, 255).astype(np.uint8)
        axes[2].imshow(val_vis)
        axes[2].set_title('Validation Mode')
        axes[2].axis('off')

        plt.tight_layout()
        comparison_path = os.path.join(self.output_dir, "mode_comparison.png")
        plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
        plt.close()
        results['mode_comparison'] = comparison_path

        return results

    def test_modality_augmentations(self, image_path, target_size=(224, 224)):
        """
        测试不同医疗模态的增强效果

        Args:
            image_path: 输入图像路径
            target_size: 目标尺寸 (height, width)
        """
        print(f"测试医疗模态增强: {image_path}")

        # 加载并调整图像尺寸
        image = self.load_and_preprocess_image(image_path, target_size)

        modalities = ['xray', 'ct', 'mri', 'pathology', 'dermatology']
        results = {}

        for modality in modalities:
            try:
                print(f"  测试 {modality} 模态...")
                transform = ImageAugmentation.get_specific_transforms(
                    image_size=target_size, mode='train', modality=modality
                )
                results[modality] = self.visualize_augmentations(
                    image, transform,
                    title=f"{modality.capitalize()} Modality Augmentation"
                )
            except Exception as e:
                print(f"  {modality} 模态测试失败: {e}")
                results[modality] = None

        return results

    def test_custom_size_augmentation(self, image_path, custom_sizes):
        """
        测试自定义尺寸的增强效果

        Args:
            image_path: 输入图像路径
            custom_sizes: 自定义尺寸列表，例如 [(128, 128), (256, 256), (512, 512)]
        """
        print(f"测试自定义尺寸增强: {image_path}")

        results = {}

        for size in custom_sizes:
            try:
                print(f"  测试尺寸 {size}...")
                # 加载并调整图像尺寸
                image = self.load_and_preprocess_image(image_path, size)

                # 获取对应尺寸的增强
                transform = ImageAugmentation.get_auto_transforms(
                    image_size=size, mode='train'
                )

                results[str(size)] = self.visualize_augmentations(
                    image, transform,
                    title=f"Custom Size Augmentation - {size}"
                )
            except Exception as e:
                print(f"  尺寸 {size} 测试失败: {e}")
                results[str(size)] = None

        return results

    def batch_test_images(self, image_paths, output_subdir="batch_test"):
        """
        批量测试多张图像

        Args:
            image_paths: 图像路径列表
            output_subdir: 输出子目录
        """
        batch_output_dir = os.path.join(self.output_dir, output_subdir)
        os.makedirs(batch_output_dir, exist_ok=True)

        original_output_dir = self.output_dir
        self.output_dir = batch_output_dir

        results = {}

        for i, image_path in enumerate(image_paths):
            try:
                print(f"\n处理图像 {i + 1}/{len(image_paths)}: {os.path.basename(image_path)}")

                # 为每张图像创建单独的测试
                image_results = self.test_single_image_augmentation(image_path)
                results[image_path] = image_results

            except Exception as e:
                print(f"图像 {image_path} 处理失败: {e}")
                results[image_path] = None

        # 恢复原始输出目录
        self.output_dir = original_output_dir

        return results

def interactive_test():
    """交互式测试函数"""
    # 获取用户输入的图像路径
    image_path = '/opt/datasets/pathmnist/output_dataset_multimodal/test/background/pathmnist_020661.png'

    if not os.path.exists(image_path):
        print("错误: 文件不存在!")
        return

    # 创建测试实例
    tester = TestImageAugmentation(output_dir="/Users/cuijinlong/Documents/workspace_py/fastai_lm/dataset/utils/interactive_test_results")

    # 加载图像获取原始尺寸
    image = tester.load_and_preprocess_image(image_path)
    original_size = image.shape[:2]

    print(f"\n图像信息:")
    print(f"  路径: {image_path}")
    print(f"  尺寸: {original_size}")
    print(f"  通道: {image.shape[2] if len(image.shape) > 2 else 1}")

    # 测试选项
    print("\n选择测试类型:")
    print("1. 单张图像完整测试")
    print("2. 医疗模态增强测试")
    print("3. 自定义尺寸测试")
    print("4. 批量测试（如果有多个图像）")

    choice = "1"

    if choice == "1":
        print("\n执行单张图像完整测试...")
        results = tester.test_single_image_augmentation(image_path)

    elif choice == "2":
        print("\n执行医疗模态增强测试...")
        target_size = input("输入目标尺寸 (格式: 224,224 或按回车使用原尺寸): ").strip()
        if target_size:
            try:
                h, w = map(int, target_size.split(','))
                target_size = (h, w)
            except:
                print("尺寸格式错误，使用原尺寸")
                target_size = original_size
        else:
            target_size = original_size

        results = tester.test_modality_augmentations(image_path, target_size)

    elif choice == "3":
        print("\n执行自定义尺寸测试...")
        sizes_input = input("输入多个尺寸 (格式: 128,128 256,256 512,512): ").strip()
        custom_sizes = []
        for size_str in sizes_input.split():
            try:
                h, w = map(int, size_str.split(','))
                custom_sizes.append((h, w))
            except:
                print(f"跳过无效尺寸: {size_str}")

        if not custom_sizes:
            print("没有有效的尺寸，使用默认尺寸")
            custom_sizes = [(128, 128), (256, 256), (512, 512)]

        results = tester.test_custom_size_augmentation(image_path, custom_sizes)

    elif choice == "4":
        print("\n执行批量测试...")
        # 获取同目录下的其他图像
        image_dir = os.path.dirname(image_path)
        image_files = [f for f in os.listdir(image_dir)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

        image_paths = [os.path.join(image_dir, f) for f in image_files]
        print(f"找到 {len(image_paths)} 个图像文件")

        results = tester.batch_test_images(image_paths)

    else:
        print("无效选择!")
        return

    print(f"\n测试完成! 结果保存在: {os.path.abspath(tester.output_dir)}")
    return results

# 便捷函数
def quick_test_image(image_path, output_dir="quick_test"):
    """
    快速测试单张图像

    Args:
        image_path: 图像路径
        output_dir: 输出目录
    """
    tester = TestImageAugmentation(output_dir=output_dir)
    return tester.test_single_image_augmentation(image_path)


if __name__ == "__main__":
    interactive_test()
    print("小图像增强配置已创建")

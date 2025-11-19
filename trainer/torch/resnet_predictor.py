import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import json
import argparse


class ResNetPredictor:
    def __init__(self, model_path, device=None):
        """
        初始化预测器

        Args:
            model_path: 训练好的模型路径
            device: 使用的设备 (cuda/cpu)
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        # 加载模型检查点
        checkpoint = torch.load(model_path, map_location=self.device)

        # 获取配置和标签字典
        self.config = checkpoint.get('config', {})
        self.label_dict = checkpoint.get('label_dict', {})

        # 反转标签字典，用于从索引到标签名的映射
        self.idx_to_label = {v: k for k, v in self.label_dict.items()}

        # 初始化模型
        self.model = self._create_model()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        # 图像预处理
        self.transform = self._get_transform()

        print(f"模型加载成功: {model_path}")
        print(f"使用设备: {self.device}")
        print(f"类别数量: {len(self.label_dict)}")
        print(f"标签映射: {self.label_dict}")

    def _create_model(self):
        """创建与训练时相同的模型结构"""
        model_name = self.config.get('model_name', 'resnet18')
        num_classes = len(self.label_dict)

        # 获取模型
        model = getattr(models, model_name)(pretrained=False)

        # 修改最后一层
        if hasattr(model, 'fc'):
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif hasattr(model, 'classifier'):
            # 某些变体使用classifier
            model.classifier = nn.Linear(model.classifier.in_features, num_classes)

        return model

    def _get_transform(self):
        """获取与训练时相同的图像预处理"""
        image_size = self.config.get('image_size', (224, 224))

        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        return transform

    def predict_image(self, image_path):
        """
        预测单张图像

        Args:
            image_path: 图像路径

        Returns:
            dict: 包含预测结果的信息
        """
        try:
            # 加载和预处理图像
            image = Image.open(image_path).convert('RGB')
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)

            # 预测
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_prob, predicted_idx = torch.max(probabilities, 1)

            # 获取预测结果
            predicted_class = predicted_idx.item()
            confidence = predicted_prob.item()
            class_name = self.idx_to_label.get(predicted_class, f"Class_{predicted_class}")

            # 获取所有类别的概率
            all_probs = probabilities.squeeze().cpu().numpy()
            class_probabilities = {
                self.idx_to_label.get(i, f"Class_{i}"): float(prob)
                for i, prob in enumerate(all_probs)
            }

            result = {
                'image_path': image_path,
                'predicted_class': class_name,
                'class_index': predicted_class,
                'confidence': confidence,
                'all_probabilities': class_probabilities,
                'success': True
            }

            return result

        except Exception as e:
            return {
                'image_path': image_path,
                'error': str(e),
                'success': False
            }

    def predict_batch(self, image_dir, extensions=None):
        """
        批量预测目录中的图像

        Args:
            image_dir: 图像目录路径
            extensions: 支持的图像扩展名

        Returns:
            list: 所有图像的预测结果
        """
        if extensions is None:
            extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

        if not os.path.exists(image_dir):
            raise ValueError(f"目录不存在: {image_dir}")

        results = []
        image_files = []

        # 收集所有图像文件
        for file in os.listdir(image_dir):
            if any(file.lower().endswith(ext) for ext in extensions):
                image_files.append(os.path.join(image_dir, file))

        print(f"找到 {len(image_files)} 张图像进行预测...")

        # 批量预测
        for image_path in image_files:
            result = self.predict_image(image_path)
            results.append(result)

            if result['success']:
                print(
                    f"预测: {os.path.basename(image_path)} -> {result['predicted_class']} (置信度: {result['confidence']:.4f})")
            else:
                print(f"错误: {os.path.basename(image_path)} -> {result['error']}")

        return results

    def predict_single_with_display(self, image_path):
        """
        预测单张图像并显示详细信息
        """
        result = self.predict_image(image_path)

        if result['success']:
            print("\n" + "=" * 50)
            print(f"图像: {os.path.basename(image_path)}")
            print(f"预测类别: {result['predicted_class']}")
            print(f"类别索引: {result['class_index']}")
            print(f"置信度: {result['confidence']:.4f}")
            print("\n所有类别概率:")
            for class_name, prob in sorted(result['all_probabilities'].items(),
                                           key=lambda x: x[1], reverse=True):
                print(f"  {class_name}: {prob:.4f}")
            print("=" * 50)
        else:
            print(f"预测失败: {result['error']}")

        return result


def main():
    parser = argparse.ArgumentParser(description='ResNet图像分类预测')
    parser.add_argument('--model_path', type=str, required=True,
                        help='训练好的模型路径')
    parser.add_argument('--image_path', type=str,
                        help='单张图像路径')
    parser.add_argument('--image_dir', type=str,
                        help='图像目录路径（批量预测）')
    parser.add_argument('--output_file', type=str,
                        help='输出结果文件路径（JSON格式）')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'],
                        help='强制使用设备')

    args = parser.parse_args()

    # 设置设备
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 初始化预测器
    predictor = ResNetPredictor(args.model_path, device)

    results = []

    # 单张图像预测
    if args.image_path:
        if os.path.isfile(args.image_path):
            print("单张图像预测模式:")
            result = predictor.predict_single_with_display(args.image_path)
            results.append(result)
        else:
            print(f"错误: 图像文件不存在: {args.image_path}")
            return

    # 批量预测
    elif args.image_dir:
        if os.path.isdir(args.image_dir):
            print("批量预测模式:")
            results = predictor.predict_batch(args.image_dir)
        else:
            print(f"错误: 目录不存在: {args.image_dir}")
            return

    else:
        print("请提供 --image_path 或 --image_dir 参数")
        return

    # 保存结果到文件
    if args.output_file and results:
        # 只保存成功的结果
        successful_results = [r for r in results if r.get('success', False)]

        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(successful_results, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存到: {args.output_file}")

    # 打印统计信息
    if results:
        successful = [r for r in results if r.get('success', False)]
        failed = [r for r in results if not r.get('success', False)]

        print(f"\n预测完成!")
        print(f"成功: {len(successful)}")
        print(f"失败: {len(failed)}")

        if successful:
            avg_confidence = sum(r['confidence'] for r in successful) / len(successful)
            print(f"平均置信度: {avg_confidence:.4f}")


# 简单使用示例
if __name__ == '__main__':
    # 如果不使用命令行参数，可以在这里直接设置
    model_path = "checkpoints/best_model.pth"  # 修改为你的模型路径

    # 方法1: 单张图像预测
    predictor = ResNetPredictor(model_path)

    # 预测单张图像

    image_path = "/opt/datasets/pathmnist/output_dataset_basic/test/background/pathmnist_000648.png"  # 修改为你的图像路径
    result = predictor.predict_single_with_display(image_path)
    print(result)
    # 方法2: 批量预测
    # image_dir = "path/to/your/images"  # 修改为你的图像目录
    # results = predictor.predict_batch(image_dir)

    # 方法3: 使用命令行
    # 在命令行中运行: python predict.py --model_path checkpoints/best_model.pth --image_path test_image.jpg
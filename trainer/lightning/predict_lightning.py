# predict_lightning.py
import torch
import pytorch_lightning as pl
from torchvision import transforms
from PIL import Image
import os
import json
import argparse
from resnet_lightning import ResNetLightning

class LightningPredictor:
    def __init__(self, checkpoint_path, device=None):
        # 自动加载模型和配置
        self.model = ResNetLightning.load_from_checkpoint(checkpoint_path)

        if device:
            self.model = self.model.to(device)
        self.model.eval()

        # 从模型获取配置信息
        self.config = self.model.hparams.config
        self.label_dict = self.model.label_dict
        self.idx_to_label = {v: k for k, v in self.label_dict.items()}

        # 图像预处理
        self.transform = self._get_transform()

        print(f"模型加载成功: {checkpoint_path}")
        print(f"类别数量: {len(self.label_dict)}")
        print(f"标签映射: {self.label_dict}")

    def _get_transform(self):
        """获取图像预处理"""
        image_size = self.config.get('image_size', (224, 224))

        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        return transform

    def predict_image(self, image_path):
        """预测单张图像"""
        try:
            # 加载和预处理图像
            image = Image.open(image_path).convert('RGB')
            input_tensor = self.transform(image).unsqueeze(0)

            # 使用模型的设备
            device = next(self.model.parameters()).device
            input_tensor = input_tensor.to(device)

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
        """批量预测目录中的图像"""
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


def main():
    parser = argparse.ArgumentParser(description='Lightning图像分类预测')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='训练好的模型检查点路径')
    parser.add_argument('--image_path', type=str,
                        help='单张图像路径')
    parser.add_argument('--image_dir', type=str,
                        help='图像目录路径（批量预测）')
    parser.add_argument('--output_file', type=str,
                        help='输出结果文件路径（JSON格式）')

    args = parser.parse_args()

    # 初始化预测器
    predictor = LightningPredictor(args.checkpoint_path)

    results = []

    # 单张图像预测
    if args.image_path:
        if os.path.isfile(args.image_path):
            print("单张图像预测模式:")
            result = predictor.predict_image(args.image_path)
            results.append(result)

            # 显示详细信息
            if result['success']:
                print("\n" + "=" * 50)
                print(f"图像: {os.path.basename(args.image_path)}")
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


if __name__ == '__main__':
    main()
from PIL import Image
import os
from pathlib import Path


def check_image_details(image_path):
    """
    检查单张图片的详细信息

    Args:
        image_path: 图片文件路径
    """
    try:
        # 检查文件是否存在
        if not os.path.exists(image_path):
            print(f"错误: 文件不存在 - {image_path}")
            return

        # 使用PIL打开图片
        with Image.open(image_path) as img:
            # 获取基本信息
            width, height = img.size
            mode = img.mode
            format_type = img.format
            filename = Path(image_path).name

            # 获取文件大小
            file_size = os.path.getsize(image_path)

            print("=" * 60)
            print(f"图片详细信息: {filename}")
            print("=" * 60)
            print(f"尺寸: {width} × {height} 像素")
            print(f"宽高比: {width / height:.2f}:1")
            print(f"色彩模式: {mode}")
            print(f"格式: {format_type}")
            print(f"文件大小: {file_size / 1024:.2f} KB ({file_size} 字节)")

            # 给出模型输入建议
            print("\n模型输入建议:")

            # 常见CNN输入尺寸
            common_sizes = [224, 256, 384, 512]
            min_dim = min(width, height)

            # 找到最接近的常见尺寸
            closest_size = min(common_sizes, key=lambda x: abs(x - min_dim))

            if min_dim >= 224:
                print(f"✅ 推荐输入尺寸: {closest_size}×{closest_size}")
                if min_dim < 512:
                    print(f"✅ 可使用高质量裁剪或保持原尺寸")
                else:
                    print(f"✅ 可考虑下采样到 {closest_size}×{closest_size}")
            else:
                print(f"⚠️  图片较小 ({min_dim}px)，建议:")
                print(f"   - 使用双线性插值上采样")
                print(f"   - 考虑使用更高分辨率的原始数据")

            # 数据增强建议
            print(f"\n数据增强建议:")
            if width == height:
                print(f"✅ 方形图片，适合标准CNN架构")
            else:
                print(f"⚠️  非方形图片，建议:")
                print(f"   - 使用随机裁剪保持宽高比")
                print(f"   - 或使用填充(padding)到方形")

            return {
                'width': width,
                'height': height,
                'mode': mode,
                'format': format_type,
                'file_size': file_size,
                'aspect_ratio': width / height
            }

    except Exception as e:
        print(f"读取图片时出错: {e}")
        return None


def check_multiple_images(image_paths):
    """
    检查多张图片的尺寸
    """
    results = []
    for path in image_paths:
        print("\n")
        result = check_image_details(path)
        if result:
            results.append(result)

    # 汇总统计
    if results:
        print("\n" + "=" * 60)
        print("汇总统计")
        print("=" * 60)

        widths = [r['width'] for r in results]
        heights = [r['height'] for r in results]

        print(f"图片数量: {len(results)}")
        print(f"宽度范围: {min(widths)} - {max(widths)} 像素")
        print(f"高度范围: {min(heights)} - {max(heights)} 像素")
        print(f"平均尺寸: {sum(widths) / len(widths):.1f} × {sum(heights) / len(heights):.1f} 像素")

        # 检查尺寸一致性
        if len(set(zip(widths, heights))) == 1:
            print("✅ 所有图片尺寸一致")
        else:
            print("⚠️  图片尺寸不一致，需要统一预处理")

    return results


# 使用示例
if __name__ == "__main__":
    # 替换为您的实际图片路径
    image_paths = [
        "/opt/datasets/pathmnist/output_dataset_basic/test/background/pathmnist_000145.png"
        # 在这里添加您的图片路径
        # "/path/to/your/image1.jpg",
        # "/path/to/your/image2.png",
        # 例如:
        # "/opt/datasets/pathmnist/output_dataset_multimodal/train/class_name/image001.jpg"
    ]

    if image_paths and image_paths[0].startswith("/"):
        # 检查单张图片
        if len(image_paths) == 1:
            check_image_details(image_paths[0])
        # 检查多张图片
        else:
            check_multiple_images(image_paths)
    else:
        print("请先编辑脚本，添加您的图片路径")
        print("\n使用方法:")
        print("1. 在 image_paths 列表中添加您的图片路径")
        print("2. 运行脚本查看图片详细信息")
        print("\n示例路径:")
        print('image_paths = ["/path/to/your/image.jpg"]')
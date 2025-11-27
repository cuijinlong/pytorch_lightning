# /opt/pytorch_lightning/trainer/yolo/yolo_inference.py
import torch
from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
import time


class YOLOInference:
    def __init__(self, model_path):
        """
        初始化YOLO推理器

        Args:
            model_path: 模型文件路径 (.pt 或 .ckpt)
        """
        try:
            # 加载模型
            self.model = YOLO(model_path)
            self.class_names = self.model.names
            print(f"模型加载成功: {model_path}")
            print(f"类别数量: {len(self.class_names)}")
        except Exception as e:
            print(f"加载模型失败: {e}")
            raise

    def predict_image(self, image_path, conf_threshold=0.25, save_result=True):
        """
        预测单张图片

        Args:
            image_path: 图片路径
            conf_threshold: 置信度阈值
            save_result: 是否保存结果图片

        Returns:
            检测结果
        """
        if not Path(image_path).exists():
            print(f"图片文件不存在: {image_path}")
            return None

        try:
            print(f"正在检测图片: {image_path}")
            start_time = time.time()

            # 执行推理
            results = self.model(image_path, conf=conf_threshold)
            inference_time = time.time() - start_time

            # 处理每个检测结果
            for i, r in enumerate(results):
                # 绘制检测结果
                im_array = r.plot()  # 返回带检测框的numpy数组

                # 显示检测信息
                print(f"检测到 {len(r.boxes)} 个目标")
                print(f"推理时间: {inference_time:.3f}秒")

                # 显示图片
                cv2.imshow('YOLO Detection', im_array)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

                # 保存结果图片
                if save_result:
                    output_path = f"detection_result_{i}.jpg"
                    cv2.imwrite(output_path, im_array)
                    print(f"结果已保存至: {output_path}")

            return results

        except Exception as e:
            print(f"图片推理失败: {e}")
            return None

    def predict_webcam(self, conf_threshold=0.25):
        """
        实时摄像头检测

        Args:
            conf_threshold: 置信度阈值
        """
        print("启动摄像头检测... (按 'q' 退出)")

        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("无法打开摄像头")
            return

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("无法读取摄像头帧")
                    break

                # 执行推理
                results = self.model(frame, conf=conf_threshold, verbose=False)

                # 绘制检测结果
                for r in results:
                    frame = r.plot()

                # 显示FPS
                cv2.putText(frame, f"YOLO Detection - Press 'q' to quit",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                cv2.imshow('YOLO Detection', frame)

                # 按'q'退出
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except Exception as e:
            print(f"摄像头检测出错: {e}")
        finally:
            # 释放资源
            cap.release()
            cv2.destroyAllWindows()
            print("摄像头检测结束")


def main():
    """
    主函数 - 推理演示
    """
    print("YOLO推理演示")

    # 模型路径选择
    model_path = "lightning_checkpoints/yolo-best.ckpt"  # 训练好的模型

    # 如果训练好的模型不存在，使用预训练模型
    if not Path(model_path).exists():
        print("训练好的模型不存在，使用预训练模型...")
        model_path = "yolov8n.pt"

    try:
        # 创建检测器
        detector = YOLOInference(model_path)

        # 选项1: 测试图片推理
        test_image = "test_image.jpg"  # 替换为你的测试图片路径

        # 如果测试图片存在，进行推理
        if Path(test_image).exists():
            detector.predict_image(test_image, conf_threshold=0.25)
        else:
            print(f"测试图片不存在: {test_image}")
            print("请将测试图片放在当前目录并命名为 'test_image.jpg'")

        # 选项2: 摄像头实时检测 (取消注释以启用)
        # print("启动摄像头检测...")
        # detector.predict_webcam()

    except Exception as e:
        print(f"推理过程出错: {e}")


if __name__ == "__main__":
    main()
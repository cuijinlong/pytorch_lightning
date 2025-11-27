# dowload.py
import os
import shutil
from pathlib import Path
from ultralytics.data.utils import download

def prepare_coco128_dataset():
    """下载并准备COCO128数据集"""
    dataset_url = "https://ultralytics.com/assets/coco128.zip"
    download_dir = Path("/opt/datasets")
    download_dir.mkdir(exist_ok=True)

    # 下载数据集
    zip_path = download_dir / "coco128.zip"
    if not zip_path.exists():
        print("下载COCO128数据集...")
        download(dataset_url, dir=download_dir, unzip=True)

    print("数据集准备完成！")
    return download_dir / "coco128"


if __name__ == "__main__":
    prepare_coco128_dataset()
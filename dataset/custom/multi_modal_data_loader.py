import os
import cv2
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import albumentations as A
from dataset.custom.utils.image_augmentation import ImageAugmentation


# ======================================================================
#                           MultiModalDataset
# ======================================================================

class MultiModalDataset(Dataset):
    """
    医疗多模态数据集（图像 + 表格）
    支持：自动列识别、训练/验证/测试模式、预处理器重用、图像增强
    """

    def __init__(
            self,
            csv_path,
            image_base_dir,
            mode,
            transform=None,
            image_col='image_path',
            label_col='label',
            numeric_cols=None,
            categorical_cols=None,
            tabular_cols=None,
            preprocessor=None,
            label_encoder=None,
            label_dict=None
    ):
        """
        Args:
            mode: ['train', 'val', 'test']
            preprocessor: 仅训练集 fit，其余 transform
            label_encoder: 同上
            label_dict: 标签到类别名称的映射字典
        """
        self.mode = mode
        self.df = pd.read_csv(csv_path)
        self.image_base_dir = Path(image_base_dir)

        self.image_col = image_col
        self.label_col = label_col
        self.transform = transform

        # --------------------------
        # 自动识别表格列
        # --------------------------
        self.numeric_cols = numeric_cols if numeric_cols else []
        self.categorical_cols = categorical_cols if categorical_cols else []
        self.tabular_cols = self.numeric_cols + self.categorical_cols

        # --------------------------
        # 预处理标签
        # --------------------------
        self.label_encoder = label_encoder
        self.label_dict = label_dict
        self.labels = self._encode_labels()

        # --------------------------
        # 表格特征处理
        # --------------------------
        self.preprocessor = preprocessor
        self.tabular_features = self._process_tabular()

        # print(f"\n[{mode}] 数据集初始化完成：共 {len(self.df)} 样本")
        # print(f"  数值特征 {len(self.numeric_cols)} 个: {self.numeric_cols}")
        # print(f"  分类特征 {len(self.categorical_cols)} 个: {self.categorical_cols}")
        if self.label_dict:
            print(f"  标签映射: {self.label_dict}")

    # ==================================================================
    # 标签编码
    # ==================================================================
    def _encode_labels(self):
        if self.label_col not in self.df.columns:
            return np.zeros(len(self.df))

        if self.label_encoder is None:
            le = LabelEncoder()
            y = le.fit_transform(self.df[self.label_col])
            self.label_encoder = le
            # 创建标签映射字典
            self.label_dict = {i: cls for i, cls in enumerate(le.classes_)}
            return y

        return self.label_encoder.transform(self.df[self.label_col])

    # ==================================================================
    # 表格特征处理 (fit 只在训练集发生)
    # ==================================================================
    def _process_tabular(self):
        if len(self.tabular_cols) == 0:
            return None

        df_tab = self.df[self.tabular_cols].copy()

        if self.preprocessor is None:
            # 数值型特征的处理管道
            numeric_tf = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),  # 中位数填充缺失值
                ('scaler', StandardScaler())  # 标准化
            ])
            # 分类型特征的处理管道
            cat_tf = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),  # 众数填充缺失值
                ('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))  # 独热编码，忽略未知类别
            ])
            # 使用ColumnTransformer将不同的处理管道应用到对应的列上
            self.preprocessor = ColumnTransformer([
                ('num', numeric_tf, self.numeric_cols),  # 数值型列使用numeric_tf管道
                ('cat', cat_tf, self.categorical_cols)  # 分类型列使用cat_tf管道
            ])

            # 在训练集上拟合预处理器并转换数据
            features = self.preprocessor.fit_transform(df_tab)
            # 学习：数值列的均值/标准差，分类列的类别映射
            # 转换：应用这些参数到训练数据
        else:
            # 在验证集或测试集上，使用已经拟合好的预处理器转换数据
            # 使用训练集学到的：
            # - 用训练集的均值填充缺失值（不是当前数据的均值）
            # - 用训练集的标准差进行标准化
            # - 用训练集的类别映射进行独热编码
            features = self.preprocessor.transform(df_tab)
        return features

    # ==================================================================
    # Dataset 必需方法
    # ==================================================================
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        # --------------------------
        # 图像加载（增强兼容CV2/PIL）
        # --------------------------
        image_path = self.df.iloc[idx][self.image_col]
        full_path = (
            Path(image_path) if os.path.isabs(image_path)
            else self.image_base_dir / image_path
        )

        image = cv2.imread(str(full_path))

        if image is None:
            # try PIL
            try:
                image = np.array(Image.open(full_path).convert("RGB"))
            except:
                image = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            if isinstance(self.transform, A.Compose):
                image = self.transform(image=image)["image"]
            else:
                image = self.transform(Image.fromarray(image))

        # --------------------------
        # 标签
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        # --------------------------
        # 表格数据
        # --------------------------
        if self.tabular_features is None:
            tab = torch.zeros(0, dtype=torch.float32)
        else:
            tab = torch.tensor(self.tabular_features[idx], dtype=torch.float32)

        return {
            "image": image,
            "tabular": tab,
            "label": label,
            # "image_path": str(full_path),
            # "original_label": self.df.iloc[idx].get(self.label_col, "unknown")
        }


# ======================================================================
#                          MultiModalDataLoader
# ======================================================================

class MultiModalDataLoader:

    def __init__(self, config):
        self.config = config

        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        self.preprocessor = None
        self.label_encoder = None
        self.label_dict = None  # 新增：标签映射字典

    # ==================================================================
    def create_data_loaders(self):
        # ---------------------------------------------------------
        #                  1) 训练集（fit 预处理器）
        # ---------------------------------------------------------
        train_ds = MultiModalDataset(
            csv_path=self.config['train_csv'],
            image_base_dir=self.config['image_base_dir'],
            mode='train',
            transform=ImageAugmentation.get_auto_transforms(
                self.config.get('image_size', (224, 224)), mode="train"
            ),
            image_col=self.config['image_col'],
            label_col=self.config['label_col'],
            tabular_cols=self.config['tabular_cols'],
            numeric_cols=self.config['numeric_cols'],
            categorical_cols=self.config['categorical_cols']
        )

        self.preprocessor = train_ds.preprocessor
        self.label_encoder = train_ds.label_encoder
        self.label_dict = train_ds.label_dict  # 新增：获取标签映射

        self.train_loader = DataLoader(
            train_ds,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config['num_workers'],
            pin_memory=torch.cuda.is_available(),
            drop_last=True
        )

        # ---------------------------------------------------------
        #                  2) 验证集
        # ---------------------------------------------------------
        if 'val_csv' in self.config:
            val_ds = MultiModalDataset(
                csv_path=self.config['val_csv'],
                image_base_dir=self.config['image_base_dir'],
                mode='val',
                transform=ImageAugmentation.get_auto_transforms(
                    self.config.get('image_size', (224, 224)), mode="val"
                ),
                image_col=self.config['image_col'],
                label_col=self.config['label_col'],
                preprocessor=self.preprocessor,
                label_encoder=self.label_encoder,
                label_dict=self.label_dict,  # 新增：传递标签映射
                tabular_cols=self.config['tabular_cols'],
                numeric_cols=self.config['numeric_cols'],
                categorical_cols=self.config['categorical_cols']
            )
            self.val_loader = DataLoader(
                val_ds,
                batch_size=self.config['batch_size'],
                shuffle=False,
                num_workers=self.config['num_workers'],
                pin_memory=torch.cuda.is_available()
            )

        # ---------------------------------------------------------
        #                  3) 测试集
        # ---------------------------------------------------------
        if 'test_csv' in self.config:
            test_ds = MultiModalDataset(
                csv_path=self.config['test_csv'],
                image_base_dir=self.config['image_base_dir'],
                mode='test',
                transform=ImageAugmentation.get_auto_transforms(
                    self.config.get('image_size', (224, 224)), mode="test"
                ),
                image_col=self.config['image_col'],
                label_col=self.config['label_col'],
                preprocessor=self.preprocessor,
                label_encoder=self.label_encoder,
                label_dict=self.label_dict,  # 新增：传递标签映射
                tabular_cols=self.config['tabular_cols'],
                numeric_cols=self.config['numeric_cols'],
                categorical_cols=self.config['categorical_cols']
            )
            self.test_loader = DataLoader(
                test_ds,
                batch_size=self.config['batch_size'],
                shuffle=False,
                num_workers=self.config['num_workers'],
                pin_memory=torch.cuda.is_available()
            )

        return self.train_loader, self.val_loader, self.test_loader

    # ==================================================================
    def get_class_weights(self):
        """用于不平衡分类"""
        if self.label_encoder is None:
            return None

        labels = self.train_loader.dataset.labels
        counts = np.bincount(labels)
        weights = len(labels) / (len(counts) * counts)
        return torch.tensor(weights, dtype=torch.float32)

    # ==================================================================
    def get_label_dict(self):
        """获取标签映射字典"""
        return self.label_dict

    # ==================================================================
    def get_num_classes(self):
        """获取类别数量"""
        if self.label_encoder is None:
            return 0
        return len(self.label_encoder.classes_)


# ======================================================================
#                              Demo
# ======================================================================
if __name__ == '__main__':

    base_dir = "/opt/datasets/pathmnist/output_dataset_multimodal"

    numeric_columns = [
        'patient_age', 'wbc_count', 'crp_level', 'temperature',
        'blood_pressure_systolic', 'blood_pressure_diastolic',
        'tumor_size'
    ]
    categorical_columns = ['patient_gender', 'biopsy_result']

    config = {
        'image_base_dir': base_dir,
        'train_csv': f"{base_dir}/train_metadata.csv",
        'val_csv': f"{base_dir}/val_metadata.csv",
        'test_csv': f"{base_dir}/test_metadata.csv",
        'image_col': "image_path",
        'label_col': "label",
        'batch_size': 64,
        'num_workers': 2,
        'image_size': (28, 28),
        'tabular_cols': numeric_columns + categorical_columns,
        'numeric_cols': numeric_columns,
        'categorical_cols': categorical_columns
    }

    loader = MultiModalDataLoader(config)
    train_loader, val_loader, test_loader = loader.create_data_loaders()

    # print("\n类别权重：", loader.get_class_weights())
    # print("标签字典：", loader.get_label_dict())
    # print("类别数量：", loader.get_num_classes())

    # 试输出 2 个 batch
    for i, batch in enumerate(train_loader):
        print("  image:", batch['image'].shape)
        print("  tabular:", batch['tabular'].shape)
        print("  label:", batch['label'].shape)
        if i == 1:
            break
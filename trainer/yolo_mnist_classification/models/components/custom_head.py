import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomClassificationHead(nn.Module):
    """
    自定义分类头
    用于YOLO分类模型的分类头
    """

    def __init__(self, in_channels, num_classes, hidden_dim=1024, dropout=0.2):
        """
        初始化自定义分类头

        Args:
            in_channels: 输入通道数
            num_classes: 类别数
            hidden_dim: 隐藏层维度
            dropout: Dropout率
        """
        super().__init__()

        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # 分类头
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入特征图 [B, C, H, W]

        Returns:
            分类logits [B, num_classes]
        """
        # 全局平均池化
        x = self.global_pool(x)  # [B, C, 1, 1]

        # 分类
        x = self.classifier(x)  # [B, num_classes]

        return x


class AttentionPooling(nn.Module):
    """
    注意力池化层
    使用注意力机制进行特征聚合
    """

    def __init__(self, in_channels):
        super().__init__()

        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, 1),
            nn.BatchNorm2d(in_channels // 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 8, 1, 1),
            nn.Sigmoid()
        )

        self.global_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入特征图 [B, C, H, W]

        Returns:
            池化后的特征 [B, C, 1, 1]
        """
        # 计算注意力权重
        attention_weights = self.attention(x)  # [B, 1, H, W]

        # 应用注意力权重
        weighted_features = x * attention_weights  # [B, C, H, W]

        # 全局平均池化
        output = self.global_pool(weighted_features)  # [B, C, 1, 1]

        return output


class MultiScaleClassificationHead(nn.Module):
    """
    多尺度分类头
    从多个尺度提取特征进行分类
    """

    def __init__(self, in_channels_list, num_classes, hidden_dim=1024):
        """
        初始化多尺度分类头

        Args:
            in_channels_list: 各尺度输入通道数列表
            num_classes: 类别数
            hidden_dim: 隐藏层维度
        """
        super().__init__()

        self.num_scales = len(in_channels_list)

        # 各尺度的池化层
        self.pools = nn.ModuleList([
            nn.AdaptiveAvgPool2d(1) for _ in range(self.num_scales)
        ])

        # 各尺度的特征变换层
        self.transformers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim // self.num_scales, 1),
                nn.BatchNorm2d(hidden_dim // self.num_scales),
                nn.ReLU(inplace=True)
            ) for in_channels in in_channels_list
        ])

        # 分类器
        total_channels = hidden_dim
        self.classifier = nn.Sequential(
            nn.Linear(total_channels, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, features_list):
        """
        前向传播

        Args:
            features_list: 各尺度特征图列表

        Returns:
            分类logits [B, num_classes]
        """
        pooled_features = []

        # 处理各尺度特征
        for i in range(self.num_scales):
            # 池化
            pooled = self.pools[i](features_list[i])  # [B, C_i, 1, 1]

            # 特征变换
            transformed = self.transformers[i](pooled)  # [B, hidden_dim/num_scales, 1, 1]

            # 展平
            flattened = transformed.flatten(1)  # [B, hidden_dim/num_scales]
            pooled_features.append(flattened)

        # 拼接各尺度特征
        combined = torch.cat(pooled_features, dim=1)  # [B, hidden_dim]

        # 分类
        output = self.classifier(combined)  # [B, num_classes]

        return output


def create_classification_head(head_type, **kwargs):
    """
    创建分类头

    Args:
        head_type: 分类头类型 ('custom', 'attention', 'multi_scale')
        **kwargs: 分类头参数

    Returns:
        分类头模块
    """
    if head_type == 'custom':
        return CustomClassificationHead(**kwargs)
    elif head_type == 'attention':
        return AttentionPooling(kwargs.get('in_channels'))
    elif head_type == 'multi_scale':
        return MultiScaleClassificationHead(**kwargs)
    else:
        raise ValueError(f"未知的分类头类型: {head_type}")
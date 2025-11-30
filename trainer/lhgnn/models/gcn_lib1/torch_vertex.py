# 2022.06.17-Changed for building ViG model
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
import numpy as np
import torch
from torch import nn
from .torch_nn import BasicConv, batched_index_select, act_layer,FFN
from .torch_edge import DenseDilatedKnnGraph, HyperedgeConstruction
from .pos_embed import get_2d_relative_pos_embed
import torch.nn.functional as F
from timm.models.layers import DropPath
import math

class LHGConv2d(nn.Module):
    """
    Max-Relative Graph Convolution (Paper: https://arxiv.org/abs/1904.03751) for dense data type
    """
    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super(LHGConv2d, self).__init__()
        self.nn = BasicConv([in_channels*3, out_channels], act, norm, bias)
        # 来获取聚类中心和隶属度权重。
        self.get_centroids = HyperedgeConstruction(in_channels)
        #self.nn_hyper = BasicConv([in_channels, in_channels], act, norm, bias)

    """
        节点7：超图卷积实现
        位置: torch_vertex.py → LHGConv2d 类
        7.1 三特征融合机制
        输入: [Batch, C, N, 1] 节点特征
        输出: [Batch, C, N, 1] 增强后的节点特征
        关键技术点:
            模糊聚类: 使用FCM算法生成超边
            三特征融合: [自身特征, 邻居特征, 聚类特征]
            软分配: 每个节点以不同程度属于多个聚类
    """
    def forward(self, x, edge_index, y=None,num_clusters=50,top_clusters=5, **kwargs):
        # 步骤7.1: 传统邻居特征提取
        x_i = batched_index_select(x, edge_index[1])
        if y is not None:
            x_j = batched_index_select(y, edge_index[0])
            
        else:
            x_j = batched_index_select(x, edge_index[0])
        x_j, _ = torch.max(x_j - x_i, -1, keepdim=True)  # 最大相对特征

        # 步骤7.2: 🎯 模糊聚类生成超边，调用C均值(FCM)聚类算法获取超边
        centroids,weights = self.get_centroids(x,num_clusters)

        # 步骤7.3: 🎯选择top-k聚类中心，根据隶属度权重选择top-k聚类中心
        weights = weights.squeeze(-2)
        _, nn_idx_centroid = torch.topk(weights, k=top_clusters, largest=True, dim=-1)

        # 步骤7.4: 构建超图连接
        b, c, n, _ = x.shape
        center_idx = torch.arange(0, n, device=x.device).repeat(b,top_clusters, 1).transpose(2, 1)
        edge_idx = torch.stack((nn_idx_centroid, center_idx), dim=0)

        # 步骤7.5: 聚类中心特征提取
        x_j_cluster = batched_index_select(centroids.unsqueeze(-1), edge_idx[0])
        x_i_cluster = batched_index_select(x, edge_idx[1])
        x_j_cluster,_ = torch.max(x_j_cluster - x_i_cluster, -1, keepdim=True)

        # 步骤7.6: 🎯 三特征拼接融合
        x = torch.cat([x.unsqueeze(2), x_j.unsqueeze(2),x_j_cluster.unsqueeze(2)], dim=2).reshape(b, 3 * c, n, -1)
        
        return self.nn(x) # 最终卷积变换
        
"""
    Max-Relative图卷积 (MRConv2d)
    计算节点与邻居的最大差值特征
    公式: max(x_j - x_i) 增强局部差异感知 
"""
class MRConv2d(nn.Module):
    """
    Max-Relative Graph Convolution (Paper: https://arxiv.org/abs/1904.03751) for dense data type
    """
    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super(MRConv2d, self).__init__()
        self.nn = BasicConv([in_channels*3, out_channels], act, norm, bias)
        self.get_centroids = HyperedgeConstruction(in_channels,'soft-kmeans')
        #self.nn_hyper = BasicConv([in_channels, in_channels], act, norm, bias)
    def forward(self, x, edge_index, y=None,num_centroids=50,H=None,W=None):
        # 🎯 使用空间池化初始化聚类中心（改进的模糊聚类）
        x_copy = x.reshape(x.shape[0],x.shape[1],H,W)
        intial_centroids = F.adaptive_avg_pool2d(x_copy, (5,10)).reshape(x.shape[0],x.shape[1],-1)

        x_i = batched_index_select(x, edge_index[1])
        if y is not None:
            x_j = batched_index_select(y, edge_index[0])
            
        else:
            x_j = batched_index_select(x, edge_index[0])
        x_j, _ = torch.max(x_j - x_i, -1, keepdim=True)

        # 🎯 使用预定义初始中心的模糊聚类
        if y is not None:
            centroid,weights = self.get_centroids(x,num_centroids,intial_centroids)
        else:
            centroid,weights = self.get_centroids(x,num_centroids,intial_centroids)

        #n bcentroid = self.nn_hyper(centroid.unsqueeze(-1)).squeeze(-1)
        weights = weights.squeeze(-2)
        _, nn_idx_centroid = torch.topk(weights, k=12, largest=True, dim=-1)
        b, c, n, _ = x.shape
        center_idx = torch.arange(0, n, device=x.device).repeat(b,12, 1).transpose(2, 1)
        edge_idx = torch.stack((nn_idx_centroid, center_idx), dim=0)
        x_j_center = batched_index_select(centroid.unsqueeze(-1), edge_idx[0])
        x_i_center = batched_index_select(x, edge_idx[1])
        x_j_center,_ = torch.max(x_j_center - x_i_center, -1, keepdim=True)

        x = torch.cat([x.unsqueeze(2), x_j.unsqueeze(2),x_j_center.unsqueeze(2)], dim=2).reshape(b, 3 * c, n, -1)
        #
        #x = torch.cat([x.unsqueeze(2),x_j.unsqueeze(2)], dim=2).reshape(b, 2 * c, n, -1)
        
        #max_value, _ = torch.max(self.nn(torch.cat([x_i_center, x_j_center - x_i_center], dim=1)), -1, keepdim=True)
        return self.nn(x)
        #return max_value


class EdgeConv2d(nn.Module):
    """
    Edge convolution layer (with activation, batch normalization) for dense data type
    """
    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super(EdgeConv2d, self).__init__()
        self.nn = BasicConv([in_channels * 2, out_channels], act, norm, bias)

    def forward(self, x, edge_index, y=None):
        x_i = batched_index_select(x, edge_index[1])
        if y is not None:
            x_j = batched_index_select(y, edge_index[0])
        else:
            x_j = batched_index_select(x, edge_index[0])
        max_value, _ = torch.max(self.nn(torch.cat([x_i, x_j - x_i], dim=1)), -1, keepdim=True)
        return max_value


class GraphSAGE(nn.Module):
    """
    GraphSAGE Graph Convolution (Paper: https://arxiv.org/abs/1706.02216) for dense data type
    """
    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super(GraphSAGE, self).__init__()
        self.nn1 = BasicConv([in_channels, in_channels], act, norm, bias)
        self.nn2 = BasicConv([in_channels*2, out_channels], act, norm, bias)

    def forward(self, x, edge_index, y=None):
        if y is not None:
            x_j = batched_index_select(y, edge_index[0])
        else:
            x_j = batched_index_select(x, edge_index[0])
        x_j, _ = torch.max(self.nn1(x_j), -1, keepdim=True)
        return self.nn2(torch.cat([x, x_j], dim=1))


class GINConv2d(nn.Module):
    """
    GIN Graph Convolution (Paper: https://arxiv.org/abs/1810.00826) for dense data type
    """
    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super(GINConv2d, self).__init__()
        self.nn = BasicConv([in_channels, out_channels], act, norm, bias)
        eps_init = 0.0
        self.eps = nn.Parameter(torch.Tensor([eps_init]))

    def forward(self, x, edge_index, y=None):
        if y is not None:
            x_j = batched_index_select(y, edge_index[0])
        else:
            x_j = batched_index_select(x, edge_index[0])
        x_j = torch.sum(x_j, -1, keepdim=True)
        return self.nn((1 + self.eps) * x + x_j)


class GraphConv2d(nn.Module):
    """
    Static graph convolution layer
    """
    def __init__(self, in_channels, out_channels, conv='edge', act='relu', norm=None, bias=True):
        super(GraphConv2d, self).__init__()
        if conv == 'edge':
            self.gconv = EdgeConv2d(in_channels, out_channels, act, norm, bias)
        elif conv == 'mr':
            """
                Max-Relative图卷积 (MRConv2d)
                计算节点与邻居的最大差值特征
                公式: max(x_j - x_i) 增强局部差异感知 
            """
            self.gconv = MRConv2d(in_channels, out_channels, act, norm, bias)
        elif conv == 'sage':
            self.gconv = GraphSAGE(in_channels, out_channels, act, norm, bias)
        elif conv == 'gin':
            self.gconv = GINConv2d(in_channels, out_channels, act, norm, bias)
        elif conv == 'lhg':
            """
            超图卷积 (LHGConv2d) - 核心创新
            融合三种特征:
                1. 节点自身特征
                2. 邻居差值特征  
                3. 聚类中心差值特征
            超图构建流程：
                soft-kmeans聚类生成超边（聚类中心）
                top-k选择从聚类中心选取代表性节点
                三特征拼接：[自身, 邻居差, 中心差]
            """
            self.gconv = LHGConv2d(in_channels, out_channels, act, norm, bias)
        else:
            raise NotImplementedError('conv:{} is not supported'.format(conv))

    def forward(self, x, edge_index, y=None,**kwargs):
        return self.gconv(x, edge_index, y,**kwargs)

class DyGraphConv2d(GraphConv2d):
    """
        动态图卷积调用链
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=9,
                 dilation=1,
                 conv='edge',
                 act='relu',
                 norm=None,
                 bias=True,
                 stochastic=False,
                 epsilon=0.0,
                 r=1):
        super(DyGraphConv2d, self).__init__(in_channels, out_channels, conv, act, norm, bias)
        self.k = kernel_size
        self.d = dilation
        self.r = r
        # DyGraphConv2d中动态计算KNN图
        """
           调用dilated_knn_graph构建图（edge_index）
               dilation膨胀率控制:（idx:backbone的Seq层数 idx=0-3:  膨胀率=1， idx=4-7:  膨胀率=2，idx=8-11: 膨胀率=3...以此类推，
               stochastic：True:随机采样 False:规则膨胀采样
               
               实际邻居选择过程:
                    模块阶段	k值	    dilation值	    候选邻居数	最终邻居数
                    第1阶段	10	    1	            10×1=10	    10
                    第2阶段	10	    2	            10×2=20	    10
                    第3阶段	10	    3	            10×3=30	    10
                对于 dilation=2 的模块:
                    候选邻居（按距离排序）: [A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T]
                    索引位置:              0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,11,12,13,14,15,16,17,18,19
                    膨胀采样（每隔2个选1个）: 0, 2, 4, 6, 8, 10, 12, 14, 16, 18
                    对应邻居:               A, C, E, G, I, K,  M,  O,  Q,  S
        """
        self.dilated_knn_graph = DenseDilatedKnnGraph(kernel_size, dilation, stochastic, epsilon)

    """
    节点6：动态图卷积核心
    位置: torch_vertex.py → DyGraphConv2d 类
        6.1 图构建过程
        输入: [Batch, C, H, W] 空间特征图
        输出: [Batch, C*channel_mul, H, W] 图卷积输出
        关键技术点:
            节点化处理: 将空间位置视为图节点
            膨胀KNN: 扩大感受野而不增加参数
            多尺度图: 通过下采样构建层次化图结构
    """
    def forward(self, x, relative_pos=None,**kwargs):
        B, C, H, W = x.shape
        # 步骤6.1: 下采样构建多尺度图
        y = None
        if self.r > 1:
            y = F.avg_pool2d(x, self.r, self.r)  # 空间下采样
            y = y.reshape(B, C, -1, 1).contiguous()

        # 步骤6.2: 展平特征为节点格式
        x = x.reshape(B, C, -1, 1).contiguous()
        """
            步骤6.3: KNN图构建
            在DyGraphConv2d中动态计算KNN图
                每层动态重建图结构
                适应不同层次的特征分布
        """
        edge_index = self.dilated_knn_graph(x, y, relative_pos)
        # 步骤6.4: 执行图卷积
        x = super(DyGraphConv2d, self).forward(x, edge_index, y, **kwargs)
        # 步骤6.5: 恢复空间格式
        return x.reshape(B, -1, H, W).contiguous()

"""
结构为 fc1 -> graph_conv -> fc2 -> drop_path：
fc1 与 fc2：均为 1×1 卷积 + BatchNorm。
    fc1：保持通道数不变，对输入特征进行线性变换，为图卷积做准备。
    fc2：将图卷积输出映射回原通道数，确保残差连接维度匹配。
    
graph_conv（动态图卷积）：根据配置使用不同的图卷积类型（由 conv 参数控制），核心是建模节点与其邻居的关系：
    Max-Relative 图卷积（MRConv2d）：计算节点 x_i 与其邻居 x_j 的差值 x_j - x_i，取最大值作为邻居特征，再与节点自身特征拼接（[x, x_j - x_i]），增强局部差异感知。
    超图卷积（LHGConv）：当 conv='lhg' 时启用，通过 soft-kmeans 聚类生成 “超边”（聚类中心），融合节点自身特征、邻居差值特征、中心差值特征（共 3 类特征），提升全局关系建模能力。

DropPath：随机深度机制，以一定概率丢弃当前模块输出，增强模型泛化能力（深层模块丢弃概率更高）。
残差连接：将输入与处理后的特征相加（x = DropPath(fc2(graph_conv(fc1(x)))) + x），缓解深层网络梯度消失问题。
"""
class Grapher(nn.Module):
    """
    Grapher module with graph convolution and fc layers
    """
    def __init__(self,
                 in_channels=1,
                 num_knn=9,
                 num_clusters=50,
                 dilation=1,
                 conv='edge',
                 act='relu',
                 norm=None,
                 bias=True,
                 stochastic=False,
                 epsilon=0.0,
                 r=1,
                 n=196,
                 drop_path=0.0,
                 relative_pos=False,
                 cluster_ratio=0.5,
                 channel_mul=1):
        super(Grapher, self).__init__()
        self.channels = in_channels
        self.n = n
        self.r = r
        channel_mul= int(channel_mul)
        self.conv = conv 
        self.num_clusters = num_clusters
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )
        """
        动态图构建细节:
            使用k近邻动态构建图结构
            支持膨胀采样扩大感受野
                > 邻居选择：通过 num_knn 控制每个节点的邻居数量（如 k=10），
                          结合膨胀采样（dilation）扩大感受野（类似 CNN 的膨胀卷积）。
                > 聚类机制：当使用超图卷积时，通过 num_clusters 设定聚类中心数量（如 10），
                          cluster_ratio 控制从中心选择的 top-k 比例（如 0.4→top4），
                          平衡局部与全局关系。
            (1) 特征归一化
            (2) KNN计算最近邻
            (3) 膨胀采样选择有效邻居
            (4) 执行图卷积操作
        """
        self.graph_conv = DyGraphConv2d(in_channels, in_channels * channel_mul,
                                        num_knn, dilation, conv,
                                        act, norm, bias, stochastic,
                                        epsilon, r)
        self.fc2 = nn.Sequential(
            nn.Conv2d(in_channels * channel_mul, in_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.relative_pos = None
        self.cluster_ratio = cluster_ratio
        # 从聚类中心选择的top-k数量
        self.top_clusters = math.ceil(num_knn * cluster_ratio)  # = ceil(10 × 0.4) = 4
        if relative_pos:
            print('using relative_pos')
            relative_pos_tensor = torch.from_numpy(np.float32(get_2d_relative_pos_embed(in_channels,
                int(n**0.5)))).unsqueeze(0).unsqueeze(1)
            relative_pos_tensor = F.interpolate(
                    relative_pos_tensor, size=(n, n//(r*r)), mode='bicubic', align_corners=False)
            self.relative_pos = nn.Parameter(-relative_pos_tensor.squeeze(1), requires_grad=False)

    def _get_relative_pos(self, relative_pos, H, W):
        if relative_pos is None or H * W == self.n:
            return relative_pos
        else:
            N = H * W
            N_reduced = N // (self.r * self.r)
            return F.interpolate(relative_pos.unsqueeze(0), size=(N, N_reduced), mode="bicubic").squeeze(0)

    """
        节点5：Grapher图卷积模块
        位置: torch_vertex.py → Grapher 类
        输入: [Batch, C, H, W] 当前stage的特征图
        输出: [Batch, C, H, W] 图卷积处理后的特征图
        关键技术点:
            动态图构建: 每层根据特征动态计算KNN图
            超图卷积: 使用模糊聚类生成超边
            残差连接: 缓解梯度消失
    """
    def forward(self, x):
        _tmp = x  # 保存残差连接
        # 步骤5.1: 线性变换
        x = self.fc1(x) # # [B,C,H,W] → [B,C,H,W] (1×1卷积)
        # 步骤5.2: 动态图卷积
        B, C, H, W = x.shape
        relative_pos = self._get_relative_pos(self.relative_pos, H, W)
        if self.conv == 'lhg':
            # 超图卷积分支
            x = self.graph_conv(x, relative_pos,num_clusters=self.num_clusters, top_clusters=self.top_clusters)
        else:
            x = self.graph_conv(x,relative_pos)
        # 步骤5.3: 线性变换恢复通道
        x = self.fc2(x) # [B,C*channel_mul,H,W] → [B,C,H,W]
        # 步骤5.4: DropPath + 残差连接
        x = self.drop_path(x) + _tmp
        x = x.reshape(B, C, H, W)
        return x
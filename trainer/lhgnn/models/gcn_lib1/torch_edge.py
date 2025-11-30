# 2022.06.17-Changed for building ViG model
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
import math
import torch
from torch import nn
import torch.nn.functional as F

""" 超边构建，用于更复杂的图结构 """
class HyperedgeConstruction(nn.Module):

    def __init__(self,in_channels,num_iters=1):

        super(HyperedgeConstruction,self).__init__()
        self.in_channels = in_channels
        
        self.num_iter = num_iters
        
    
    def forward(self,x,num_centroids):
        """
        实现模糊C均值(FCM)聚类算法
        Inputs:
            x: 输入特征 [B, C, H, W]
            num_centroids: 聚类中心数量
        Outputs:
            centroids: (B,C,num_centroids)
            weights: (B,H*W,1,num_centroids) soft assignment of each node to centroids
        """

        b,c,h,w = x.shape
        x_copy = x.reshape(b,c,h,w)
        x = x.reshape(b,h*w,c)  # 展平为点云格式 [B, N, C]
        m = 2 # 🎯 模糊系数，控制聚类的模糊程度

        with torch.no_grad():
            # 1. 随机初始化聚类中心
            centroids = torch.randn((b, c, num_centroids), device=x.device, dtype=x.dtype)
            # 2. 🎯 模糊聚类迭代过程
            for i in range(self.num_iter):
                # 计算每个点到聚类中心的距离
                dist_to_centers = torch.cdist(x, centroids.transpose(1, 2))
                # 🎯 模糊隶属度计算 (FCM核心公式)
                inv_dist = 1.0 / (dist_to_centers + 1e-10)
                power = 2 / (m - 1)
                membership = (inv_dist / inv_dist.sum(dim=-1, keepdim=True)).pow(power)
                # 🎯 更新聚类中心 (加权平均)
                weights = membership.pow(m).unsqueeze(2)
                centroids = torch.sum(weights * x.unsqueeze(-1), dim=1) / weights.sum(dim=1)
            # 返回聚类中心和隶属度权重
            return centroids,weights
            
        
def pairwise_distance(x):
    """
    Compute pairwise distance of a point cloud.
    Args:
        x: tensor (batch_size, num_points, num_dims)
    Returns:
        pairwise distance: (batch_size, num_points, num_points)
    """
    with torch.no_grad():
        x_inner = -2*torch.matmul(x, x.transpose(2, 1))
        x_square = torch.sum(torch.mul(x, x), dim=-1, keepdim=True)
        return x_square + x_inner + x_square.transpose(2, 1)


def part_pairwise_distance(x, start_idx=0, end_idx=1):
    """
    Compute pairwise distance of a point cloud.
    Args:
        x: tensor (batch_size, num_points, num_dims)
    Returns:
        pairwise distance: (batch_size, num_points, num_points)
    """
    with torch.no_grad():
        x_part = x[:, start_idx:end_idx]
        x_square_part = torch.sum(torch.mul(x_part, x_part), dim=-1, keepdim=True)
        x_inner = -2*torch.matmul(x_part, x.transpose(2, 1))
        x_square = torch.sum(torch.mul(x, x), dim=-1, keepdim=True)
        return x_square_part + x_inner + x_square.transpose(2, 1)


def xy_pairwise_distance(x, y):
    """
    Compute pairwise distance of a point cloud.
    Args:
        x: tensor (batch_size, num_points, num_dims)
    Returns:
        pairwise distance: (batch_size, num_points, num_points)
    """
    with torch.no_grad():
        xy_inner = -2*torch.matmul(x, y.transpose(2, 1))
        x_square = torch.sum(torch.mul(x, x), dim=-1, keepdim=True)
        y_square = torch.sum(torch.mul(y, y), dim=-1, keepdim=True)
        return x_square + xy_inner + y_square.transpose(2, 1)


def dense_knn_matrix(x, k=16, relative_pos=None):
    """Get KNN based on the pairwise distance.
    Args:
        x: (batch_size, num_dims, num_points, 1)
        k: int
    Returns:
        nearest neighbors: (batch_size, num_points, k) (batch_size, num_points, k)
    """
    with torch.no_grad():
        x = x.transpose(2, 1).squeeze(-1)
        batch_size, n_points, n_dims = x.shape
        ### memory efficient implementation ###
        n_part = 10000
        if n_points > n_part:
            nn_idx_list = []
            groups = math.ceil(n_points / n_part)
            for i in range(groups):
                start_idx = n_part * i
                end_idx = min(n_points, n_part * (i + 1))
                dist = part_pairwise_distance(x.detach(), start_idx, end_idx)
                if relative_pos is not None:
                    dist += relative_pos[:, start_idx:end_idx]
                _, nn_idx_part = torch.topk(-dist, k=k)
                nn_idx_list += [nn_idx_part]
            nn_idx = torch.cat(nn_idx_list, dim=1)
        else:
            dist = pairwise_distance(x.detach())
            if relative_pos is not None:
                dist += relative_pos
            _, nn_idx = torch.topk(-dist, k=k) # b, n, k
        ######
        center_idx = torch.arange(0, n_points, device=x.device).repeat(batch_size, k, 1).transpose(2, 1)
    return torch.stack((nn_idx, center_idx), dim=0)


def xy_dense_knn_matrix(x, y, k=16, relative_pos=None):
    """Get KNN based on the pairwise distance.
    Args:
        x: (batch_size, num_dims, num_points, 1)
        k: int
    Returns:
        nearest neighbors: (batch_size, num_points, k) (batch_size, num_points, k)
    """
    with torch.no_grad():
        x = x.transpose(2, 1).squeeze(-1)
        y = y.transpose(2, 1).squeeze(-1)
        batch_size, n_points, n_dims = x.shape
        """
           计算x和y之间的成对欧氏距离
           dist = ||x||² - 2x·yᵀ + ||y||²
       """
        dist = xy_pairwise_distance(x.detach(), y.detach())
        if relative_pos is not None:
            dist += relative_pos
        _, nn_idx = torch.topk(-dist, k=k)
        center_idx = torch.arange(0, n_points, device=x.device).repeat(batch_size, k, 1).transpose(2, 1)
    return torch.stack((nn_idx, center_idx), dim=0)


class DenseDilated(nn.Module):
    """
    膨胀采样公式
    edge_index: (2, batch_size, num_points, k)
    """
    def __init__(self, k=9, dilation=1, stochastic=False, epsilon=0.0):
        super(DenseDilated, self).__init__()
        self.dilation = dilation
        self.stochastic = stochastic
        self.epsilon = epsilon
        self.k = k

    def forward(self, edge_index):
        if self.stochastic:
            if torch.rand(1) < self.epsilon and self.training:
                # 随机采样
                num = self.k * self.dilation
                randnum = torch.randperm(num)[:self.k]
                edge_index = edge_index[:, :, :, randnum]
            else:
                # 规则膨胀采样
                edge_index = edge_index[:, :, :, ::self.dilation]
        else:
            # 先选择 k*dilation 个邻居，然后每隔dilation个采样
            """
                节点A的10个最近邻: [A, B, E, D, C, ...]  ← 按距离排序
                索引位置: [0, 1, 2, 3, 4, 5]
            """
            edge_index = edge_index[:, :, :, ::self.dilation]
        return edge_index

""" 
    使用k近邻和膨胀策略构建图结构 
    目标: 为每个节点选择K个最相似的邻居，使用膨胀采样扩大感受野
    输入: 节点特征 [B, C, N, 1]
    输出: 邻居索引 [2, B, N, k]
"""
class DenseDilatedKnnGraph(nn.Module):
    """
    Find the neighbors' indices based on dilated knn
    """
    def __init__(self, k=9, dilation=1, stochastic=False, epsilon=0.0):
        super(DenseDilatedKnnGraph, self).__init__()
        # dilation
        # 膨胀率控制（ idx:backbone的Seq层数
        # idx=0-3:  膨胀率=1，
        # idx=4-7:  膨胀率=2，
        # idx=8-11: 膨胀率=3...以此类推，
        # 但不超过max_dilation）
        self.dilation = dilation
        self.stochastic = stochastic
        self.epsilon = epsilon
        self.k = k  # 传入的k值
        self._dilated = DenseDilated(k, dilation, stochastic, epsilon)

    def forward(self, x, y=None, relative_pos=None):
        # 第一步：选择 k * dilation 个候选邻居
        if y is not None:
            #### normalize
            """
            1、特征归一化：效果: 所有特征向量长度变为1，只保留方向信息
                归一化前: [1.2, 2.3, 0.8] → 长度 =√(1.2²+2.3²+0.8²) = 2.7
                归一化后: [0.44, 0.85, 0.29] → 长度 = 1
            """
            x = F.normalize(x, p=2.0, dim=1) # L2归一化
            y = F.normalize(y, p=2.0, dim=1)
            """
                2、计算距离矩阵
                计算距离矩阵：dist(i,j) = ||x_i||² - 2·x_i·x_jᵀ + ||x_j||²
                节点: A, B, C, D, E
                距离矩阵:
                    A   B   C   D   E
                A   0   0.2 0.8 0.5 0.3
                B   0.2 0   0.7 0.6 0.4
                C   0.8 0.7 0   0.9 0.7
                D   0.5 0.6 0.9 0   0.8
                E   0.3 0.4 0.7 0.8 0
                3、选择K个最近邻
                    假设 k=3, 为每个节点选择距离最小的3个邻居
                    节点A的邻居: [A(0), B(0.2), E(0.3)]  ← 距离从小到大
                    节点B的邻居: [B(0), A(0.2), E(0.4)]
                    节点C的邻居: [C(0), B(0.7), E(0.7)]
                    ...
            """
            edge_index = xy_dense_knn_matrix(x, y, self.k * self.dilation, relative_pos)
        else:
            #### normalize
            x = F.normalize(x, p=2.0, dim=1)
            ####
            edge_index = dense_knn_matrix(x, self.k * self.dilation, relative_pos)
            """
                实际邻居选择过程:
                    模块阶段	k值	dilation值	候选邻居数	最终邻居数
                    第1阶段	10	1	            10×1=10	10
                    第2阶段	10	2	            10×2=20	10
                    第3阶段	10	3	            10×3=30	10
                对于 dilation=2 的模块:
                    候选邻居（按距离排序）: [A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T]
                    索引位置:              0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,11,12,13,14,15,16,17,18,19
                    膨胀采样（每隔2个选1个）: 0, 2, 4, 6, 8, 10, 12, 14, 16, 18
                    对应邻居:               A, C, E, G, I, K,  M,  O,  Q,  S
            """
        # 第二步：膨胀采样，每隔 dilation 个选一个
        return self._dilated(edge_index)
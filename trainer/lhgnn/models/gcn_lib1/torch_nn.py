# 2022.06.17-Changed for building ViG model
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
import torch
from torch import nn
from torch.nn import Sequential as Seq, Linear as Lin, Conv2d
from timm.models.layers import DropPath

""" 各种基础卷积、归一化、激活函数等 """
##############################
#    Basic layers
##############################
def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
    act = act.lower()
    if act == 'relu':
        """
            ReLU(x)=max(0,x)
            主流场景：CNN（卷积神经网络）、MLP（多层感知机）的隐藏层
            经典模型：AlexNet（2012 年 ImageNet 冠军）首次大规模验证其有效性，成为深度学习标配
            适合：数据量充足、网络较深的模型（梯度稳定）
        """
        layer = nn.ReLU(inplace)
    elif act == 'leakyrelu':
        """
        LeakyReLU(x) = { x  x≥0,  
                         αx x<0
                        },
            其中(α)是固定超参数（通常取 0.01，即负区间保留 1% 的信号）
            形状：分段线性，负区间有小斜率（α)
            正区间：同 ReLU（斜率 = 1）
            负区间：斜率为(α)的直线（例(α=0.01)时，(y=0.01x)）
            应用场景
                解决死亡 ReLU：适用于 ReLU 训练中出现大量神经元失活的场景
                推荐模型：GAN（生成对抗网络）的生成器层、RNN 隐藏层
                适合：负区间仍需保留微弱梯度的模型
        """
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act == 'gelu':
        """
            计算复杂度：高（近似含指数 / 三角）	
            GELU(x)=x⋅Φ(x)，其中Φ(x)是标准正态分布的累积分布函数（CDF），计算复杂。
            形状：平滑的 S 形曲线，介于 ReLU 和 Sigmoid 之间
            关键特性：均值接近 0（输出对称分布）
                    负区间非零（但衰减快）
                    梯度连续且平滑（无拐点）
                    梯度：在(x=0)处梯度最大，向两侧逐渐衰减
            应用场景
                    Transformer 模型：BERT、GPT、T5 等大语言模型的核心激活函数
                    NLP 任务：文本分类、机器翻译、问答系统等
                    需要高精度的场景：模型容量大、追求极限性能的任务
        """
        layer = nn.GELU()
    elif act == 'hswish':
        layer = nn.Hardswish(inplace)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act)
    return layer


def norm_layer(norm, nc):
    # normalization layer 2d
    norm = norm.lower()
    if norm == 'batch':
        layer = nn.BatchNorm2d(nc, affine=True)
    elif norm == 'instance':
        layer = nn.InstanceNorm2d(nc, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm)
    return layer


class MLP(Seq):
    def __init__(self, channels, act='relu', norm=None, bias=True):
        m = []
        for i in range(1, len(channels)):
            m.append(Lin(channels[i - 1], channels[i], bias))
            if act is not None and act.lower() != 'none':
                m.append(act_layer(act))
            if norm is not None and norm.lower() != 'none':
                m.append(norm_layer(norm, channels[-1]))
        super(MLP, self).__init__(*m)


class BasicConv(Seq):
    def __init__(self, channels, act='relu', norm=None, bias=True, drop=0.):
        m = []
        for i in range(1, len(channels)):
            m.append(Conv2d(channels[i - 1], channels[i], 1, bias=bias, groups=4))
            if norm is not None and norm.lower() != 'none':
                m.append(norm_layer(norm, channels[-1]))
            if act is not None and act.lower() != 'none':
                m.append(act_layer(act))
            if drop > 0:
                m.append(nn.Dropout2d(drop))

        super(BasicConv, self).__init__(*m)

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def batched_index_select(x, idx):
    r"""fetches neighbors features from a given neighbor idx

    Args:
        x (Tensor): input feature Tensor
                :math:`\mathbf{X} \in \mathbb{R}^{B \times C \times N \times 1}`.
        idx (Tensor): edge_idx
                :math:`\mathbf{X} \in \mathbb{R}^{B \times N \times l}`.
    Returns:
        Tensor: output neighbors features
            :math:`\mathbf{X} \in \mathbb{R}^{B \times C \times N \times k}`.
    """
    batch_size, num_dims, num_vertices_reduced = x.shape[:3]
    _, num_vertices, k = idx.shape
    idx_base = torch.arange(0, batch_size, device=idx.device).view(-1, 1, 1) * num_vertices_reduced
    idx = idx + idx_base
    idx = idx.contiguous().view(-1)

    x = x.transpose(2, 1)
    feature = x.contiguous().view(batch_size * num_vertices_reduced, -1)[idx, :]
    feature = feature.view(batch_size, num_vertices, k, num_dims).permute(0, 3, 1, 2).contiguous()
    return feature

class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act='relu',drop_path=0.0):
        super().__init__()
        out_features = out_features if out_features is not None else in_features
        hidden_features = hidden_features if hidden_features is not None else in_features
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.act = act_layer(act)
        self.fc1 = Seq(nn.Conv2d(in_features,hidden_features,1,stride=1,bias=False,padding=0),
                          nn.BatchNorm2d(hidden_features))
        self.fc2 = Seq(nn.Conv2d(hidden_features,out_features,1,stride=1,bias=False,padding=0),
                          nn.BatchNorm2d(out_features))
        
    def forward(self, x):
        shortcut = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop_path(x) + shortcut
       
        return x
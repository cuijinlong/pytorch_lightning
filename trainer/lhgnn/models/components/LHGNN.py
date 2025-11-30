import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq
from trainer.lhgnn.models.gcn_lib1.torch_vertex import Grapher
from trainer.lhgnn.models.gcn_lib1.torch_nn import act_layer, norm_layer, MLP, BasicConv
from trainer.lhgnn.models.utils.model_utils import FFN,ResDWC,ConvFFN,Stem_conv,DownSample

def init_layer(layer):
    if layer.weight.ndimension() == 4:
        (n_out, n_in, height, width) = layer.weight.size()
        n = n_in * height * width
    elif layer.weight.ndimension() == 2:
        (n_out, n) = layer.weight.size()

    std = math.sqrt(2. / n)
    scale = std * math.sqrt(3.)
    layer.weight.data.uniform_(-scale, scale)

    if layer.bias is not None:
        layer.bias.data.fill_(0.)

def init_bn(bn):
    bn.weight.data.fill_(1.)

class LHGNN(nn.Module):
    # trainer/lhgnn/models/tagging_module.py -> net
    # net -> trainer/lhgnn/configs/model/LHGNN.yaml
    def __init__(self,
                 # --------------------------------------- 模型规模大小 ---------------------------------------------------------
                 size='s',  # small、media、big
                 # --------------------------------------- 图结构特性 ---------------------------------------------------------
                 k=10, # 每个节点的邻居数量，控制图的稀疏程度，值越大感受野越大
                 conv='mr', # 选择图卷积类型（'mr': Max-Relative图卷积 | 'edge': 边卷积 | 'sage': GraphSAGE卷积 | 'gin': 图同构网络卷积 | 'lhg': 超图卷积（使用聚类中心））
                 clusters=10,  # 作用: 超图卷积中的聚类中心数量，仅当 conv='lhg' 时生效
                 cluster_ratio=0.4,  # 作用: 控制从聚类中心选择的top-k比例，top_clusters = ceil(k * cluster_ratio)
                 epsilon=0.2, # 用于DenseDilatedKnnGraph中的随机性，控制随机选择邻居的概率
                 act='gelu', # 激活函数类型（'relu' | 'gelu' | 'leakyrelu' | 'prelu'等）
                 norm='batch', # 归一化层类型，如'batch','instance'等。
                 bias=True, # 是否在卷积层中使用偏置。
                 # --------------------------------------- 图结构特性 ---------------------------------------------------------
                 dropout=0.0, # 丢弃率，用于预测层中的Dropout。
                 drop_path=0.1, # DropPath的丢弃率，用于随机深度。
                 dilation=True,# 作用: 是否在图卷积中使用膨胀采样,效果: 增加感受野而不增加参数
                 # --------------------------------------- 输入输出规格 ---------------------------------------------------------
                 num_class=10, # 作用: 类任务的类别数量，输出维度: [batch_size, num_class]
                 emb_dims=1024, # 作用: 最终嵌入维度（prediction层中间维度）
                 freq_num=128, # 作用: 输入频谱图的频率轴大小
                 time_num=256, # 作用: 输入频谱图的时间轴大小
                 ):
        super(LHGNN,self).__init__()
        if size == 's':
            self.blocks = [2, 2, 6, 2] # 每个stage的Grapher模块数量
            #self.channels = [64, 128, 320, 512]
            self.channels = [80, 160, 400, 640] # 每个阶段的通道数，根据size设置。
            self.emb_dims = 1024
        elif size == 'm':
            self.blocks = [2,2,16,2] # 每个stage的Grapher模块数量
            self.channels = [96, 192, 384, 768] # 每个阶段的通道数，根据size设置。
            self.emb_dims = 1024
        else:
            self.blocks = [2,2,18,2]
            self.channels = [128, 256, 512, 1024]
        self.k = int(k) # 每个节点的邻居数量，控制图的稀疏程度，值越大感受野越大
        self.conv = conv # 选择图卷积类型（'mr': Max-Relative图卷积 | 'edge': 边卷积 | 'sage': GraphSAGE卷积 | 'gin': 图同构网络卷积 | 'lhg': 超图卷积（使用聚类中心））
        self.act = act # 激活函数类型（'relu' | 'gelu' | 'leakyrelu' | 'prelu'等）
        self.norm = norm # 归一化层类型，如'batch','instance'等。
        self.bias = bias # 是否在卷积层中使用偏置。
        self.drop_path = drop_path # DropPath的丢弃率，用于随机深度。
        self.num_class = num_class # 作用: 类任务的类别数量，输出维度: [batch_size, num_class]
        self.emb_dims = emb_dims # 作用: 最终嵌入维度（prediction层中间维度）
        self.freq_num = freq_num # 作用: 输入频谱图的频率轴大小
        self.time_num = time_num # 作用: 输入频谱图的时间轴大小
        self.epsilon = epsilon # 用于DenseDilatedKnnGraph中的随机性，控制随机选择邻居的概率
        self.dilation = dilation # 作用: 是否在图卷积中使用膨胀采样,效果: 增加感受野而不增加参数
        self.dropout = dropout # 丢弃率，用于预测层中的Dropout。
        stochastic = False # 作用: 是否在KNN中使用随机采样
        self.num_blocks = sum(self.blocks)
        self.cluster_ratio = cluster_ratio # 控制从聚类中心选择的top-k比例
        if conv == 'lhg':
            channel_mul = 3 # 因为LHGConv会concat 3种特征
        else:
            channel_mul = 2 # 其他卷积concat 2种特征
        reduce_ratios = [4,2,1,1] # 作用: 每个stage的下采样比率，影响: 控制每个阶段的下采样程度，早期下采样更多。
        num_clusters = [int (x.item()) for x in torch.linspace(clusters,clusters,self.num_blocks)] # 每个块使用的聚类数，这里每个块都固定为clusters。
        num_knn = [int(x.item()) for x in torch.linspace(k,k,self.num_blocks)] # 每个块使用的knn数，这里每个块都固定为k。
        max_dilation = 128//max(num_clusters) # 最大膨胀系数，根据聚类数计算。
        self.stem = Stem_conv(1, self.channels[0], act=act)  # 这个 Stem 模块相当于传统 CNN 中的"特征金字塔"的底层
        self.pos_embed = nn.Parameter(torch.zeros(1, self.channels[0], freq_num//4, time_num//4))
        self.HW = freq_num//4 * time_num//4  # Num nodes after the stem bloc
        # dpr为每个模块分配不同的drop_path率，深层模块丢弃概率更高。
        dpr = [x.item() for x in torch.linspace(0, drop_path, self.num_blocks)]
        # 初始化模型主干（backbone），主干由多个阶段（stage）组成，每个阶段包含一个下采样模块（DownSample）和多个Grapher模块（每个Grapher模块后面跟着一个ConvFFN模块）
        self.backbone = nn.ModuleList([])
        idx = 0
        for i in range(len(self.blocks)):
            if i > 0:
                self.backbone.append(DownSample(self.channels[i-1], self.channels[i]))
                self.HW = self.HW // 4
            for j in range(self.blocks[i]):
                self.backbone += [
                    Seq(Grapher(self.channels[i],
                                num_knn[idx],
                                num_clusters[idx],
                                min(idx // 4 + 1, max_dilation), # 膨胀率控制（idx=0-3: 膨胀率=1，idx=4-7: 膨胀率=2，idx=8-11: 膨胀率=3...以此类推，但不超过max_dilation）
                                self.conv,
                                self.act,
                                self.norm,
                                self.bias,
                                stochastic,
                                epsilon,
                                reduce_ratios[i],
                                n = self.HW,
                                drop_path = dpr[idx],
                                relative_pos = True,
                                cluster_ratio = self.cluster_ratio,
                                channel_mul = channel_mul),
                                ConvFFN(in_features = self.channels[i],  # 输入通道
                                        hidden_features = self.channels[i] * 4, # 隐藏层=4倍通道
                                        out_features = self.channels[i], # 输出通道不变
                                        act = act, # 激活函数
                                        drop_path = dpr[idx] # 相同的随机深度
                                        )
                         )]
                idx += 1
        self.backbone = Seq(*self.backbone)
        # 预测层：将特征映射转换为分类结果 self.channels[-1]：获取数值中最后一个数
        self.prediction = Seq(nn.Conv2d(self.channels[-1],
                                        1024,
                                        1,
                                        bias=True),
                              nn.BatchNorm2d(1024),
                              act_layer(act),
                              nn.Dropout(self.dropout),
                              nn.Conv2d(1024,
                                        self.num_class,
                                        1,
                                        bias=True))
        # 初始化模型权重
        self.model_init()
    
    def model_init(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True
                
    def forward(self,inputs):

        """ 整体流程： 输入 → Stem卷积 → 位置编码 → 多阶段GNN Backbone → 全局池化 → 分类输出 """

        """ 1.输入处理 """
        # 增加一个维度，从[Batch, Freq, Time] 到 [Batch, 1, Freq, Time]
        inputs = inputs.unsqueeze(1)
        # 变为[Batch, 1, Time, Freq] 注意：在音频中，通常是[Batch, Freq, Time]，这里转置后变 [Batch, 1, Time, Freq]，但后续的stem卷积会调整通道和尺寸。
        inputs = inputs.transpose(2,3)

        """ 2.stem卷积 """
        x = self.stem(inputs) + self.pos_embed
        # Batch, Chennel, Heigth, Width
        B, C, H, W = x.shape

        """ 3.经过backbone（多个Grapher和DownSample）"""
        for i in range(len(self.backbone)):
            x = self.backbone[i](x)

        """ 4.全局平均池化 """
        x = F.adaptive_avg_pool2d(x, 1)

        """ 5.预测层 """
        x = self.prediction(x)
            
        #preds = torch.sigmoid(x)
        preds = x.squeeze(-1).squeeze(-1)

        return preds
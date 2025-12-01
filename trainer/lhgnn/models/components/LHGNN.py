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
"""
    图卷积架构-调用链
    LHGNN → torch_vertex.Grapher → DyGraphConv2d → GraphConv2d → LHGConv2d/MRConv2d
"""
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
                 num_classes=10, # 作用: 类任务的类别数量，输出维度: [batch_size, num_class]
                 emb_dims=1024, # 作用: 最终嵌入维度（prediction层中间维度）
                 freq_num=64, # 作用: 输入频谱图的频率轴大小
                 time_num=256, # 作用: 输入频谱图的时间轴大小
                 ):
        super(LHGNN,self).__init__()
        if size == 's':
            self.blocks = [2, 2, 6, 2] #  # 各stage模块数量，每个stage的Grapher模块数量
            #self.channels = [64, 128, 320, 512]
            self.channels = [80, 160, 400, 640] # 通道数渐进增加，每个阶段的通道数，根据size设置。
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
        self.num_classes = num_classes # 作用: 类任务的类别数量，输出维度: [batch_size, num_class]
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
        # 方法A：
        num_clusters = [clusters] * self.num_blocks # 每个块使用的聚类数，这里每个块都固定为clusters。
        # 方法B：实现真正的渐进变化（如果需要）
        # num_clusters = [
        #     int(torch.linspace(clusters // 2, clusters * 2, self.num_blocks)[i].item())
        #     for i in range(self.num_blocks)
        # ]
        num_knn = [k] * self.num_blocks # 每个块使用的knn数，这里每个块都固定为k。
        max_dilation = 128//max(num_clusters) # 最大膨胀系数，根据聚类数计算。
        """
            Stem_conv 通俗解释：
            原来：一大袋面粉（原始信号：64行×256列 = 16384个像素点） 
                原始频谱图：
                      64×256个点，每个点只有1个值（灰度）
                      [■■■■■■■■■■■■] 64行
                      [■■■■■■■■■■■■] 
                      ... (共256列)
            现在：一小团调好味的面糊（高级特征：变成了80种调料的混合（80个特征通道） 尺寸缩小到16×64 = 1024个像素点）
                16×64个点，每个点有80个特征
                      [■■] → 这不再是一个灰度值，而是
                              包含80个数字的向量：
                              [0.2, 0.8, -0.1, ..., 0.5]
                              这个向量包含：
                                  1. 频率特征
                                  2. 时间特征  
                                  3. 纹理特征
                                  4. 边缘特征
                                  ... (共80种)
            作用：快速下采样，提取低级特征（边缘、纹理等）
            输入: [B, 1, 1024, 128] → 输出: [B, 80, 256, 32] (分辨率变为1/4)
            3层卷积结构：两次下采样 + 一次特征提取
            效果：将空间分辨率降至1/4，为图神经网络准备合适尺寸的输入
        """
        self.stem = Stem_conv(1, self.channels[0], act=act)  # 这个 Stem 模块相当于传统 CNN 中的"特征金字塔"的底层
        self.pos_embed = nn.Parameter(torch.zeros(1, self.channels[0], freq_num//4, time_num//4))
        self.HW = freq_num//4 * time_num//4  # Num nodes after the stem bloc
        # dpr为每个模块分配不同的drop_path率，深层模块丢弃概率更高。
        dpr = [x.item() for x in torch.linspace(0, drop_path, self.num_blocks)]
        # 初始化模型主干（backbone），主干由多个阶段（stage）组成，每个阶段包含一个下采样模块（DownSample）和多个Grapher模块（每个Grapher模块后面跟着一个ConvFFN模块）
        self.backbone = nn.ModuleList([])
        idx = 0
        for i in range(len(self.blocks)):
            if i > 0: # 阶段间下采样
                self.backbone.append(DownSample(self.channels[i-1], self.channels[i]))
                self.HW = self.HW // 4
            for j in range(self.blocks[i]):
                """
                每个模块 = Grapher + ConvFFN，其中：
                    Grapher: 图卷积核心
                        位于 (Grapher in torch_vertex.py)
                        结构: fc1 -> graph_conv -> fc2 -> drop_path -> 残差连接
                        fc1/fc2: 1×1卷积 + BatchNorm（通道变换）
                        graph_conv: 动态图卷积，支持多种卷积类型
                        DropPath: 随机深度机制，增强泛化能力
                    ConvFFN: 前馈网络  
                        结构: 1×1卷积 → 激活 → 深度可分离卷积 → 1×1卷积
                        隐藏层通道数 = 输入通道 × 4
                        位于每个Grapher之后
                        增强非线性表达能力
                        包含残差连接
                输入: [Batch, 80, H, W] Stem输出特征
                输出: [Batch, 640, H/32, W/32] 深层特征图 (todo 32)
                关键技术点:
                    渐进式通道扩展: 80→160→400→640
                    分层下采样: 空间分辨率逐步降低
                    模块化设计: 每个stage包含多个(Grapher + ConvFFN)单元
                """
                self.backbone += [
                    Seq(Grapher(self.channels[i],
                                num_knn[idx],
                                num_clusters[idx],
                                # dilation
                                   # 膨胀率控制（
                                   # idx=0-3: 膨胀率=1，
                                   # idx=4-7: 膨胀率=2，
                                   # idx=8-11: 膨胀率=3...以此类推，
                                   # 但不超过max_dilation）
                                min(idx // 4 + 1, max_dilation),
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
        self.prediction = Seq(
                              # 通道扩展
                              nn.Conv2d(self.channels[-1],
                                        1024,
                                        1,
                                        bias=True),
                              nn.BatchNorm2d(1024),
                              act_layer(act),
                              nn.Dropout(self.dropout),
                              # 最终分类
                              nn.Conv2d(1024,
                                        self.num_classes,
                                        1,
                                        bias=True))
        # 初始化模型权重
        self.model_init()
    
    def model_init(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                # 使用更稳定的初始化
                torch.nn.init.kaiming_normal_(m.weight)
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True
                
    def forward(self,inputs):

        """ 整体流程： 输入 → Stem卷积 → 位置编码 → 多阶段GNN Backbone → 全局池化 → 分类输出 """

        """ 
            节点1：输入预处理(维度调整)
                [Batch, Freq, Time] 原始音频频谱图
                [Batch, 1, Time, Freq] 调整后的4D张量
                关键技术点:
                    增加通道维度适配卷积网络
                    转置操作确保时间、频率轴正确对应卷积核
        """
        # 步骤1.1: 增加通道维度 shape：torch.Size([8, 256, 64])
        inputs = inputs.unsqueeze(1) # [Batch, Freq, Time] → [Batch, 1, Freq, Time]
        # 步骤1.2: 转置适配卷积操作 shape：torch.Size([8, 1, 64, 256])
        inputs = inputs.transpose(2,3) # [Batch, 1, Freq, Time] → [Batch, 1, Time, Freq]

        """ 
            节点2：Stem卷积特征提取(快速下采样)
                位置: model_utils.py → Stem_conv 类
                输入: [Batch, 1, Time, Freq] 预处理后的频谱图
                输出: [Batch, 80, Time/4, Freq/4] 下采样特征图
                关键技术点:
                    两次下采样: 空间分辨率降至1/4
                    通道扩展: 1→40→80通道
                    GELU激活: 更平滑的非线性变换
                特征金字塔构建
                Stem 卷积：作为模型的 “特征金字塔底层”，使用 1×1 卷积将输入通道
                          从 1 映射到第一个 stage 的通道数（如 80），同时通过下采样将空间尺寸
                          缩小为原来的 1/4（如 freq_num//4 * time_num//4），初步提取低频特征。
                位置编码：通过可学习参数 pos_embed 为特征添加位置信息（形状与 Stem 输出一致），
                        帮助模型感知时空位置关系，类似 Transformer 的位置编码。
                输出: [Batch, C0, H/4, W/4] 分辨率: 1/4, 通道: 80
        """
        x = self.stem(inputs) + self.pos_embed # inputs：torch.Size([8, 1, 64, 256]) pos_embed：torch.Size([1, 80, 16, 64]) x：torch.Size([8, 80, 16, 64])
        """ 
            节点3：位置编码融合(todo位置编码)
                输入: 
                    Stem输出: [Batch, 80, H, W]
                    位置编码: [1, 80, H, W] (可学习参数)
                输出: [Batch, 80, H, W] 带位置信息的特征图
                关键技术点:
                    可学习位置编码: 类似Transformer的位置感知
                    逐元素相加: 简单有效的融合方式
        """
        # Batch, Chennel, Heigth, Width
        B, C, H, W = x.shape
        """ 
            节点4：多阶段GNN Backbone
                输入: [B,80,H,W]
                输出: [B,640,H/32,W/32]
            渐进式特征抽象:
                分辨率递减：1024 → 512 → 256 → 128 → 64像素
                通道递增：80 → 160 → 400 → 640通道
                感受野递增：局部 → 中等 → 全局
                语义层次递增：边缘纹理 → 部件 → 物体 → 语义
        """
        for i in range(len(self.backbone)): # [2, 2, 6, 2]
            x = self.backbone[i](x) # x: [8, 80, 16, 64]
        """
            节点5：Grapher图卷积模块(动态图卷积)
                位置: torch_vertex.py → Grapher 类
                输入: [Batch, C, H, W] 当前stage的特征图
                输出: [Batch, C, H, W] 图卷积处理后的特征图
                调用：节点6_DyGraphConv2d(“节点6”、“节点7” 属于 “节点5”)
                关键技术点:
                    动态图构建: 每层根据特征动态计算KNN图
                    超图卷积: 使用模糊聚类生成超边
                    残差连接: 缓解梯度消失
                    
            节点6：动态图卷积核心(KNN图构建)
                位置: torch_vertex.py → DyGraphConv2d 类
                输入: [Batch, C, H, W] 空间特征图
                输出: [Batch, C*channel_mul, H, W] 图卷积输出
                调用： DenseDilatedKnnGraph(torch_edge.py) -> GraphConv2d -> 节点7_LHGConv2d
                关键技术点:
                    节点化处理: 将空间位置视为图节点
                    膨胀KNN: 扩大感受野而不增加参数
                    多尺度图: 通过下采样构建层次化图结构
                    
            节点7：超图卷积实现(超图三特征融合)
                位置: torch_vertex.py → LHGConv2d 类
                输入: [Batch, C, N, 1] 节点特征
                输出: [Batch, C, N, 1] 增强后的节点特征
                调用：HyperedgeConstruction(torch_edge.py)
                关键技术点:
                    模糊聚类: 使用FCM算法生成超边
                    三特征融合: [自身特征, 邻居特征, 聚类特征]
                    软分配: 每个节点以不同程度属于多个聚类
                    
            节点8：ConvFFN前馈网络
                位置: model_utils.py → ConvFFN 类
                输入: [Batch, C, H, W] Grapher输出
                输出: [Batch, C, H, W] 非线性变换后的特征
                关键技术点:
                    通道扩展: 隐藏层通道数=4×输入通道
                    深度卷积: 保持空间关系的轻量卷积
                    残差学习: 稳定深层网络训练
        """

        """ 节点9：全局池化与分类输出
                全局池化：通过 adaptive_avg_pool2d 将最终特征图压缩为 [Batch, C, 1, 1]，聚合全局信息。
                分类层：由两个 1×1 卷积组成，先将通道映射到 1024 维，经激活和 Dropout 后，最终输出 [Batch, num_class] 的分类结果。 
            输入: [Batch, 640, H, W] Backbone输出
            输出: [Batch, 640, 1, 1] 全局特征向量
            关键技术点:
                自适应池化: 无论输入尺寸如何，输出固定尺寸
                全局信息聚合: 将空间信息压缩为通道描述符
        """
        x = F.adaptive_avg_pool2d(x, 1) #  [B,640,H,W] → [B,640,1,1]

        """ 节点10：分类预测层 
            输入: [Batch, 640, 1, 1] 池化后的全局特征
            输出: [Batch, num_class] 分类logits
            关键技术点:
                1×1卷积: 等效全连接但参数更少
                通道扩展: 640→1024增强表示能力
                Dropout: 正则化防止过拟合
        """
        x = self.prediction(x) # [B,640,1,1] → [B,num_classes,1,1]
        #preds = torch.sigmoid(x)
        preds = x.squeeze(-1).squeeze(-1) # [B,num_classes,1,1] → [B,num_classes]
        return preds
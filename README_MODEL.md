核心工具函数调用
函数	                            位置	            调用者	                    功能
batched_index_select	    torch_nn.py	        LHGConv2d, MRConv2d	        批量索引选择邻居特征
act_layer	                torch_nn.py	        所有模块	                    动态创建激活函数
norm_layer	                torch_nn.py	        所有模块	                    动态创建归一化层
get_2d_relative_pos_embed	pos_embed.py	    Grapher	                    生成相对位置编码


LHGNN (主控制器)
    ├── Stem_conv (特征金字塔底层)
    ├── Grapher (图卷积核心)
    │   ├── DyGraphConv2d (动态图构建)
    │   │   ├── DenseDilatedKnnGraph (KNN图)
    │   │   └── LHGConv2d (超图卷积)
    │   │       └── HyperedgeConstruction (模糊聚类)
    │   └── BasicConv (特征变换)
    ├── ConvFFN (前馈网络)
    │   └── ResDWC (深度卷积)
    └── Prediction (分类器)


数据流调用顺序
    输入预处理 (LHGNN.forward)
    初始特征提取 (Stem_conv)
    位置编码融合 (LHGNN.forward)
    多阶段图卷积 (循环调用 Grapher + ConvFFN)
    全局信息聚合 (adaptive_avg_pool2d)
    分类预测 (Prediction)

关键技术调用点
    🎯 模糊聚类: LHGConv2d → HyperedgeConstruction.forward
    🎯 动态图构建: Grapher → DyGraphConv2d → DenseDilatedKnnGraph
    🎯 三特征融合: LHGConv2d.forward 中的 torch.cat
    🎯 随机深度: Grapher 和 ConvFFN 中的 DropPath
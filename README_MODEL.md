输入：[Batch, Freq, Time] 音频频谱图  
├─ 节点1：输入预处理  
│  ├─ 增加通道维度：unsqueeze(1) → [Batch, 1, Freq, Time]  
│  └─ 转置适配卷积：transpose(2,3) → [Batch, 1, Time, Freq]  
│  
├─ 节点2：Stem卷积（快速下采样）  
│  ├─ 3层卷积（含2次下采样）→ 分辨率降至1/4，通道扩展至80  
│  └─ 输出：[Batch, 80, Time/4, Freq/4]  
│  
├─ 节点3：位置编码融合  
│  ├─ 可学习位置编码：pos_embed [1, 80, H, W]  
│  └─ 逐元素相加 → [Batch, 80, H, W]（带位置信息的特征图）  
│  
├─ 节点4：多阶段GNN Backbone（核心模块链）  
│  ├─ 分4个stage，每个stage包含：  
│  │  ├─ 下采样模块（DownSample）：分辨率减半，通道数加倍（如80→160）  
│  │  └─ N个(Grapher + ConvFFN)单元（数量由blocks配置，如[2,2,6,2]）  
│  │  
│  ├─ 核心单元：Grapher（节点5）  
│  │  ├─ 步骤1：FC1（1×1卷积）→ 特征线性变换  
│  │  ├─ 步骤2：动态图卷积（节点6：DyGraphConv2d）  
│  │  │  ├─ 子步骤1：特征节点化 → [Batch, C, N, 1]（N=H×W）  
│  │  │  ├─ 子步骤2：KNN图构建（DenseDilatedKnnGraph）  
│  │  │  │  ├─ 特征归一化（L2归一化）  
│  │  │  │  ├─ 计算距离矩阵（成对欧氏距离）  
│  │  │  │  ├─ 选k×dilation个候选邻居（如k=10，dilation=2→20个）  
│  │  │  │  └─ 膨胀采样（每隔dilation选1个，最终保留k个）  
│  │  │  │  
│  │  │  └─ 子步骤3：超图卷积（节点7：LHGConv2d，当conv='lhg'时）  
│  │  │     ├─ 超边构建（HyperedgeConstruction）：  
│  │  │     │  ├─ FCM模糊聚类：随机初始化中心→迭代计算隶属度→更新中心  
│  │  │     │  ├─ 输出：聚类中心centroids [B,C,num_clusters]、隶属度weights [B,N,1,num_clusters]  
│  │  │     │  └─ 筛选top_clusters（ceil(k×cluster_ratio)）  
│  │  │     │  
│  │  │     └─ 三特征融合：[自身特征, 邻居差值特征, 聚类中心差值特征] → 1×1卷积输出  
│  │  │  
│  │  ├─ 步骤3：FC2（1×1卷积）→ 恢复通道数  
│  │  └─ 步骤4：DropPath + 残差连接 → 输出[Batch, C, H, W]  
│  │  
│  └─ 核心单元：ConvFFN（节点8）  
│     ├─ 1×1卷积→通道扩展至4C→GELU激活→1×1卷积压缩回C  
│     └─ 残差连接 → 输出[Batch, C, H, W]  
│  
├─ 节点9：全局池化  
│  └─ adaptive_avg_pool2d → [Batch, 640, 1, 1]（聚合全局特征）  
│  
└─ 节点10：分类预测层  
   ├─ 1×1卷积→640→1024（激活+Dropout）→1×1卷积→num_class  
   └─ 输出：[Batch, num_class]（分类结果）


关键子模块细节补充
KNN 图构建（DenseDilatedKnnGraph）采用 “候选邻居筛选→膨胀采样” 策略，例如 dilation=2 时，从 20 个候选邻居中每隔 1 个选 1 个，最终保留 10 个，扩大感受野同时控制计算量。
超图构建（HyperedgeConstruction）基于 FCM 算法：通过模糊隶属度（每个节点以不同概率属于多个聚类）生成超边，将节点与全局聚类中心关联，增强长距离特征依赖建模。
多阶段设计通道数逐步扩展（80→160→400→640），分辨率逐步降低（1/4→1/8→1/32），平衡局部细节与全局语义提取。

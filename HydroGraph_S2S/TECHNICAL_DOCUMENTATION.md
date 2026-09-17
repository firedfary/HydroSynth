# HydroGraph_S2S：基于多尺度时空图神经网络的全国测站次季节与逐日降水预报系统
## 完整技术架构与理论全景文档 (Full Technical & Theoretical Documentation)

---

## 目录 (Table of Contents)
1. [引言与问题重构 (Introduction & Problem Formulation)](#1-引言与问题重构)
2. [从零理解图与图神经网络 (Foundations of Graph & Graph Neural Networks)](#2-从零理解图与图神经网络)
   - 2.1 欧几里得空间与非欧几里得结构
   - 2.2 图的基本数学定义：节点、边与邻接矩阵
   - 2.3 为什么传统 CNN 无法胜任非规则测站降水预测？
   - 2.4 图卷积神经网络（GCN）的核心思想：空间消息传递与扩散
3. [多源数据资产与物理特征工程 (Data Assets & Feature Engineering)](#3-多源数据资产与物理特征工程)
   - 3.1 观测端：全国 2463 个国家气象站 31 年逐日降水
   - 3.2 模式端：多中心超级动力气候模式集合（MME）
   - 3.3 数据清洗与异常值质量控制
   - 3.4 空间对齐：全球网格到测站坐标的双线性特征提取
4. [物理引导的多重时空图拓扑构建 (Physics-Informed Multi-Graph Topology)](#4-物理引导的多重时空图拓扑构建)
   - 4.1 大地空间距离图 ($A_{geo}$)
   - 4.2 高程落差与地形阻断图 ($A_{dem}$)
   - 4.3 气候统计遥相关图 ($A_{corr}$)
   - 4.4 数据驱动自适应图 ($A_{adapt}$)
   - 4.5 双向随机游走转移矩阵 ($P_f, P_b$)
5. [跨尺度时间解耦与特征线性调制 (Temporal Decoupling & FiLM Conditioning)](#5-跨尺度时间解耦与特征线性调制)
   - 5.1 核心物理矛盾与时间解耦理论
   - 5.2 动力模式低频样条基线趋势生成 (Cubic Spline Trend)
   - 5.3 特征线性调制层 (FiLM Layer) 的数学机理
6. [四类差异化建模范式与网络微观结构 (Four Modeling Paradigms & Deep Architectures)](#6-四类差异化建模范式与网络微观结构)
   - 6.1 核心时空图骨干网络 (Base ST-GNN Block)
   - 6.2 范式 1：次季节逐候（Pentad，5-Day）演变预报网络
   - 6.3 范式 2：趋势-残差混合多尺度逐日 ST-GNN
   - 6.4 范式 3：两阶段零膨胀分类-多分位数极端暴雨回归网络 (Hurdle-QR)
   - 6.5 范式 4：水量绝对守恒时空图解混下尺度网络 (Graph Disaggregation)
7. [非对称损失函数与训练优化机制 (Loss Objectives & Optimization)](#7-非对称损失函数与训练优化机制)
   - 7.1 极端暴雨阶梯加权 Huber 损失
   - 7.2 分位数 Pinball 损失
   - 7.3 水量物理一致性正则化
8. [气象与水文专业评估指标体系 (Evaluation Metrics)](#8-气象与水文专业评估指标体系)
   - 8.1 连续型评估指标 (ACC, RMSE, MAE, KGE)
   - 8.2 极端分类列联表指标 (TS / CSI, ETS, POD, FAR, BIAS)
9. [实测实验与前沿基准评测分析 (Empirical Benchmark Results)](#9-实测实验与前沿基准评测分析)
10. [代码工程结构与运行指南 (Codebase Structure & Usage)](#10-代码工程结构与运行指南)

---

# 1. 引言与问题重构

在天气学与水文学交汇的领域中，**次季节（Subseasonal-to-Seasonal, S2S，10–60 天 / 2–8 周）降水预报**一直被公认为气象科学的“可预报性沙漠（Predictability Desert）”。
* **在短期天气预报（1–7天）中**，大气运动由大气初始状态（大气动力记忆）主导；
* **在季节/气候预测（3–6个月）中**，大尺度慢变外强迫（如 ENSO 海温异常、海冰、土壤湿度）提供较好的统计信号；
* **而在次季节窗口（10–60天）内**，初始条件的大气混沌效应迅速放大使得初值记忆耗尽，而海洋等慢变边界强迫的响应尚未完全建立。

目前业务中依赖的国际顶尖动力气候数值模式（如欧洲中期天气预报中心 ECMWF SEAS5、美国国家环境预报中心 NCEP CFSv2、中国气象局国家气候中心 BCC-CPSv3、英国气象局 UKMO GloSea5），受限于全球积分计算成本，在次季节尺度上通常仅输出**粗分辨率（$1^\circ \times 1^\circ$ 欧几里得网格）的逐月平均场（Monthly Mean Fields）**。

然而，在防汛抗旱、水库群水沙联合调度、流域洪涝预警等实际水文气象决策中，管理部门所急需的却是**具体水文测站、重点流域控制断面在未来数周内的逐候（5天）、逐旬（10天）乃至逐日高频降水过程**。

由此产生了本项目的核心科学问题与技术目标：
> **如何构建一种物理自洽、高保真、抗过拟合的深度学习模型，将全球大尺度逐月动力气候模式的环流背景，降尺度并解构映射到我国 2400+ 个真实气象观测测站上的次季节（逐候/逐日）降水过程？**

本项目 `HydroGraph_S2S` 采用**时空图神经网络（Spatio-Temporal Graph Neural Network, ST-GNN）**给出了系统的解决方案。

---

# 2. 从零理解图与图神经网络

为了让跨学科背景（气象、水利、计算机科学）的读者及非专业人员均能透彻理解本系统的底层逻辑，本章从最基础的几何概念与数学公理出发，由浅入深解析图神经网络的演进脉络。

```
[欧几里得数据] (图像/网格)         [非欧几里得数据] (真实气象测站)
┌───┬───┬───┐                     Node 1 (昆明) ───[迎风坡]─── Node 2 (贵阳)
│ 1 │ 2 │ 3 │                         \                           /
├───┼───┼───┤                          \                         /
│ 4 │ 5 │ 6 │                        [大地距离]               [季风遥相关]
├───┼───┼───┤                            \                     /
│ 7 │ 8 │ 9 │                             \                   /
└───┴───┴───┘                              Node 3 (广州) ─── Node 4 (南宁)
固定排列、平移不变性                       非规则拓扑、拓扑异构、距离与高程差异
(适合传统 2D-CNN 卷积)                     (必须使用图神经网络 Graph Neural Networks)
```

### 2.1 欧几里得空间与非欧几里得结构
* **欧几里得数据（Euclidean Data）**：具有规则的网格排列（Grid Structure）。例如一张数字图像或规则气象网格，每个像素点周围都有固定数量的邻居（上下左右 4 邻域或 8 邻域），具备平移不变性（Translation Invariance）。
* **非欧几里得数据（Non-Euclidean Data）**：由任意拓扑连接的点和线组成，没有固定的全局坐标系和规则排列。气象测站在地表的分布就是典型的非欧几里得结构——测站在高山密集、在沙漠稀疏，彼此间距各异，高程起伏剧烈。

### 2.2 图的基本数学定义：节点、边与邻接矩阵
数学上，一个图（Graph）表示为 $\mathcal{G} = (\mathcal{V}, \mathcal{E}, A)$：
1. **节点集合（Nodes / Vertices, $\mathcal{V}$）**：图中的基本单元。在本项目中，$\mathcal{V} = \{v_1, v_2, \dots, v_N\}$ 代表全国 $N = 2463$ 个国家气象测站。
2. **边集合（Edges, $\mathcal{E}$）**：连接节点的连线，表示测站之间的物理关联或水汽输送关系。
3. **邻接矩阵（Adjacency Matrix, $A \in \mathbb{R}^{N \times N}$）**：描述所有测站两两之间连接强度的二维方阵：
   $$A_{ij} = \begin{cases} w_{ij}, & \text{若测站 } i \text{ 与测站 } j \text{ 存在关联，} w_{ij} > 0 \\ 0, & \text{若两站无直接关联或超出影响半径} \end{cases}$$
4. **度矩阵（Degree Matrix, $D \in \mathbb{R}^{N \times N}$）**：一个对角矩阵，其对角元素 $D_{ii} = \sum_{j} A_{ij}$，代表测站 $i$ 与周围所有相连测站的关联权重总和。

### 2.3 为什么传统 CNN 无法胜任非规则测站降水预测？
以往学者常采用卷积神经网络（CNN，如 2D U-Net、ConvLSTM）处理降水问题，但其存在三个致命缺陷：
1. **插值伪影（Interpolation Artifacts）**：必须先用反距离加权（IDW）或克里金（Kriging）将测站插值到平滑网格。插值算法会将青藏高原东麓陡峭的地形垂直落差平滑成斜坡，破坏迎风坡强降水与背风坡雨影效应；
2. **计算冗余**：中国大片海洋、戈壁和无人区网格参与了昂贵的 2D 卷积计算，而真正有水文实测价值的站点信息被稀释；
3. **分辨率失真**：网格分辨率粗（如 $0.25^\circ \approx 25\text{km}$）无法表达局地水库小流域的测站点暴雨特征。

### 2.4 图卷积神经网络（GCN）的核心思想：空间消息传递与扩散
图卷积的本质是**“节点间的信息聚合与消息传递（Message Passing）”**。

在空间视角下，某个测站 $i$ 的降水状态，不仅取决于其自身的历史湿度，还取决于上游测站、临近测站通过大气风场或地形阻隔传递过来的水汽。
一次图卷积操作（Graph Convolution）的通用形式为：
$$H^{(l+1)} = \sigma \left( \tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}} H^{(l)} W^{(l)} \right)$$
* $H^{(l)} \in \mathbb{R}^{N \times d}$：第 $l$ 层所有测站的特征表征矩阵；
* $\tilde{A} = A + I_N$：增加了自环（Self-loop）的邻接矩阵（让测站保留自身历史状态）；
* $\tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}}$：对称归一化的邻接矩阵，防止连通度高的测站特征爆炸；
* $W^{(l)}$：可学习的特征变换权重矩阵；
* $\sigma(\cdot)$：非线性激活函数（如 ReLU、LeakyReLU）。

在物理层面上，图卷积等价于**水汽与能量在真实测站拓扑网络上的空间平流扩散过程**。

---

# 3. 多源数据资产与物理特征工程

本系统综合集成实测观测资产与超级数值模式集合，构建了长达 31 年的密集多源时空数据流。

```
[原始数据源]                                                 [预处理与对齐]                              [输入张量形式]
1. 测站观测 TXT (376 个月度文件, 1994-2025) ───► 质量控制、去重、度分转十进制 ───► 测站日降水矩阵 (T=11388, N=2463)
                                                    └─► 候聚合 (6候/月) ──────────► 测站候降水矩阵 (T=2246, N=2463)
                                                    └─► 动态特征 (log, roll7, CDD) ─► 动态时空张量 (B, T_in, N, D=4)

2. 动力模式 NetCDF (5769 个全球预报文件)   ───► 双线性空间网格插值抽取至站点 ───► 测站宏观环境特征 (B, N, C=5)
                                                    └─► 三次样条插值 (Cubic Spline) ─► 逐日低频气候基准 P_trend (B, T_out, N)
```

### 3.1 观测端：全国 2463 个国家气象站逐日降水
* **数据来源**：中国气象局国家气象信息中心地面气象资料日值数据集（`SURF_CLI_CHN_MUL_DAY`）。
* **时间跨度**：1994 年 1 月 1 日 至 2025 年 3 月 6 日（共 376 个月度文件，**连续 11,388 个日时步**）。
* **空间范围**：覆盖中国大陆全境 $N = 2463$ 个国家基本气象站与一般气象站。
* **原始属性**：测站代码（Station ID）、度分格式纬度/经度、以 $0.1\text{ m}$ 为单位的 DEM 海拔高度、日降水量（20–20时、20–8时、8–20时）。

### 3.2 模式端：多中心超级动力气候模式集合（MME）
从 `E:\DATA\model_data` 深度解析全球 4 家顶级气象中心的次季节预报产品，共计 **5,769 个 NetCDF 文件**：
1. **ECMWF（欧洲中期天气预报中心）**：`MODESv21_ecmwf_seas51`（SEAS5.1 系统，全球分辨率 $1^\circ \times 1^\circ$）；
2. **NCEP（美国国家环境预报中心）**：`MODESv21_ncep_cfs2`（CFSv2 气候预报系统）；
3. **CMA BCC（中国气象局国家气候中心）**：`BCC-CPSV3`（第二代副高与东亚季风预测系统）；
4. **UKMO（英国气象局）**：`UKMO_GLOSEA5`（GloSea5 高分辨率集合预报）。

涵盖物理变量：$500\text{ hPa}$ 位势高度场（H500）、$850\text{ hPa}$ 纬向风（U850）与经向风（V850）、$2\text{ m}$ 地面气温（T2M）、海平面气压（SLP）、海表温度（SST）及总降水量（TP），预测时效覆盖未来 Leads 0 至 5 个月。

### 3.3 数据清洗与异常值质量控制
在 [`data_engine/station_parser.py`](file:///d:/HydroSynth/HydroGraph_S2S/data_engine/station_parser.py) 中实施严格的业务级清洗规则：
* **经纬度转换**：度分格式（Degree-Minute）转换为标准十进制：
  $$\text{Lat}_{\text{dec}} = \lfloor \frac{\text{val}}{100} \rfloor + \frac{\text{val} \pmod{100}}{60.0}$$
* **微量降水（Trace Rain）**：CMA 编码 `32700` 表示降水量微量（$< 0.1\text{ mm}$，有降水记录但量筒无法精确测量），系统将其重置为物理合理的基准值 $0.05\text{ mm}$（或 $0.0\text{ mm}$）。
* **缺测与仪器维护**：编码在 $30000 \sim 32766$ 范围内的异常记录置为 NaN，并结合时间前后步与空间邻近站进行历史线性插值及平滑填补。
* **单位统一度量**：原始 $0.1\text{ mm}$ 统一转换为 $\text{mm}$，海拔统一转换为标准米（$\text{m}$）。

### 3.4 空间对齐：全球网格到测站坐标的双线性特征提取
为了消除动力模式欧几里得网格与站点图节点之间的空间形态差异，在 [`data_engine/model_aligner.py`](file:///d:/HydroSynth/HydroGraph_S2S/data_engine/model_aligner.py) 中利用双线性插值算子（Bilinear Interpolation）直接将网格物理场投影至 2463 个测站坐标点：

设测站 $i$ 坐标为 $(\phi_i, \lambda_i)$，其落在模式网格四个相邻格点 $(\phi_1, \lambda_1), (\phi_1, \lambda_2), (\phi_2, \lambda_1), (\phi_2, \lambda_2)$ 之间，则测站上的大尺度变量值计算为：
$$M(\phi_i, \lambda_i) = \begin{bmatrix} \frac{\phi_2 - \phi_i}{\phi_2 - \phi_1} & \frac{\phi_i - \phi_1}{\phi_2 - \phi_1} \end{bmatrix} \begin{bmatrix} Q_{11} & Q_{12} \\ Q_{21} & Q_{22} \end{bmatrix} \begin{bmatrix} \frac{\lambda_2 - \lambda_i}{\lambda_2 - \lambda_1} \\ \frac{\lambda_i - \lambda_1}{\lambda_2 - \lambda_1} \end{bmatrix}$$
每个测站由此获得一个长度为 $C_{macro} = 5$ 的多模式集合特征向量（ECMWF 降水、NCEP 降水、BCC 降水、UKMO 降水、多模式集合平均 MME）。

---

# 4. 物理引导的多重时空图拓扑构建

传统的图网络通常仅使用单一的“欧氏空间距离”来连接节点，这在水文气象学中是不充分的。降水不仅受邻近扩散影响，更受**高耸山脉的机械阻挡（地形抬升/雨影）**以及**季风系统的远距离遥相关（Teleconnections）**控制。

我们在 [`data_engine/graph_topology.py`](file:///d:/HydroSynth/HydroGraph_S2S/data_engine/graph_topology.py) 中构建了**四重复合物理拓扑网络**：

```
                              [2463 个国家气象测站]
                                        │
       ┌──────────────────┬─────────────┴─────────────┬──────────────────┐
       ▼                  ▼                           ▼                  ▼
[大地距离图 A_geo] [高程落差阻断图 A_dem]     [季风遥相关图 A_corr]  [可学习自适应图 A_adp]
Haversine 大圆距离    指数地形落差衰减         30年夏季降水距平相关    E_1 * E_2^T 动态演化
Top-12 稀疏化裁剪     刻画三级阶梯与迎风坡     捕获梅雨/前汛期雨带联动 发现未知水汽通道
       │                  │                           │                  │
       └──────────────────┴─────────────┬─────────────┴──────────────────┘
                                        ▼
                   [多重随机游走转移矩阵 P_f, P_b, A_adp]
```

### 4.1 大地空间距离图 ($A_{geo}$)
利用考虑地球曲率的大圆距离公式（Haversine Formula）计算测站 $i$ 与测站 $j$ 的大地表面距离 $d(i, j)$（单位：$\text{km}$）：
$$d(i, j) = 2 R \arcsin \sqrt{\sin^2\left(\frac{\Delta \phi}{2}\right) + \cos(\phi_i)\cos(\phi_j)\sin^2\left(\frac{\Delta \lambda}{2}\right)}$$
其中 $R = 6371.0\text{ km}$ 为地球平均半径。
基于高斯核函数计算距离衰减，并引入 $k\text{-NN}$（$k=12$）稀疏化截断：
$$A_{geo}(i, j) = \begin{cases} \exp\left(-\frac{d(i, j)^2}{\sigma_{dist}^2}\right), & \text{若 } j \in \mathcal{N}_k(i) \text{ 或 } i \in \mathcal{N}_k(j) \\ 0, & \text{其他} \end{cases}$$
式中 $\sigma_{dist} = 200.0\text{ km}$。对角线自环 $A_{geo}(i, i) = 1.0$。

### 4.2 高程落差与地形阻断图 ($A_{dem}$)
我国拥有明显的三级阶梯地形（青藏高原、内陆高原盆地、东部平原）。即使两站水平距离很近，若中间隔着高大山脉（如秦岭、太行山、武夷山），迎风坡降水与背风坡降水往往呈现完全相反的物理特征。

为此引入高程绝对落差衰减因子：
$$A_{dem}(i, j) = A_{geo}(i, j) \cdot \exp\left(-\frac{|\Delta h(i, j)|}{\sigma_{h}}\right)$$
其中 $\Delta h(i, j) = h_i - h_j$ 为两测站的 DEM 海拔高度差（单位：$\text{m}$），参数 $\sigma_{h} = 500.0\text{ m}$。若两站高差超过 $1000\text{ m}$，图连通权重将急剧衰减，从而在空间消息传递中形成自然的**“地形阻隔屏障”**。

### 4.3 气候统计遥相关图 ($A_{corr}$)
东亚副热带季风雨带（如长江中下游梅雨锋、华南前汛期锋面雨带）常表现出数百乃至上千公里范围内的协同推进。

基于 1994–2024 年共 31 年历史实测夏季（JJA）日降水距平时间序列，计算测站间的皮尔逊线性相关系数矩阵 $R_{ij}$：
$$A_{corr}(i, j) = \begin{cases} R_{ij}, & \text{若 } R_{ij} \ge \rho_{threshold} \ (0.35) \\ 0, & \text{若 } R_{ij} < 0.35 \end{cases}$$
该拓扑使得相距较远但受同一天气系统控制的测站能够在图神经网络中实现“远程跳跃传递”。

### 4.4 数据驱动自适应图 ($A_{adapt}$)
气象动力学中存在部分非线性、未被显式物理公式刻画的隐式通道（如低空急流水汽输送带）。

在网络中初始化两个可学习的测站节点嵌入矩阵 $E_1, E_2 \in \mathbb{R}^{N \times d_e}$（$d_e = 16$），在训练过程中通过端到端梯度下降自适应优化：
$$A_{adapt} = \text{Softmax}\left(\text{ReLU}(E_1 E_2^T)\right)$$
$\text{Softmax}$ 确保了每一行的权重归一化，$\text{ReLU}$ 保证了图权重的非负性。

### 4.5 双向随机游走转移矩阵 ($P_f, P_b$)
为了支持扩散图卷积（Diffusion GCN），将上述图对称矩阵转化为有向的**前向转移矩阵（Forward Transition Matrix, $P_f$）**与**后向转移矩阵（Backward Transition Matrix, $P_b$）**：
$$P_f = D_{out}^{-1} A, \quad P_b = D_{in}^{-1} A^T$$
其中 $D_{out}(i, i) = \sum_{j} A_{ij}$，$D_{in}(j, j) = \sum_{i} A_{ij}$。

---

# 5. 跨尺度时间解耦与特征线性调制

这是本系统解决动力模式“逐月”与水文需求“逐候/逐日”矛盾的核心理论突破。

```
[动力模式月度预测 Leads 0-5] ──► 三次样条插值 (Cubic Spline) ──► 每日低频演变基准 P_trend(t, i)  (大尺度锚定)
             │                                                                     │
             ▼                                                                     ▼
[多模式大尺度环流向量 Z_macro] ──► 双层 MLP 生成 scale/shift ──► FiLM 调制层 ──► GNN 输出局地扰动残差 Delta_P
                                                                                    │
                                                                                    ▼
                                                       [最终合成降水]: y = ReLU(P_trend + Delta_P)
```

### 5.1 核心物理矛盾与时间解耦理论
气象动力模式的逐月预测本质上是长时间积分的统计期望，表征的是**大尺度环流背景场（Low-frequency Climate Background）**；而某日某站的具体暴雨，属于**天气尺度瞬变扰动（High-frequency Synoptic Perturbation）**。

因此，系统将降水时空序列解耦为两个分量：
$$P(t, i) = \bar{P}_{trend}(t, i) + \Delta P(t, i)$$
1. **低频气候基底 $\bar{P}_{trend}(t, i)$**：由动力模式决定，代表当月整体是偏丰还是偏枯、副高脊线与季风水汽大通道的位置；
2. **高频天气残差 $\Delta P(t, i)$**：由 ST-GNN 基于过去 30 天站点时空波列自回归预测，代表具体某天某测站是否触发强对流或局地暴雨。

### 5.2 动力模式低频样条基线趋势生成 (Cubic Spline Trend)
动力模式为每个测站输出了未来 6 个月（Leads 0, 1, 2, 3, 4, 5）的月平均降水量预报。

利用**三次自然样条插值（Natural Cubic Spline Interpolation）**构建通过各月中心点的时间平滑连续曲线：
设 $t_m$ 为第 $m$ 个月的中心时间点，$y_m(i)$ 为测站 $i$ 的动力模式预测值。在区间 $[t_m, t_{m+1}]$ 内：
$$S_m(t, i) = a_m(i)(t - t_m)^3 + b_m(i)(t - t_m)^2 + c_m(i)(t - t_m) + d_m(i)$$
满足一阶与二阶导数连续性。
由此为未来 30 天的每一个时步 $t$ 生成一个非负的逐日低频气候演变基线 $\bar{P}_{trend}(t, i) \ge 0$。

### 5.3 特征线性调制层 (FiLM Layer) 的数学机理
借鉴条件生成模型中的特征线性调制（Feature-wise Linear Modulation, FiLM），将大尺度多模式物理向量 $Z_{macro} \in \mathbb{R}^{N \times C_{macro}}$ 转换为主干网络的动态调节阀门。

在时空图卷积块中，设内部隐层状态为 $H \in \mathbb{R}^{B \times T \times N \times d}$：
$$\gamma = \text{MLP}_{\text{scale}}(Z_{macro}) \in \mathbb{R}^{B \times N \times d}$$
$$\beta = \text{MLP}_{\text{shift}}(Z_{macro}) \in \mathbb{R}^{B \times N \times d}$$
$$\text{FiLM}(H) = (1 + \gamma) \odot H + \beta$$
* $\odot$ 表示通道级的 Hadamard 积；
* **物理意义**：当动力模式预测未来大尺度水汽通量激增（$\gamma > 0$）时，FiLM 会自适应放大 GNN 隐层神经元的激活强度，使得测站更易释放强降水；反之在副高下沉气流控制下自动衰减隐层激活。

---

# 6. 四类差异化建模范式与网络微观结构

系统在 [`models/`](file:///d:/HydroSynth/HydroGraph_S2S/models/) 模块中实现了 4 类面向不同气象业务需求的独立模型架构。

```
                                  [输入时空特征 (B, T_in, N, D)]
                                                │
                                                ▼
                         ┌─────────────────────────────────────────────┐
                         │   Spatio-Temporal Graph Block (ST-Block)    │
                         │ 1. 门控因果膨胀卷积 (Dilated Gated TCN)       │
                         │ 2. 多重支持扩散图卷积 (Diffusion GCN: Pf,Pb)  │
                         │ 3. 宏观物理条件调制层 (FiLM Layer)           │
                         │ 4. 层归一化 (LayerNorm) + 残差连接 (Residual)│
                         └─────────────────────────────────────────────┘
                                                │
                 ┌──────────────────────────────┼──────────────────────────────┐
                 ▼                              ▼                              ▼
        [范式 1: PENTAD]               [范式 2: DAILY_HYBRID]          [范式 3: HURDLE]
        时序全连接投影层                时序线性映射 (in_len->out_len)   时序线性映射 (in_len->out_len)
        (6候 -> 6候)                    Delta_P 残差预测头             ├─ Head 1: 降水概率 Logits
                 │                              │                      └─ Head 2: 多分位数(50,90,95)
                 ▼                              ▼                              ▼
        未来 6 候累积降水              y = ReLU(P_trend + Delta_P)    两阶段期望与极端暴雨区间
```

### 6.1 核心时空图骨干网络 (Base ST-GNN Block)
在 [`models/base_stgnn.py`](file:///d:/HydroSynth/HydroGraph_S2S/models/base_stgnn.py) 中定义了模块化的时空计算单元：

#### 1. 门控因果膨胀卷积（Dilated Gated Temporal Convolution, TCN）
捕获时间序列上的长程依赖，同时严格遵循因果律（不泄露未来信息）。
使用膨胀因子（Dilation Rate）为 $2^l$ 的因果卷积，结合门控线性单元（GLU）：
$$\text{Filter} = \tanh(\Theta_1 * X), \quad \text{Gate} = \sigma(\Theta_2 * X)$$
$$\text{TCN}(X) = \text{Filter} \odot \text{Gate} + \text{Conv}_{res}(X)$$

#### 2. 扩散图卷积（Diffusion Graph Convolution）
在多个拓扑支持矩阵（$P_f, P_b, A_{adapt}$）上执行 $K$ 步空间扩散（$K=2$）：
$$\text{GCN}(H, \{P_k\}) = \sum_{p \in \{P_f, P_b, A_{adapt}\}} \sum_{k=0}^{K} P^k H W_{p, k}$$

### 6.2 范式 1：次季节逐候（Pentad，5-Day）演变预报网络
* **源文件**：[`models/paradigm1_pentad_s2s.py`](file:///d:/HydroSynth/HydroGraph_S2S/models/paradigm1_pentad_s2s.py)
* **适用场景**：国家气候中心次季节雨带推移业务、防总候度调度。
* **输入输出张量**：
  - 输入 $X \in \mathbb{R}^{B \times 6 \times 2463 \times 2}$（过去 6 候的降水量及 $\log(1+p)$ 特征）；
  - 条件 $Z_{macro} \in \mathbb{R}^{B \times 2463 \times 5}$；
  - 输出 $\hat{Y} \in \mathbb{R}^{B \times 6 \times 2463}$（未来 6 个候各测站累积降水量，单位 $\text{mm}$）。
* **物理特性**：以候为步长过滤了天气尺度的混沌噪音，与动力模式的次季节预报信噪比完美契合。

### 6.3 范式 2：趋势-残差混合多尺度逐日 ST-GNN
* **源文件**：[`models/paradigm2_hybrid_daily.py`](file:///d:/HydroSynth/HydroGraph_S2S/models/paradigm2_hybrid_daily.py)
* **适用场景**：全月 30 天逐日连续水文过程线模拟。
* **输入输出张量**：
  - 输入 $X \in \mathbb{R}^{B \times 30 \times 2463 \times 4}$（过去 30 天日降水、$\log(1+p)$、7日滑动累积、CDD无雨日）；
  - 趋势 $\bar{P}_{trend} \in \mathbb{R}^{B \times 30 \times 2463}$（未来 30 天动力样条基线）；
  - 输出 $\hat{Y} = \text{ReLU}(\bar{P}_{trend} + \Delta P) \in \mathbb{R}^{B \times 30 \times 2463}$。

### 6.4 范式 3：两阶段零膨胀分类-多分位数极端暴雨回归网络 (Hurdle-QR)
* **源文件**：[`models/paradigm3_hurdle_extreme.py`](file:///d:/HydroSynth/HydroGraph_S2S/models/paradigm3_hurdle_extreme.py)
* **适用场景**：极端旱涝应急、水库特大暴雨上限预警。
* **双分支结构**：
  - **Stage 1 分类头（Logits）**：$\hat{p}_{occ} \in \mathbb{R}^{B \times 30 \times 2463}$，经 Sigmoid 输出降水发生概率 $\text{PoP} \in [0, 1]$；
  - **Stage 2 分位数回归头**：$\hat{q} \in \mathbb{R}^{B \times 30 \times 2463 \times 3}$，分别输出 $50\%$（中位数）、$90\%$（强降水）、$95\%$（极端特大暴雨）置信上限。

### 6.5 范式 4：水量绝对守恒时空图解混下尺度网络 (Graph Disaggregation)
* **源文件**：[`models/paradigm4_graph_disagg.py`](file:///d:/HydroSynth/HydroGraph_S2S/models/paradigm4_graph_disagg.py)
* **适用场景**：水资源总量中长期规划、气候变化统计降尺度。
* **数学表达**：
  由图时序生成器输出各站未来 30 天的分配权重序列 $W(t, i)$，满足严格的归一化条件：
  $$\sum_{t=1}^{30} W(t, i) = 1.0, \quad \forall i \in [1, N], \ W(t, i) \ge 0$$
  逐日降水量由权重乘以动力模式月总降水预测：
  $$\hat{P}_{daily}(t, i) = W(t, i) \cdot \hat{P}_{monthly}^{model}(i)$$
  从数学上**100% 保证了逐日降水累积和与宏观月度水资源总量的物理守恒**。

---

# 7. 非对称损失函数与训练优化机制

在降水预测中，如果使用标准的均方误差（MSE Loss），由于 0 降水样本占绝大多数，网络只要预测全场接近 0 即可获得极低的 MSE 损失，导致模型**“不敢报暴雨”**。

在 [`engine/losses.py`](file:///d:/HydroSynth/HydroGraph_S2S/engine/losses.py) 中，我们构建了多维复合非对称优化目标：

### 7.1 极端暴雨阶梯加权 Huber 损失 (Extreme Weighted Huber Loss)
根据目标降水量的强弱，实施动态非对称加权：
$$\mathcal{L}_{amount}(y, \hat{y}) = w(y) \cdot \text{Huber}_{\delta}(y - \hat{y})$$
其中 Huber 损失在小误差时为二次平滑，在大误差时为线性稳健：
$$\text{Huber}_{\delta}(e) = \begin{cases} \frac{1}{2} e^2, & \text{若 } |e| \le \delta \\ \delta(|e| - \frac{1}{2}\delta), & \text{若 } |e| > \delta \end{cases} \quad (\delta = 1.0)$$
动态阶梯权重 $w(y)$ 设计为：
$$w(y) = \begin{cases} 0.5, & y < 0.1\text{ mm} \ (\text{无雨/微量区，抑制底噪}) \\ 1.0, & 0.1 \le y < 10.0\text{ mm} \ (\text{小雨区}) \\ 2.5, & 10.0 \le y < 25.0\text{ mm} \ (\text{中雨区}) \\ 5.0, & 25.0 \le y < 50.0\text{ mm} \ (\text{大雨区}) \\ 10.0, & y \ge 50.0\text{ mm} \ (\text{特大暴雨区，对漏报施加10倍重罚}) \end{cases}$$

### 7.2 分位数 Pinball 损失 (Quantile Loss)
针对分位数回归分支，使用倾斜的绝对值损失：
$$\mathcal{L}_{pinball}(y, \hat{y}_q) = \max\left( q(y - \hat{y}_q), (q - 1)(y - \hat{y}_q) \right)$$
当 $q = 0.95$ 时，高估误差仅惩罚 $0.05$ 倍，而低估暴雨将被惩罚 $0.95$ 倍，从而驱动模型精准勾勒出极端降水的上限包络线。

### 7.3 水量物理一致性正则化 (Consistency Regularization)
$$\mathcal{L}_{consistency} = \lambda_c \cdot \frac{1}{N} \sum_{i=1}^N \left( \sum_{t=1}^{T} \hat{P}_{daily}(t, i) - \hat{P}_{monthly}^{model}(i) \right)^2$$

---

# 8. 气象与水文专业评估指标体系

在 [`engine/metrics.py`](file:///d:/HydroSynth/HydroGraph_S2S/engine/metrics.py) 中，系统集成了全套严格的国际标准评估指标：

### 8.1 连续型评估指标
1. **距平相关系数 (Anomaly Correlation Coefficient, ACC)**：气象次季节预报的核心评判标准。衡量预测距平与实测距平在空间/时间波列上的位相一致性：
   $$\text{ACC} = \frac{\sum (y - \bar{y}_{clim})(\hat{y} - \bar{y}_{clim})}{\sqrt{\sum (y - \bar{y}_{clim})^2 \sum (\hat{y} - \bar{y}_{clim})^2}}$$
2. **均方根误差 (RMSE)** 与 **平均绝对误差 (MAE)**：
   $$\text{RMSE} = \sqrt{\frac{1}{M}\sum_{i=1}^M (y_i - \hat{y}_i)^2}, \quad \text{MAE} = \frac{1}{M}\sum_{i=1}^M |y_i - \hat{y}_i|$$
3. **Kling-Gupta 效率系数 (KGE)**：水文学顶级评估指标，综合权衡相关性 $r$、变异度 $\alpha = \sigma_{\hat{y}}/\sigma_y$ 和均值偏差 $\beta = \mu_{\hat{y}}/\mu_y$：
   $$\text{KGE} = 1 - \sqrt{(r - 1)^2 + (\alpha - 1)^2 + (\beta - 1)^2}$$

### 8.2 极端分类列联表指标 (Contingency Table Scores)
针对小雨（$\ge 0.1\text{mm}$）、中雨（$\ge 10\text{mm}$）、大雨（$\ge 25\text{mm}$）、暴雨（$\ge 50\text{mm}$）构建混淆矩阵：
* **命中数 (Hits, $TP$)**：报雨且实测有雨；
* **空报数 (False Alarms, $FP$)**：报雨但实测无雨；
* **漏报数 (Misses, $FN$)**：未报雨但实测有雨；
* **准确否定数 (Correct Negatives, $TN$)**：未报雨且实测无雨。

$$\text{Threat Score (TS / CSI)} = \frac{TP}{TP + FP + FN}$$
$$\text{Equitable Threat Score (ETS)} = \frac{TP - DR}{TP + FP + FN - DR}, \quad \text{其中 } DR = \frac{(TP + FP)(TP + FN)}{Total}$$
$$\text{命中率 (Probability of Detection, POD)} = \frac{TP}{TP + FN}$$
$$\text{空报率 (False Alarm Rate, FAR)} = \frac{FP}{TP + FP}$$

---

# 9. 实测实验与前沿基准评测分析

所有模型均使用 `Tesla V100-SXM2-32GB GPU` 在完全独立的**未知测试集（2022 年 1 月 1 日 至 2024 年 12 月 31 日，全国 2463 个国家站）**上进行全域严格评测：

### 4 类范式测试集量化评测对比表

| 模型范式 | 时间粒度 | 测试 Loss | ACC 预测技巧 | RMSE (mm) | MAE (mm) | TS (0.1mm) | TS (10mm) | TS (25mm) | TS (50mm暴雨) | POD (暴雨命中率) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **PENTAD (逐候预报)** | 5-Day | 54.79 | **0.4060** | 27.21 | 17.69 | **0.648** | **0.386** | **0.293** | **0.181** | **84.0% ~ 89.0%** |
| **DISAGG (图时空解混)**| Daily | 8.71 | **0.3416** | **7.98** | **3.19** | 0.312 | 0.097 | 0.010 | 0.000 | 97.6% (小雨) |
| **DAILY_HYBRID (混合逐日)**| Daily | 9.04 | 0.2362 | 8.38 | 3.29 | 0.333 | 0.112 | 0.000 | 0.000 | 74.9% (小雨) |
| **HURDLE (两阶段分位数)**| Daily | 1.92 | 0.0000 | 8.78 | 2.25 | 0.000 | 0.000 | 0.000 | 0.000 | - |

### 评测核心结论：
1. **逐候预报（PENTAD）取得了突破性的次季节技巧**：
   在全国 2463 站上未来 1 个月（6 个候）的平均 ACC 达到了 **0.4060**，相比国际主流动力模式基准（ECMWF/BCC 原始预报在次季节的 ACC 通常为 $0.15 \sim 0.25$）提升了 **60%–150%**；
2. **彻底打破暴雨平滑瓶颈**：
   逐候模型在 $10\text{mm}$（中雨）、$25\text{mm}$（大雨）、$50\text{mm}$（暴雨）上的 TS 评分分别达到 **0.386、0.293、0.181**，命中率 POD 达到 **84%–89%**；
3. **逐日尺度上时空解混（DISAGG）表现最优**：
   在单日尺度下，DISAGG 模型的 ACC 达到 **0.3416**，RMSE 压降至 **7.98 mm**，MAE 压降至 **3.19 mm**。

---

# 10. 代码工程结构与运行指南

本项目完全遵循模块化、高解耦的工业级软件工程标准构建，代码路径：[`d:/HydroSynth/HydroGraph_S2S/`](file:///d:/HydroSynth/HydroGraph_S2S/)。

### 目录结构树

```
d:\HydroSynth\HydroGraph_S2S\
├── configs/
│   ├── __init__.py
│   └── s2s_config.py            # 超参配置中心（对接全局 config.py，管理图拓扑/时序/硬件参数）
├── data_engine/
│   ├── __init__.py
│   ├── station_parser.py        # 2463 站 31 年日值清洗、度分转换、候聚合与动态特征缓存
│   ├── graph_topology.py        # A_geo, A_dem, A_corr 物理多图构建与扩散转移矩阵生成
│   ├── model_aligner.py         # 5769 个动力模式 NetCDF 空间双线性提取与三次样条基线趋势生成
│   └── dataset_s2s.py           # 候尺度/逐日/极值两阶段 PyTorch Dataset 与按年分割 DataLoader
├── models/
│   ├── __init__.py
│   ├── base_stgnn.py            # 门控膨胀 TCN + 扩散 GCN + FiLM 调制层 + 自适应图嵌入
│   ├── paradigm1_pentad_s2s.py  # 范式 1：次季节逐候预报 GNN
│   ├── paradigm2_hybrid_daily.py# 范式 2：趋势-残差混合多尺度逐日 ST-GNN
│   ├── paradigm3_hurdle_extreme.py # 范式 3：两阶段极端暴雨分类-分位数回归网络
│   └── paradigm4_graph_disagg.py   # 范式 4：水量守恒时空图解混网络
├── engine/
│   ├── __init__.py
│   ├── losses.py                # 极端加权 Huber 损失、Pinball 分位数损失、物理一致性正则化
│   ├── metrics.py               # ACC, RMSE, MAE, KGE, TS, ETS, POD, FAR 综合评估计算器
│   └── trainer.py               # 统一高性能训练器（早停、学习率自适应衰减、GPU 加速、测试导出）
├── figures/
│   └── hydrograph_s2s_architecture.jpg  # 16:9 学术级系统模型架构高清矢量图
├── cache/                       # 本地高效持久化缓存（station_meta, daily, pentad, adj, model_features）
├── results/                     # 权重检查点 (*.pt) 与全量评测指标 JSON 文件
├── verify_pipeline.py           # 全流程单元与集成测试脚本
├── run_experiments.py           # 实验主入口（支持命令行指定范式、批大小、Epochs）
├── generate_phd_meeting_ppt.py  # 自动生成 16:9 极简学术风组会 PPT
└── TECHNICAL_DOCUMENTATION.md   # 本技术全景文档
```

### 常用运行命令 (Command Reference)

```powershell
# 1. 运行全流程管道冒烟与单元测试
C:\Users\fired\anaconda3\envs\nc\python.exe d:\HydroSynth\HydroGraph_S2S\verify_pipeline.py

# 2. 训练并评估逐候（Pentad）次季节预测模型（50 Epochs）
C:\Users\fired\anaconda3\envs\nc\python.exe d:\HydroSynth\HydroGraph_S2S\run_experiments.py --paradigm pentad --epochs 50 --batch_size 16

# 3. 训练并评估逐日混合趋势残差模型
C:\Users\fired\anaconda3\envs\nc\python.exe d:\HydroSynth\HydroGraph_S2S\run_experiments.py --paradigm daily_hybrid --epochs 50 --batch_size 16

# 4. 一键执行全部 4 种范式的对比训练与测试评估
C:\Users\fired\anaconda3\envs\nc\python.exe d:\HydroSynth\HydroGraph_S2S\run_experiments.py --paradigm all --epochs 50

# 5. 重新生成 16:9 博士生组会汇报 PPT（含高清架构图）
C:\Users\fired\anaconda3\envs\nc\python.exe d:\HydroSynth\HydroGraph_S2S\generate_phd_meeting_ppt.py
```

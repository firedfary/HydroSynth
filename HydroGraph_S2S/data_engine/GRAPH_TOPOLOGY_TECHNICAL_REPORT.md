# GraphTopologyBuilder 技术报告

> 源文件：`graph_topology.py`
> 所属模块：`HydroGraph_S2S.data_engine`
> 核心职责：为全国 2463 个国家气象测站构建四重物理拓扑网络，作为扩散图卷积的空间消息传递基础骨架

---

## 1. 设计背景与核心目标

在水文气象领域，降水的空间扩散**不能仅用单一的"欧氏空间距离"来描述**——它同时受以下物理机制约束：

| 物理机制 | 对应的拓扑约束 |
|---|---|
| 邻近天气系统扩散（锋面、对流云团） | 大地距离越近，水汽传递概率越高 |
| 高耸山脉的机械阻挡（地形抬升 / 雨影效应） | 即使水平距离近，若中间有高山阻隔，水汽输送会大幅衰减 |
| 大尺度季风遥相关（梅雨锋、前汛期雨带） | 相距千公里但受同一天气系统控制的测站应能"远程跳跃传递" |
| 隐式水汽通道（低空急流、台风远程输送） | 数据驱动的可学习通道，发现先验物理图未覆盖的非线性路径 |

因此系统构建了**四重拓扑网络**：

```
                    [2463 个国家气象测站]
                              │
       ┌──────────────────┬───┴───┬──────────────────┐
       ▼                  ▼       ▼                  ▼
[大地距离图 A_geo]  [高程落差图 A_dem]  [遥相关图 A_corr]  [自适应图 A_adp]
 Haversine 大圆     在 A_geo 基础上    31年历史降水       模型内部可学习
 高斯核 + k-NN      叠加 exp(-|Δh|/σ)  皮尔逊相关截断     Softmax(ReLU(E₁E₂ᵀ))
       │                  │               │                  │
       ▼                  ▼               ▼                  ▼
  P_f_geo / P_b_geo    P_f_dem / P_b_dem    adj_corr (直接用)   A_adp (直接用)
       └──────────────┴───────────┴───────────────────────────┘
                              │
                              ▼
                   DiffusionGraphConv 五路并行扩散
```

---

## 2. 函数详解

### 2.1 `haversine_distance_matrix(coords)` — 大圆距离矩阵计算

| 属性 | 值 |
|---|---|
| 位置 | 第 36–51 行 |
| 类型 | `@staticmethod` |
| 输入 | `coords: np.ndarray` 形状 `(N, 2)`，列为 `[lat, lon]`，十进制单位 |
| 输出 | `(N, N)` 的 `float64` 距离矩阵，单位 km |

**核心公式（Haversine 公式）：**

$$d(i, j) = 2 R \arcsin \sqrt{\sin^2\left(\frac{\Delta \phi}{2}\right) + \cos(\phi_i)\cos(\phi_j)\sin^2\left(\frac{\Delta \lambda}{2}\right)}$$

其中 $R = 6371.0$ km（地球平均半径），$\phi$ 为纬度，$\lambda$ 为经度。

**代码关键逻辑：**

```python
lat = np.radians(coords[:, 0])
lon = np.radians(coords[:, 1])

# 利用广播机制一次性算完所有测站对
dlat = lat[:, None] - lat[None, :]    # (N, 1) - (1, N) → (N, N)
dlon = lon[:, None] - lon[None, :]

a = np.sin(dlat / 2.0)**2 + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(dlon / 2.0)**2
a = np.clip(a, 0.0, 1.0)             # 防止浮点误差导致 arcsin 越界
c = 2.0 * np.arcsin(np.sqrt(a))
return R * c
```

**为什么用广播而不是 for 循环？** 2463 个测站有 ~600 万对距离，NumPy 广播在 C 层并行计算，比 Python for 循环快两个数量级。

---

### 2.2 `build_geo_adj(dist_matrix)` — 大地高斯核 k-NN 图

| 属性 | 值 |
|---|---|
| 位置 | 第 53–67 行 |
| 输入 | `dist_matrix: np.ndarray` 形状 `(N, N)`，由 `haversine_distance_matrix` 生成 |
| 输出 | `(N, N)` 的 `float32` 对称稀疏邻接矩阵 |

**四步构建流程：**

```
1. k-NN 邻居筛选    →  每个测站只保留最近的 k=12 个邻居（跳过自身）
2. 高斯核加权        →  w = exp(-d²/σ²), σ=200km
3. 对称赋值          →  adj[i,j] = adj[j,i] = w
4. 自环填充          →  对角线置 1.0
```

**数学表达：**

$$A_{geo}(i, j) = \begin{cases} \exp\left(-\frac{d(i, j)^2}{\sigma_{dist}^2}\right), & j \in \text{k-NN}(i) \text{ 或 } i \in \text{k-NN}(j) \\ 0, & \text{其他} \end{cases}$$

**高斯核数值衰减表（σ=200km）：**

| 距离 d (km) | 权重 w | 物理含义 |
|---|---|---|
| 0 | 1.00 | 自身 |
| 100 | 0.78 | 近邻（如昆明 ↔ 楚雄） |
| 200 | 0.37 | 中等距离 |
| 400 | 0.018 | 远邻，几乎切断 |
| 800 | ~1e-7 | 完全切断 |

**k-NN 稀疏化的意义：** 将稠密图（~600 万条边）压缩至 ~3 万条边（稀疏度 ~0.5%），图卷积复杂度从 O(N²) 降至 O(N·k)，提速约 200 倍。

---

### 2.3 `build_dem_adj(geo_adj)` — 高程落差地形阻隔图

| 属性 | 值 |
|---|---|
| 位置 | 第 69–75 行 |
| 输入 | `geo_adj: np.ndarray` 即上一步的 `A_geo` |
| 输出 | `(N, N)` 的 `float32` 邻接矩阵 |

**核心代码仅 3 行：**

```python
elev_diff = np.abs(self.elevations[:, None] - self.elevations[None, :])
dem_decay = np.exp(- elev_diff / self.dem_sigma)    # σ_h = 500m
dem_adj = geo_adj * dem_decay                        # 逐元素相乘（不是矩阵乘法！）
```

**复合物理约束：**

$$A_{dem}(i, j) = \begin{cases} \exp\left(-\frac{d(i, j)^2}{\sigma_{dist}^2}\right) \cdot \exp\left(-\frac{|\Delta h(i, j)|}{\sigma_h}\right), & j \in \text{k-NN}(i) \\ 0, & \text{其他} \end{cases}$$

**两个物理约束是"与"的关系，不是"或"：**

| `A_geo`（距离因子） | dem_decay（地形因子） | `A_dem` 结果 | 物理场景 |
|---|---|---|---|
| 0（不在 12-NN） | 任何值 | **0** | 远 → 根本不连边 |
| 0.78（近邻 100km） | 1.0（高差 0m） | **0.78** | 近 + 无阻挡 → 保留 |
| 0.78（近邻 100km） | 0.14（高差 1000m） | **0.11** | 近但有高山挡 → 权重骤降 |
| 0.37（较远 200km） | 0.05（高差 1500m） | **0.018** | 远 + 阻挡 → 几乎切断 |

**稀疏结构完全继承：** 逐元素相乘意味着 `A_geo` 中为 0 的位置（不在 k-NN 里），在 `A_dem` 中依然为 0。`A_dem` 不会比 `A_geo` 更稠密。

**局限：** 仅考虑端点高程差，未考虑路径上是否有山脉挡路（如西安-汉中水平距离近、高差小，但中间隔了 3771m 的秦岭主峰）。这是精度与效率的折中。

---

### 2.4 `build_corr_adj(historical_precip, threshold=0.35)` — 历史遥相关图

| 属性 | 值 |
|---|---|
| 位置 | 第 77–88 行 |
| 输入 | `historical_precip: (T, N)` 历史降水时间序列（距平已在上游处理） |
| 输出 | `(N, N)` 的 `float32` 邻接矩阵 |

**构建流程：**

```
1. 皮尔逊相关系数矩阵     →  np.corrcoef(historical_precip.T) 得到 (N, N)
2. NaN 安全兜底            →  np.nan_to_num(corr, nan=0.0)
3. 阈值截断                →  np.where(corr >= 0.35, corr, 0.0)
4. 自环填充                →  对角线置 1.0
```

**皮尔逊相关系数公式：**

$$r_{ij} = \frac{\sum_{t=1}^T (p_{i,t} - \bar{p}_i)(p_{j,t} - \bar{p}_j)}{\sqrt{\sum_{t=1}^T (p_{i,t} - \bar{p}_i)^2 \sum_{t=1}^T (p_{j,t} - \bar{p}_j)^2}}$$

**和 A_geo / A_dem 的根本差异：**

| 维度 | A_geo / A_dem | A_corr |
|---|---|---|
| 建图依据 | 物理空间（距离 + 地形） | 数据统计（时间序列同步性） |
| 邻居选择 | 强制 k-NN（每站恰好 12 个邻居） | 自适应阈值（邻居数不固定） |
| 边权重含义 | 高斯核距离衰减 | 皮尔逊相关系数本身 |
| 典型连边 | 昆明 ↔ 楚雄（100km 内） | 武汉 ↔ 南京（梅雨带，~600km） |
| 截断策略 | 负权重截断（全部非负） | 仅保留正相关（≥ 0.35） |

**threshold = 0.35 的选择依据：** 对于 31 年样本量（~11000 天），$r > 0.35$ 对应 $p < 0.05$（95% 置信度）下显著非零，过滤掉统计噪声。

**未纳入扩散转移矩阵：** `A_corr` 在 `get_all_topologies` 中未被传入 `compute_transition_matrices`，它作为独立邻接矩阵直接使用，不做行/列归一化——因为遥相关本身是双向的（武汉和南京互相关），且阈值截断后度数分布极不均匀（梅雨区稠密、西北干旱区稀疏），强制归一化会稀释有效信号。

---

### 2.5 `compute_transition_matrices(adj)` — 前向/后向随机游走转移矩阵

| 属性 | 值 |
|---|---|
| 位置 | 第 90–111 行 |
| 类型 | `@staticmethod` |
| 输入 | `adj: (N, N)` 对称非负邻接矩阵 |
| 输出 | `List[np.ndarray]` 两个 `(N, N)` 的 `float32` 矩阵 `[P_f, P_b]` |

**核心公式：**

$$P_f = D_{out}^{-1} A, \quad P_b = D_{in}^{-1} A^T$$

**五步构建流程：**

```
1. 负权重截断          →  np.maximum(adj, 0.0) 防御性处理
2. 加自环              →  adj = adj + np.eye(N)  对角线权重翻倍
3. 前向转移 P_f        →  行归一化：每行和 = 1，模拟正向扩散概率
4. 后向转移 P_b        →  先转置再行归一化，模拟反向回流概率
5. 返回 [P_f, P_b]
```

**物理意义：**

| 矩阵 | 含义 | 物理场景 |
|---|---|---|
| $P_f[i, j]$ | 从节点 $i$ 出发沿边走到 $j$ 的概率 | 水汽从昆明正向传递到楚雄 |
| $P_b[i, j]$ | 从节点 $i$ 出发逆边走到 $j$ 的概率 | 楚雄的降水条件反向影响昆明上游水汽 |

**与 GCN 对称归一化 $\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}$ 的区别：**

| 维度 | GCN 对称归一化 | 扩散 GCN 双向转移矩阵 |
|---|---|---|
| 对称性 | 保持对称 | 非对称（概率转移矩阵） |
| 特征值 | $[-1, 1]$，适合谱卷积近似 | 行/列和为 1，合法概率分布 |
| 物理意义 | 无向"特征平均" | 有向**随机游走扩散** |
| 降水应用 | 不适用（扩散有方向性） | 适用（水汽有输送方向） |

**为什么 A_adp 跳过此函数？** 因为 `A_adp = Softmax(ReLU(E₁E₂^T))` 本身就是**行归一化的转移矩阵**（Softmax 强制每行和为 1），可以直接和物理预计算的 `P_f` / `P_b` 拼接进 `DiffusionGraphConv`。

---

### 2.6 `get_all_topologies(historical_precip, force_recompute=False)` — 总装调度入口

| 属性 | 值 |
|---|---|
| 位置 | 第 113–147 行 |
| 输入 | `historical_precip: Optional[np.ndarray]`、`force_recompute: bool` |
| 输出 | `Dict[str, np.ndarray]` 7 个 `(N, N)` 的 `float32` 数组 |

**完整执行流程：**

```
① 缓存检查
   │
   ├─ cache_file 存在且 force_recompute=False → 直接 .np.load 返回，跳过所有计算
   │
   └─ 缓存缺失或强制重算 → 继续 ↓

② Haversine 距离矩阵
   dist_mat = haversine_distance_matrix(coords)

③ 三张静态物理邻接图
   adj_geo = build_geo_adj(dist_mat)        # 依赖 dist_mat
   adj_dem = build_dem_adj(adj_geo)          # 依赖 adj_geo（继承稀疏结构）
   adj_corr = build_corr_adj(historical_precip) # 独立生成

④ 四张扩散转移矩阵
   trans_geo = compute_transition_matrices(adj_geo)   → pf_geo, pb_geo
   trans_dem = compute_transition_matrices(adj_dem)   → pf_dem, pb_dem
   （adj_corr 未做转移矩阵转换）

⑤ 打包 + 持久化
   topologies = {
       "adj_geo": adj_geo,      "adj_dem": adj_dem,    "adj_corr": adj_corr,
       "pf_geo":  pf_geo,       "pb_geo":  pb_geo,
       "pf_dem":  pf_dem,       "pb_dem":  pb_dem,
   }
   np.savez_compressed(cache_file, **topologies)
   return topologies
```

**缓存设计要点：**
- 文件名编码关键参数：`graph_adj_N{num_nodes}_k{knn_k}.npz`，防止不同配置误读
- 使用 `np.savez_compressed` 压缩，节省 30–50% 磁盘空间
- 物理先验图是静态的，训练过程中不会变化，缓存一次即可

---

## 3. 四图融合进入扩散图卷积

在模型端（`base_stgnn.py`），`BaseSTGNN.extract_features` 将预计算的物理转移矩阵和模型内部在线生成的自适应图拼接：

```python
adp = self.adaptive_adj()              # Softmax(ReLU(E₁E₂ᵀ)), 每次前向传播动态更新
all_supports = list(supports) + [adp]  # 拼接

# supports 来自 get_all_topologies() 返回值:
# [pf_geo, pb_geo, pf_dem, pb_dem]  ← 4 张静态物理转移矩阵
# + [adp]                           ← 1 张动态学习图
# = 共 5 个 support 并行进入 DiffusionGraphConv
```

`DiffusionGraphConv` 对每个 support 做 K=2 步扩散：

$$\text{GCN}(H, \{P_k\}) = \sum_{p \in \{P_f^{geo}, P_b^{geo}, P_f^{dem}, P_b^{dem}, A_{adp}\}} \sum_{k=0}^{K} P^k H W_{p, k}$$

---

## 4. 四重拓扑对比总结表

| 维度 | A_geo | A_dem | A_corr | A_adp |
|---|---|---|---|---|
| **物理约束** | 大地空间距离 | 距离 + 高程落差 | 历史降水遥相关 | 数据驱动隐式通道 |
| **构建方式** | 预计算（静态） | 预计算（静态） | 预计算（静态） | 模型内部在线生成 |
| **稀疏结构** | k-NN 强制（每站 12 邻居） | 继承 A_geo | 自适应阈值 0.35 | Softmax 行归一化 |
| **边权重来源** | 高斯核 exp(-d²/σ²) | 高斯核 × 地形衰减 | 皮尔逊相关系数 | E₁·E₂ᵀ 点积相似度 |
| **进入扩散 GCN** | ✅ P_f_geo, P_b_geo | ✅ P_f_dem, P_b_dem | ❌ 未转换（直接用邻接矩阵） | ✅ 本身就是转移矩阵 |
| **能否被训练优化** | ❌ 完全固定 | ❌ 完全固定 | ❌ 完全固定 | ✅ E₁, E₂ 可学习 |
| **捕捉的物理场景** | 邻近锋面、对流云团扩散 | 地形阻挡/雨影效应 | 梅雨锋、季风远距联动 | 台风远程水汽、低空急流 |

---

## 5. 配置参数速查

| 参数 | 默认值 | 所在位置 | 含义 |
|---|---|---|---|
| `knn_k` | 12 | `__init__` 第 20 行 | k-NN 邻居数 |
| `geo_sigma` | 200.0 km | `__init__` 第 21 行 | 地理高斯核尺度 |
| `dem_sigma` | 500.0 m | `__init__` 第 22 行 | 地形衰减尺度 |
| `threshold` | 0.35 | `build_corr_adj` 默认参数 | 遥相关截断阈值 |
| `embed_dim` | 16 | `AdaptiveAdjacency.__init__` | 自适应图嵌入维度 |
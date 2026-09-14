# U_Net_3D 多提前期降水预测模型诊断、迭代与工程重构报告

报告日期：2026-09-12  
研究对象：中国区域月降水空间异常预测，Lead 0-5  
独立评估时段：2023-01 至 2024-09，共 21 个月

## 摘要

本轮工作从两个异常现象出发：除 Lead 0 外，自动模型选择几乎总是选择 Ridge；同时五折交叉验证 ACC 明显高于独立 Test ACC。复核表明，这两个现象并不能简单归因于 Ridge 能力不足。约 321 个训练月份需要支撑 170 余维预测因子和高维空间输出，线性强正则模型天然比高自由度非线性模型稳定；更重要的是，原验证流程使用不符合时间外推任务的随机交叉验证，并在切分前拟合了使用目标信息的 MCA/PCA，导致验证信息泄漏和乐观偏差。模型选择还存在多候选比较的赢家偏差，而 2023-2024 与历史时期之间存在明显年代漂移。

围绕这些问题，工作依次完成了时间序列验证、折内监督降维、ECMWF 基线约束、OOF 集成、观测重建、稳健异常变换、多模式“模式即样本”迁移和训练样本时间衰减。当前推荐产品的宏平均 ACC 从初始 0.0936 提升到 0.1980，ECMWF 基线为 0.0953；六个 lead 的 ACC 和 RMSE 点估计均优于 ECMWF。当前结果说明，真正有效的改进来自验证制度、目标质量、物理先验和样本构造，而不是单纯增加网络复杂度。

与此同时，项目完成了数据与工作区重构。Python 源码不再硬编码本地盘符；原始数据、缓存和实验结果分别由 `HYDRO_DATA_DIR`、`paths.cache_dir` 和 `paths.get_exp_dir()` 管理。约 10.03 GiB 历史实验产物和 0.69 GiB 缓存已迁出代码仓库，所有重要结果均得到保留。

## 1. 问题背景与初始结果

初始程序对每个 lead 分别构造预测因子主成分和目标 PCA，再在 Ridge、核 Ridge、SVR、随机森林、LightGBM、XGBoost 和 FiLM-UNet 之间按五折验证 ACC 选择模型。用户提供的结果显示：

| Lead | 选择模型 | CV ACC | Test ACC | ECMWF ACC | 模型 RMSE | ECMWF RMSE |
|---:|---|---:|---:|---:|---:|---:|
| 0 | FiLM-UNet | 0.27658 | 0.28560 | 0.33175 | 0.79161 | 0.77869 |
| 1 | KernelRidge-Poly | 0.17725 | 0.00645 | 0.02092 | 0.85233 | 0.83250 |
| 2 | Ridge | 0.18108 | 0.10472 | 0.03726 | 0.87607 | 0.83025 |
| 3 | Ridge | 0.19540 | 0.05746 | 0.01811 | 0.88537 | 0.83088 |
| 4 | Ridge | 0.19405 | 0.07648 | -0.00068 | 0.86261 | 0.82869 |
| 5 | Ridge | 0.21351 | 0.03095 | 0.06274 | 0.88286 | 0.82160 |

初始宏平均 Test ACC 为 0.09361，ECMWF 为 0.07835，绝对增益只有 0.01526；模型 RMSE 在所有 lead 上均没有优于 ECMWF。Lead 1 的 CV ACC 为 0.17725，但 Test ACC 仅为 0.00645；Lead 5 的 CV ACC 为 0.21351，但 Test ACC 为 0.03095。这种系统性落差说明，验证分数不能可靠代表未来年代的泛化能力。

## 2. 如何定位问题

### 2.1 为什么多数 lead 选择 Ridge

Ridge 被频繁选中主要由数据结构决定，而非程序必然偏爱线性模型。

1. 样本量与维度不匹配。每个 lead 约有 321 个训练样本，但预测因子已达到 171-174 维，目标仍是 120 x 140 空间场的低维表示。样本数相对模型自由度过少。
2. PCA/MCA 后的变量接近连续低维模态。主成分之间的映射通常先呈现近似线性关系，Ridge 正好适合在共线条件下做稳定收缩。
3. 树模型和深度模型方差较大。随机森林、Boosting 和 FiLM-UNet 需要更多独立样本才能稳定学习区域性和非线性关系。月度序列虽然有三百余项，但相邻月份和同季节年份并非完全独立。
4. 输出端已经强降维。目标 PCA 只保留少量分量时，复杂模型能利用的局地非线性信息已经有限，而参数估计方差仍然存在。
5. 降水长提前期信噪比较低。Lead 1-5 中，稳健收缩往往比追逐训练期局部规律更有利，因此 Ridge 成为合理的基线赢家。

因此，解决方案不是强制排除 Ridge，而是修正验证流程、加入原始模式基线，并让复杂模型只有在严格 OOF 证据支持时才获得较大权重。

### 2.2 为什么 CV ACC 明显高于 Test ACC

代码检查和复现实验确认了五类原因。

| 原因 | 机制 | 后果 |
|---|---|---|
| 随机 KFold | 未来月份可进入较早月份的训练集，同一季节和相邻年代被打散 | 验证任务比真实未来外推容易 |
| 监督 MCA 在切分前拟合 | MCA 使用预测因子与目标的协方差，验证目标参与了空间基底估计 | 验证目标信息泄漏进特征 |
| 预处理未完全折内拟合 | 气候态、标准化、PCA 若读取验证期统计量，会降低分布差异 | CV 分数被系统性抬高 |
| 多模型择优偏差 | 在多个模型、PCA 阈值和超参数中取最大值 | 最优 CV 分数包含赢家偏差 |
| 年代漂移和短测试集 | 2023-2024 环流背景与历史分布不同，且测试仅 21 个月 | Test 方差大，历史最优权重可能失效 |

Lead 5 提供了清楚的年代漂移证据：某一阶段历史 OOF 集成 ACC 为 0.08045，历史 ECMWF 为 0.01873；但在 2023-2024 上，集成与 ECMWF 分别为 0.06802 和 0.08098，优劣发生反转。这表明继续依据这 21 个月调整权重会形成测试集调参，不能作为可信改进。

## 3. 验证制度重建

第一轮实质性改动不是更换模型，而是重建评估制度。

### 3.1 时间顺序切分

随机 KFold 被替换为 `TimeSeriesSplit(n_splits=5, test_size=21)`。每折只用过去训练、用连续的未来 21 个月验证，使折内任务与最终 21 个月外推任务保持一致。近期折采用更高权重，以降低过老年代对当前模型选择的支配。

对应脚本：`train_pcr_multilead.py`（AMS 的时间切分、近期折加权与候选选择）；`experiment_model_as_sample_transfer.py`（迁移模型的独立滚动切分）。

### 3.2 所有监督步骤移入折内

MCA、目标 PCA、气候态、标准化和回归器均在每个训练折中重新拟合。验证月份只允许经过由训练折确定的变换，不能参与基底或统计量估计。该改动直接消除了原流程最主要的乐观偏差。

对应脚本：`train_pcr_multilead.py`（折内 MCA、目标 PCA 和 AMS 训练）；`analyze_multimodel_raw.py`（多模式训练期气候态与日期对齐）；`experiment_model_as_sample_transfer.py`（迁移实验的折内模式气候态、PCA 和 Ridge）。

### 3.3 ECMWF 成为显式候选和融合锚点

ECMWF 原始预报不再只是事后比较对象，而是进入模型选择流程。每个候选订正模型都与 ECMWF 在 0.0-1.0 权重网格上融合：权重 0 表示完全使用 ECMWF，权重 1 表示完全使用订正模型。只有 OOF 证据支持时，订正模型才偏离原始动力模式。

对应脚本：`train_pcr_multilead.py`（ECMWF 候选和逐 lead 融合权重）；`build_fixed_ensemble.py`（后续 AMS/stacking 的 OOF 融合与安全回退）。

### 3.4 保存 OOF 预测

五个连续验证折共形成 105 个 OOF 月份。后续的模型集成、幅度校准、安全回退和 bootstrap 均只读取这些历史 OOF 预测，不读取 2023-2024 标签。这样可以将模型训练、模型选择和最终测试分开审计。

对应脚本：`train_pcr_multilead.py` 和 `experiment_model_as_sample_transfer.py`（生成各自 OOF）；`experiment_seasonal_stacking.py`（stacking OOF）；`build_fixed_ensemble.py`、`build_multisource_oof_ensemble.py`（用 OOF 选融合权重）；`evaluate_multilead_metrics.py`（测试指标和块 bootstrap）。

修正验证后，近期加权 AMS 的宏平均 Test ACC 为 0.0949，ECMWF 为 0.0784。结果只小幅改善，但可信度显著提高，并清楚暴露 Lead 5 的负迁移。这个阶段的重要产出是得到可信诊断，而不是追求表面高分。

## 4. 评价指标扩展

单一平均空间 ACC 不足以描述降水预报质量，因此新增了完整的面积加权评估模块。主要指标包括：

| 指标 | 含义 | 作用 |
|---|---|---|
| Spatial ACC | 每月预报与观测空间型态相关 | 主优化目标 |
| Pooled RMSE | 汇总月份与有效格点的面积加权误差 | 衡量绝对误差幅度 |
| MSESS/RMSE skill | 相对 ECMWF 的均方误差技巧 | 判断订正是否真正降低误差 |
| MAE | 对极端误差较不敏感的幅度指标 | 补充 RMSE |
| Bias | 区域平均系统偏差 | 诊断整体偏湿或偏干 |
| Centered RMSE | 去除平均偏差后的型态误差 | 区分均值和空间结构问题 |
| TCC | 每个格点的时间相关并做 Fisher 汇总 | 衡量时间变化一致性 |
| Willmott index | 综合幅度和一致性的拟合指数 | 提供非相关型综合评价 |
| CSI 与事件频率 | 湿、强湿和干事件命中能力 | 检查阈值事件表现 |

不确定性使用三个月移动块 bootstrap 估计。块抽样保留相邻月份的部分自相关，优于把 21 个月当作独立同分布样本。报告同时给出 95% 区间和增益为正的概率，避免只比较点估计。

对应脚本：`evaluate_multilead_metrics.py` 计算全国指标和块 bootstrap；`evaluate_stratified_skill.py` 按季节、区域拆分诊断。

## 5. 数据质量修复

### 5.1 站点坐标编码错误

原预处理曾把 CMA 度分坐标直接除以 100。例如 `4015` 被解释为 40.15 度，但正确含义是 40 度 15 分，即 40.25 度。误差最高约 0.4 度，会影响沿海、山地和站点稀疏区的插值。

`rebuild_station_observations.py` 实现了度分到十进制度的正确转换，按站点和月份重建月降水，并在固定站网凸包内插值。重建产品覆盖 1994-01 至 2024-12，共 372 个月，稳定有效格点 5243 个。

### 5.2 固定参考期，避免目标泄漏

站点月气候态固定使用 1994-2010，每个站点和月份至少需要 10 个有效参考年。2023-2024 不参与气候态估计。新旧目标在训练期和测试期的空间 ACC 分别约为 0.903 和 0.877，说明修复改变了空间结构，不能通过简单缩放替代重训。

对应脚本：`rebuild_station_observations.py` 同时承担坐标转换、固定参考期和网格重建；`train_pcr_multilead.py`、`experiment_seasonal_stacking.py` 使用重建后的观测重新训练或验证。

### 5.3 稳健百分比异常

原始异常为：

```text
x = (P - climatology) / climatology
```

干旱区和少雨月份的气候态接近零，会产生极端比例值，导致 RMSE 被少数格点支配。最终采用固定、无参数的有符号对数变换：

```text
z = sign(x) * log1p(abs(x))
```

该变换同时应用于观测、ECMWF、季节 stacking 和评估，不读取测试期统计量。采用此变换后，六个 lead 的 ACC 和 RMSE 点估计首次全部优于 ECMWF，宏平均 ACC 从 0.1112 提升到 0.1503。

对应脚本：`train_pcr_multilead.py`（`AMS_TARGET_TRANSFORM=signed_log1p`）、`experiment_seasonal_stacking.py`（`--observation-transform signed_log1p`）、`build_fixed_ensemble.py`（同口径融合）、`evaluate_multilead_metrics.py`（同口径评估）。

## 6. 现有数据审计与取舍

`HYDRO_DATA_DIR` 下的资料并非都适合直接拼入当前模型。审计按“是否为带真实初值的季节预测、是否覆盖测试期、时间尺度是否一致、系统版本是否稳定”进行判断。

对应脚本与记录：`analyze_multimodel_raw.py` 实测季节模式覆盖与空间技巧；`experiment_jja_march_multimodel.py` 评估 March-JJA 三模式；HiCIPC 的压缩包盘点和用途判断记录于 `DATA_AUDIT.md`，当前没有将 HiCIPC 接入 Lead 0-5 的训练脚本。

| 数据源 | 审计结论 | 最终用途 |
|---|---|---|
| ECMWF SEAS5 | 覆盖完整，是最强动力基线 | 主输入、幅度锚点和回退基线 |
| ERSSTv5 | 历史连续，适合提取遥相关 | ENSO/IOD 等海温预测因子 |
| NCEP CFSv2 | 与 ECMWF 误差互补，但存在版本缺口 | 低权重 stacking 和迁移样本 |
| JMA CPS3 | 部分 lead 有互补性，近期 hindcast 不完整 | 迁移训练样本，不直接填补测试输入 |
| NCC/UKMO | 历史样本可用，末期覆盖不完整 | 扩展迁移和年代稳健性实验 |
| ECMWF System4 | 时段较老，和 SEAS5 域偏移明显 | 实验证明不宜直接并入 |
| BCC-CPSV3 | 主要为每年 3 月起报 | 独立 March-JJA 专项实验 |
| CMA 日站资料 | 目标质量最高，可延长至 2025-03 | 重建观测、季节累计和极端标签 |
| HiCIPC | CMIP6 SSP 年度气候影响指数，不是初始化预报 | 长期风险产品或静态空间预训练 |

### 6.1 HiCIPC 的内容与边界

HiCIPC 包含 20 个 CMIP6 模式、ssp126/245/370/585 四种情景、1979-2099 年度数据，共审计到 1263 个 NetCDF。空间分辨率约 0.1 度，覆盖中国及周边。22 类指数包括 CDD、CWD、R10、R20、RX1DAY、SDII、多个降水百分位总量、热浪、生长季和高温日数等。

这些资料适合长期气候变化、洪旱风险、农业热害、不同 SSP 情景比较，以及学习极端指数的空间 EOF 或自编码器基底。它们不适合直接作为 Lead 0-5 月季节预报输入，因为 SSP 情景模拟没有对应目标月份的真实初始化状态；直接混用会造成任务定义错误，甚至形成概念性泄漏。

## 7. 模型迭代路线与实验决策

整个改进过程遵循同一原则：每轮只提出一个可检验假设，使用历史 OOF 选择参数，再由固定测试集评估；没有稳定增益的方案不进入推荐产品。

| 阶段 | 核心假设与改动 | 对应程序脚本 | 宏 Test ACC | 结论 |
|---|---|---|---:|---|
| 初始 AMS | 随机 CV，多候选直接择优 | `train_pcr_multilead.py`（原始版本；现文件已改造） | 0.0936 | 验证偏乐观，RMSE 全面落后 ECMWF |
| 无泄漏时间 CV | TimeSeriesSplit、折内 MCA/PCA、ECMWF 融合 | `train_pcr_multilead.py` | 0.0949 | 分数更可信，发现 Lead 5 负迁移 |
| 固定 OOF 集成 | AMS 与 seasonal stacking 互补 | `train_pcr_multilead.py`、`experiment_seasonal_stacking.py` → `build_fixed_ensemble.py` | 约 0.0979 | RMSE 总体改善，但并非所有 lead 均领先 |
| 订正观测重训 | 修正坐标、固定气候态、重建掩膜 | `rebuild_station_observations.py` → `train_pcr_multilead.py`、`experiment_seasonal_stacking.py` → `build_fixed_ensemble.py` | 0.1112 | 目标更可信，Lead 2/5 仍不稳定 |
| 有符号对数异常 | 抑制干旱区极端比例值 | `train_pcr_multilead.py`、`experiment_seasonal_stacking.py` → `build_fixed_ensemble.py` | 0.1503 | 六个 lead 的 ACC/RMSE 点估计均领先 |
| 三模式样本迁移 | NCEP/JMA 作为额外训练样本 | `analyze_multimodel_raw.py` → `experiment_model_as_sample_transfer.py` → `build_multisource_oof_ensemble.py` | 0.1737 | 宏增益区间首次完全高于零 |
| 五源扩展迁移 | 再加入 NCC/UKMO | `experiment_model_as_sample_transfer.py`（配置训练源）→ `build_multisource_oof_ensemble.py` | 0.1830 | 宏 ACC 更高，但 Lead 3 JJA 明显退化 |
| 加入 System4 | 用旧系统扩大样本 | `analyze_multimodel_raw.py`、`experiment_model_as_sample_transfer.py`（配置训练源）→ `build_multisource_oof_ensemble.py` | 0.1769 | 域偏移大于样本收益，否决 |
| 时间衰减迁移 | 按 lead 选择 0/5/10 年半衰期 | `experiment_model_as_sample_transfer.py` → `build_fixed_ensemble.py` | 0.1980 | 当前推荐，均衡性和年代适应性最好 |

表中数值用于描述研发过程，但并非每一行都是只改变一个变量的严格消融。观测重建和有符号对数变换改变了目标定义，初始 0.0936 与当前 0.1980 因而不能被解释为单一算法在完全相同数据口径下的净提升。可以严格横向比较的是同一行实验中的模型与其 ECMWF 基线，以及使用相同目标和测试窗的相邻消融实验。

箭头表示先生成上游候选或 OOF，再由后续脚本融合；`evaluate_multilead_metrics.py` 是各阶段统一的 Test ACC、RMSE 和置信区间评估入口，不负责训练。初始版本的运行输出已保存，但当前 `train_pcr_multilead.py` 是改造后的版本，不能直接执行它来重现原先含随机 CV 的行为。五源与 System4 行通过 `AMS_TRANSFER_SOURCES` 配置同一迁移脚本，并非各有独立的 Python 文件。

### 7.1 季节 stacking

季节 stacking 使用 ECMWF、NCEP、最近三期合法观测滞后和去年同月观测，按 DJF、MAM、JJA、SON 选择非负且和为 1 的权重。权重只能由连续 OOF 月份确定，并向 ECMWF 收缩。这个分支与 AMS 的误差结构不同，因此可用于集成，而不是替代主模型。

固定 `0.5 * AMS + 0.5 * seasonal stacking` 后，仅用 105 个 OOF 月拟合全国平均分量和空间异常分量的非负幅度系数。该方法改善了 Lead 1、3、4 的 MSESS，但 Lead 2 ACC 和 Lead 5 RMSE 尚未超过 ECMWF，说明统一固定权重不够稳健。

对应脚本：`experiment_seasonal_stacking.py` 生成季节预测及 OOF；`train_pcr_multilead.py` 生成 AMS 及 OOF；`build_fixed_ensemble.py` 选择权重、做幅度校准或季节安全回退；`evaluate_multilead_metrics.py` 评估最终数组。

### 7.2 物理桥梁特征

从现有环流和 SST 中构建了 WNPSH、南亚高压、东亚急流、850 hPa 季风、索马里急流、海平面气压梯度、ENSO、IOD、热带印度洋和暖池等 11 个低维指数。其思想是把可预报的大尺度环流作为局地降水的物理桥梁。

低维物理指数保留为候选特征，但直接追加高维预报 SST 场会使 Lead 5 的 OOF 和测试技巧下降；物理指数与年循环的交互项也使宏 ACC 从 0.1503 降至 0.1467。因此这两类扩展默认关闭，避免在小样本中增加无效自由度。

对应脚本：`train_pcr_multilead.py` 中的 `build_physical_indices()` 和 `AMS_USE_PHYSICAL_INDICES`、`AMS_USE_FORECAST_SST`、`AMS_USE_SEASON_INTERACTIONS` 开关；`test_physical_indices.py` 检查特征构造；`build_fixed_ensemble.py` 与 `evaluate_multilead_metrics.py` 用于产物融合和比较。

### 7.3 高维残差订正为何被否决

多模式 EOF/Ridge 残差模型尝试直接学习 ECMWF 的高维空间误差。其宏 Test ACC 为 0.08163，低于 ECMWF 的 0.08293，Lead 4/5 明显退化。结果说明约 300 个训练月份不足以稳定估计高维、年代变化的模式误差。该路线没有进入正式产品。

对应脚本：`experiment_multimodel_residual.py`（训练与滚动验证），输入模式场由 `analyze_multimodel_raw.py` 整理。

### 7.4 BCC March-JJA 专项实验

BCC-CPSV3 只有每年 3 月起报的资料，因此被单独对齐到 JJA 三个月累计，而没有强行加入全年逐月模型。2011-2020 OOF 选择得到 ECMWF/NCEP/BCC 权重 0.15/0.75/0.10；历史 OOF ACC 从 ECMWF 的 0.0162 提高到 0.0659，但 2021-2024 独立测试中集合 ACC 为 0.2238，低于 ECMWF 的 0.2830。

该失败案例说明，只有 10 个 OOF 年和 4 个测试年时，权重非常不稳定。BCC 暂时只保留为 JJA 参考成员或用于提取 WNPSH/南亚高压等少量指数。

对应脚本：`experiment_jja_march_multimodel.py` 完成 March 起报文件对齐、JJA 累计、OOF 权重选择和测试评估；它不参与全年逐月最终产品。

### 7.5 “模式即样本”迁移

前沿研究的共同做法不是简单把多个模式拼成更高维输入，而是把每个模式 hindcast 分别与同一观测配对，作为额外训练样本。本项目据此实现以下流程：

1. ECMWF、NCEP、JMA 分别形成“模式预报-验证观测”训练对。
2. 验证和最终测试始终只在 ECMWF 域上进行，避免改变业务输入定义。
3. 每个时间折独立重算模式气候态、输入 PCA 和目标 PCA。
4. PCA 维数、Ridge alpha、辅助模式样本权重和输出映射权重全部由 OOF 选择。
5. 多数 lead 选择较低辅助权重和 `alpha=1000`，反映小样本下强收缩仍然必要。

三模式迁移与 seasonal stacking 的平衡产品达到宏 ACC 0.1737。Lead 1-5 的 JJA ACC 分别为 0.302、0.332、0.303、0.321、0.317，已进入所比较中国夏季降水研究的常见范围。

对应脚本链：`analyze_multimodel_raw.py` 读取并对齐各模式 → `experiment_model_as_sample_transfer.py` 训练迁移候选和保存 OOF → `build_multisource_oof_ensemble.py` 根据历史 OOF 融合 AMS、迁移候选与 seasonal stacking → `evaluate_multilead_metrics.py` 和 `evaluate_stratified_skill.py` 评估全国及 JJA 技巧。五源扩展和 System4 对照使用相同训练入口，通过 `AMS_TRANSFER_SOURCES` 改变训练源。

### 7.6 时间衰减迁移

为了应对年代漂移，训练样本新增无衰减、5 年半衰期和 10 年半衰期三个候选。时间权重与其他超参数一起仅由滚动 OOF 选择：

- Lead 0、1、2、5 选择 10 年半衰期；
- Lead 3 选择 5 年半衰期；
- Lead 4 保留无衰减。

这说明非平稳性具有 lead 依赖，不能使用一个统一的近期窗口。四折选择、第五折完全留出的保守实验仍得到宏 Test ACC 0.1904，六个 lead 在历史留出折均超过 ECMWF，为总体迁移增益提供了额外年代外推证据。

对应脚本链：`experiment_model_as_sample_transfer.py` 中的 `RECENCY_HALFLIVES` 和 `AMS_SELECTION_FOLD_COUNT` 生成五折或四折选择的迁移预测与 OOF → `build_fixed_ensemble.py` 根据迁移 OOF 与 signed-log stacking OOF 生成 `ensemble_safe.npy` → `evaluate_multilead_metrics.py` 核验 ACC/RMSE，`evaluate_stratified_skill.py` 诊断 DJF/JJA。

## 8. 当前推荐模型与最终结果

当前推荐产品位于：

```text
<HYDRO_WORKSPACE>/results/U_Net_3D/model_as_sample_transfer_recency_run/
multi_lead_predict_results_ensemble_safe.npy
```

模型可概括为：以 ECMWF 为业务域和幅度锚点，以 ERSST、合法观测滞后和大尺度环流为主预测信息；使用 NCEP/JMA 扩充训练样本；通过折内 PCA、强 Ridge 正则和按 lead 时间衰减控制方差；最后利用历史 OOF 进行受约束融合。

对应的最终生成顺序是 `rebuild_station_observations.py`（目标）→ `train_pcr_multilead.py` 和 `experiment_seasonal_stacking.py`（基线分支）→ `experiment_model_as_sample_transfer.py`（时间衰减迁移候选）→ `build_fixed_ensemble.py`（当前 `ensemble_safe.npy`）→ `evaluate_multilead_metrics.py`（全国指标）。`build_multisource_oof_ensemble.py` 属于上一阶段三源/五源对照，不是当前推荐文件的直接生成脚本。

| Lead | 最终 ACC | ECMWF ACC | ACC 增益 | 最终 RMSE | ECMWF RMSE | RMSE 技巧 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.3932 | 0.3593 | +0.0339 | 0.5272 | 0.5343 | +1.33% |
| 1 | 0.1447 | 0.0318 | +0.1129 | 0.5701 | 0.5837 | +2.32% |
| 2 | 0.1744 | 0.0557 | +0.1187 | 0.5665 | 0.5798 | +2.28% |
| 3 | 0.1436 | 0.0325 | +0.1111 | 0.5686 | 0.5813 | +2.18% |
| 4 | 0.1577 | 0.0186 | +0.1391 | 0.5670 | 0.5780 | +1.89% |
| 5 | 0.1743 | 0.0740 | +0.1003 | 0.5648 | 0.5725 | +1.34% |

宏平均 ACC 为 0.1980，ECMWF 为 0.0953；宏平均 RMSE 为 0.5607，ECMWF 为 0.5716。研发过程中的宏 Test ACC 从初始 0.0936 增长到当前 0.1980；由于中途修正了观测和异常定义，这一变化应理解为整个数据-验证-模型系统的综合改进，而不是单一模型结构的严格净增益。

10,000 次三个月移动块 bootstrap 中，六个 lead 的 ACC 增益 95% 区间下界均高于零；宏 ACC 增益区间为 `[0.0209, 0.1932]`，增益为正概率为 99.6%。Lead 1/2 的 MSESS 区间完全高于零；其他 lead 的 RMSE 点估计改善，但区间仍跨零。

按季节看，JJA Lead 1-5 ACC 分别为 0.291、0.354、0.300、0.337、0.337。DJF 已较早期版本明显改善，但 Lead 4 冬季 ACC 仍为 -0.012，是当前主要短板之一。

## 9. 与前沿工作的比较

检索到的相关研究通常通过以下途径获得较高分数：

| 研究 | 方法 | 高分的关键 | 与本项目的差异 |
|---|---|---|---|
| Jin et al. (2022) | 多 GCM hindcast 预训练，再用 ERA5 和降水迁移 | 大量模式样本、先学模式再观测订正 | PCC 0.71 主要是同期环流重建，不等同 1-5 月真实预报 |
| Ling et al. (2022) | 直接降水订正与间接环流订正两条深度路径 | 将集合成员视为训练样本 | 目标是中国夏季降水，不是全年逐月 |
| Deng et al. (2023) | U-Net 空间订正 JJA 模式降水 | 动力场强先验、季节累计更平滑 | 主要约 1 个月 lead |
| Lu et al. (2023) | Attention U-Net、PCA、浅层网络、早停 | 遥相关注意力与小样本控制 | 任务集中于夏季和 1-3 月 lead |
| Yang et al. (2023) | 两阶段 TU-Net，先预测 WNPSH 再预测降水 | 物理瓶颈降低局地预测难度 | 区域和季节范围更窄 |

相关文献：

- Jin et al. (2022): https://doi.org/10.1007/s13351-022-1174-7
- Ling et al. (2022): https://doi.org/10.1088/1748-9326/aca68a
- Deng et al. (2023): https://doi.org/10.1016/j.aosl.2022.100322
- Lu et al. (2023): https://doi.org/10.1029/2023EA003129
- Yang et al. (2023): https://doi.org/10.1175/AIES-D-22-0078.1

当前 JJA 结果已达到所比较研究的常见 ACC 范围，但不能直接宣称全面领先。不同论文的目标季节、空间区域、提前期、异常定义和测试年份并不一致。尤其是部分高 PCC 来自同期环流重建，而不是严格的未来季节预测。

## 10. 工程重构过程

模型实验产生了大量数组、OOF、日志和指标，原项目同时存在 `E:/DATA`、旧 `E:/workplace` 和仓库目录混用的问题。数据路径和代码逻辑耦合后，会造成以下风险：

- 换机器后脚本无法运行；
- 结果可能写入源码目录并被误提交；
- 同名实验覆盖历史结果；
- 缓存、原始数据和最终产品的所有权不清；
- 复现实验时难以确认输入版本。

### 10.1 统一路径模块

新增 `U_Net_3D/project_paths.py`，内部统一实例化：

```python
from utils.paths import SubprojectPaths

paths = SubprojectPaths(__file__, subproject_name="U_Net_3D")
```

路径职责被明确划分：

| 内容 | 解析方式 | 实际命名空间 |
|---|---|---|
| 原始模式和 ERSST | `paths.get_raw_data(...)` | `<HYDRO_DATA_DIR>/...` |
| 处理后数组和站点观测 | `paths.cache_dir` | `<HYDRO_WORKSPACE>/cache/U_Net_3D` |
| 模型、OOF、指标和图件 | `paths.get_exp_dir(name)` | `<HYDRO_WORKSPACE>/results/U_Net_3D/<name>` |

`utils.paths` 现在要求 `.env` 明确提供 `HYDRO_WORKSPACE` 和 `HYDRO_DATA_DIR`，不再静默使用机器相关的默认盘符。`get_raw_data()` 拒绝绝对路径和包含 `..` 的越界路径。

### 10.2 脚本改造

完成路径改造的主要程序包括：

- `train_pcr_multilead.py`
- `prepare_data_v4.py`
- `train_pcr_best.py`
- `analyze_multimodel_raw.py`
- `rebuild_station_observations.py`
- `experiment_seasonal_stacking.py`
- `experiment_model_as_sample_transfer.py`
- `experiment_jja_march_multimodel.py`
- `build_fixed_ensemble.py`
- `build_multisource_oof_ensemble.py`
- `build_oof_model_hybrid.py`
- `build_spatial_skill_blend.py`
- `evaluate_multilead_metrics.py`
- `evaluate_stratified_skill.py`

所有输出参数仍允许命令行显式覆盖，但默认值均落到受管理的实验目录。显式输出路径也会自动创建父目录。NetCDF 读取改用当前 `nc` 环境已有的 `xarray`，消除了对缺失 `h5py` 的启动依赖。

### 10.3 产物迁移

历史数据没有删除，而是按语义迁移：

| 原位置或类型 | 新位置 |
|---|---|
| 旧 `*_run` 实验目录 | `results/U_Net_3D/<原实验名>` |
| `lr_unet`、`hr_unet`、SST 中间数组 | `cache/U_Net_3D/prepared` |
| 重建观测和站点 CSV | `cache/U_Net_3D/station_observations` |
| 旧根目录多 lead 数组 | `results/U_Net_3D/legacy_default` |
| 历史 OOF、指标和日志 | `results/U_Net_3D/artifact_archive` 或对应实验目录 |
| 旧权重、图片和训练日志 | `results/U_Net_3D/legacy_unet` |

迁移后缓存包含 8 个文件、约 0.69 GiB；实验结果包含 277 个文件、约 10.03 GiB。旧工作区已无剩余内容，`U_Net_3D` 源码目录不再包含 `.npy`、`.npz`、`.csv`、`.json`、`.log`、权重或 `__pycache__`。

## 11. 验证与可复现性

工程重构完成后进行了以下验证：

1. 23 个 Python 文件完成内存编译检查，无语法错误。
2. 12 个无 fixture 回归测试全部通过，包括 OOF 权重选择、幅度恢复、季节安全门、物理指数和路径越界保护。
3. 10 个主要 CLI 入口通过 `--help` 启动检查。
4. 使用真实 ECMWF NetCDF 验证读取结果，得到 `(6, 60, 70)` 的 `float32` 有限数组。
5. 迁移后的当前最佳产品完成端到端评估，ACC/RMSE 与迁移前一致。
6. Python 源码扫描未发现本地盘符、`workplace` 或 `/Users` 路径硬编码。

`nc` 环境没有安装 pytest，因此测试函数通过同一 Python 环境直接加载执行，而不是使用 pytest runner。该限制不影响函数断言结果，但后续建议在环境依赖文件中显式加入 pytest，以恢复标准测试命令。

关键文件：

- 路径管理：`U_Net_3D/project_paths.py`
- 全局路径工具：`utils/paths.py`
- 主训练程序：`U_Net_3D/train_pcr_multilead.py`
- 观测重建：`U_Net_3D/rebuild_station_observations.py`
- 模式样本迁移：`U_Net_3D/experiment_model_as_sample_transfer.py`
- OOF 集成：`U_Net_3D/build_multisource_oof_ensemble.py`
- 完整评估：`U_Net_3D/evaluate_multilead_metrics.py`
- 数据审计：`U_Net_3D/DATA_AUDIT.md`
- 工作区说明：`U_Net_3D/WORKSPACE_LAYOUT.md`

## 12. 尚不能回避的限制

1. 2023-2024 只有 21 个测试月份，单 lead 置信区间仍可能较宽。
2. 该测试期已经参与多轮开发期诊断。即使每次参数选择只读取历史 OOF，它也不再是完全未见的确认性测试集。
3. 月降水空间异常在 DJF 和部分长江区域的可预报性仍然偏低。
4. 多模式资料存在系统版本、变量口径和时间覆盖差异，不能把“更多数据”直接等同于“更多有效样本”。
5. 当前 RMSE 是有符号对数百分比异常空间中的无量纲误差，不能解释为毫米。
6. 当前结果优于本项目 ECMWF 基线，但与论文的任务定义不同，不能只凭 ACC 数值宣称绝对前沿领先。

## 13. 下一阶段建议

### 13.1 建立真正封存的确认性测试

优先使用 2024-10 至 2025-03 新增站点观测，或预先划定另一个完全封存的历史年代窗。模型、特征和权重在揭示标签前必须冻结。这是将开发期结果转化为论文确认性证据的最高优先级工作。

### 13.2 建立 JJA 和三个月累计专项产品

现有结果中 JJA 最稳定，也最接近相关文献任务。应把 JJA/汛期累计、华南前汛期、长江梅雨和华北雨季拆成单独任务，而不是继续增加全年统一模型自由度。

### 13.3 从日站资料构造更有业务意义的目标

可增加三个月累计降水、湿日数、R10、R20、RX1day、最长连续干期，以及华南、长江和华北雨带指数。极端事件指标应与 HiCIPC 的长期风险产品保持任务边界清晰。

### 13.4 强化物理中间任务

参考 TU-Net 路线，先预测 WNPSH、南亚高压、东亚夏季风或 ENSO/IOD 状态，再将其作为降水模型的中间监督或低维条件。相比直接追加高维场，这种物理瓶颈更符合当前样本规模。

### 13.5 保持严格的实验治理

后续实验应预先登记目标、候选模型、OOF 规则和最终测试窗；所有产物继续写入独立实验目录；每次结果必须同时报告 ACC、ECMWF 对照、RMSE/MSESS、置信区间和失败 lead。任何依据测试表现新增的回退规则都应标记为探索性，并在新测试窗复核。

## 14. 总结

本轮工作的核心认识是：初始结果中的 Ridge 集中选择是小样本、高维和强噪声条件下的合理反应，而 CV-Test 落差主要暴露了验证制度和年代稳定性问题。真正有效的路线不是不断增加黑箱复杂度，而是依次修复验证泄漏、目标数据、异常分布、动力模式约束和训练样本构造。

经过多轮可证伪实验，项目从初始宏 Test ACC 0.0936 提升到 0.1980，并在六个 lead 上同时取得高于 ECMWF 的 ACC 和更低的 RMSE 点估计。多模式“模式即样本”与按 lead 时间衰减是当前最有效的模型改进；高维残差、预报 SST 全场、System4 直接拼接和过多交互项则被实验证明不适合当前样本规模。

工程层面，代码、原始数据、缓存和实验结果已经彻底解耦，历史产物得到分类保留，默认脚本可在统一工作区中复现。下一步的关键不再是继续优化已经反复查看的 21 个月测试集，而是建立新的封存测试和更聚焦的 JJA/三个月累计产品。

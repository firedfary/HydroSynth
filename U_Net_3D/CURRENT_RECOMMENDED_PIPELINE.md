# 当前推荐模型完整流程

统一入口为 `run_recommended_pipeline.py`。该程序只负责编排、校验、记录和恢复；各阶段的科学算法仍保留在原脚本中，避免形成两套实现。

## 1. 一键运行

所有命令使用项目规定的 `nc` Conda 环境：

```powershell
& 'C:\Users\fired\anaconda3\envs\nc\python.exe' `
  'D:\HydroSynth\U_Net_3D\run_recommended_pipeline.py'
```

默认产物位于：

```text
<HYDRO_WORKSPACE>/results/U_Net_3D/current_recommended_pipeline/
```

程序默认启用 `--resume`。已存在的阶段只有在通过内容校验后才会跳过；不存在、损坏或口径不匹配的产物会重新生成。`--force` 会显式重跑并覆盖所选阶段的同名产物。

只查看解析后的完整命令：

```powershell
& 'C:\Users\fired\anaconda3\envs\nc\python.exe' `
  'D:\HydroSynth\U_Net_3D\run_recommended_pipeline.py' --plan
```

从迁移阶段继续到最终评估：

```powershell
& 'C:\Users\fired\anaconda3\envs\nc\python.exe' `
  'D:\HydroSynth\U_Net_3D\run_recommended_pipeline.py' `
  --from-stage transfer --through-stage stratified
```

## 2. 模型数据流

```text
CMA 站点月降水
  └─ 固定 1994–2010 气候态，重建无泄漏观测目标
       ├─ ECMWF + ERSST + 合法观测滞后 → signed-log AMS 基础产品
       ├─ ECMWF + NCEP + 近期/年际观测 → signed-log seasonal stacking
       └─ ECMWF / NCEP / JMA 分别与同一观测配对
            └─ 每折重算模式气候态、输入 PCA、目标 PCA
                 └─ Ridge 强正则 + 0/5/10 年半衰期 OOF 选择
                      └─ 时间衰减迁移候选

时间衰减迁移 OOF + seasonal stacking OOF
  └─ 每个 lead 选择非负融合权重
       └─ ECMWF 幅度锚定 + 季节安全回退
            └─ multi_lead_predict_results_ensemble_safe.npy
                 ├─ 全国 ACC/RMSE/MAE/TCC/Willmott/事件指标
                 └─ 季节和区域分层诊断
```

## 3. 七个执行阶段

| 阶段 | 直接执行脚本 | 核心职责 | 主要产物 |
| --- | --- | --- | --- |
| `observations` | `rebuild_station_observations.py` | 修正度分坐标；以 1994–2010 为固定参考期；生成站点网格异常 | 缓存中的观测 NPZ 和训练对齐 NPY |
| `base_ams` | `train_pcr_multilead.py` | signed-log AMS；时间滚动验证；所有监督变换折内拟合；保存基础场和 OOF | `base_ams/*.npy`、`ams_oof_patterns.npz` |
| `seasonal_stacking` | `experiment_seasonal_stacking.py` | 按季节组合 ECMWF、NCEP、近期观测和去年同月观测 | `seasonal_stacking_test_patterns.npz` |
| `transfer` | `experiment_model_as_sample_transfer.py` | 将 ECMWF/NCEP/JMA 作为独立训练配对；折内 PCA/Ridge；按 lead 选择时间衰减 | 迁移候选、`model_as_sample_transfer_oof.npz`、参数 JSON |
| `ensemble` | `build_fixed_ensemble.py` | 用迁移 OOF 与 stacking OOF 选非负权重；应用 ECMWF 锚点和安全回退 | `final/multi_lead_predict_results_ensemble_safe.npy` 及 JSON |
| `metrics` | `evaluate_multilead_metrics.py` | 全国面积加权指标和三个月移动块 bootstrap | 全国指标 CSV、置信区间 JSON |
| `stratified` | `evaluate_stratified_skill.py` | 按 DJF/MAM/JJA/SON 和区域诊断 | `evaluation/stratified_metrics.csv` |

`analyze_multimodel_raw.py` 的日期对齐、模式场读取和异常计算函数由 stacking 与迁移脚本直接复用，因此不需要单独生成一个中间阶段。

## 4. 固定的当前推荐口径

- 目标参考期：1994–2010。
- 目标变换：`signed_log1p`。
- 业务域：ECMWF；最终验证和测试不切换到辅助模式域。
- 迁移训练源：ECMWF、NCEP、JMA。
- 表示学习：每个滚动折独立计算模式气候态、输入 PCA 和目标 PCA。
- 回归器：Ridge；候选 `alpha` 为 1、10、100、1000，当前各 lead 选择结果以强收缩为主。
- 时间适应：无衰减、5 年半衰期、10 年半衰期由 OOF 分 lead 选择。
- 融合：迁移候选与 seasonal stacking 的非负 OOF 权重。
- 幅度：沿用 ECMWF 幅度约定；当前产品不启用额外 OOF 幅度校准。
- 安全机制：只有历史 OOF 支持的季节订正才偏离 ECMWF。
- 最终测试：21 个月；全国评估默认执行 10,000 次移动块 bootstrap。

## 5. 脚本职责边界

当前最终文件由 `build_fixed_ensemble.py` 直接生成。`build_multisource_oof_ensemble.py` 用于此前三源、五源扩展实验，不在当前产品链中；`experiment_multimodel_residual.py`、System4 扩展、高维预报 SST 和额外交互项均为已否决或默认关闭的研究分支。

基础 AMS 的 OOF 用于保留完整的基线审计链；最终融合必须读取 `transfer/model_as_sample_transfer_oof.npz`，不能误用基础 AMS 的 OOF。

## 6. 可复现性与故障恢复

每个阶段的标准输出和错误输出写入 `<run_dir>/logs/<stage>.log`。`pipeline_manifest.json` 保存实际 Python、参数、阶段命令、阶段专用环境变量、耗时、产物路径和文件大小。运行失败时，清单会记录失败阶段和异常；修复输入后以相同实验名重新运行即可从最后一个通过校验的阶段继续。

所有原始数据只从 `HYDRO_DATA_DIR` 读取；重建观测写入 `paths.cache_dir`，模型和评估产物写入 `paths.get_exp_dir()`。程序不会把数据、权重、日志或结果写入代码仓库。

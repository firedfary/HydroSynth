# U_Net_3D: ReMAP 次季节降水预测工程

`U_Net_3D` 子项目承载了 HydroSynth 系统中的核心次季节降水预测模型 **ReMAP**（Recency-decayed Model-As-sample Precipitation，时变衰减“模式即样本”次季节降水预测模型）。

模型面向中国区域月度降水空间异常预测任务（提前 0 至 5 个月，即 Lead 0–5），在 2023–2024 年连续 21 个月独立盲测集中取得了宏平均 ACC 0.20（相对 ECMWF 业务基线提升 +107.8%）、RMSE 全面下降的突破性表现。

---

## 1. 项目核心文档导航

- [**系统架构设计规范 (ARCHITECTURE.md)**](file:///d:/HydroSynth/U_Net_3D/ARCHITECTURE.md)：
  详尽阐述 ReMAP 模型的设计哲学、端到端数据流、无参 signed-log1p 变换数理推导、Engine 1（MAS 跨模式迁移）与 Engine 2（自适应季节堆叠）深度架构、空间物理振幅无偏恢复及典型业务案例剖析。
- [**研发进展与成果交付报告 (PROGRESS.md)**](file:///d:/HydroSynth/U_Net_3D/PROGRESS.md)：
  系统汇总 21 个月盲测集基准评测成果、逐 Lead 点估计对比、夏季主汛期与冬季弱强迫分层评估、极端旱涝 6 级离散 ETS 评分、全要素消融实验实证与超参数敏感性规律。
- [**工作区与路径管理规范 (WORKSPACE_LAYOUT.md)**](file:///d:/HydroSynth/U_Net_3D/WORKSPACE_LAYOUT.md)：
  说明代码仓库纯净原则、`HYDRO_DATA_DIR` 原始数据接入与 `<HYDRO_WORKSPACE>` 缓存和结果落盘机制。

---

## 2. 生产流水线核心程序

子项目的核心可执行生产脚本高度收敛，职责分明：

| 源码程序 | 职责定位 | 标准调用命令 |
|---|---|---|
| [`rebuild_station_observations.py`](file:///d:/HydroSynth/U_Net_3D/rebuild_station_observations.py) | 解析台站度分坐标，拟合 1994–2010 固定基准期，重建 0.5° 无泄漏观测真值 | `& 'C:\Users\fired\AppData\Local\anaconda3\envs\nc\python.exe' U_Net_3D/rebuild_station_observations.py` |
| [`train_final_model.py`](file:///d:/HydroSynth/U_Net_3D/train_final_model.py) | ReMAP 生产模型端到端训练、双引擎滚动拟合、OOF 凸融合、振幅恢复与盲测评测 | `& 'C:\Users\fired\AppData\Local\anaconda3\envs\nc\python.exe' U_Net_3D/train_final_model.py --exp-name final_model` |
| [`plot_all_cases.py`](file:///d:/HydroSynth/U_Net_3D/plot_all_cases.py) | 绘制 42 幅全季度空间检验图与宏观 ACC 增益矩阵，自动同步至桌面科研中枢 | `& 'C:\Users\fired\AppData\Local\anaconda3\envs\nc\python.exe' U_Net_3D/plot_all_cases.py` |
| [`test_train_final_model.py`](file:///d:/HydroSynth/U_Net_3D/test_train_final_model.py) | 覆盖 ReMAP 数学内核、奇函数对称性、面积加权、单纯形网格、冬季动力锚定与退化边界的单元测试（14项） | `& 'C:\Users\fired\AppData\Local\anaconda3\envs\nc\python.exe' -m unittest U_Net_3D/test_train_final_model.py` |

---

## 3. 运行环境与快速上手

根据根目录 `AGENTS.md` 规范，所有脚本统一运行于 Conda 专有科学计算环境 `nc`（解释器规范路径为 `C:\Users\fired\anaconda3\envs\nc\python.exe`，本机实际安装位于 `C:\Users\fired\AppData\Local\anaconda3\envs\nc\python.exe`）：

```powershell
# 1. 切换至工作区根目录
Set-Location -Path "d:\HydroSynth"

# 2. 运行单元测试验证数学内核与边界条件
& 'C:\Users\fired\AppData\Local\anaconda3\envs\nc\python.exe' `
  -m unittest 'd:\HydroSynth\U_Net_3D\test_train_final_model.py'

# 3. 执行端到端训练与评估一体化流程
& 'C:\Users\fired\AppData\Local\anaconda3\envs\nc\python.exe' `
  'd:\HydroSynth\U_Net_3D\train_final_model.py' --exp-name final_model
```

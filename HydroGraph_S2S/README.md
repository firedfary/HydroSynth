# HydroGraph_S2S: Spatio-Temporal Graph Neural Network for Subseasonal Precipitation Forecasting

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

面向全国 **2463 个国家气象测站** 的次季节与逐日降水时空图神经网络预报系统（HydroGraph_S2S）。

---

## 📑 完整技术全景文档 (Full Technical Documentation)

本项目提供极其详尽、从零基础到进阶推导的学术级技术文档：
👉 **[点击查阅 TECHNICAL_DOCUMENTATION.md](file:///d:/HydroSynth/HydroGraph_S2S/TECHNICAL_DOCUMENTATION.md)**

文档包含以下章节：
1. **引言与问题重构**：次季节（S2S）“可预报性沙漠”与尺度错位矛盾
2. **从零理解图与图神经网络**：欧氏空间 vs 非欧空间、邻接矩阵、消息传递与扩散 GCN
3. **多源数据资产与物理特征工程**：2463 站 31 年逐日观测 + 5,769 个 NetCDF 动力模式
4. **物理引导的多重时空图拓扑构建**：$A_{geo}$ (大地距离)、$A_{dem}$ (地形阻隔)、$A_{corr}$ (遥相关)、$A_{adapt}$ (自适应图)
5. **跨尺度时间解耦与特征线性调制**：三次样条基线趋势 + FiLM 宏观环境调制
6. **四类差异化建模范式与网络微观结构**：
   - 范式 1：次季节逐候（Pentad，5-Day）演变预报 GNN
   - 范式 2：趋势-残差混合多尺度逐日 ST-GNN
   - 范式 3：两阶段极端暴雨零膨胀分类-多分位数回归网络
   - 范式 4：水量绝对守恒时空图解混下尺度网络
7. **非对称损失函数与训练优化机制**：极端暴雨阶梯加权 Huber 损失、Pinball 损失
8. **气象与水文专业评估指标体系**：ACC, RMSE, MAE, KGE, TS / CSI, ETS, POD, FAR
9. **实测实验与前沿基准评测分析**：未知测试集（2022–2024）全量评测
10. **代码工程结构与运行指南**

---

## 📊 核心评测结果 (Benchmark Results on 2022–2024 Test Set)

| 模型范式 | 时间粒度 | ACC 预测技巧 | RMSE (mm) | MAE (mm) | TS (0.1mm) | TS (10mm) | TS (25mm) | TS (50mm暴雨) | POD (暴雨命中率) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **PENTAD (逐候预报)** | 5-Day | **0.4060** | 27.21 | 17.69 | **0.648** | **0.386** | **0.293** | **0.181** | **84.0% ~ 89.0%** |
| **DISAGG (图时空解混)**| Daily | **0.3416** | **7.98** | **3.19** | 0.312 | 0.097 | 0.010 | 0.000 | 97.6% (小雨) |
| **DAILY_HYBRID (混合逐日)**| Daily | 0.2362 | 8.38 | 3.29 | 0.333 | 0.112 | 0.000 | 0.000 | 74.9% (小雨) |
| **HURDLE (两阶段分位数)**| Daily | 0.0000 | 8.78 | 2.25 | 0.000 | 0.000 | 0.000 | 0.000 | - |

---

## 🚀 快速开始 (Quick Start)

```powershell
# 1. 运行单元与集成测试
C:\Users\fired\anaconda3\envs\nc\python.exe d:\HydroSynth\HydroGraph_S2S\verify_pipeline.py

# 2. 训练并评估逐候（Pentad）次季节预测模型
C:\Users\fired\anaconda3\envs\nc\python.exe d:\HydroSynth\HydroGraph_S2S\run_experiments.py --paradigm pentad --epochs 50 --batch_size 16

# 3. 一键执行全部 4 种范式的对比训练与测试评估
C:\Users\fired\anaconda3\envs\nc\python.exe d:\HydroSynth\HydroGraph_S2S\run_experiments.py --paradigm all --epochs 50

# 4. 生成 16:9 极简学术风组会 PPT（含高清架构图）
C:\Users\fired\anaconda3\envs\nc\python.exe d:\HydroSynth\HydroGraph_S2S\generate_phd_meeting_ppt.py
```

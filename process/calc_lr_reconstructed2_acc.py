import os
import sys
import numpy as np
import pandas as pd
import torch

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import config
from utils import utils

def main():
    lr_file = r"E:\workplace\U_Net_3D\lr_unet\lr_data_reconstructed2.npy"
    hr_file = r"E:\workplace\U_Net_3D\hr_unet\hr_data.npy"

    lr = np.load(lr_file) # [366, 10, 120, 140]
    hr = np.load(hr_file) # [366, 120, 140]

    precip_pred = lr[:, 0].copy() # Channel 0: [366, 120, 140]
    obs_target = hr.copy()        # [366, 120, 140]

    # Build exact matching 366 dates
    dates = pd.date_range("1994-01-01", "2024-09-01", freq="MS")
    exclude = pd.to_datetime(["2011-09-01", "2011-10-01", "2017-01-01"])
    dates = pd.DatetimeIndex([d for d in dates if d not in exclude])

    # Valid China mainland station interpolation mask
    mask = ~np.isnan(obs_target[0])
    precip_pred_masked = precip_pred.copy()
    precip_pred_masked[:, ~mask] = np.nan
    obs_target[:, ~mask] = np.nan

    # 1. Spatial ACC per time step
    spatial_acc = utils.cal_acc(torch.tensor(obs_target), torch.tensor(precip_pred_masked)).numpy()
    df_acc = pd.DataFrame({
        "date": dates,
        "year": dates.year,
        "month": dates.month,
        "acc": spatial_acc
    })

    # 2. Temporal ACC at each grid cell
    p_valid = precip_pred_masked[:, mask] # [366, 5301]
    t_valid = obs_target[:, mask]         # [366, 5301]
    p_anom = p_valid - np.nanmean(p_valid, axis=0, keepdims=True)
    t_anom = t_valid - np.nanmean(t_valid, axis=0, keepdims=True)
    cov = np.nansum(p_anom * t_anom, axis=0)
    std_p = np.sqrt(np.nansum(p_anom**2, axis=0) + 1e-12)
    std_t = np.sqrt(np.nansum(t_anom**2, axis=0) + 1e-12)
    temp_acc_grid = cov / (std_p * std_t)

    print("=" * 75)
    print("lr_data_reconstructed2.npy (Channel 0: 降水距平百分率) ACC 计算结果")
    print("=" * 75)
    print(f"数据路径: {lr_file}")
    print(f"数据形状: {lr.shape}, 评估通道: Channel 0 (降水距平百分率)")
    print(f"对照基准 (Ground Truth): {hr_file} ({hr.shape})")
    print(f"评估时段: {dates[0].strftime('%Y-%m')} 至 {dates[-1].strftime('%Y-%m')}")
    print(f"有效评估月数: {len(df_acc)} 个月")
    print(f"中国区域有效格点数: {int(mask.sum())} (120x140 网格分辨率)")
    print("-" * 75)
    print(f"全时段平均空间 ACC (Spatial ACC Mean):  {df_acc['acc'].mean():.4f}")
    print(f"空间 ACC 中位数 (Median):              {df_acc['acc'].median():.4f}")
    print(f"空间 ACC 标准差 (Std):                 {df_acc['acc'].std():.4f}")
    print(f"空间 ACC 最大值 (Max):                 {df_acc['acc'].max():.4f} ({df_acc.loc[df_acc['acc'].idxmax(), 'date'].strftime('%Y-%m')})")
    print(f"空间 ACC 最小值 (Min):                 {df_acc['acc'].min():.4f} ({df_acc.loc[df_acc['acc'].idxmin(), 'date'].strftime('%Y-%m')})")
    print(f"空间 ACC 正相关比例 (ACC > 0):         {(df_acc['acc'] > 0).mean() * 100:.1f}%")
    print(f"空间 ACC 高技巧比例 (ACC >= 0.4):      {(df_acc['acc'] >= 0.4).mean() * 100:.1f}%")
    print(f"格点时间序列相关系数空间平均 (Temporal ACC): {np.nanmean(temp_acc_grid):.4f}")
    print("-" * 75)
    print(f"训练集时段 (前 {len(df_acc)-21} 月, 1994-01 至 2022-12) 空间 ACC: {df_acc['acc'].iloc[:-21].mean():.4f}")
    print(f"测试集时段 (后 21 个月, 2023-01 至 2024-09) 空间 ACC:          {df_acc['acc'].iloc[-21:].mean():.4f}")
    print("-" * 75)

    print("\n[各月份平均空间 ACC 详情]")
    month_names = ["1月 (Jan)", "2月 (Feb)", "3月 (Mar)", "4月 (Apr)", "5月 (May)", "6月 (Jun)", 
                   "7月 (Jul)", "8月 (Aug)", "9月 (Sep)", "10月 (Oct)", "11月 (Nov)", "12月 (Dec)"]
    for m in range(1, 13):
        grp = df_acc[df_acc['month'] == m]
        m_mean = grp['acc'].mean()
        m_std = grp['acc'].std()
        print(f"  {month_names[m-1]:<12}: 平均 ACC = {m_mean:+.4f} (标准差: {m_std:.4f}, 样本数: {len(grp)})")

    print("\n[各季节平均空间 ACC]")
    seasons = {
        '春季 (MAM 3-5月)': [3, 4, 5],
        '夏季 (JJA 6-8月)': [6, 7, 8],
        '秋季 (SON 9-11月)': [9, 10, 11],
        '冬季 (DJF 12-2月)': [12, 1, 2]
    }
    for s_name, s_months in seasons.items():
        s_acc = df_acc[df_acc['month'].isin(s_months)]['acc'].mean()
        print(f"  {s_name:<18}: 平均 ACC = {s_acc:+.4f}")
    print("=" * 75)

if __name__ == "__main__":
    main()

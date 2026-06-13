import os
import sys

# 确保项目根目录在 sys.path 中
_curr_file = os.path.abspath(__file__)
_proj_root = os.path.dirname(os.path.dirname(_curr_file))
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import xarray as xr
from tqdm import tqdm

def main():
    print("开始计算降水距平百分率...")
    
    # 1. 查找并排序所有 NetCDF 原始数据文件
    data_path = 'D:/MODESv21_ecmwf_seas51'
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"未找到数据目录: {data_path}")
        
    files = [f for f in os.listdir(data_path) if f.endswith('.nc')]
    files.sort()
    
    tp_list = []
    month_list = []
    
    print("读取 NetCDF 文件中 (提取 lead 0)...")
    for fname in tqdm(files, desc="处理文件"):
        ym_str = fname.split('_')[-3]
        year = int(ym_str[:4])
        month = int(ym_str[4:6])
        
        # 仅选择 199401 至 202409 范围内的月份
        if year < 1994 or year > 2024:
            continue
        if year == 2024 and month > 9:
            continue
            
        # 排除破损或变量缺失的月份
        if (year == 2011 and month in [9, 10]) or (year == 2017 and month == 1):
            continue
            
        fpath = os.path.join(data_path, fname)
        try:
            ds = xr.open_dataset(fpath)
            # 只选取降水变量 tp，时间选择 lead 0 (time=0)，并进行空间切片
            selected = ds['tp'].sel(
                longitude=slice(70, 140), 
                latitude=slice(60, 0)
            )
            # 转为 numpy, 形状为 [60, 70]
            nth_tp = selected.isel(time=0).to_numpy()
            tp_list.append(nth_tp)
            month_list.append(month)
        except Exception as e:
            print(f"\n读取文件 {fname} 失败: {e}")
            
    tp_dataset = np.stack(tp_list)  # [样本数, 60, 70]
    month_arr = np.array(month_list)
    print(f"\n堆叠后降水数据形状: {tp_dataset.shape}")
    
    # 2. 双三次插值到高分辨率
    print("双三次插值放大中...")
    tp_tensor = torch.from_numpy(tp_dataset).unsqueeze(1)  # 添加通道维度 [样本数, 1, 60, 70]
    tp_interpolated = F.interpolate(tp_tensor, scale_factor=2.0, mode='bicubic').squeeze(1).numpy()  # [样本数, 120, 140]
    print(f"插值后数据形状: {tp_interpolated.shape}")
    
    # 3. 单位转换为 mm/month
    print("单位转换 (m -> mm/month)...")
    tp_data = tp_interpolated * 31 * 24 * 60 * 60 * 1000
    
    # 4. 计算降水距平百分率
    print("计算降水距平百分率...")
    anomaly = np.zeros_like(tp_data)
    for m in range(1, 13):
        idx = np.where(month_arr == m)[0]
        # 计算该月的多年气候态均值
        clim = tp_data[idx].mean(axis=0)
        # 距平百分率公式
        anomaly[idx] = (tp_data[idx] - clim) / (clim + 1e-6)
    
    print(f"距平百分率计算完成，形状: {anomaly.shape}")
    
    # 5. 保存结果
    save_path = os.path.join(_proj_root, 'data', 'anomaly.npy')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, anomaly)
    print(f"降水距平百分率数据已保存至: {save_path}")

if __name__ == '__main__':
    main()
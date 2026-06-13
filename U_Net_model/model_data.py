import os
import sys

# 确保项目根目录在 sys.path 中，以便正确导入 config 和 process.observe_norm
_curr_file = os.path.abspath(__file__)
_proj_root = os.path.dirname(os.path.dirname(_curr_file))
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

import glob
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import xarray as xr
from tqdm import tqdm

import config
from utils.observe_norm import DataNormalizer

def min_max_normalize_channels_4D(lr_data):
    """
    对 4D 数据做全局 min-max 归一化，使得每个通道的值在 [-1, 1] 之间。
    """
    channel_mins = torch.amin(lr_data, dim=(0, 2, 3), keepdim=True)
    channel_maxs = torch.amax(lr_data, dim=(0, 2, 3), keepdim=True)
    normalized_data = 2 * (lr_data - channel_mins) / (channel_maxs - channel_mins + 1e-8) - 1
    return normalized_data

def main():
    print("开始执行降水数据归一化重建程序...")
    
    # 1. 查找并排序所有 NetCDF 原始数据文件
    data_path = 'D:/MODESv21_ecmwf_seas51'
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"未找到数据目录: {data_path}")
        
    files = [f for f in os.listdir(data_path) if f.endswith('.nc')]
    files.sort()
    
    cond_list = []
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
            # 选取10个通道，时间选择 lead 0 (time=0)，并进行空间切片
            selected = ds[['tp', 'h200', 'h500', 'slp', 't2m', 't850', 'u200', 'u850', 'v200', 'v850']].sel(
                longitude=slice(70, 140), 
                latitude=slice(60, 0)
            )
            # 转为 numpy, 形状为 [10, 60, 70]
            nth_cond = selected.isel(time=0).to_array().to_numpy()
            cond_list.append(nth_cond)
            month_list.append(month)
        except Exception as e:
            print(f"\n读取文件 {fname} 失败: {e}")
            
    cond_dataset = np.stack(cond_list) # [366, 10, 60, 70]
    month_arr = np.array(month_list)
    print(f"\n堆叠后数据形状: {cond_dataset.shape}")
    
    # 2. 双三次插值到高分辨率
    print("双三次插值放大中...")
    cond_tensor = torch.from_numpy(cond_dataset)
    lr_data = F.interpolate(cond_tensor, scale_factor=2.0, mode='bicubic').numpy() # [366, 10, 120, 140]
    print(f"插值后数据形状: {lr_data.shape}")
    
    # 3. 计算降水距平百分率 (Channel 0)
    print("计算降水距平百分率...")
    # 单位转换为 mm/month
    lr_data[:, 0] = lr_data[:, 0] * 31*24*60*60*1000
    
    anomaly_lead0 = np.zeros_like(lr_data[:, 0])
    for m in range(1, 13):
        idx = np.where(month_arr == m)[0]
        # 计算该月的多年气候态均值
        clim = lr_data[idx, 0].mean(axis=0)
        # 距平百分率公式
        anomaly_lead0[idx] = (lr_data[idx, 0] - clim) / (clim + 1e-6)
        
    # 4. 对距平进行 Z-Score + 3-Sigma 裁剪归一化
    print("对降水距平百分率进行 Z-score + 3-sigma 裁剪归一化...")
    normalizer = DataNormalizer(clip_sigma=3.0)
    normalizer.fit(anomaly_lead0)
    norm_anomaly = normalizer.transform(anomaly_lead0)
    
    # 将归一化后的数据放回 Channel 0
    lr_data[:, 0] = norm_anomaly
    
    # 5. 全局通道归一化
    print("进行所有通道的全局 min-max 归一化...")
    lr_tensor = torch.from_numpy(lr_data)
    lr_norm = min_max_normalize_channels_4D(lr_tensor).numpy()
    
    # 6. 保存重建的数据
    save_dir = config.modelconfig['lr_path']
    save_path = os.path.join(save_dir, 'lr_data_reconstructed2.npy')
    np.save(save_path, lr_norm)
    print(f"重建完成！数据已保存至: {save_path}")
    
    # 7. 自动校验与原始 lr_data1.npy 的误差
    target_path = os.path.join(save_dir, 'lr_data1.npy')
    if os.path.exists(target_path):
        print("\n检测到原始的归一化数据文件，开始进行校验...")
        data1 = np.load(target_path)
        
        # 逐通道计算皮尔逊相关系数和平均绝对误差
        print("-" * 55)
        print(f"{'通道':<6} | {'皮尔逊相关系数 (Correlation)':<24} | {'平均绝对误差 (MAE)':<16}")
        print("-" * 55)
        for c in range(10):
            t_flat = data1[:, c].flatten()
            r_flat = lr_norm[:, c].flatten()
            
            # 使用 numpy 稳健计算相关系数，避免 SegFault
            x_mean = np.mean(t_flat)
            y_mean = np.mean(r_flat)
            num = np.sum((t_flat - x_mean) * (r_flat - y_mean))
            den = np.sqrt(np.sum((t_flat - x_mean)**2) * np.sum((r_flat - y_mean)**2))
            corr = num / (den + 1e-8) if den != 0 else 0.0
            
            mae = np.mean(np.abs(t_flat - r_flat))
            print(f"Ch {c:<3} | {corr:<28.6f} | {mae:<16.8f}")
        print("-" * 55)
        
        # 校验成功判定
        overall_mae = np.mean(np.abs(data1 - lr_norm))
        if overall_mae < 1e-6:
            print(f"校验成功！总体平均绝对误差 (MAE) = {overall_mae:.8f}，重建数据与原数据完全一致。")
        else:
            print(f"校验警告：存在偏差。总体平均绝对误差 (MAE) = {overall_mae:.8f}。")
    else:
        print(f"\n未找到原始文件 {target_path}，无法进行自动校对。")

if __name__ == '__main__':
    main()

import os
import sys
import glob
import re
import time
import numpy as np
import pandas as pd
import xarray as xr
from concurrent.futures import ThreadPoolExecutor
from scipy.interpolate import RegularGridInterpolator, griddata
import warnings
warnings.filterwarnings("ignore")

# Set up paths
MODEL_DIR = r"E:\DATA\model_data\MODESv21_ecmwf_seas51"
OBS_DIR = r"E:\DATA\原始站点资料（有华南）"

def read_single_station_txt(filepath):
    """Reads one monthly observation TXT file."""
    try:
        data = pd.read_csv(
            filepath, sep=r'\s+', header=None, usecols=[0, 1, 2, 4, 5, 9],
            names=['Stn_No', 'Lat', 'Long', 'Year', 'Month', 'Precip'],
            encoding='utf-8', engine='c'
        )
        # Filter invalid precipitation codes
        data['Precip'] = data['Precip'].replace([32700, 32766, 32001, 30001, 9999900], np.nan)
        # Convert units: 0.1 mm -> mm
        data['Precip'] = data['Precip'] / 10.0
        
        # Monthly total per station
        monthly = data.groupby(['Stn_No', 'Lat', 'Long', 'Year', 'Month'], as_index=False)['Precip'].sum(min_count=1)
        return monthly
    except Exception as e:
        print(f"Warning reading obs file {os.path.basename(filepath)}: {e}")
        return None

def load_all_obs():
    """Loads all observation TXT files and computes station anomalies."""
    print(">>> 1. 读取原始站点观测数据 (E:\\DATA\\原始站点资料（有华南）)...")
    t0 = time.time()
    obs_files = sorted(glob.glob(os.path.join(OBS_DIR, "SURF_CLI_CHN_MUL_DAY-PRE-13011-*.TXT")))
    
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(read_single_station_txt, obs_files))
    
    dfs = [df for df in results if df is not None and not df.empty]
    all_obs = pd.concat(dfs, ignore_index=True)
    
    # Normalize coordinates (from DDDMM or scaled float to standard degrees)
    if all_obs['Long'].abs().max() > 180:
        all_obs['Long'] = all_obs['Long'] / 100.0
    if all_obs['Lat'].abs().max() > 90:
        all_obs['Lat'] = all_obs['Lat'] / 100.0
        
    all_obs['Year'] = all_obs['Year'].astype(int)
    all_obs['Month'] = all_obs['Month'].astype(int)
    all_obs['time_str'] = all_obs['Year'].astype(str) + all_obs['Month'].apply(lambda m: f"{m:02d}")
    
    # Compute multi-year climatology per station per month
    clim = all_obs.groupby(['Stn_No', 'Month'])['Precip'].mean().reset_index().rename(columns={'Precip': 'clim_precip'})
    all_obs = pd.merge(all_obs, clim, on=['Stn_No', 'Month'], how='left')
    
    # Percentage anomaly: (P - Clim) / (Clim + eps)
    all_obs['anoma'] = (all_obs['Precip'] - all_obs['clim_precip']) / (all_obs['clim_precip'] + 1e-4)
    # Absolute anomaly (mm): P - Clim
    all_obs['anoma_abs'] = all_obs['Precip'] - all_obs['clim_precip']
    
    t1 = time.time()
    print(f"    观测数据读取完成: 共 {len(all_obs)} 条站点-月记录, {all_obs['Stn_No'].nunique()} 个独立站点, 耗时 {t1-t0:.2f}s")
    return all_obs

def load_all_model_lead0():
    """Loads all model Lead-0 forecast total precipitation from NetCDF files."""
    print(">>> 2. 读取原始 SEAS5 气候模式数据 Lead-0 (E:\\DATA\\model_data\\MODESv21_ecmwf_seas51)...")
    t0 = time.time()
    model_files = sorted(glob.glob(os.path.join(MODEL_DIR, "MODESv21_ecmwf_seas51_*_monthly_em.nc")))
    
    model_raw = {}
    ref_lats, ref_lons = None, None
    corrupt_count = 0
    
    for f in model_files:
        m = re.search(r'(\d{6})', os.path.basename(f))
        if not m:
            continue
        ym = m.group(1)
        try:
            with xr.open_dataset(f) as ds:
                # Lead 0 is time index 0
                tp_da = ds['tp'].isel(time=0)
                if 'latitude' in ds.coords:
                    lats = ds['latitude'].values
                    lons = ds['longitude'].values
                else:
                    lats = ds['lat'].values
                    lons = ds['lon'].values
                    
                if ref_lats is None:
                    ref_lats, ref_lons = lats, lons
                model_raw[ym] = tp_da.values.astype(np.float32)
        except Exception:
            corrupt_count += 1
            continue
            
    model_dates = sorted(model_raw.keys())
    
    # Compute Model Climatology per month (1 to 12)
    model_stack = {m: [] for m in range(1, 13)}
    for ym in model_dates:
        month = int(ym[4:6])
        model_stack[month].append(model_raw[ym])
    
    model_clim = {m: np.mean(np.stack(model_stack[m]), axis=0) for m in range(1, 13)}
    
    # Compute Model Lead-0 Percentage Anomaly and Absolute Anomaly
    model_anom_pct = {}
    model_anom_abs = {}
    for ym in model_dates:
        month = int(ym[4:6])
        clim = model_clim[month]
        val = model_raw[ym]
        model_anom_pct[ym] = (val - clim) / (clim + 1e-8)
        model_anom_abs[ym] = val - clim
        
    t1 = time.time()
    print(f"    模式数据读取完成: 共 {len(model_dates)} 个有效月份 ({model_dates[0]} 至 {model_dates[-1]}), 跳过异常文件 {corrupt_count} 个, 耗时 {t1-t0:.2f}s")
    return model_anom_pct, model_anom_abs, ref_lats, ref_lons

def compute_station_level_acc(all_obs, model_anom_pct, ref_lats, ref_lons):
    """Computes ACC directly at observation station locations."""
    print(">>> 3. 计算台站尺度 (Station-level) Lead-0 空间 ACC...")
    
    # Ensure latitudes in ascending order for RegularGridInterpolator
    if ref_lats[0] > ref_lats[-1]:
        lat_order = -1
        interp_lats = ref_lats[::-1]
    else:
        lat_order = 1
        interp_lats = ref_lats
        
    common_dates = sorted(list(set(all_obs['time_str'].unique()) & set(model_anom_pct.keys())))
    
    records = []
    
    for ym in common_dates:
        mod_grid = model_anom_pct[ym]
        if lat_order == -1:
            mod_grid = mod_grid[::-1, :]
            
        interp_fn = RegularGridInterpolator(
            (interp_lats, ref_lons), mod_grid,
            bounds_error=False, fill_value=np.nan
        )
        
        obs_m = all_obs[all_obs['time_str'] == ym].dropna(subset=['Lat', 'Long', 'anoma'])
        # Filter China bounds (roughly Lat 15-55, Lon 70-138)
        obs_m = obs_m[(obs_m['Lat'] >= 15) & (obs_m['Lat'] <= 55) & (obs_m['Long'] >= 70) & (obs_m['Long'] <= 138)]
        
        if len(obs_m) < 50:
            continue
            
        pts = obs_m[['Lat', 'Long']].to_numpy()
        mod_stn_vals = interp_fn(pts)
        obs_stn_vals = obs_m['anoma'].to_numpy()
        
        valid = np.isfinite(mod_stn_vals) & np.isfinite(obs_stn_vals)
        if valid.sum() < 50:
            continue
            
        m_vals = mod_stn_vals[valid]
        o_vals = obs_stn_vals[valid]
        
        m_anom = m_vals - np.mean(m_vals)
        o_anom = o_vals - np.mean(o_vals)
        
        cov = np.sum(m_anom * o_anom)
        var_m = np.sum(m_anom**2)
        var_o = np.sum(o_anom**2)
        
        if var_m > 0 and var_o > 0:
            acc = cov / np.sqrt(var_m * var_o)
        else:
            acc = 0.0
            
        records.append({
            'time_str': ym,
            'year': int(ym[:4]),
            'month': int(ym[4:]),
            'n_stations': int(valid.sum()),
            'acc_station': float(acc)
        })
        
    df_res = pd.DataFrame(records)
    return df_res

def compute_grid_level_acc(all_obs, model_anom_pct, ref_lats, ref_lons):
    """Computes ACC interpolated onto standard 0.5 x 0.5 degree China grid."""
    print(">>> 4. 计算网格尺度 (0.5°x0.5° Grid-level) Lead-0 空间 ACC...")
    
    target_lons = np.arange(70, 140, 0.5)
    target_lats = np.arange(60, 0, -0.5)
    grid_lon2d, grid_lat2d = np.meshgrid(target_lons, target_lats)
    
    if ref_lats[0] > ref_lats[-1]:
        lat_order = -1
        interp_lats = ref_lats[::-1]
    else:
        lat_order = 1
        interp_lats = ref_lats
        
    common_dates = sorted(list(set(all_obs['time_str'].unique()) & set(model_anom_pct.keys())))
    records = []
    
    for ym in common_dates:
        mod_grid = model_anom_pct[ym]
        if lat_order == -1:
            mod_grid = mod_grid[::-1, :]
            
        interp_fn = RegularGridInterpolator(
            (interp_lats, ref_lons), mod_grid,
            bounds_error=False, fill_value=np.nan
        )
        
        # Grid model onto 0.5 x 0.5 target
        grid_pts = np.stack([grid_lat2d.flatten(), grid_lon2d.flatten()], axis=-1)
        mod_05 = interp_fn(grid_pts).reshape(grid_lat2d.shape)
        
        # Grid observations onto 0.5 x 0.5 target
        obs_m = all_obs[all_obs['time_str'] == ym].dropna(subset=['Lat', 'Long', 'anoma'])
        if len(obs_m) < 50:
            continue
            
        try:
            obs_05 = griddata(
                (obs_m['Long'], obs_m['Lat']),
                obs_m['anoma'],
                (grid_lon2d, grid_lat2d),
                method='linear'
            )
        except Exception:
            continue
            
        valid = np.isfinite(obs_05) & np.isfinite(mod_05)
        if valid.sum() < 500:
            continue
            
        m_v = mod_05[valid]
        o_v = obs_05[valid]
        
        m_a = m_v - np.mean(m_v)
        o_a = o_v - np.mean(o_v)
        
        cov = np.sum(m_a * o_a)
        var_m = np.sum(m_a**2)
        var_o = np.sum(o_a**2)
        
        if var_m > 0 and var_o > 0:
            acc = cov / np.sqrt(var_m * var_o)
        else:
            acc = 0.0
            
        records.append({
            'time_str': ym,
            'year': int(ym[:4]),
            'month': int(ym[4:]),
            'n_valid_grid': int(valid.sum()),
            'acc_grid': float(acc)
        })
        
    df_res = pd.DataFrame(records)
    return df_res

def main():
    print("=" * 75)
    print("读取原始气象数据计算 SEAS (ECMWF SEAS5) Lead-0 中国区域降水 ACC")
    print(f"模式源路径: {MODEL_DIR}")
    print(f"观测源路径: {OBS_DIR}")
    print("=" * 75)
    
    all_obs = load_all_obs()
    model_anom_pct, model_anom_abs, ref_lats, ref_lons = load_all_model_lead0()
    
    # 1. Station-level ACC
    df_stn = compute_station_level_acc(all_obs, model_anom_pct, ref_lats, ref_lons)
    
    # 2. Grid-level ACC
    df_grd = compute_grid_level_acc(all_obs, model_anom_pct, ref_lats, ref_lons)
    
    # Merge both evaluations
    df_all = pd.merge(df_stn, df_grd[['time_str', 'acc_grid']], on='time_str', how='inner')
    
    print("\n" + "=" * 75)
    print("【计算结果汇总: SEAS 气候模式 Lead-0 中国降水 ACC】")
    print("=" * 75)
    print(f"共同评估时段: {df_all['time_str'].min()} 至 {df_all['time_str'].max()}")
    print(f"有效评估月数: {len(df_all)} 个月")
    print(f"全国有效观测站点数/月: 平均 {df_all['n_stations'].mean():.0f} 个站点")
    print("-" * 75)
    print(f"1. 台站尺度 (Station-level) 平均空间 ACC:  {df_all['acc_station'].mean():.4f}")
    print(f"   - 中位数 (Median):                     {df_all['acc_station'].median():.4f}")
    print(f"   - 标准差 (Std):                        {df_all['acc_station'].std():.4f}")
    print(f"   - 极值范围 (Min ~ Max):                {df_all['acc_station'].min():.4f} ~ {df_all['acc_station'].max():.4f}")
    print(f"   - 正相关比例 (ACC > 0):                {(df_all['acc_station'] > 0).mean() * 100:.1f}%")
    print(f"   - 高技巧比例 (ACC >= 0.4):             {(df_all['acc_station'] >= 0.4).mean() * 100:.1f}%")
    print("-" * 75)
    print(f"2. 网格尺度 (0.5° Grid-level) 平均空间 ACC: {df_all['acc_grid'].mean():.4f}")
    print(f"   - 中位数 (Median):                     {df_all['acc_grid'].median():.4f}")
    print(f"   - 标准差 (Std):                        {df_all['acc_grid'].std():.4f}")
    print(f"   - 极值范围 (Min ~ Max):                {df_all['acc_grid'].min():.4f} ~ {df_all['acc_grid'].max():.4f}")
    print(f"   - 正相关比例 (ACC > 0):                {(df_all['acc_grid'] > 0).mean() * 100:.1f}%")
    print(f"   - 高技巧比例 (ACC >= 0.4):             {(df_all['acc_grid'] >= 0.4).mean() * 100:.1f}%")
    print("-" * 75)
    
    print("\n【各月份平均 ACC (台站 vs 网格)】")
    print(f"  {'月份':<8} | {'台站 ACC':<12} | {'网格 ACC':<12} | {'样本月数':<8}")
    print("-" * 50)
    month_names = ["1月 (Jan)", "2月 (Feb)", "3月 (Mar)", "4月 (Apr)", "5月 (May)", "6月 (Jun)", 
                   "7月 (Jul)", "8月 (Aug)", "9月 (Sep)", "10月 (Oct)", "11月 (Nov)", "12月 (Dec)"]
    for m in range(1, 13):
        sub = df_all[df_all['month'] == m]
        print(f"  {month_names[m-1]:<8} | {sub['acc_station'].mean():+.4f}       | {sub['acc_grid'].mean():+.4f}       | {len(sub):<8}")
        
    print("\n【各季节平均 ACC】")
    seasons = {
        '春季 (MAM 3-5月)': [3, 4, 5],
        '夏季 (JJA 6-8月)': [6, 7, 8],
        '秋季 (SON 9-11月)': [9, 10, 11],
        '冬季 (DJF 12-2月)': [12, 1, 2]
    }
    for s_name, s_months in seasons.items():
        sub = df_all[df_all['month'].isin(s_months)]
        print(f"  {s_name:<18}: 台站 ACC = {sub['acc_station'].mean():+.4f} | 网格 ACC = {sub['acc_grid'].mean():+.4f}")
        
    # Save CSV output
    out_csv = r"D:\HydroSynth\process\seas_lead0_acc_from_raw.csv"
    df_all.to_csv(out_csv, index=False)
    print(f"\n月度明细数据已保存至: {out_csv}")
    print("=" * 75)

if __name__ == '__main__':
    main()

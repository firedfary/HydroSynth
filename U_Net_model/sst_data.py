import numpy as np
import pandas as pd
import xarray as xr
import sys
import os

# Ensure the parent of the HydroSynth package is on sys.path.
_curr_file = os.path.abspath(__file__)
_project_root = os.path.dirname(os.path.dirname(_curr_file))
_project_parent = os.path.dirname(_project_root)
if _project_parent not in sys.path:
    sys.path.insert(0, _project_parent)

from HydroSynth import config



# 1. 构造时间列表（去掉缺失月）
all_months = []
for year in range(1994, 2025):
    for month in range(1, 13):
        if (year == 2011 and month in [9, 10]) or (year == 2017 and month == 1):
            continue
        # 截止到2024年9月
        if year == 2024 and month > 9:
            break
        all_months.append((year, month))
all_months = np.array(all_months)  # shape: [n_time, 2]

# 2. 构造每个时间点的前三个月列表
def prev_months(year, month, n=6):
    months = []
    for i in range(n, 0, -1):  # i=3,2,1
        y = year
        m = month - i
        while m <= 0:
            y -= 1
            m += 12
        months.append((y, m))
    return months

# 3. 读取数据并合并
data_dir = os.getenv("ERSST_DATA_PATH")
sample_file = os.path.join(data_dir, f"ersst.v5.{all_months[0,0]:04d}{all_months[0,1]:02d}.nc")
ds = xr.open_dataset(sample_file)
lat = ds['lat'].values
lon = ds['lon'].values
ds.close()

n_time = len(all_months)
n_chan = 6  # Store 6 months for each time point
n_lat = len(lat)
n_lon = len(lon)
result = np.zeros((n_time, n_chan, n_lat, n_lon), dtype=np.float32)

for i, (year, month) in enumerate(all_months):
    months3 = prev_months(year, month, 6)
    for j, (y, m) in enumerate(months3):
        nc_file = os.path.join(data_dir, f"ersst.v5.{y:04d}{m:02d}.nc")
        if not os.path.exists(nc_file):
            print(f"缺失文件: {nc_file}")
            result[i, j, :, :] = np.nan
            continue
        ds = xr.open_dataset(nc_file)
        result[i, j, :, :] = ds['ssta'].values
        ds.close()

# 4. 保存为npy
np.save(os.path.join(config.modelconfig["base_data_path"], "sst_6chan.npy"), result)
print("保存 shape:", result.shape)
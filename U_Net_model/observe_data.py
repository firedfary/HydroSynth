import sys
import os

# Ensure the parent of the HydroSynth package is on sys.path.
_curr_file = os.path.abspath(__file__)
_project_root = os.path.dirname(os.path.dirname(_curr_file))
_project_parent = os.path.dirname(_project_root)
if _project_parent not in sys.path:
    sys.path.insert(0, _project_parent)
import pandas as pd

import numpy as np
import HydroSynth.utils.utils as utils
import matplotlib.pyplot as plt
import matplotlib as mpl
import HydroSynth.utils.observe_norm as observe_norm
from HydroSynth import config
import torch


result = pd.read_csv(os.path.join(_project_root, "utils", "observe_data24.csv"))
result['Long'] = result['Long']/100
result['Lat'] = result['Lat']/100

def filter_data_by_date(start_date, end_date, df, exclude_dates=None):
    # 创建日期序列
    date_range = pd.date_range(start=start_date, end=end_date, freq='MS')
    
    # 将time列转换为datetime格式
    df['time'] = pd.to_datetime(df['time'])
    
    # 过滤数据
    filtered_df = df[df['time'].isin(date_range)]
    
    # 如果提供了exclude_dates参数，删除对应日期的数据
    if exclude_dates:
        exclude_dates = pd.to_datetime(exclude_dates)
        filtered_df = filtered_df[~filtered_df['time'].isin(exclude_dates)]
    
    return filtered_df


# 示例调用
start_date = '1994-01-01'
end_date = '2024-09-01'#
filtered_df = filter_data_by_date(start_date, end_date, result, exclude_dates=['2017-01-01', '2011-09-01', '2011-10-01'])
print(filtered_df)

grid_data = utils.gred_time_site_to_net(df=filtered_df, to_xr=False, gred_var='anoma')
utils.check_nan_status(grid_data)
np.save(os.path.join(config.modelconfig['hr_path'], 'hr_data.npy'), grid_data)

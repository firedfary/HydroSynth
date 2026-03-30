# -*- coding: utf-8 -*-
"""
ERA5 monthly-split downloader for multiple pressure levels and surface variables
中国西南地区数据 (20°N-35°N, 95°E-110°E)
下载月平均数据

输出形式说明：
- 月文件：YYYYMM_China.nc，例如：199301_China.nc
- 年文件：ERA5_YYYY_China.nc，例如：ERA5_1993_China.nc

Features:
- Split requests by month to avoid CDS "cost limits exceeded" errors.
- Download per-month netCDF, then merge into one annual netCDF.
- Log progress, record failed months, support resume/skip existing files.
- Compress final annual file.
- 不再删除月文件（CLEAN_MONTHS = False）。
"""

import os
import datetime
import time
import traceback

import cdsapi
import xarray as xr
import numpy as np
import pandas as pd
import requests

# -----------------------------
# ========== CONFIG ===========
# -----------------------------
FOLDER_OUT = r'D:\ERA5'  # 输出目录，可自行修改
os.makedirs(FOLDER_OUT, exist_ok=True)

# 中国西南地区范围 [lat_max, lon_min, lat_min, lon_max]
# 纬度：20°N - 35°N，经度：95°E - 110°E
AREA = [60, 70, 0, 140]  # 中国西南地区范围，精确匹配您的要求

# 变量与气压层配置
PRESSURE_LEVEL_VARS = [
    'geopotential',
    'u_component_of_wind',
    'v_component_of_wind',
    'temperature'
]
PRESSURE_LEVELS = ['200', '500', '850']  # 根据Excel文件，需要200hPa, 500hPa, 850hPa

# 地面变量
SINGLE_LEVEL_VARS = [
    'mean_sea_level_pressure',  # slp
    'sea_surface_temperature',  # sst
    '2m_temperature',  # t2m
    'total_precipitation'  # tp
]

# YEARS to download: 1993-2024 (inclusive)
YEARS = [str(y) for y in range(1993, 1994)]  # 1993到2024年
MONTHS = [str(m).zfill(2) for m in range(1, 13)]

# CDS client settings
CDS_RETRY_MAX = 20  # cdsapi.Client 的 retry_max（内部重试）
CDS_TIMEOUT = 3600  # seconds

# 自己再包一层 Python 级下载重试（针对 SSL/网络抖动）
DOWNLOAD_RETRY_MAX = 3  # 单月下载最大尝试次数
DOWNLOAD_RETRY_SLEEP = 10  # 每次失败后等待秒数

# Post-processing
MERGE_ANNUAL = False  # 是否合并成年文件
CLEAN_MONTHS = False  # ✅ 不删除月文件
COMPLEVEL = 4  # netCDF 压缩等级 (0-9)

# logging
LOG_FILE = os.path.join(FOLDER_OUT, 'download_log.txt')


# -----------------------------
# ===== helper functions ======
# -----------------------------
def write_log(msg):
    ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f'[{ts}] {msg}'
    print(line)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def month_fname(year, month):
    """
    月文件命名：YYYYMM_China.nc
    例如：199301_China.nc
    """
    return os.path.join(FOLDER_OUT, f'{year}{month}_China.nc')


def annual_fname(year):
    """
    年文件命名：ERA5_YYYY_China.nc
    例如：ERA5_1993_China.nc
    """
    return os.path.join(FOLDER_OUT, f'ERA5_{year}_China.nc')


def failed_months_fname(year):
    return os.path.join(FOLDER_OUT, f'failed_months_{year}.txt')


# -----------------------------
# ===== download per-month ====
# -----------------------------
def download_pressure_month(year, month):
    """
    下载单个月的 ERA5 压力层数据（仅限中国西南地区）
    下载月平均数据
    """
    out_file = month_fname(year, month)
    if os.path.exists(out_file):
        write_log(f"Month file exists, skipping: {out_file}")
        return out_file

    # 首先下载压力层数据
    write_log(f"Downloading monthly mean pressure level data for {year}-{month} (SW China region)")

    # 压力层数据请求 - 月平均
    pressure_request = {
        'product_type': 'monthly_averaged_reanalysis',
        'format': 'netcdf',
        'variable': PRESSURE_LEVEL_VARS,
        'pressure_level': PRESSURE_LEVELS,
        'year': year,
        'month': month,
        'time': '00:00',  # 月平均数据只需要一个时间点
        'area': AREA,  # 使用中国西南地区范围
    }

    # 地面数据请求 - 月平均
    surface_request = {
        'product_type': 'monthly_averaged_reanalysis_by_hour_of_day',
        'format': 'netcdf',
        'variable': SINGLE_LEVEL_VARS,
        'year': year,
        'month': month,
        'time': '00:00',  # 月平均数据只需要一个时间点
        'area': AREA,  # 使用中国西南地区范围
    }

    # 临时文件名
    temp_pressure_file = os.path.join(FOLDER_OUT, f'temp_pressure_{year}{month}.nc')
    temp_surface_file = os.path.join(FOLDER_OUT, f'temp_surface_{year}{month}.nc')

    # Python 级重试循环 - 下载压力层数据
    for attempt in range(1, DOWNLOAD_RETRY_MAX + 1):
        write_log(
            f"Requesting monthly mean pressure level data {year}-{month} (attempt {attempt}/{DOWNLOAD_RETRY_MAX})")
        start = time.time()
        try:
            c = cdsapi.Client(retry_max=CDS_RETRY_MAX, timeout=CDS_TIMEOUT)
            c.retrieve('reanalysis-era5-pressure-levels-monthly-means', pressure_request, temp_pressure_file)
            dur = time.time() - start
            write_log(f"Downloaded monthly mean pressure level data {year}-{month} in {dur:.1f}s")
            break
        except requests.exceptions.SSLError as e:
            write_log(f"SSLError downloading pressure data {year}-{month}: {repr(e)}")
            if os.path.exists(temp_pressure_file):
                os.remove(temp_pressure_file)
            if attempt == DOWNLOAD_RETRY_MAX:
                write_log(f"Failed to download pressure data for {year}-{month}")
                return None
            time.sleep(DOWNLOAD_RETRY_SLEEP)
        except Exception as e:
            write_log(f"ERROR downloading pressure data {year}-{month}: {e}")
            if os.path.exists(temp_pressure_file):
                os.remove(temp_pressure_file)
            if attempt == DOWNLOAD_RETRY_MAX:
                write_log(f"Failed to download pressure data for {year}-{month}")
                return None
            time.sleep(DOWNLOAD_RETRY_SLEEP)

    # Python 级重试循环 - 下载地面数据
    for attempt in range(1, DOWNLOAD_RETRY_MAX + 1):
        write_log(f"Requesting monthly mean surface data {year}-{month} (attempt {attempt}/{DOWNLOAD_RETRY_MAX})")
        start = time.time()
        try:
            c = cdsapi.Client(retry_max=CDS_RETRY_MAX, timeout=CDS_TIMEOUT)
            c.retrieve('reanalysis-era5-single-levels-monthly-means', surface_request, temp_surface_file)
            dur = time.time() - start
            write_log(f"Downloaded monthly mean surface data {year}-{month} in {dur:.1f}s")
            break
        except requests.exceptions.SSLError as e:
            write_log(f"SSLError downloading surface data {year}-{month}: {repr(e)}")
            if os.path.exists(temp_surface_file):
                os.remove(temp_surface_file)
            if attempt == DOWNLOAD_RETRY_MAX:
                write_log(f"Failed to download surface data for {year}-{month}")
                # 清理已下载的压力层数据
                if os.path.exists(temp_pressure_file):
                    os.remove(temp_pressure_file)
                return None
            time.sleep(DOWNLOAD_RETRY_SLEEP)
        except Exception as e:
            write_log(f"ERROR downloading surface data {year}-{month}: {e}")
            if os.path.exists(temp_surface_file):
                os.remove(temp_surface_file)
            if attempt == DOWNLOAD_RETRY_MAX:
                write_log(f"Failed to download surface data for {year}-{month}")
                # 清理已下载的压力层数据
                if os.path.exists(temp_pressure_file):
                    os.remove(temp_pressure_file)
                return None
            time.sleep(DOWNLOAD_RETRY_SLEEP)

    # 合并压力层数据和地面数据
    write_log(f"Merging monthly mean pressure and surface data for {year}-{month}")
    try:
        # 打开两个数据集
        ds_pressure = xr.open_dataset(temp_pressure_file)
        ds_surface = xr.open_dataset(temp_surface_file)

        # 重命名变量以匹配Excel文件中的名称
        # 创建一个新的数据集
        ds_combined = xr.Dataset()

        # 复制坐标
        ds_combined['longitude'] = ds_pressure['longitude']
        ds_combined['latitude'] = ds_pressure['latitude']
        ds_combined['time'] = ds_pressure['time']

        # 提取并重命名200hPa变量
        ds_combined['h200'] = ds_pressure['z'].sel(level=200)
        ds_combined['u200'] = ds_pressure['u'].sel(level=200)
        ds_combined['v200'] = ds_pressure['v'].sel(level=200)

        # 提取并重命名500hPa变量
        ds_combined['h500'] = ds_pressure['z'].sel(level=500)

        # 提取并重命名850hPa变量
        ds_combined['u850'] = ds_pressure['u'].sel(level=850)
        ds_combined['v850'] = ds_pressure['v'].sel(level=850)
        ds_combined['t850'] = ds_pressure['t'].sel(level=850)

        # 添加地面变量（重命名以匹配Excel）
        ds_combined['slp'] = ds_surface['msl']  # mean sea level pressure
        ds_combined['sst'] = ds_surface['sst']  # sea surface temperature
        ds_combined['t2m'] = ds_surface['t2m']  # 2m temperature
        ds_combined['tp'] = ds_surface['tp']  # total precipitation

        # 设置变量属性（单位等）
        ds_combined['h200'].attrs = {'units': 'm**2 s**-2', 'long_name': 'Geopotential at 200hPa (monthly mean)'}
        ds_combined['u200'].attrs = {'units': 'm s**-1', 'long_name': 'U component of wind at 200hPa (monthly mean)'}
        ds_combined['v200'].attrs = {'units': 'm s**-1', 'long_name': 'V component of wind at 200hPa (monthly mean)'}
        ds_combined['h500'].attrs = {'units': 'm**2 s**-2', 'long_name': 'Geopotential at 500hPa (monthly mean)'}
        ds_combined['u850'].attrs = {'units': 'm s**-1', 'long_name': 'U component of wind at 850hPa (monthly mean)'}
        ds_combined['v850'].attrs = {'units': 'm s**-1', 'long_name': 'V component of wind at 850hPa (monthly mean)'}
        ds_combined['t850'].attrs = {'units': 'K', 'long_name': 'Temperature at 850hPa (monthly mean)'}
        ds_combined['slp'].attrs = {'units': 'Pa', 'long_name': 'Mean sea level pressure (monthly mean)'}
        ds_combined['sst'].attrs = {'units': 'K', 'long_name': 'Sea surface temperature (monthly mean)'}
        ds_combined['t2m'].attrs = {'units': 'K', 'long_name': '2 metre temperature (monthly mean)'}
        ds_combined['tp'].attrs = {'units': 'm s**-1', 'long_name': 'Mean total precipitation rate (monthly mean)'}

        # 设置全局属性
        ds_combined.attrs = {
            'Conventions': 'CF-1.6',
            'history': f'Downloaded from CDS on {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
            'source': 'ERA5 monthly mean reanalysis',
            'title': f'ERA5 monthly mean data for {year}-{month} (Southwest China region)',
            'region': f'Southwest China: {AREA[0]}°N to {AREA[2]}°N, {AREA[1]}°E to {AREA[3]}°E',
            'time_resolution': 'monthly mean',
            'institution': 'European Centre for Medium-Range Weather Forecasts'
        }

        # 保存合并后的文件
        comp = dict(zlib=True, complevel=COMPLEVEL)
        encoding = {var: comp for var in ds_combined.data_vars}

        ds_combined.to_netcdf(out_file, encoding=encoding)
        write_log(f"Merged monthly mean data saved: {out_file}")

        # 关闭数据集
        ds_pressure.close()
        ds_surface.close()
        ds_combined.close()

        # 清理临时文件
        if os.path.exists(temp_pressure_file):
            os.remove(temp_pressure_file)
        if os.path.exists(temp_surface_file):
            os.remove(temp_surface_file)

        write_log(f"Successfully downloaded and merged monthly mean data for {year}-{month}")
        return out_file

    except Exception as e:
        write_log(f"ERROR merging monthly mean data for {year}-{month}: {e}")
        write_log(traceback.format_exc())

        # 清理临时文件
        for temp_file in [temp_pressure_file, temp_surface_file]:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass
        return None


# -----------------------------
# ===== merge monthly files ===
# -----------------------------
def merge_monthly_to_annual(year, month_files, out_annual):
    """
    将某年的所有月文件合并成一个年度文件，并使用压缩。
    """
    write_log(f"Merging {len(month_files)} monthly files -> {out_annual}")
    try:
        # 打开多文件
        ds = xr.open_mfdataset(
            month_files,
            combine='by_coords',
            parallel=False,
            engine='netcdf4'
        )

        # 构造 encoding，对真正的数据变量压缩
        comp = dict(zlib=True, complevel=COMPLEVEL)
        encoding = {var: comp for var in ds.data_vars}

        # 写年度文件
        ds.to_netcdf(out_annual, encoding=encoding)
        write_log(f"Annual monthly mean file saved: {out_annual}")
        ds.close()
        return True

    except Exception as e:
        write_log(f"ERROR merging to annual for {year}: {e}")
        write_log(traceback.format_exc())
        return False


# -----------------------------
# ===== main processing =======
# -----------------------------
def process_year_monthly(year):
    """
    处理单个年份：
    - 逐月下载月平均数据（带自动重试）
    - 记录失败月份
    - 合并成年文件
    返回：failed_months（列表，如 ['03','10']）
    """
    write_log(f"===== START YEAR {year} (Monthly Mean) =====")
    failed_months = []
    month_files = []

    for m in MONTHS:
        mf = month_fname(year, m)
        if os.path.exists(mf):
            write_log(f"Month file exists, skipping: {mf}")
            month_files.append(mf)
            continue

        # 下载该月的月平均数据
        out = download_pressure_month(year, m)
        if out is None:
            write_log(f"Download FAILED for {year}-{m}")
            failed_months.append(m)
        else:
            month_files.append(out)

    # 记录失败月份
    if failed_months:
        ffn = failed_months_fname(year)
        with open(ffn, 'w', encoding='utf-8') as fh:
            fh.write('\n'.join(failed_months))
        write_log(f"Some months failed for {year}: {failed_months}. See {ffn} to retry later.")
    else:
        write_log(f"All months downloaded for {year}.")

    # 合并成年文件
    if MERGE_ANNUAL and month_files:
        # 按文件名排序（实际上也就是按月份排序）
        month_files_sorted = sorted(month_files)
        annual_out = annual_fname(year)
        ok = merge_monthly_to_annual(year, month_files_sorted, annual_out)
        if ok:
            # 写元数据说明文件
            try:
                meta_fname = os.path.join(FOLDER_OUT, f'ERA5_metadata_{year}_China.txt')
                with open(meta_fname, 'w', encoding='utf-8') as mf:
                    mf.write(f"Generated: {datetime.datetime.now().isoformat()}\n")
                    mf.write(f"Year: {year}\n")
                    mf.write(f"Region: Southwest China\n")
                    mf.write(f"Area: {AREA[0]}°N to {AREA[2]}°N, {AREA[1]}°E to {AREA[3]}°E\n")
                    mf.write(f"Years range: 1993-2024\n")
                    mf.write(f"Time resolution: Monthly mean\n\n")
                    mf.write("Variables included:\n")
                    mf.write("Pressure level variables (monthly mean):\n")
                    mf.write(" - h200: Geopotential at 200hPa (m**2 s**-2)\n")
                    mf.write(" - u200: U component of wind at 200hPa (m s**-1)\n")
                    mf.write(" - v200: V component of wind at 200hPa (m s**-1)\n")
                    mf.write(" - h500: Geopotential at 500hPa (m**2 s**-2)\n")
                    mf.write(" - u850: U component of wind at 850hPa (m s**-1)\n")
                    mf.write(" - v850: V component of wind at 850hPa (m s**-1)\n")
                    mf.write(" - t850: Temperature at 850hPa (K)\n\n")
                    mf.write("Surface variables (monthly mean):\n")
                    mf.write(" - slp: Mean sea level pressure (Pa)\n")
                    mf.write(" - sst: Sea surface temperature (K)\n")
                    mf.write(" - t2m: 2 metre temperature (K)\n")
                    mf.write(" - tp: Mean total precipitation rate (m s**-1)\n\n")
                    mf.write("Source monthly files:\n")
                    for f in month_files_sorted:
                        mf.write(f" - {f}\n")
                    mf.write(f"\nAnnual merged: {annual_out}\n")
                write_log(f"metadata saved -> {meta_fname}")
            except Exception as e:
                write_log(f"ERROR writing metadata: {e}")
                write_log(traceback.format_exc())

    write_log(f"===== END YEAR {year} =====\n")
    return failed_months


# -----------------------------
# ===== entry point ===========
# -----------------------------
if __name__ == '__main__':
    write_log("=== ERA5 monthly-split download (Southwest China region) - Monthly Mean Data ===")
    write_log(f"Output folder: {FOLDER_OUT}")
    write_log(f"Region: Southwest China")
    write_log(f"Area: {AREA[0]}°N to {AREA[2]}°N, {AREA[1]}°E to {AREA[3]}°E")
    write_log(f"Years: {YEARS[0]} to {YEARS[-1]} (共{len(YEARS)}年)")
    write_log(f"Time resolution: Monthly mean")
    write_log(f"Pressure-level variables: {PRESSURE_LEVEL_VARS}")
    write_log(f"Pressure levels: {PRESSURE_LEVELS}")
    write_log(f"Single-level variables: {SINGLE_LEVEL_VARS}")
    write_log("---------------------------------------------------")

    all_failed = []  # 汇总所有年份失败的 (year, month)

    for y in YEARS:
        try:
            failed_m = process_year_monthly(y)
            if failed_m:
                all_failed.extend([(y, m) for m in failed_m])
        except Exception as e:
            write_log(f"FATAL error processing year {y}: {e}")
            write_log(traceback.format_exc())

    if all_failed:
        write_log(f"=== SUMMARY: some year-months FAILED (please retry or download separately): {all_failed}")
    else:
        write_log("=== SUMMARY: all requested months downloaded & merged successfully ===")

    write_log("=== ALL YEARS PROCESSED ===")
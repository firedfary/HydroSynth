import requests
import os
import time

# start/end in YYYYMM format, e.g. 202507 and 202602
start_yyyymm = 202507
end_yyyymm = 202602

# build inclusive month list between start_yyyymm and end_yyyymm
start_year = start_yyyymm // 100
start_month = start_yyyymm % 100
end_year = end_yyyymm // 100
end_month = end_yyyymm % 100

if not (1 <= start_month <= 12 and 1 <= end_month <= 12):
    raise ValueError("Month must be between 01 and 12 in YYYYMM.")

months = []
year, month = start_year, start_month
while (year < end_year) or (year == end_year and month <= end_month):
    months.append(f"{year}{month:02d}")
    month += 1
    if month > 12:
        month = 1
        year += 1

# 创建下载目录
download_dir = r"D:/ersst_data"
os.makedirs(download_dir, exist_ok=True)

# 下载函数
def download_file(url, filename):
    try:
        with requests.get(url, stream=True) as response:
            response.raise_for_status()
            with open(filename, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        return True
    except Exception as e:
        print(f"Error downloading {url}: {str(e)}")
        return False

# 遍历所有月份进行下�?
for month in months:
    filename = os.path.join(download_dir, f"ersst.v5.{month}.nc")
    url = f"https://www.ncei.noaa.gov/pub/data/cmb/ersst/v5/netcdf/ersst.v5.{month}.nc"
    
    # 跳过已存在的文件
    if os.path.exists(filename):
        print(f"File {filename} already exists. Skipping.")
        continue
    
    # 尝试下载
    print(f"Downloading {month}...")
    if download_file(url, filename):
        print(f"Successfully downloaded {filename}")
    else:
        print(f"Failed to download {filename}")
    
    # 请求间隔
    time.sleep(1)

print("Download process completed.")

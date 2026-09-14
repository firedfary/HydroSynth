# -*- coding: utf-8 -*-
"""
CSU-PCAST NOAA GFS Analysis & GEFS Baseline Downloader
------------------------------------------------------
Downloads operational initial conditions (GFS Analysis) and baseline forecasts (GEFS)
from NOAA Open Data Dissemination (NODD / AWS S3) via public HTTPS:
- GFS Analysis (0.25-deg resolution):
    Used as operational initial condition at 00, 06, 12, 18 UTC cycles (f000).
- GEFS Forecasts (0.50-deg resolution):
    Used as operational baseline for verification (30 members, lead times 6h to 360h).
    Downloads compact pgrb2a product containing total precipitation (APCP).

Output paths are dynamically managed by utils.paths.get_raw_data_dir():
- GFS:  E:/DATA/gfs/pcast_gfs
- GEFS: E:/DATA/gefs/pcast_gefs
"""

import sys
import os
import argparse
import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple
import requests
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.paths import get_raw_data_dir

def disable_proxy():
    """Remove proxy environment variables to force direct connection without consuming VPN quota."""
    for var in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"]:
        os.environ.pop(var, None)
    os.environ["NO_PROXY"] = "*"

# AWS Open Data Public Endpoints (Free, High Bandwidth, No Credentials Needed)
NOAA_GFS_S3_BASE = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"
NOAA_GEFS_S3_BASE = "https://noaa-gefs-pds.s3.amazonaws.com"


def get_gfs_output_dir() -> Path:
    out_dir = get_raw_data_dir() / "gfs" / "pcast_gfs"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def get_gefs_output_dir() -> Path:
    out_dir = get_raw_data_dir() / "gefs" / "pcast_gefs"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def download_single_file(url: str, target_path: Path, min_size_bytes: int = 100 * 1024, force: bool = False) -> bool:
    """Download a file with streaming and resume check."""
    if target_path.exists() and target_path.stat().st_size > min_size_bytes and not force:
        return True

    temp_path = target_path.with_suffix(".tmp")
    try:
        with requests.get(url, stream=True, timeout=30) as r:
            if r.status_code == 404:
                return False
            r.raise_for_status()
            with open(temp_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=2 * 1024 * 1024):
                    if chunk:
                        f.write(chunk)
        temp_path.replace(target_path)
        return True
    except Exception as e:
        if temp_path.exists():
            temp_path.unlink()
        print(f"[Error] Failed to download {target_path.name}: {e}")
        return False


def download_gfs_analyses(dates: List[str], cycles: List[str], max_workers: int = 4, force: bool = False):
    """
    Download GFS 0.25-deg operational analysis fields (f000).
    Dates formatted as YYYYMMDD. Cycles e.g. ['00', '06', '12', '18'].
    """
    out_dir = get_gfs_output_dir()
    tasks = []

    for d in dates:
        for c in cycles:
            # S3 path: gfs.YYYYMMDD/HH/atmos/gfs.tHHz.pgrb2.0p25.f000
            filename = f"gfs.{d}.t{c}z.pgrb2.0p25.f000"
            url = f"{NOAA_GFS_S3_BASE}/gfs.{d}/{c}/atmos/gfs.t{c}z.pgrb2.0p25.f000"
            target = out_dir / filename
            tasks.append((url, target))

    print(f"\n[GFS Analysis] Total cycles to download: {len(tasks)}")
    success = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(download_single_file, url, target, 50 * 1024 * 1024, force): target for url, target in tasks}
        with tqdm(total=len(futures), desc="Downloading GFS Analysis") as pbar:
            for fut in as_completed(futures):
                if fut.result():
                    success += 1
                pbar.update(1)
    print(f"[Done] Downloaded {success}/{len(tasks)} GFS files to {out_dir}")


def download_gefs_forecast(date_str: str, cycle: str, max_lead_hours: int = 360,
                           members: int = 30, max_workers: int = 8, force: bool = False):
    """
    Download GEFS 0.50-deg operational ensemble forecasts for a specific initialization.
    date_str: YYYYMMDD, cycle: '00', '06', '12', or '18'.
    """
    out_dir = get_gefs_output_dir() / f"{date_str}_{cycle}z"
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = []
    # Lead steps: 6-hourly up to max_lead_hours (e.g. 72h or 360h)
    lead_steps = range(6, max_lead_hours + 1, 6)

    # Members: Control (gec00) + Perturbed (gep01 to gep30)
    mem_list = ["gec00"] + [f"gep{m:02d}" for m in range(1, members + 1)]

    for mem in mem_list:
        for f in lead_steps:
            # S3: gefs.YYYYMMDD/HH/atmos/pgrb2ap5/gepXX.tHHz.pgrb2a.0p50.fFFF
            filename = f"{mem}.t{cycle}z.pgrb2a.0p50.f{f:03d}"
            url = f"{NOAA_GEFS_S3_BASE}/gefs.{date_str}/{cycle}/atmos/pgrb2ap5/{filename}"
            target = out_dir / filename
            tasks.append((url, target))

    print(f"\n[GEFS Forecast] Total forecast steps to download: {len(tasks)} ({members} members x {len(lead_steps)} steps)")
    success = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(download_single_file, url, target, 5 * 1024 * 1024, force): target for url, target in tasks}
        with tqdm(total=len(futures), desc=f"Downloading GEFS {date_str} {cycle}z") as pbar:
            for fut in as_completed(futures):
                if fut.result():
                    success += 1
                pbar.update(1)
    print(f"[Done] Downloaded {success}/{len(tasks)} GEFS files to {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="CSU-PCAST NOAA GFS & GEFS Downloader (AWS Open Data)")
    parser.add_argument("--mode", choices=["gfs", "gefs", "both"], default="both", help="Download mode")
    parser.add_argument("--dates", type=str, nargs="+", default=["20231014"],
                        help="Dates in YYYYMMDD format (default: 20231014 for Typhoon Sanba)")
    parser.add_argument("--cycles", type=str, nargs="+", default=["06"],
                        help="Cycle hours: 00, 06, 12, 18 (default: 06)")
    parser.add_argument("--gefs-lead-hours", type=int, default=72,
                        help="Max forecast lead hours for GEFS (default: 72h for case study, up to 360h for 15-day)")
    parser.add_argument("--gefs-members", type=int, default=30, help="Number of ensemble members (default: 30)")
    parser.add_argument("--workers", type=int, default=6, help="Concurrent workers")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    parser.add_argument("--no-proxy", action="store_true", default=True,
                        help="Bypass all system/terminal proxies to save VPN quota (default: True)")
    args = parser.parse_args()

    if args.no_proxy:
        disable_proxy()
        print("[Network] Proxy bypassed: downloading directly via local internet (consuming 0 proxy quota).")

    print("=== CSU-PCAST NOAA GFS & GEFS Downloader ===")
    if args.mode in ("gfs", "both"):
        download_gfs_analyses(args.dates, args.cycles, max_workers=args.workers, force=args.force)

    if args.mode in ("gefs", "both"):
        for d in args.dates:
            for c in args.cycles:
                download_gefs_forecast(d, c, max_lead_hours=args.gefs_lead_hours,
                                       members=args.gefs_members, max_workers=args.workers, force=args.force)


if __name__ == "__main__":
    main()

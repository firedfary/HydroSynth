# -*- coding: utf-8 -*-
"""
CSU-PCAST Master Data Download & Environment Verification Pipeline
------------------------------------------------------------------
Central orchestration script for downloading all datasets required by CSU-PCAST:
1. Static Invariant Fields (LSM, SOIL, ORO) via CDS API
2. ERA5 Reanalysis Atmospheric & Surface Predictors (57 channels) via CDS API
3. GPM IMERG V07 Final Precipitation Labels via NASA GES DISC
4. NOAA GFS Analysis & GEFS Baseline Forecasts via NOAA AWS Open Data

All data paths are dynamically routed to HYDRO_DATA_DIR (E:/DATA) via utils.paths.
"""

import sys
import os
import shutil
import argparse
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.paths import get_raw_data_dir, get_workspace_root


def check_environment():
    """Verify disk space and authentication credentials for all services."""
    print("=" * 65)
    print("       CSU-PCAST Data Pipeline Environment & Health Check        ")
    print("=" * 65)

    # 1. Check Data Root & Free Disk Space
    try:
        data_dir = get_raw_data_dir()
        data_dir.mkdir(parents=True, exist_ok=True)
        total, used, free = shutil.disk_usage(data_dir)
        print(f"[Storage] Target Data Directory : {data_dir}")
        print(f"[Storage] Available Disk Space : {free / (1024**4):.2f} TB free ({free / (1024**3):.1f} GB) out of {total / (1024**4):.2f} TB total")
    except Exception as e:
        print(f"[Storage Error] Failed to inspect target directory: {e}")

    # 2. Check CDS API (ERA5)
    print("\n--- Checking ECMWF CDS API (ERA5) ---")
    cdsapirc_path = Path.home() / ".cdsapirc"
    if cdsapirc_path.exists():
        print(f"[OK] Found ~/.cdsapirc config: {cdsapirc_path}")
        try:
            import cdsapi
            client = cdsapi.Client()
            print(f"[OK] CDS API client initialized successfully. Endpoint: {client.url}")
        except Exception as e:
            print(f"[Warn] CDS API client error: {e}")
    else:
        print(f"[Missing] ~/.cdsapirc file not found. ERA5 downloads require CDS API key.")

    # 3. Check NASA Earthdata (.netrc for IMERG)
    print("\n--- Checking NASA Earthdata Credentials (IMERG) ---")
    netrc_path = Path.home() / ".netrc"
    _netrc_path = Path.home() / "_netrc"
    found_netrc = None
    for p in (netrc_path, _netrc_path):
        if p.exists():
            found_netrc = p
            break

    if found_netrc:
        print(f"[OK] Found Earthdata netrc: {found_netrc}")
        content = found_netrc.read_text(encoding="ascii", errors="ignore")
        if "urs.earthdata.nasa.gov" in content:
            print("[OK] Confirmed entry for 'urs.earthdata.nasa.gov'.")
        else:
            print("[Warn] 'urs.earthdata.nasa.gov' not explicitly found inside .netrc.")
    else:
        print(f"[Missing] Neither ~/.netrc nor ~/_netrc found. NASA IMERG requires Earthdata credentials.")

    # 4. Check NOAA AWS Open Data (GFS/GEFS)
    print("\n--- Checking NOAA Open Data AWS Connectivity (GFS/GEFS) ---")
    try:
        import requests
        res = requests.head("https://noaa-gfs-bdp-pds.s3.amazonaws.com/?prefix=", timeout=8)
        if res.status_code in (200, 403, 307):
            print("[OK] Direct access to NOAA Open Data on AWS S3 is active (No credentials needed).")
        else:
            print(f"[Status] NOAA S3 responded with status: {res.status_code}")
    except Exception as e:
        print(f"[Warn] Connection to NOAA AWS Open Data failed: {e}")

    print("=" * 65)


def run_command(cmd_list):
    """Execute a python command using the active environment."""
    python_exe = sys.executable
    full_cmd = [python_exe] + cmd_list
    print(f"\n[Executing] {' '.join(full_cmd)}")
    ret = subprocess.run(full_cmd)
    if ret.returncode != 0:
        print(f"[Warning] Process exited with non-zero code: {ret.returncode}")


def main():
    parser = argparse.ArgumentParser(description="CSU-PCAST Master Data Downloader")
    parser.add_argument("--check-env", action="store_true", help="Run health check and exit")
    parser.add_argument("--dataset", choices=["all", "static", "era5", "imerg", "gfs", "gefs"],
                        default="all", help="Target dataset to download")
    parser.add_argument("--start-year", type=int, default=2018, help="ERA5 start year (default: 2018)")
    parser.add_argument("--end-year", type=int, default=2022, help="ERA5 end year (default: 2022)")
    parser.add_argument("--months", type=int, nargs="+", default=list(range(1, 13)), help="Months to download")
    parser.add_argument("--start-date", type=str, default="2023-10-14", help="IMERG start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default="2023-10-21", help="IMERG end date (YYYY-MM-DD)")
    parser.add_argument("--date", type=str, default="20231014", help="GFS/GEFS date in YYYYMMDD")
    parser.add_argument("--cycle", type=str, default="06", help="GFS/GEFS cycle hour (00, 06, 12, 18)")
    parser.add_argument("--gefs-lead-hours", type=int, default=72, help="GEFS lead hours (default: 72h)")
    parser.add_argument("--workers", type=int, default=4, help="Download threads")
    parser.add_argument("--connections", type=int, default=8, help="Aria2 connections per file for ERA5 (default: 8)")
    parser.add_argument("--no-proxy", action="store_true", default=True,
                        help="Bypass all system/terminal proxies to save VPN quota (default: True)")
    args = parser.parse_args()

    if args.no_proxy:
        for var in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"]:
            os.environ.pop(var, None)
        os.environ["NO_PROXY"] = "*"
        print("[Network] Proxy bypassed: all downloads will connect directly via local internet (0 VPN quota used).")

    if args.check_env:
        check_environment()
        return

    check_environment()

    process_dir = REPO_ROOT / "process"
    proxy_flag = ["--no-proxy"] if args.no_proxy else []

    # 1. Static Invariant Fields
    if args.dataset in ("all", "static"):
        print("\n>>> Task 1: Downloading ERA5 Static Invariant Fields (LSM, SOIL, ORO) <<<")
        run_command([str(process_dir / "download_pcast_era5.py"), "--static-only"] + proxy_flag)

    # 2. ERA5 Multi-variable Data
    if args.dataset in ("all", "era5"):
        print(f"\n>>> Task 2: Downloading ERA5 Atmospheric & Surface Data ({args.start_year}-{args.end_year}) <<<")
        era5_cmd = [
            str(process_dir / "download_pcast_era5.py"),
            "--start-year", str(args.start_year),
            "--end-year", str(args.end_year),
            "--connections", str(args.connections),
            "--months"
        ] + [str(m) for m in args.months] + proxy_flag
        run_command(era5_cmd)

    # 3. GPM IMERG Precipitation
    if args.dataset in ("all", "imerg"):
        print(f"\n>>> Task 3: Downloading GPM IMERG Precipitation ({args.start_date} to {args.end_date}) <<<")
        imerg_cmd = [
            str(process_dir / "download_pcast_imerg.py"),
            "--start-date", args.start_date,
            "--end-date", args.end_date,
            "--workers", str(args.workers)
        ] + proxy_flag
        run_command(imerg_cmd)

    # 4. GFS Operational Analysis
    if args.dataset in ("all", "gfs"):
        print(f"\n>>> Task 4: Downloading NOAA GFS Operational Analysis ({args.date} {args.cycle}z) <<<")
        gfs_cmd = [
            str(process_dir / "download_pcast_gefs_gfs.py"),
            "--mode", "gfs",
            "--dates", args.date,
            "--cycles", args.cycle,
            "--workers", str(args.workers)
        ] + proxy_flag
        run_command(gfs_cmd)

    # 5. GEFS Baseline Forecast
    if args.dataset in ("all", "gefs"):
        print(f"\n>>> Task 5: Downloading NOAA GEFS Baseline Forecasts ({args.date} {args.cycle}z, lead 0-{args.gefs_lead_hours}h) <<<")
        gefs_cmd = [
            str(process_dir / "download_pcast_gefs_gfs.py"),
            "--mode", "gefs",
            "--dates", args.date,
            "--cycles", args.cycle,
            "--gefs-lead-hours", str(args.gefs_lead_hours),
            "--workers", str(args.workers)
        ] + proxy_flag
        run_command(gefs_cmd)

    print("\n[Done] All requested download tasks have been processed.")


if __name__ == "__main__":
    main()

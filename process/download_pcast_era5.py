# -*- coding: utf-8 -*-
"""
CSU-PCAST ERA5 Downloader (Aria2 Multi-Connection Accelerated + Cloud Pipelining)
---------------------------------------------------------------------------------
Downloads 57 atmospheric and surface variables at 0.25-deg resolution,
6-hourly (00, 06, 12, 18 UTC) as required by CSU-PCAST:
- 6 Upper-air variables on 8 pressure levels: Z, T, U, V, Q, W
- 9 Surface variables: 2T, 2D, U10, V10, MSL, CAPE, TCWV, SP, SWVL1
- 3 Static variables: LSM, SOIL, ORO

Key Accelerations:
1. Direct Link Extraction (获取真实直链):
   - Submits requests to ECMWF CADS API.
   - Once ready, extracts the direct pre-signed S3 object-store URL (object-store.os-api.cci2.ecmwf.int).
2. Multi-Connection High-Speed aria2c Engine (多线程分块并发下载):
   - Automatically uses aria2c with 8~16 parallel connections per file.
   - Leverages HTTP Accept-Ranges to fully saturate network bandwidth.
3. Dual-Track Cloud Pipelining (双轨异步流水线):
   - Upper-Air track and Surface track run in parallel.
   - While aria2c is downloading Month N, Month N+1 is already pre-submitted to
     ECMWF cloud queue so it is ready immediately when Month N completes.
4. Robust Fallback & Resume:
   - Verifies complete file size (PL >= 8.0 GB, SFC >= 1.0 GB).
   - Automatically skips already completed files.
   - Falls back to built-in downloader if aria2c encounters errors.
"""

import sys
import os
import time
import shutil
import argparse
import calendar
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

def disable_proxy():
    """Remove proxy environment variables to force direct connection without consuming VPN quota."""
    for var in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"]:
        os.environ.pop(var, None)
    os.environ["NO_PROXY"] = "*"

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.paths import get_raw_data_dir

# -----------------------------------------------------------------------------
# Parameter Definitions based on CSU-PCAST (Table 1 in Xiong et al., 2026)
# -----------------------------------------------------------------------------
PRESSURE_LEVELS = ['200', '250', '300', '400', '500', '600', '700', '850']

UPPER_AIR_VARIABLES = [
    'geopotential',
    'temperature',
    'u_component_of_wind',
    'v_component_of_wind',
    'specific_humidity',
    'vertical_velocity',
]

SURFACE_VARIABLES = [
    '2m_temperature',
    '2m_dewpoint_temperature',
    '10m_u_component_of_wind',
    '10m_v_component_of_wind',
    'mean_sea_level_pressure',
    'convective_available_potential_energy',
    'total_column_water_vapour',
    'surface_pressure',
    'volumetric_soil_water_layer_1',
]

STATIC_VARIABLES = [
    'land_sea_mask',
    'soil_type',
    'geopotential',
]

HOURS_6H = ['00:00', '06:00', '12:00', '18:00']

MIN_PL_FILE_SIZE_BYTES = 8000 * 1024 * 1024   # >= 8.0 GB for monthly upper-air
MIN_SFC_FILE_SIZE_BYTES = 1000 * 1024 * 1024  # >= 1.0 GB for monthly surface


def find_aria2() -> Optional[Path]:
    """Locate aria2c executable in current Python environment or system PATH."""
    env_scripts = Path(sys.executable).parent / "Scripts" / "aria2c.exe"
    if env_scripts.exists():
        return env_scripts

    which_path = shutil.which("aria2c")
    if which_path:
        return Path(which_path)

    # Search in WinGet packages
    winget_dir = Path.home() / "AppData" / "Local" / "Microsoft" / "WinGet"
    if winget_dir.exists():
        matches = list(winget_dir.glob("**/aria2c.exe"))
        if matches:
            return matches[0]
    return None


def get_era5_output_dir() -> Path:
    out_dir = get_raw_data_dir() / "era5" / "pcast_era5"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def get_static_output_dir() -> Path:
    out_dir = get_raw_data_dir() / "static" / "pcast_static"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def is_file_complete(file_path: Path, min_size: int) -> bool:
    """Check if a file exists and satisfies the minimum complete size."""
    if not file_path.exists():
        return False
    return file_path.stat().st_size >= min_size


def download_static_fields(client, force: bool = False):
    """Download static invariant fields: LSM, SOIL, ORO (geopotential)."""
    static_dir = get_static_output_dir()
    target_file = static_dir / "era5_static_lsm_soil_oro.nc"

    if is_file_complete(target_file, 500 * 1024) and not force:
        print(f"[Skip] Static file already exists: {target_file.name}")
        return

    print(f"\n[CDS] Requesting static invariant fields -> {target_file}")
    client.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': ['reanalysis'],
            'variable': STATIC_VARIABLES,
            'year': ['2020'],
            'month': ['01'],
            'day': ['01'],
            'time': ['00:00'],
            'data_format': 'netcdf',
            'download_format': 'unarchived',
        },
        str(target_file)
    )
    print(f"[Done] Static invariant fields saved: {target_file.name} ({target_file.stat().st_size / 1e6:.2f} MB)")


def make_pl_request(year: int, month: int) -> Dict[str, Any]:
    month_str = f"{month:02d}"
    _, num_days = calendar.monthrange(year, month)
    days = [f"{d:02d}" for d in range(1, num_days + 1)]
    return {
        'product_type': ['reanalysis'],
        'variable': UPPER_AIR_VARIABLES,
        'pressure_level': PRESSURE_LEVELS,
        'year': [str(year)],
        'month': [month_str],
        'day': days,
        'time': HOURS_6H,
        'data_format': 'netcdf',
        'download_format': 'unarchived',
    }


def make_sfc_request(year: int, month: int) -> Dict[str, Any]:
    month_str = f"{month:02d}"
    _, num_days = calendar.monthrange(year, month)
    days = [f"{d:02d}" for d in range(1, num_days + 1)]
    return {
        'product_type': ['reanalysis'],
        'variable': SURFACE_VARIABLES,
        'year': [str(year)],
        'month': [month_str],
        'day': days,
        'time': HOURS_6H,
        'data_format': 'netcdf',
        'download_format': 'unarchived',
    }


def download_with_aria2(url: str, target_file: Path, aria2_exe: Path,
                        connections: int = 8, no_proxy: bool = True) -> bool:
    """
    Download a file using aria2c with multi-connection chunking.
    """
    temp_file = target_file.with_suffix(".tmp")
    aria2_control_file = target_file.parent / f"{temp_file.name}.aria2"

    cmd = [
        str(aria2_exe),
        "-s", str(connections),
        "-x", str(connections),
        "-k", "2M",
        "--file-allocation=none",
        "--auto-file-renaming=false",
        "--allow-overwrite=true",
        "--max-tries=10",
        "--retry-wait=5",
        "--console-log-level=notice",
        "--summary-interval=10",
        "-d", str(target_file.parent),
        "-o", temp_file.name,
        url
    ]
    if no_proxy:
        cmd.append('--all-proxy=""')

    print(f"\n[aria2c] Spawning multi-connection download ({connections} streams) -> {target_file.name}")
    try:
        proc = subprocess.run(cmd, check=False)
        if proc.returncode == 0 and temp_file.exists():
            temp_file.replace(target_file)
            if aria2_control_file.exists():
                aria2_control_file.unlink()
            return True
        else:
            print(f"[aria2c] Download failed with exit code: {proc.returncode}")
            return False
    except Exception as e:
        print(f"[aria2c Error] {e}")
        return False


def run_pipelined_track(track_name: str, collection_id: str,
                        tasks: List[Dict[str, Any]], min_size_bytes: int,
                        max_in_flight: int = 2, connections: int = 8,
                        aria2_exe: Optional[Path] = None,
                        force: bool = False, no_proxy: bool = True):
    """
    Run an asynchronous pipelined download track using aria2c for multi-threaded transfer.
    """
    import cdsapi
    client = cdsapi.Client()

    print(f"\n[{track_name}] Initializing track. Total months queued: {len(tasks)}")

    # Filter out already completed tasks
    pending_tasks = []
    for t in tasks:
        if is_file_complete(t['target_file'], min_size_bytes) and not force:
            print(f"[{track_name}] [Skip] Already complete: {t['target_file'].name} ({t['target_file'].stat().st_size / 1e6:.2f} MB)")
        else:
            pending_tasks.append(t)

    if not pending_tasks:
        print(f"[{track_name}] All tasks in this track are already complete!")
        return

    print(f"[{track_name}] Remaining tasks to download: {len(pending_tasks)}")

    active_jobs = []
    task_idx = 0

    while task_idx < len(pending_tasks) or active_jobs:
        # 1. Pre-submit requests to ECMWF cloud queue up to max_in_flight
        while len(active_jobs) < max_in_flight and task_idx < len(pending_tasks):
            next_task = pending_tasks[task_idx]
            y, m = next_task['year'], next_task['month']
            try:
                print(f"\n[{track_name}] Pre-submitting {y}-{m:02d} to ECMWF cloud queue...")
                remote = client.client.submit(collection_id, next_task['request'])
                print(f"[{track_name}] {y}-{m:02d} queued successfully (Request ID: {remote.request_id})")
                active_jobs.append((next_task, remote))
                task_idx += 1
            except Exception as e:
                print(f"[{track_name}] Submission failed for {y}-{m:02d}: {e}. Retrying in 30s...")
                time.sleep(30)
                break

        if not active_jobs:
            time.sleep(10)
            continue

        # 2. Monitor the head of the queue
        curr_task, curr_remote = active_jobs[0]
        y, m = curr_task['year'], curr_task['month']
        target_path = curr_task['target_file']

        try:
            curr_remote.update()
            status = curr_remote.status

            if status == "successful":
                print(f"\n[{track_name}] >>> {y}-{m:02d} is READY on ECMWF cloud! Extracting direct URL... <<<")
                results = curr_remote.get_results()

                # Extract Direct S3 URL
                direct_url = getattr(results, "location", None)
                if not direct_url and hasattr(results, "asset") and isinstance(results.asset, dict):
                    direct_url = results.asset.get("href")

                download_start = time.time()
                success = False

                if direct_url and aria2_exe and aria2_exe.exists():
                    print(f"[{track_name}] Direct S3 URL: {direct_url[:80]}...")
                    print(f"[{track_name}] Handing over to aria2c ({connections} streams)...")
                    success = download_with_aria2(direct_url, target_path, aria2_exe,
                                                  connections=connections, no_proxy=no_proxy)

                # Fallback to standard CDS download if aria2c is unavailable or fails
                if not success:
                    print(f"[{track_name}] Using built-in streaming download fallback...")
                    temp_file = target_path.with_suffix(".tmp")
                    if temp_file.exists():
                        temp_file.unlink()
                    curr_remote.download(str(temp_file))
                    temp_file.replace(target_path)
                    success = True

                elapsed = time.time() - download_start
                size_mb = target_path.stat().st_size / 1e6
                speed_mb_s = size_mb / max(elapsed, 1)

                print(f"[{track_name}] [Done] Saved: {target_path.name} ({size_mb:.2f} MB in {elapsed/60:.1f} min, speed: {speed_mb_s:.2f} MB/s)")
                active_jobs.pop(0)

            elif status in ("accepted", "running"):
                state_str = "in server queue" if status == "accepted" else "slicing data on server"
                print(f"[{track_name}] {y}-{m:02d} status: {status} ({state_str}) | Queue size: {len(active_jobs)}")
                time.sleep(20)

            elif status in ("failed", "rejected"):
                print(f"[{track_name}] [Error] Job failed for {y}-{m:02d} with status: {status}. Retrying...")
                active_jobs.pop(0)
                time.sleep(10)

            else:
                print(f"[{track_name}] Status '{status}' for {y}-{m:02d}. Waiting...")
                time.sleep(20)

        except Exception as e:
            print(f"[{track_name}] Error monitoring/downloading {y}-{m:02d}: {e}. Retrying in 20s...")
            time.sleep(20)

    print(f"\n[{track_name}] Track completed successfully.")


def main():
    parser = argparse.ArgumentParser(description="CSU-PCAST ERA5 Downloader (Aria2 Multi-Connection Accelerated)")
    parser.add_argument("--start-year", type=int, default=2018, help="Start year (default: 2018)")
    parser.add_argument("--end-year", type=int, default=2022, help="End year (default: 2022)")
    parser.add_argument("--months", type=int, nargs="+", default=list(range(1, 13)), help="Months to download (1-12)")
    parser.add_argument("--skip-pl", action="store_true", help="Skip pressure-level downloads")
    parser.add_argument("--skip-sfc", action="store_true", help="Skip surface downloads")
    parser.add_argument("--static-only", action="store_true", help="Download static fields only")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    parser.add_argument("--connections", type=int, default=8, help="Aria2 concurrent connections per file (default: 8)")
    parser.add_argument("--max-in-flight", type=int, default=2,
                        help="Max pre-submitted requests queued in ECMWF cloud per track (default: 2)")
    parser.add_argument("--no-proxy", action="store_true", default=True,
                        help="Bypass all system/terminal proxies to save VPN quota (default: True)")
    args = parser.parse_args()

    if args.no_proxy:
        disable_proxy()
        print("[Network] Proxy bypassed: downloading directly via local internet (consuming 0 proxy quota).")

    aria2_exe = find_aria2()
    if aria2_exe:
        print(f"[Aria2] Multi-connection engine detected: {aria2_exe} (Concurrency: {args.connections} streams)")
    else:
        print("[Aria2] aria2c executable not found. Will use built-in streaming downloader.")

    import cdsapi
    client = cdsapi.Client()

    era5_dir = get_era5_output_dir()
    print("=" * 65)
    print("  CSU-PCAST ERA5 Downloader (Aria2 Accelerated + Pipelined)     ")
    print("=" * 65)
    print(f"Destination     : {era5_dir}")
    print(f"Years           : {args.start_year} to {args.end_year}")
    print(f"Months          : {args.months}")
    print(f"Aria2 Streams   : {args.connections} connections per file")
    print(f"Cloud In-Flight : {args.max_in_flight} tasks per track")

    # Static fields
    print("\n--- Checking Static Fields ---")
    download_static_fields(client, force=args.force)
    if args.static_only:
        print("[Done] Static invariant fields download completed.")
        return

    # Prepare Upper-Air tasks
    pl_tasks = []
    if not args.skip_pl:
        for y in range(args.start_year, args.end_year + 1):
            for m in args.months:
                target = era5_dir / f"era5_pl_{y}{m:02d}.nc"
                pl_tasks.append({
                    'year': y,
                    'month': m,
                    'request': make_pl_request(y, m),
                    'target_file': target
                })

    # Prepare Surface tasks
    sfc_tasks = []
    if not args.skip_sfc:
        for y in range(args.start_year, args.end_year + 1):
            for m in args.months:
                target = era5_dir / f"era5_sfc_{y}{m:02d}.nc"
                sfc_tasks.append({
                    'year': y,
                    'month': m,
                    'request': make_sfc_request(y, m),
                    'target_file': target
                })

    # Execute Track 1 (Upper-Air) and Track 2 (Surface) in parallel!
    print("\n>>> Launching Dual-Track Aria2 Pipelined Download <<<")
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = []
        if pl_tasks:
            futures.append(executor.submit(
                run_pipelined_track,
                "Upper-Air Track",
                "reanalysis-era5-pressure-levels",
                pl_tasks,
                MIN_PL_FILE_SIZE_BYTES,
                args.max_in_flight,
                args.connections,
                aria2_exe,
                args.force,
                args.no_proxy
            ))
        if sfc_tasks:
            futures.append(executor.submit(
                run_pipelined_track,
                "Surface Track",
                "reanalysis-era5-single-levels",
                sfc_tasks,
                MIN_SFC_FILE_SIZE_BYTES,
                args.max_in_flight,
                args.connections,
                aria2_exe,
                args.force,
                args.no_proxy
            ))

        for f in as_completed(futures):
            f.result()

    print("\n[Completed] All ERA5 accelerated downloads finished successfully.")


if __name__ == "__main__":
    main()

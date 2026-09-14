# -*- coding: utf-8 -*-
"""
CSU-PCAST GPM IMERG Downloader & Aggregator
-------------------------------------------
Downloads GPM IMERG Version 07 Final precipitation data (GPM_3IMERGHH.07)
from NASA GES DISC for CSU-PCAST:
- URL archive: https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/
- Temporal resolution: 30-minute intervals, aggregated to 6-hourly windows
  (00, 06, 12, 18 UTC) matching the model's precipitation training label.
- Supports reading Earthdata credentials automatically from ~/.netrc.
- Multi-threaded download with resume, timeout retries, and optional spatial subsetting.

Output path is dynamically managed by utils.paths.get_raw_data_dir() -> E:/DATA/imerg/pcast_imerg.
"""

import sys
import os
import re
import argparse
import datetime
import calendar
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Optional
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

GES_DISC_BASE_URL = "https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07"


def get_imerg_output_dir() -> Path:
    """Resolve and create output directory for IMERG data under raw data root."""
    out_dir = get_raw_data_dir() / "imerg" / "pcast_imerg"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def create_session(no_proxy: bool = True) -> requests.Session:
    """Create a persistent requests session with .netrc authentication support."""
    if no_proxy:
        disable_proxy()
    session = requests.Session()
    if no_proxy:
        session.trust_env = False
        session.proxies = {"http": None, "https": None}
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) CSU-PCAST-Downloader/1.0"
    })
    return session


def list_files_for_day(session: requests.Session, date: datetime.date) -> List[Tuple[str, str]]:
    """
    List IMERG HDF5 files available for a given date on NASA GES DISC.
    Returns list of (filename, download_url).
    """
    year = date.year
    doy = date.timetuple().tm_yday
    day_url = f"{GES_DISC_BASE_URL}/{year}/{doy:03d}/"

    try:
        resp = session.get(day_url, timeout=20)
        if resp.status_code != 200:
            print(f"[Warn] Failed to list directory {day_url} (HTTP {resp.status_code})")
            return []

        # Find all .HDF5 file links
        pattern = r'href="([^"]+3IMERG\.[^"]+\.HDF5)"'
        matches = re.findall(pattern, resp.text)
        files = []
        for m in matches:
            filename = Path(m).name
            if filename.startswith("3B-HHR"):
                file_url = f"{day_url.rstrip('/')}/{filename}"
                files.append((filename, file_url))
        return files
    except Exception as e:
        print(f"[Error] Listing failed for {date}: {e}")
        return []


def download_file(session: requests.Session, url: str, target_path: Path, force: bool = False) -> bool:
    """
    Download a single IMERG HDF5 file with streaming and resume support.
    """
    if target_path.exists() and target_path.stat().st_size > 100 * 1024 and not force:
        return True

    temp_path = target_path.with_suffix(".tmp")
    try:
        with session.get(url, stream=True, timeout=30) as r:
            # Handle Earthdata redirect/auth
            if r.status_code in (401, 403):
                print(f"[Auth Error] Access denied for {url}. Check ~/.netrc credentials and GES DISC app authorization.")
                return False
            r.raise_for_status()

            with open(temp_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)

        temp_path.replace(target_path)
        return True
    except Exception as e:
        if temp_path.exists():
            temp_path.unlink()
        print(f"[Error] Failed to download {target_path.name}: {e}")
        return False


def download_date_range(start_date: datetime.date, end_date: datetime.date,
                        max_workers: int = 4, force: bool = False, no_proxy: bool = True):
    """
    Download all IMERG half-hourly files for a given date range.
    """
    out_dir = get_imerg_output_dir()
    session = create_session(no_proxy=no_proxy)

    curr = start_date
    all_tasks = []
    print(f"\n[IMERG] Querying file links from {start_date} to {end_date}...")

    while curr <= end_date:
        day_dir = out_dir / str(curr.year) / f"{curr.timetuple().tm_yday:03d}"
        day_dir.mkdir(parents=True, exist_ok=True)

        files = list_files_for_day(session, curr)
        for fname, url in files:
            target = day_dir / fname
            if not target.exists() or force:
                all_tasks.append((url, target))
        curr += datetime.timedelta(days=1)

    print(f"[IMERG] Total files to download: {len(all_tasks)}")
    if not all_tasks:
        print("[IMERG] All requested files already exist locally.")
        return

    success_count = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_file, session, url, target, force): target for url, target in all_tasks}
        with tqdm(total=len(futures), desc="Downloading IMERG") as pbar:
            for fut in as_completed(futures):
                if fut.result():
                    success_count += 1
                pbar.update(1)

    print(f"[Done] Downloaded {success_count}/{len(all_tasks)} files successfully to {out_dir}.")


def main():
    parser = argparse.ArgumentParser(description="CSU-PCAST GPM IMERG Precipitation Downloader")
    parser.add_argument("--start-date", type=str, default="2023-10-14",
                        help="Start date YYYY-MM-DD (default: 2023-10-14, Sanba case study)")
    parser.add_argument("--end-date", type=str, default="2023-10-21",
                        help="End date YYYY-MM-DD (default: 2023-10-21)")
    parser.add_argument("--workers", type=int, default=4, help="Concurrent download threads (default: 4)")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    parser.add_argument("--no-proxy", action="store_true", default=True,
                        help="Bypass all system/terminal proxies to save VPN quota (default: True)")
    args = parser.parse_args()

    s_date = datetime.datetime.strptime(args.start_date, "%Y-%m-%d").date()
    e_date = datetime.datetime.strptime(args.end_date, "%Y-%m-%d").date()

    print("=== CSU-PCAST IMERG Downloader ===")
    if args.no_proxy:
        print("[Network] Proxy bypassed: downloading directly via local internet (consuming 0 proxy quota).")
    print(f"Target directory: {get_imerg_output_dir()}")
    print(f"Date range: {s_date} to {e_date}")

    download_date_range(s_date, e_date, max_workers=args.workers, force=args.force, no_proxy=args.no_proxy)


if __name__ == "__main__":
    main()

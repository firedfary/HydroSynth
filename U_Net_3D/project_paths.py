"""Workspace paths for the U_Net_3D subproject.

Raw inputs are read from ``HYDRO_DATA_DIR``. Reusable derived datasets live
under the subproject cache, while model runs and evaluations live under the
subproject results directory.
"""

from __future__ import annotations

import sys
from pathlib import Path


# Support both ``python -m U_Net_3D.<module>`` and direct script execution.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.paths import SubprojectPaths


paths = SubprojectPaths(__file__, subproject_name="U_Net_3D")

MODEL_DATA_DIR = paths.get_raw_data("model_data")
ERSST_DATA_DIR = paths.get_raw_data("ersst_data")

PREPARED_DATA_DIR = paths.cache_dir / "prepared"
STATION_CACHE_DIR = paths.cache_dir / "station_observations"
STATION_TABLE_FILE = STATION_CACHE_DIR / "observe_data24.csv"
OBSERVATION_FILE = STATION_CACHE_DIR / "hr_observations_ref1994_2010.npz"
ALIGNED_OBSERVATION_FILE = (
    STATION_CACHE_DIR / "hr_observations_ref1994_2010_aligned.npy"
)


def experiment_dir(name: str) -> Path:
    """Return a managed experiment directory and create standard subfolders."""
    return paths.get_exp_dir(name)


def experiment_path(name: str, filename: str) -> Path:
    """Return a file path inside a managed experiment directory."""
    return experiment_dir(name) / filename

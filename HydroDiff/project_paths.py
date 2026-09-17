"""Workspace and experiment path management for the HydroDiff subproject.

Enforces workspace isolation rules:
- Codebase (d:\\HydroSynth\\HydroDiff) remains strictly clean.
- Intermediate caches -> <HYDRO_WORKSPACE>/cache/HydroDiff/
- Experiment outputs, checkpoints, figures, logs -> <HYDRO_WORKSPACE>/results/HydroDiff/<exp_name>/
- Shared raw data -> read via paths.get_raw_data()
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.paths import SubprojectPaths

paths = SubprojectPaths(__file__, subproject_name="HydroDiff")


def experiment_dir(name: str) -> Path:
    """Return a managed experiment directory and auto-create standard subfolders."""
    return paths.get_exp_dir(name)


def experiment_path(name: str, filename: str) -> Path:
    """Return a file path inside a managed experiment directory."""
    return experiment_dir(name) / filename

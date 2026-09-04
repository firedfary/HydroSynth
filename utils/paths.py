"""
HydroSynth Global Path Management & Workspace Auto-Namespace Engine.

This module provides unified, cross-platform path resolution for HydroSynth:
1. Reads environment variables from .env (e.g., HYDRO_WORKSPACE, HYDRO_DATA_DIR).
2. Provides `SubprojectPaths(caller_file)` which automatically derives isolated
   `cache/` and `results/` directories for any subproject (e.g., HydroGraph_S2S, FNO, U_Net_3D)
   in the external workspace.
3. Completely decouples codebase from heavy binary data, caching, checkpoints, and logs.
"""

import os
from pathlib import Path
from typing import Optional, Union, Dict, Any

# Root directory of the HydroSynth git repository
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENV_FILE = REPO_ROOT / ".env"


def load_env(env_file: Optional[Union[str, Path]] = None, override: bool = False) -> Dict[str, str]:
    """Load KEY=VALUE pairs from a .env file into os.environ."""
    env_path = Path(env_file).expanduser() if env_file is not None else DEFAULT_ENV_FILE
    loaded = {}

    if not env_path.exists():
        return loaded

    with env_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")

            if not key:
                continue
            if override or key not in os.environ:
                os.environ[key] = value
            loaded[key] = os.environ.get(key, value)

    return loaded


# Automatically load .env on module import
load_env()


def get_workspace_root() -> Path:
    """
    Get the root directory for external data workspace (cache, results, runs).
    The root must be configured explicitly in the repository ``.env`` or the
    process environment. This avoids silently writing to a machine-specific
    fallback location.
    """
    env_ws = os.getenv("HYDRO_WORKSPACE")
    if not env_ws:
        raise RuntimeError(
            "HYDRO_WORKSPACE is not configured. Set it in the repository .env."
        )
    return Path(env_ws).expanduser().resolve()


def get_raw_data_dir() -> Path:
    """
    Get the shared raw datasets directory (SST, ERA5, station data, etc.).
    The raw-data root must be configured explicitly so source code never
    invents or writes to a local data location.
    """
    env_data = os.getenv("HYDRO_DATA_DIR")
    if not env_data:
        raise RuntimeError(
            "HYDRO_DATA_DIR is not configured. Set it in the repository .env."
        )
    return Path(env_data).expanduser().resolve()


class SubprojectPaths:
    """
    Subproject-aware path manager.
    
    Given the caller's `__file__` (or explicit name), it automatically discovers
    the subproject name and constructs dedicated cache and result paths.
    
    Usage:
        paths = SubprojectPaths(__file__)
        cache_file = paths.cache_dir / "graph_adj.npz"
        exp_dir = paths.get_exp_dir("baseline_v1")
        raw_sst = paths.get_raw_data("sst_6chan.npy")
    """
    def __init__(self, caller_file_or_name: Union[str, Path], subproject_name: Optional[str] = None):
        if subproject_name:
            self.subproject_name = subproject_name
        else:
            self.subproject_name = self._infer_subproject_name(caller_file_or_name)

        self.workspace_root = get_workspace_root()
        self.raw_data_dir = get_raw_data_dir()

        # Dedicated cache directory for this subproject
        self.cache_dir = self.workspace_root / "cache" / self.subproject_name
        
        # Dedicated results directory for this subproject
        self.results_dir = self.workspace_root / "results" / self.subproject_name

    def _infer_subproject_name(self, caller_ref: Union[str, Path]) -> str:
        caller_path = Path(caller_ref).resolve()
        try:
            rel = caller_path.relative_to(REPO_ROOT)
            parts = rel.parts
            if parts and parts[0] not in ("utils", "process", "ref", "tmp", ".agents"):
                return parts[0]
        except ValueError:
            pass
        
        # Fallback if caller is outside repo or at root level
        parent_name = caller_path.parent.name
        if parent_name and parent_name not in ("utils", "HydroSynth", ""):
            return parent_name
        return "common"

    def ensure_dirs(self) -> "SubprojectPaths":
        """Ensure base cache and results directories exist."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        return self

    def get_cache_file(self, filename: str) -> Path:
        """Get path to a file inside the subproject's cache directory (creates dir if needed)."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        return self.cache_dir / filename

    def get_exp_dir(self, exp_name: str, create_subdirs: bool = True) -> Path:
        """
        Get an isolated experiment output directory.
        Optionally creates 'checkpoints', 'figures', and 'logs' subdirectories.
        """
        exp_dir = self.results_dir / exp_name
        if create_subdirs:
            (exp_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
            (exp_dir / "figures").mkdir(parents=True, exist_ok=True)
            (exp_dir / "logs").mkdir(parents=True, exist_ok=True)
        else:
            exp_dir.mkdir(parents=True, exist_ok=True)
        return exp_dir

    def get_raw_data(self, rel_path: Union[str, Path]) -> Path:
        """Get path to a file or folder in the shared raw datasets directory."""
        relative = Path(rel_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Raw-data path must be relative: {rel_path}")
        return self.raw_data_dir / relative

    def __repr__(self) -> str:
        return (
            f"SubprojectPaths(subproject='{self.subproject_name}', "
            f"cache_dir='{self.cache_dir}', results_dir='{self.results_dir}')"
        )


def get_subproject_paths(caller_file_or_name: Union[str, Path], subproject_name: Optional[str] = None) -> SubprojectPaths:
    """Convenience factory function for SubprojectPaths."""
    return SubprojectPaths(caller_file_or_name, subproject_name).ensure_dirs()

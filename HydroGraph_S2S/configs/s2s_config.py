import os
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import torch

# Ensure HydroSynth root is accessible for importing global config
HYDROSYNTH_ROOT = Path(__file__).resolve().parents[2]
if str(HYDROSYNTH_ROOT) not in sys.path:
    sys.path.insert(0, str(HYDROSYNTH_ROOT))

try:
    from utils.paths import SubprojectPaths, get_raw_data_dir
    _paths = SubprojectPaths(__file__)
    _DEFAULT_CACHE_DIR = str(_paths.cache_dir)
    _DEFAULT_RESULTS_DIR = str(_paths.results_dir)
    _DEFAULT_RAW_DIR = get_raw_data_dir()
except ImportError:
    _DEFAULT_CACHE_DIR = str(Path(__file__).resolve().parents[1] / "cache")
    _DEFAULT_RESULTS_DIR = str(Path(__file__).resolve().parents[1] / "results")
    _DEFAULT_RAW_DIR = Path(r"E:\DATA")

try:
    from config import modelconfig, _resolve_device
except ImportError:
    modelconfig = {}
    def _resolve_device():
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@dataclass
class S2SConfig:
    # Data paths (automatically derived from unified workspace or overridden via .env)
    project_root: Path = Path(__file__).resolve().parents[1]
    station_data_dir: str = field(
        default_factory=lambda: os.getenv("STATION_DATA_PATH", str(_DEFAULT_RAW_DIR / "原始站点资料（有华南）"))
    )
    model_data_dir: str = field(
        default_factory=lambda: os.getenv("MODEL_DATA_PATH", str(_DEFAULT_RAW_DIR / "model_data"))
    )
    cache_dir: str = field(default_factory=lambda: _DEFAULT_CACHE_DIR)
    results_dir: str = field(default_factory=lambda: _DEFAULT_RESULTS_DIR)
    
    # Selected dynamical models to use
    dynamic_models: List[str] = field(default_factory=lambda: [
        "MODESv21_ecmwf_seas51",
        "MODESv21_ncep_cfs2",
        "BCC-CPSV3",
        "UKMO_GLOSEA5"
    ])
    
    # Station and graph settings
    num_stations: int = 2371
    knn_k: int = 12
    geo_sigma: float = 200.0  # km
    dem_sigma: float = 500.0  # meters
    adaptive_dim: int = 16
    
    # Temporal task settings
    # Paradigm 1: Pentad (5-day) S2S forecast
    pentad_in_len: int = 6    # Past 6 pentads (30 days)
    pentad_out_len: int = 6   # Future 6 pentads (30 days)
    
    # Paradigm 2: Daily Hybrid S2S forecast
    daily_in_len: int = 30    # Past 30 days
    daily_out_len: int = 30   # Future 30 days
    
    # Date splitting
    train_years: tuple = (1994, 2018)
    val_years: tuple = (2019, 2021)
    test_years: tuple = (2022, 2024)
    
    # Model architecture parameters
    node_in_dim: int = 4      # Dynamic precip, log_precip, rolling_7d, cdd
    node_static_dim: int = 4  # Lat, Lon, Elevation, Dist_to_coast
    macro_in_dim: int = 16    # Dimension of extracted dynamical model features per station
    hidden_dim: int = 64
    tcn_layers: int = 4
    gcn_order: int = 2
    dropout: float = 0.2
    
    # Training parameters
    device: torch.device = field(default_factory=_resolve_device)
    batch_size: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 100
    early_stop_patience: int = 15
    grad_clip: float = 5.0
    
    # Quantiles for Hurdle & Quantile regression
    quantiles: List[float] = field(default_factory=lambda: [0.50, 0.90, 0.95])
    
    def __post_init__(self):
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Sync with global HydroSynth modelconfig if present
        if "device" in modelconfig and isinstance(modelconfig["device"], torch.device):
            self.device = modelconfig["device"]
        if "batch_size" in modelconfig and isinstance(modelconfig["batch_size"], int):
            self.batch_size = modelconfig["batch_size"]
        if "lr" in modelconfig and isinstance(modelconfig["lr"], (int, float)):
            self.learning_rate = float(modelconfig["lr"])


def get_s2s_config(**kwargs) -> S2SConfig:
    """Factory function to get or override S2S configuration."""
    return S2SConfig(**kwargs)

import torch
import numpy as np
import os
import sys

# Add repo root to sys.path
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from Seas2Rain.train import prepare_data

try:
    data = prepare_data()
    print("Data loaded successfully.")
    
    seas_anom = data["seas_anom"]
    obs_target = data["obs_target"]
    
    print(f"seas_anom: shape={seas_anom.shape}, min={seas_anom.min():.2f}, max={seas_anom.max():.2f}, mean={seas_anom.mean():.2f}, std={seas_anom.std():.2f}")
    print(f"obs_target: shape={obs_target.shape}, min={obs_target.min():.2f}, max={obs_target.max():.2f}, mean={obs_target.mean():.2f}, std={obs_target.std():.2f}")
    
    # Check for NaNs
    print(f"seas_anom NaNs: {np.isnan(seas_anom).sum()}")
    print(f"obs_target NaNs: {np.isnan(obs_target).sum()}")
    
    # Percentiles
    print(f"seas_anom 1st/99th percentile: {np.percentile(seas_anom, [1, 99])}")
    print(f"obs_target 1st/99th percentile: {np.percentile(obs_target, [1, 99])}")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

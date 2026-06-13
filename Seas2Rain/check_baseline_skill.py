import os
import sys
import numpy as np
import torch
import pandas as pd

# Ensure repo parent is on sys.path
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_repo_root = os.path.normpath(_repo_root)
_repo_parent = os.path.dirname(_repo_root)
if _repo_parent not in sys.path:
    sys.path.insert(0, _repo_parent)

from HydroSynth.Seas2Rain.train import prepare_data, LEADS

def calculate_baseline_acc():
    data = prepare_data()
    ec_base = data["ec_base"] # [T, LEADS, 1, H, W]
    obs_target = data["obs_target"] # [T, LEADS, 1, H, W]
    obs_mask = data["obs_mask"] # [T, LEADS, 1, H, W]
    split_indices = data["split_indices"]
    
    val_idx = split_indices["val"]
    
    ec_base_val = ec_base[val_idx, :, 0]
    obs_target_val = obs_target[val_idx, :, 0]
    obs_mask_val = obs_mask[val_idx, :, 0]
    
    def get_acc(pred, target, mask):
        m = mask.astype(np.float32)
        count = m.sum(axis=(1, 2), keepdims=True)
        count = np.maximum(count, 1.0)
        
        pm = (pred * m).sum(axis=(1, 2), keepdims=True) / count
        tm = (target * m).sum(axis=(1, 2), keepdims=True) / count
        
        pa = (pred - pm) * m
        ta = (target - tm) * m
        
        num = (pa * ta).sum(axis=(1, 2))
        den = np.sqrt((pa**2).sum(axis=(1, 2)) * (ta**2).sum(axis=(1, 2)) + 1e-12)
        
        acc = num / den
        valid_mask = (m.sum(axis=(1, 2)) > 0)
        if valid_mask.sum() == 0:
            return 0.0
        return acc[valid_mask].mean()

    results = []
    for l in range(LEADS):
        acc = get_acc(ec_base_val[:, l], obs_target_val[:, l], obs_mask_val[:, l])
        results.append(acc)
    
    print(f"Baseline (ECMWF SEAS5) VAL ACC: {results}")

if __name__ == "__main__":
    calculate_baseline_acc()

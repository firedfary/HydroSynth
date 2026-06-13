import os
import sys
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt

# Ensure repo parent is on sys.path
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_repo_root = os.path.normpath(_repo_root)
_repo_parent = os.path.dirname(_repo_root)
if _repo_parent not in sys.path:
    sys.path.insert(0, _repo_parent)

from HydroSynth.Seas2Rain.train import prepare_data, LEADS, COND_VARS

def diagnostic_analysis():
    data = prepare_data()
    cond = data["cond"] # [T, LEADS, C, H, W]
    ec_base = data["ec_base"][:, :, 0] # [T, LEADS, H, W]
    obs_target = data["obs_target"][:, :, 0] # [T, LEADS, H, W]
    obs_mask = data["obs_mask"][:, :, 0] # [T, LEADS, H, W]
    split_indices = data["split_indices"]
    
    train_idx = split_indices["train"]
    
    error = obs_target[train_idx] - ec_base[train_idx]
    mask = obs_mask[train_idx]
    
    print(f"Analyzing {len(train_idx)} training samples...")
    
    for c_idx, var_name in enumerate(COND_VARS):
        c_val = cond[train_idx, :, c_idx]
        
        # Calculate correlation between var and error at each lead
        for l in range(LEADS):
            v = c_val[:, l]
            e = error[:, l]
            m = mask[:, l]
            
            # Global mean correlation (spatial mean of both)
            v_mean = (v * m).sum(axis=(1, 2)) / m.sum(axis=(1, 2)).clip(1.0)
            e_mean = (e * m).sum(axis=(1, 2)) / m.sum(axis=(1, 2)).clip(1.0)
            
            corr = np.corrcoef(v_mean, e_mean)[0, 1]
            print(f"Lead {l} Var {var_name:5} Corr with Error: {corr: .4f}")

if __name__ == "__main__":
    diagnostic_analysis()

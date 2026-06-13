import os
import sys
import numpy as np
import pandas as pd
import torch
import tqdm
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from typing import Dict, List, Tuple

# Ensure repo root is on sys.path
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from HydroSynth import config
# Reuse the high-quality data preparation from the previous pipeline
from Seas2Rain.train import prepare_data, init_metric_state, update_metrics, finalize_metrics

def run_knn_experiment(k: int = 10, n_pcs: int = 20, weight_mode: str = "distance"):
    print(f"Starting SeasAnalog (kNN) Experiment: k={k}, n_pcs={n_pcs}...")
    
    # 1. Prepare Data
    data = prepare_data()
    splits = data["split_indices"]
    
    # Extract training (history library) and validation (target)
    train_idx = splits["train"]
    val_idx = splits["val"]
    
    # [T, L, C, H, W] -> T samples
    cond = data["cond"] 
    obs_target = data["obs_target"]
    obs_mask = data["obs_mask"]
    ec_base = data["ec_base"]
    init_month = data["init_month"]
    
    T_total, LEADS, C, H, W = cond.shape
    
    # 2. Feature Extraction (PCA/EOF)
    # We flatten spatial dims for each lead and each variable to find a global analog
    # To keep it simple but physical, we'll extract PCA features from cond variables
    # We do this for each lead independently as error modes change
    
    final_preds = np.zeros((len(val_idx), LEADS, H, W))
    
    for lead in range(LEADS):
        print(f"Processing Lead {lead}...")
        
        # Prepare feature matrix for this lead: [Time, C * H * W]
        # Only use atmospheric variables (first 5 in COND_VARS) for matching
        # Or all 6 if we want SST impact
        train_features = cond[train_idx, lead].reshape(len(train_idx), -1)
        val_features = cond[val_idx, lead].reshape(len(val_idx), -1)
        
        # Standardize features
        mean = train_features.mean(axis=0)
        std = train_features.std(axis=0) + 1e-6
        train_features = (train_features - mean) / std
        val_features = (val_features - mean) / std
        
        # PCA for dimensionality reduction
        pca = PCA(n_components=min(n_pcs, len(train_idx)))
        train_pca = pca.fit_transform(train_features)
        val_pca = pca.transform(val_features)
        
        # 3. kNN Search with Monthly Constraint
        for i, v_idx in enumerate(val_idx):
            target_month = init_month[v_idx]
            
            # Constraint: search only in same month +/- 1 month in history
            # init_month ranges from 1 to 12
            month_diff = np.abs(init_month[train_idx] - target_month)
            # Handle circularity of months
            month_diff = np.minimum(month_diff, 12 - month_diff)
            
            # Allow +/- 1 month window
            eligible_train_mask = (month_diff <= 1)
            eligible_indices = np.where(eligible_train_mask)[0]
            
            if len(eligible_indices) < k:
                # Fallback to all training if window too narrow
                eligible_indices = np.arange(len(train_idx))
            
            # Fit kNN on eligible historical samples
            knn = NearestNeighbors(n_neighbors=min(k, len(eligible_indices)), metric='cosine')
            knn.fit(train_pca[eligible_indices])
            
            distances, neighbors = knn.kneighbors(val_pca[i].reshape(1, -1))
            
            # 4. Aggregation
            # Get historical observed precipitation anomalies for these neighbors
            # neighbors returns indices relative to eligible_indices
            real_train_indices = train_idx[eligible_indices[neighbors[0]]]
            
            hist_obs = obs_target[real_train_indices, lead, 0] # [k, H, W]
            
            if weight_mode == "distance":
                # Inverse distance weighting
                weights = 1.0 / (distances[0] + 1e-6)
                weights /= weights.sum()
                pred = np.sum(hist_obs * weights[:, None, None], axis=0)
            else:
                # Simple average
                pred = hist_obs.mean(axis=0)
            
            final_preds[i, lead] = pred

    # 5. Evaluation
    device = torch.device("cpu")
    val_state = init_metric_state(LEADS, device=device)
    base_state = init_metric_state(LEADS, device=device)
    
    # Prepare tensors for eval
    pred_t = torch.from_numpy(final_preds).float()
    target_t = torch.from_numpy(obs_target[val_idx, :, 0]).float()
    mask_t = torch.from_numpy(obs_mask[val_idx, :, 0]).float()
    base_t = torch.from_numpy(ec_base[val_idx, :, 0]).float()
    
    update_metrics(val_state, pred_t, target_t, mask_t)
    update_metrics(base_state, base_t, target_t, mask_t)
    
    res = finalize_metrics(val_state)
    base_res = finalize_metrics(base_state)
    
    print("\n" + "="*50)
    print("FINAL RESULTS: SeasAnalog (kNN)")
    print(f"Lead | Baseline ACC | SeasAnalog ACC | Improvement")
    print("-" * 50)
    for l in range(LEADS):
        imp = res['acc'][l] - base_res['acc'][l]
        print(f" {l}   | {base_res['acc'][l]:.4f}      | {res['acc'][l]:.4f}       | {imp:+.4f} {'***' if imp > 0.1 else ''}")
    print("="*50)

if __name__ == "__main__":
    # Try different hyperparams
    # Small k helps retain variance, larger n_pcs captures more detailed modes
    run_knn_experiment(k=15, n_pcs=30, weight_mode="distance")

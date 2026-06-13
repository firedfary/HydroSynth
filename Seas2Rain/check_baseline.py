import torch
import numpy as np
import os
import sys

# Add repo root to sys.path
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from HydroSynth import config
from Seas2Rain.train import prepare_data, build_batch_store, evaluate_baseline, format_metric_line

def check():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    data = prepare_data()
    batch_store = build_batch_store(data, train_device=device)
    
    baseline_val = evaluate_baseline(batch_store, split="val")
    baseline_test = evaluate_baseline(batch_store, split="test")
    
    print(format_metric_line("Baseline VAL RMSE", baseline_val["rmse"]))
    print(format_metric_line("Baseline TEST RMSE", baseline_test["rmse"]))
    print(format_metric_line("Baseline VAL ACC", baseline_val["acc"]))
    print(format_metric_line("Baseline TEST ACC", baseline_test["acc"]))

if __name__ == "__main__":
    check()

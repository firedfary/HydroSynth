import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent
HYDROSYNTH_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(HYDROSYNTH_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from configs.s2s_config import get_s2s_config
from data_engine.station_parser import StationParser
from data_engine.graph_topology import GraphTopologyBuilder
from data_engine.model_aligner import DynamicalModelAligner
from data_engine.dataset_s2s import create_s2s_dataloaders
from models.paradigm1_pentad_s2s import PentadS2S_GNN
from models.paradigm2_hybrid_daily import HybridDailySTGNN
from models.paradigm3_hurdle_extreme import HurdleExtremeSTGNN
from models.paradigm4_graph_disagg import GraphDisaggregationNet
from engine.trainer import S2STrainer
from engine.metrics import S2SMetrics


def run_experiment_pipeline(
    paradigm: str = "pentad",
    epochs: int = 50,
    batch_size: int = 16,
    lr: float = 1e-3,
    fast_dev_run: bool = False
):
    print("=" * 70)
    print(f"HydroGraph_S2S Experiment Runner | Paradigm: {paradigm.upper()}")
    print("=" * 70)
    
    config = get_s2s_config()
    if fast_dev_run:
        epochs = 2
        print("[FAST DEV RUN] Epochs reduced to 2 for quick verification.")

    # 1. Parse and load station data
    parser = StationParser(data_dir=config.station_data_dir, cache_dir=config.cache_dir)
    parser.parse_and_cache()
    num_nodes = len(parser.station_ids)
    
    # 2. Build multi-graph topologies
    geo_builder = GraphTopologyBuilder(
        coords=parser.coords,
        elevations=parser.elevations,
        knn_k=config.knn_k,
        cache_dir=config.cache_dir
    )
    topologies = geo_builder.get_all_topologies(historical_precip=parser.daily_precip[:365])
    
    supports = [
        torch.tensor(topologies["pf_geo"], dtype=torch.float32),
        torch.tensor(topologies["pb_geo"], dtype=torch.float32)
    ]

    # 3. Model Align & Feature Extraction
    aligner = DynamicalModelAligner(coords=parser.coords, model_dir=config.model_data_dir, cache_dir=config.cache_dir)
    model_feat_dict = aligner.build_or_load_station_features(target_dates=parser.dates)
    
    # 4. Prepare datasets & DataLoaders
    data_dict = {
        "daily_precip": parser.daily_precip,
        "daily_dates": parser.dates,
        "monthly_dates": pd.to_datetime(model_feat_dict["dates_monthly"]),
        "macro_features": model_feat_dict["macro_features"],
        "daily_trend": model_feat_dict["daily_trend"]
    }
    
    if paradigm == "pentad":
        pentad_mat, pentad_dates = parser.get_pentad_precip()
        data_dict["pentad_precip"] = pentad_mat
        data_dict["pentad_dates"] = pentad_dates
        dataset_type = "pentad"
    elif paradigm == "daily_hybrid":
        data_dict["daily_features"] = parser.compute_dynamic_features(parser.daily_precip)
        dataset_type = "daily_hybrid"
    elif paradigm == "hurdle":
        data_dict["daily_features"] = parser.compute_dynamic_features(parser.daily_precip)
        dataset_type = "hurdle"
    elif paradigm == "disagg":
        data_dict["daily_features"] = parser.compute_dynamic_features(parser.daily_precip)
        dataset_type = "daily_hybrid"  # Reuse daily hybrid format for disagg targets
    else:
        raise ValueError(f"Unknown paradigm: {paradigm}")

    train_loader, val_loader, test_loader = create_s2s_dataloaders(
        dataset_type=dataset_type,
        data_dict=data_dict,
        batch_size=batch_size,
        train_years=config.train_years,
        val_years=config.val_years,
        test_years=config.test_years,
        num_workers=0
    )

    # 5. Initialize Model
    device = config.device
    print(f"Initializing model for paradigm '{paradigm}' on device {device}...")
    
    if paradigm == "pentad":
        model = PentadS2S_GNN(
            num_nodes=num_nodes,
            in_dim=2,
            hidden_dim=config.hidden_dim,
            macro_dim=5,
            in_len=config.pentad_in_len,
            out_len=config.pentad_out_len,
            num_layers=config.tcn_layers,
            dropout=config.dropout
        )
    elif paradigm == "daily_hybrid":
        model = HybridDailySTGNN(
            num_nodes=num_nodes,
            in_dim=config.node_in_dim,
            hidden_dim=config.hidden_dim,
            macro_dim=5,
            in_len=config.daily_in_len,
            out_len=config.daily_out_len,
            num_layers=config.tcn_layers,
            dropout=config.dropout
        )
    elif paradigm == "hurdle":
        model = HurdleExtremeSTGNN(
            num_nodes=num_nodes,
            in_dim=config.node_in_dim,
            hidden_dim=config.hidden_dim,
            macro_dim=5,
            in_len=config.daily_in_len,
            out_len=config.daily_out_len,
            quantiles=config.quantiles,
            num_layers=config.tcn_layers,
            dropout=config.dropout
        )
    elif paradigm == "disagg":
        model = GraphDisaggregationNet(
            num_nodes=num_nodes,
            in_dim=config.node_in_dim,
            hidden_dim=config.hidden_dim,
            macro_dim=5,
            in_len=config.daily_in_len,
            out_len=config.daily_out_len,
            num_layers=config.tcn_layers,
            dropout=config.dropout
        )

    # 6. Train and Evaluate
    exp_dir = os.path.join(config.results_dir, f"exp_{paradigm}")
    os.makedirs(exp_dir, exist_ok=True)

    trainer = S2STrainer(
        model=model,
        paradigm=paradigm,
        supports=supports,
        device=device,
        learning_rate=lr,
        weight_decay=config.weight_decay,
        grad_clip=config.grad_clip,
        early_stop_patience=config.early_stop_patience,
        save_dir=exp_dir
    )

    max_batches = 30 if fast_dev_run else None
    history = trainer.fit(train_loader, val_loader, epochs=epochs, max_batches=max_batches)
    
    print(f"\nEvaluating '{paradigm}' on unseen Test Set ({config.test_years[0]} - {config.test_years[1]})...")
    test_loss, test_metrics = trainer.evaluate(test_loader, max_batches=max_batches)
    
    print("\n" + "=" * 50)
    print(f"TEST EVALUATION RESULTS [{paradigm.upper()}]:")
    print("=" * 50)
    for k, v in test_metrics.items():
        print(f"  {k:20s}: {v:.4f}")
        
    # Save results to JSON
    result_record = {
        "paradigm": paradigm,
        "test_loss": float(test_loss),
        "test_metrics": test_metrics,
        "train_years": config.train_years,
        "test_years": config.test_years,
        "num_stations": num_nodes,
        "epochs": epochs
    }
    
    out_json = os.path.join(exp_dir, f"evaluation_{paradigm}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result_record, f, indent=4, ensure_ascii=False)
    print(f"Results saved to {out_json}")
    
    return result_record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run HydroGraph_S2S Experiments")
    parser.add_argument(
        "--paradigm",
        type=str,
        default="pentad",
        choices=["pentad", "daily_hybrid", "hurdle", "disagg", "all"],
        help="Model paradigm to train and evaluate"
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--fast_dev_run", action="store_true", help="Fast smoke test run")
    
    args = parser.parse_args()
    
    if args.paradigm == "all":
        results = {}
        for p in ["pentad", "daily_hybrid", "hurdle", "disagg"]:
            results[p] = run_experiment_pipeline(
                paradigm=p,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                fast_dev_run=args.fast_dev_run
            )
        print("\nAll paradigms executed successfully!")
    else:
        run_experiment_pipeline(
            paradigm=args.paradigm,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            fast_dev_run=args.fast_dev_run
        )

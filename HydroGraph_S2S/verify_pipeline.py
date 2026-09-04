import os
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent
HYDROSYNTH_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(HYDROSYNTH_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from configs.s2s_config import get_s2s_config
from data_engine.station_parser import StationParser
from data_engine.graph_topology import GraphTopologyBuilder
from data_engine.model_aligner import DynamicalModelAligner
from data_engine.dataset_s2s import PentadDataset, DailyHybridDataset, HurdleDataset, create_s2s_dataloaders
from models.paradigm1_pentad_s2s import PentadS2S_GNN
from models.paradigm2_hybrid_daily import HybridDailySTGNN
from models.paradigm3_hurdle_extreme import HurdleExtremeSTGNN
from models.paradigm4_graph_disagg import GraphDisaggregationNet
from engine.losses import ExtremeWeightedLoss, HurdleLoss, QuantileLoss
from engine.metrics import S2SMetrics
from engine.trainer import S2STrainer


def test_station_and_topology():
    print("=" * 60)
    print("1. Testing Station Parser and Graph Topology Builder...")
    config = get_s2s_config()
    
    # Initialize parser
    parser = StationParser(data_dir=config.station_data_dir, cache_dir=config.cache_dir)
    parser.parse_and_cache()
    
    num_nodes = len(parser.station_ids)
    print(f"-> Parsed stations: {num_nodes}")
    assert num_nodes >= 2000, f"Expected ~2371 stations, got {num_nodes}"
    assert parser.coords.shape == (num_nodes, 2)
    assert parser.elevations.shape == (num_nodes,)
    assert parser.daily_precip.shape[1] == num_nodes
    
    # Test Pentad aggregation
    pentad_precip, pentad_dates = parser.get_pentad_precip()
    print(f"-> Pentads: {len(pentad_dates)}, shape: {pentad_precip.shape}")
    assert pentad_precip.shape[1] == num_nodes
    
    # Test Graph Topologies
    geo_builder = GraphTopologyBuilder(
        coords=parser.coords,
        elevations=parser.elevations,
        knn_k=config.knn_k,
        cache_dir=config.cache_dir
    )
    topologies = geo_builder.get_all_topologies(historical_precip=parser.daily_precip[:365])
    print(f"-> Topologies built: {list(topologies.keys())}")
    assert "adj_geo" in topologies and "pf_geo" in topologies and "pb_geo" in topologies
    print("[PASS] Station & Topology test passed!")
    return parser, topologies, config


def test_models_forward_backward(parser, topologies, config):
    print("=" * 60)
    print("2. Testing 4 Paradigms Forward and Backward Propagation...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_nodes = len(parser.station_ids)
    supports = [
        torch.tensor(topologies["pf_geo"], dtype=torch.float32).to(device),
        torch.tensor(topologies["pb_geo"], dtype=torch.float32).to(device)
    ]
    
    B = 2
    macro_z = torch.randn(B, num_nodes, 5, device=device)
    
    # --- Paradigm 1: Pentad S2S ---
    print("-> Testing Paradigm 1 (Pentad S2S GNN)...")
    m1 = PentadS2S_GNN(num_nodes=num_nodes, in_dim=2, hidden_dim=32, macro_dim=5, in_len=6, out_len=6, num_layers=2).to(device)
    x1 = torch.randn(B, 6, num_nodes, 2, device=device)
    y1_target = torch.rand(B, 6, num_nodes, device=device)
    y1_pred, adp1 = m1(x1, macro_z, supports)
    loss1 = ExtremeWeightedLoss()(y1_pred, y1_target)
    loss1.backward()
    assert y1_pred.shape == (B, 6, num_nodes)
    print("   [PASS] Paradigm 1 forward/backward passed!")
    
    # --- Paradigm 2: Daily Hybrid ST-GNN ---
    print("-> Testing Paradigm 2 (Daily Hybrid ST-GNN with Trend + Residual)...")
    m2 = HybridDailySTGNN(num_nodes=num_nodes, in_dim=4, hidden_dim=32, macro_dim=5, in_len=30, out_len=30, num_layers=2).to(device)
    x2 = torch.randn(B, 30, num_nodes, 4, device=device)
    trend2 = torch.rand(B, 30, num_nodes, device=device)
    y2_target = torch.rand(B, 30, num_nodes, device=device)
    y2_pred, delta2, adp2 = m2(x2, trend2, macro_z, supports)
    loss2 = ExtremeWeightedLoss()(y2_pred, y2_target)
    loss2.backward()
    assert y2_pred.shape == (B, 30, num_nodes)
    print("   [PASS] Paradigm 2 forward/backward passed!")
    
    # --- Paradigm 3: Hurdle Extreme ST-GNN ---
    print("-> Testing Paradigm 3 (Two-Stage Hurdle Extreme ST-GNN)...")
    m3 = HurdleExtremeSTGNN(num_nodes=num_nodes, in_dim=4, hidden_dim=32, macro_dim=5, in_len=30, out_len=30, num_layers=2).to(device)
    x3 = torch.randn(B, 30, num_nodes, 4, device=device)
    y3_amt = torch.rand(B, 30, num_nodes, device=device) * 10.0
    y3_occ = (y3_amt >= 0.1).float()
    occ_logits, quantiles_pred, adp3 = m3(x3, macro_z, supports)
    loss3 = HurdleLoss()(occ_logits, quantiles_pred, y3_occ, y3_amt)
    loss3.backward()
    assert occ_logits.shape == (B, 30, num_nodes)
    assert quantiles_pred.shape == (B, 30, num_nodes, 3)
    print("   [PASS] Paradigm 3 forward/backward passed!")
    
    # --- Paradigm 4: Graph Disaggregation Net ---
    print("-> Testing Paradigm 4 (Graph Disaggregation / Downscaling Net)...")
    m4 = GraphDisaggregationNet(num_nodes=num_nodes, in_dim=4, hidden_dim=32, macro_dim=5, in_len=30, out_len=30, num_layers=2).to(device)
    x4 = torch.randn(B, 30, num_nodes, 4, device=device)
    m_total = torch.rand(B, num_nodes, device=device) * 100.0
    y4_target = torch.rand(B, 30, num_nodes, device=device)
    daily_pred, weights, adp4 = m4(x4, m_total, macro_z, supports)
    loss4 = ExtremeWeightedLoss()(daily_pred, y4_target)
    loss4.backward()
    assert daily_pred.shape == (B, 30, num_nodes)
    print("   [PASS] Paradigm 4 forward/backward passed!")


def test_metrics():
    print("=" * 60)
    print("3. Testing S2S Evaluation Metrics (ACC, RMSE, KGE, TS, ETS, POD, FAR)...")
    y_true = np.random.exponential(scale=5.0, size=(10, 6, 2371)).astype(np.float32)
    y_pred = y_true + np.random.normal(scale=2.0, size=y_true.shape).astype(np.float32)
    y_pred = np.maximum(y_pred, 0.0)
    
    metrics = S2SMetrics.compute_all(y_pred, y_true)
    print("Computed Metrics Sample:")
    for k, v in list(metrics.items())[:8]:
        print(f"   {k}: {v:.4f}")
    assert "ACC" in metrics and "RMSE" in metrics and "KGE" in metrics and "TS_Light_0.1mm" in metrics
    print("[PASS] Metrics calculation passed!")


if __name__ == "__main__":
    print("Starting HydroGraph_S2S Pipeline Verification...\n")
    parser, topologies, config = test_station_and_topology()
    test_models_forward_backward(parser, topologies, config)
    test_metrics()
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)

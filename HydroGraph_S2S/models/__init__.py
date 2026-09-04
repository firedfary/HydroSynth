# HydroGraph_S2S models package
from .base_stgnn import BaseSTGNN, DiffusionGraphConv, TemporalGatedConv, FiLMLayer
from .paradigm1_pentad_s2s import PentadS2S_GNN
from .paradigm2_hybrid_daily import HybridDailySTGNN
from .paradigm3_hurdle_extreme import HurdleExtremeSTGNN
from .paradigm4_graph_disagg import GraphDisaggregationNet

__all__ = [
    "BaseSTGNN",
    "DiffusionGraphConv",
    "TemporalGatedConv",
    "FiLMLayer",
    "PentadS2S_GNN",
    "HybridDailySTGNN",
    "HurdleExtremeSTGNN",
    "GraphDisaggregationNet"
]

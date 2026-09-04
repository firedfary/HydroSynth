import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
from .base_stgnn import BaseSTGNN, DiffusionGraphConv


class GraphDisaggregationNet(nn.Module):
    """
    Paradigm 4: Spatio-Temporal Graph Disaggregation / Downscaling Network.
    Disaggregates monthly total dynamic model forecast into daily station sequences
    while preserving temporal weather patterns and mass conservation:
    P_daily(t, i) = Disagg_Weights(t, i) * P_monthly_total(i)
    """
    def __init__(
        self,
        num_nodes: int = 2371,
        in_dim: int = 4,
        hidden_dim: int = 64,
        macro_dim: int = 5,
        in_len: int = 30,
        out_len: int = 30,
        num_layers: int = 3,
        dropout: float = 0.2
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.in_len = in_len
        self.out_len = out_len
        
        self.backbone = BaseSTGNN(
            num_nodes=num_nodes,
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            macro_dim=macro_dim,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # Generator for temporal disaggregation weights across out_len days
        self.temporal_disagg_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, out_len)
        )
        
        # Refinement graph convolution
        self.refine_gcn = DiffusionGraphConv(hidden_dim, hidden_dim, num_supports=2)

    def forward(
        self,
        x: torch.Tensor,
        monthly_total: torch.Tensor,
        macro_z: torch.Tensor,
        supports: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x: (B, in_len, N, in_dim) historical daily dynamics
        monthly_total: (B, N) monthly total precipitation forecast from dynamic model
        macro_z: (B, N, macro_dim)
        supports: list of graph matrices
        Returns:
            daily_pred: (B, out_len, N) disaggregated daily precipitation
            weights: (B, out_len, N) temporal disaggregation distribution
            adp: (N, N) learned adaptive graph
        """
        B, T, N, D = x.shape
        
        # Extract spatiotemporal features: (B, in_len, N, hidden_dim)
        h, adp = self.backbone.extract_features(x, supports, macro_z)
        
        # Temporal pooling: pool over history to get station weather state: (B, N, hidden_dim)
        h_pool = torch.mean(h, dim=1)
        
        # Generate temporal distribution logits over out_len: (B, N, out_len)
        weights_logits = self.temporal_disagg_head(h_pool)
        weights = F.softmax(weights_logits, dim=-1)  # (B, N, out_len), sum_t = 1.0
        
        # Permute to (B, out_len, N)
        weights_daily = weights.permute(0, 2, 1)
        
        # Disaggregate monthly total: (B, out_len, N)
        monthly_expanded = monthly_total.unsqueeze(1)  # (B, 1, N)
        daily_pred = weights_daily * monthly_expanded
        
        return daily_pred, weights_daily, adp

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
from .base_stgnn import BaseSTGNN


class HybridDailySTGNN(nn.Module):
    """
    Paradigm 2: Multi-Scale Hybrid Daily Spatio-Temporal Graph Neural Network.
    Combines monthly dynamical model base trend (via cubic spline) with daily ST-GNN residual perturbations:
    y_pred = ReLU(P_trend + Delta_P)
    """
    def __init__(
        self,
        num_nodes: int = 2371,
        in_dim: int = 4,           # [p, log_p, roll7, cdd]
        hidden_dim: int = 64,
        macro_dim: int = 5,
        in_len: int = 30,          # Past 30 days
        out_len: int = 30,         # Future 30 days
        num_layers: int = 4,
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
        
        # Temporal projection from in_len (30d) to out_len (30d)
        self.temporal_proj = nn.Sequential(
            nn.Linear(in_len, out_len),
            nn.LeakyReLU(0.2)
        )
        
        # Residual perturbation head
        self.residual_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(
        self,
        x: torch.Tensor,
        trend: torch.Tensor,
        macro_z: torch.Tensor,
        supports: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x: (B, in_len, N, in_dim)
        trend: (B, out_len, N) base spline trend from dynamical models
        macro_z: (B, N, macro_dim)
        supports: list of graph matrices
        Returns:
            y_pred: (B, out_len, N) total predicted daily precipitation (mm)
            delta_p: (B, out_len, N) predicted residual perturbation
            adp: (N, N) learned adaptive graph
        """
        B, T, N, D = x.shape
        
        # Extract features: (B, in_len, N, hidden_dim)
        h, adp = self.backbone.extract_features(x, supports, macro_z)
        
        # Project temporal dimension across in_len -> out_len:
        # (B, hidden_dim, N, in_len) -> (B, hidden_dim, N, out_len)
        h_perm = h.permute(0, 3, 2, 1)
        h_future = self.temporal_proj(h_perm).permute(0, 3, 2, 1)  # (B, out_len, N, hidden_dim)
        
        # Compute residual perturbation
        delta_p = self.residual_head(h_future).squeeze(-1)  # (B, out_len, N)
        
        # Additive hybrid fusion: P_trend + Delta_P
        y_pred = F.relu(trend + delta_p)
        
        return y_pred, delta_p, adp

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
from .base_stgnn import BaseSTGNN


class PentadS2S_GNN(nn.Module):
    """
    Paradigm 1: Subseasonal Pentad (5-Day) Spatio-Temporal Graph Neural Network.
    Forecasts the next 6 pentads (Pentad 1 to Pentad 6, covering 30 days) across 2371 stations.
    """
    def __init__(
        self,
        num_nodes: int = 2371,
        in_dim: int = 2,           # [pentad_precip, log1p_precip]
        hidden_dim: int = 64,
        macro_dim: int = 5,        # Multi-model channels
        in_len: int = 6,           # Past 6 pentads
        out_len: int = 6,          # Future 6 pentads
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
        
        # Temporal projection from in_len steps to out_len steps
        self.temporal_proj = nn.Linear(in_len * hidden_dim, out_len * hidden_dim)
        
        # Readout head
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(
        self,
        x: torch.Tensor,
        macro_z: torch.Tensor,
        supports: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B, in_len, N, in_dim)
        macro_z: (B, N, macro_dim)
        supports: list of graph matrices
        Returns:
            y_pred: (B, out_len, N) predicted pentad cumulative precipitation (mm)
            adp: (N, N) learned adaptive graph
        """
        B, T, N, D = x.shape
        
        # Extract spatio-temporal features: (B, in_len, N, hidden_dim)
        h, adp = self.backbone.extract_features(x, supports, macro_z)
        
        # Flatten time and hidden: (B, N, in_len * hidden_dim)
        h_flat = h.permute(0, 2, 1, 3).reshape(B, N, -1)
        
        # Project to future time steps: (B, N, out_len, hidden_dim)
        h_future = self.temporal_proj(h_flat).view(B, N, self.out_len, -1)
        
        # Permute to (B, out_len, N, hidden_dim)
        h_future = h_future.permute(0, 2, 1, 3)
        
        # Readout to precipitation amounts (mm)
        out = self.head(h_future).squeeze(-1)  # (B, out_len, N)
        y_pred = F.relu(out)  # Precipitation >= 0
        
        return y_pred, adp

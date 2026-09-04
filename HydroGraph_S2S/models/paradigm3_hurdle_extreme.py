import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
from .base_stgnn import BaseSTGNN


class HurdleExtremeSTGNN(nn.Module):
    """
    Paradigm 3: Two-Stage Hurdle and Multi-Quantile Spatio-Temporal Graph Neural Network
    for Extreme Precipitation Forecasting.
    - Stage 1: Binary Classification for Precipitation Occurrence (Probability of Precipitation)
    - Stage 2: Multi-Quantile Regression (e.g., 50%, 90%, 95%) for Intensity under heavy rain
    """
    def __init__(
        self,
        num_nodes: int = 2371,
        in_dim: int = 4,
        hidden_dim: int = 64,
        macro_dim: int = 5,
        in_len: int = 30,
        out_len: int = 30,
        quantiles: List[float] = [0.50, 0.90, 0.95],
        num_layers: int = 4,
        dropout: float = 0.2
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.in_len = in_len
        self.out_len = out_len
        self.quantiles = quantiles
        self.num_quantiles = len(quantiles)
        
        self.backbone = BaseSTGNN(
            num_nodes=num_nodes,
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            macro_dim=macro_dim,
            num_layers=num_layers,
            dropout=dropout
        )
        
        self.temporal_proj = nn.Sequential(
            nn.Linear(in_len, out_len),
            nn.LeakyReLU(0.2)
        )
        
        # Stage 1: Occurrence Classification Head (Logits)
        self.occ_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Stage 2: Multi-Quantile Regression Head
        self.quantile_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, self.num_quantiles)
        )

    def forward(
        self,
        x: torch.Tensor,
        macro_z: torch.Tensor,
        supports: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x: (B, in_len, N, in_dim)
        macro_z: (B, N, macro_dim)
        supports: list of graph matrices
        Returns:
            occ_logits: (B, out_len, N) logits for rain occurrence
            quantiles_pred: (B, out_len, N, num_quantiles) predicted precipitation amounts per quantile
            adp: (N, N) learned adaptive graph
        """
        B, T, N, D = x.shape
        
        # Feature extraction
        h, adp = self.backbone.extract_features(x, supports, macro_z)
        
        # Temporal projection to out_len: (B, out_len, N, hidden_dim)
        h_perm = h.permute(0, 3, 2, 1)
        h_future = self.temporal_proj(h_perm).permute(0, 3, 2, 1)
        
        # Stage 1: Occurrence logits
        occ_logits = self.occ_head(h_future).squeeze(-1)  # (B, out_len, N)
        
        # Stage 2: Quantiles (Non-negative precipitation)
        quantiles_raw = self.quantile_head(h_future)  # (B, out_len, N, num_quantiles)
        quantiles_pred = F.relu(quantiles_raw)
        
        return occ_logits, quantiles_pred, adp

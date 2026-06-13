from __future__ import annotations
import torch
import torch.nn as nn
from typing import Optional

class Seas2RainModel(nn.Module):
    def __init__(
        self,
        cond_channels: int = 6,
        hidden_dim: int = 128,
        dropout: float = 0.5,
        **kwargs
    ):
        super().__init__()
        # 1. Global Context Extractor
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Total global features per lead:
        # cond (6 variables) + sst_hist (window=12) = 18 global scalar indices
        in_features = cond_channels + 12
        
        # 2. Decoder from Global to Spatial
        # Target shape is 60x70 = 4200
        self.H, self.W = 60, 70
        self.fc = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, self.H * self.W)
        )
        
        # 3. Learnable scaling for ECMWF baseline
        self.base_weight = nn.Parameter(torch.ones(1))
        
        # Zero-init the final layer so correction starts at exactly zero
        nn.init.zeros_(self.fc[-1].weight)
        nn.init.zeros_(self.fc[-1].bias)

    def forward(
        self,
        cond: torch.Tensor,
        seas_anom: torch.Tensor,
        ec_base: torch.Tensor,
        sst_hist: torch.Tensor,
        init_month: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        B, T, C, H, W = cond.shape
        
        # --- 1. Extract Global Climate Indices ---
        # cond_global: [B, T, C]
        cond_global = self.global_pool(cond.view(B*T, C, H, W)).view(B, T, C)
        
        # sst_global: [B, 12, H_sst, W_sst] -> [B, 12]
        sst_global = self.global_pool(sst_hist).view(B, 12)
        # Repeat sst_global for all lead times: [B, T, 12]
        sst_global = sst_global.unsqueeze(1).expand(B, T, 12)
        
        # Combine global indices: [B*T, 18]
        global_features = torch.cat([cond_global, sst_global], dim=-1).view(B * T, -1)
        
        # --- 2. Decode Global State into Spatial Correction ---
        correction = self.fc(global_features).view(B, T, self.H, self.W) # [B, T, H, W]
        
        # --- 3. Modulate Baseline ---
        base = ec_base[:, :, 0] # [B, T, H, W]
        
        return self.base_weight * base + correction

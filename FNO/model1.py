from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _num_groups(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(_num_groups(out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(_num_groups(out_ch), out_ch)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.norm1(self.conv1(x)))
        x = self.act(self.norm2(self.conv2(x)))
        return x


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.block = ConvBlock(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.block(x)


class TemporalConvFusion(nn.Module):
    def __init__(self, token_dim: int):
        super().__init__()
        self.dw = nn.Conv1d(token_dim, token_dim, kernel_size=3, padding=1, groups=token_dim)
        self.pw = nn.Conv1d(token_dim, token_dim, kernel_size=1)
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(token_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: [B, T, D]
        x = tokens.transpose(1, 2)  # [B, D, T]
        x = self.dw(x)
        x = self.act(x)
        x = self.pw(x)
        x = x.transpose(1, 2)
        return self.norm(tokens + x)


class GlobalContextEncoder(nn.Module):
    def __init__(self, in_ch: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 16, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 24, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(24, 32, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(32, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B*T, C, H, W]
        x = self.net(x).flatten(1)
        return self.proj(x)


class FiLMGenerator(nn.Module):
    def __init__(
        self,
        global_dim: int,
        pcs_dim: int,
        lead_embed_dim: int,
        leads: int,
        film_channels: Sequence[int],
    ):
        super().__init__()
        self.pcs_dim = pcs_dim
        self.leads = leads
        self.film_channels = list(film_channels)
        self.lead_embed = nn.Embedding(leads, lead_embed_dim)
        in_dim = global_dim + pcs_dim + lead_embed_dim
        hidden = 128
        out_dim = 2 * sum(self.film_channels)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(
        self,
        global_tokens: torch.Tensor,
        sst_pcs: Optional[torch.Tensor],
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        # global_tokens: [B, T, D]
        bsz, tdim, _ = global_tokens.shape
        if tdim != self.leads:
            raise ValueError(f"Expected lead dimension {self.leads}, got {tdim}.")

        if sst_pcs is None:
            sst_pcs = global_tokens.new_zeros((bsz, tdim, self.pcs_dim))
        if sst_pcs.dim() != 3:
            raise ValueError(f"sst_pcs must be [B,T,K], got shape {tuple(sst_pcs.shape)}")
        if sst_pcs.shape[:2] != (bsz, tdim):
            raise ValueError(
                f"sst_pcs batch/time mismatch, expected {(bsz, tdim)}, got {tuple(sst_pcs.shape[:2])}"
            )
        if sst_pcs.shape[2] != self.pcs_dim:
            raise ValueError(f"sst_pcs feature mismatch: expect {self.pcs_dim}, got {sst_pcs.shape[2]}")

        lead_idx = torch.arange(tdim, device=global_tokens.device).unsqueeze(0).expand(bsz, -1)
        lead_emb = self.lead_embed(lead_idx)
        token = torch.cat([global_tokens, sst_pcs, lead_emb], dim=-1)
        token = token.reshape(bsz * tdim, -1)
        raw = self.mlp(token).reshape(bsz, tdim, -1)

        params: List[Tuple[torch.Tensor, torch.Tensor]] = []
        offset = 0
        for ch in self.film_channels:
            gamma = raw[:, :, offset : offset + ch]
            offset += ch
            beta = raw[:, :, offset : offset + ch]
            offset += ch
            params.append((gamma, beta))
        return params


class GlobalResidualUNet6(nn.Module):
    """
    Lightweight dual-branch model for 6-lead precipitation correction.

    Inputs:
      cond_global: [B, 6, Cc, 180, 360]
      ec_base:     [B, 6, 1, 120, 140]
      sst_pcs:     [B, 6, K]
    Output:
      pred:        [B, 6, 120, 140]
    """

    def __init__(
        self,
        cond_channels: int = 8,
        leads: int = 6,
        pcs_dim: int = 8,
        channels: Tuple[int, int, int, int] = (32, 48, 64, 96),
        lead_embed_dim: int = 8,
        global_dim: int = 64,
    ):
        super().__init__()
        if len(channels) != 4:
            raise ValueError(f"channels must be length 4, got {channels}")
        c1, c2, c3, c4 = channels

        self.cond_channels = cond_channels
        self.leads = leads
        self.pcs_dim = pcs_dim
        self.out_hw = (120, 140)

        self.global_encoder = GlobalContextEncoder(cond_channels, global_dim)
        self.temporal_fusion = TemporalConvFusion(global_dim)
        self.film_generator = FiLMGenerator(
            global_dim=global_dim,
            pcs_dim=pcs_dim,
            lead_embed_dim=lead_embed_dim,
            leads=leads,
            film_channels=[c4, c3, c2],
        )

        self.local_lead_embed = nn.Embedding(leads, lead_embed_dim)
        self.pool = nn.AvgPool2d(2)

        self.enc1 = ConvBlock(1 + lead_embed_dim, c1)
        self.enc2 = ConvBlock(c1, c2)
        self.enc3 = ConvBlock(c2, c3)
        self.bottleneck = ConvBlock(c3, c4)

        self.up2 = UpBlock(c4, c3, c3)
        self.up1 = UpBlock(c3, c2, c2)
        self.up0 = UpBlock(c2, c1, c1)
        self.head = nn.Conv2d(c1, 1, kernel_size=1)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    @staticmethod
    def _apply_film(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        # Keep FiLM modulation bounded for stability.
        g = 1.0 + 0.1 * torch.tanh(gamma).unsqueeze(-1).unsqueeze(-1)
        b = 0.1 * beta.unsqueeze(-1).unsqueeze(-1)
        return x * g + b

    def forward(
        self,
        cond_global: torch.Tensor,
        ec_base: Optional[torch.Tensor] = None,
        sst_pcs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if cond_global.dim() != 5:
            raise ValueError(f"cond_global must be [B,T,C,H,W], got shape {tuple(cond_global.shape)}")
        bsz, tdim, cdim, hdim, wdim = cond_global.shape
        if tdim != self.leads:
            raise ValueError(f"Lead mismatch: expected {self.leads}, got {tdim}")
        if cdim != self.cond_channels:
            raise ValueError(f"cond channel mismatch: expected {self.cond_channels}, got {cdim}")

        if ec_base is None:
            ec_base = cond_global.new_zeros((bsz, tdim, 1, self.out_hw[0], self.out_hw[1]))
        if ec_base.dim() != 5:
            raise ValueError(f"ec_base must be [B,T,1,H,W], got shape {tuple(ec_base.shape)}")
        if ec_base.shape[:2] != (bsz, tdim) or ec_base.shape[2] != 1:
            raise ValueError(f"ec_base shape mismatch: got {tuple(ec_base.shape)}")

        # Global branch.
        g = self.global_encoder(cond_global.reshape(bsz * tdim, cdim, hdim, wdim))
        g = g.reshape(bsz, tdim, -1)
        g = self.temporal_fusion(g)
        film_params = self.film_generator(g, sst_pcs)

        # Local residual branch.
        out_delta = []
        lead_ids = torch.arange(tdim, device=cond_global.device)
        lead_embed = self.local_lead_embed(lead_ids)  # [T, E]
        out_h, out_w = ec_base.shape[-2], ec_base.shape[-1]

        for t in range(tdim):
            base_t = ec_base[:, t]  # [B,1,H,W]
            lead_map = lead_embed[t].view(1, -1, 1, 1).expand(bsz, -1, out_h, out_w)
            x = torch.cat([base_t, lead_map], dim=1)

            s1 = self.enc1(x)
            x = self.pool(s1)
            s2 = self.enc2(x)
            x = self.pool(s2)
            s3 = self.enc3(x)
            x = self.pool(s3)

            x = self.bottleneck(x)
            g0, b0 = film_params[0]
            x = self._apply_film(x, g0[:, t], b0[:, t])

            x = self.up2(x, s3)
            g1, b1 = film_params[1]
            x = self._apply_film(x, g1[:, t], b1[:, t])

            x = self.up1(x, s2)
            g2, b2 = film_params[2]
            x = self._apply_film(x, g2[:, t], b2[:, t])

            x = self.up0(x, s1)
            delta = self.head(x).squeeze(1)
            out_delta.append(delta)

        delta = torch.stack(out_delta, dim=1)
        pred = ec_base[:, :, 0] + delta
        return torch.nan_to_num(pred, nan=0.0, posinf=1e3, neginf=-1e3)


def masked_huber_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    valid = mask.bool()
    if not torch.any(valid):
        return pred.new_tensor(0.0)
    return F.huber_loss(pred[valid], target[valid], delta=delta, reduction="mean")


def masked_mse_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = mask.bool()
    if not torch.any(valid):
        return pred.new_tensor(0.0)
    return F.mse_loss(pred[valid], target[valid], reduction="mean")

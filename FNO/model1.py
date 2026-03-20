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


class TemporalLeadMixer(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.dw = nn.Conv3d(
            channels,
            channels,
            kernel_size=(3, 1, 1),
            padding=(1, 0, 0),
            groups=channels,
        )
        self.pw = nn.Conv3d(channels, channels, kernel_size=1)
        self.act = nn.GELU()
        self.norm = nn.GroupNorm(_num_groups(channels), channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C, H, W]
        bsz, tdim, ch, h, w = x.shape
        y = x.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]
        y = self.dw(y)
        y = self.act(y)
        y = self.pw(y)
        y = y.permute(0, 2, 1, 3, 4).contiguous()
        y = self.norm(y.view(bsz * tdim, ch, h, w)).view(bsz, tdim, ch, h, w)
        return x + y


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
        lead_ids: Optional[torch.Tensor] = None,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        # global_tokens: [B, T, D]
        bsz, tdim, _ = global_tokens.shape
        if lead_ids is None and tdim != self.leads:
            raise ValueError(f"Expected lead dimension {self.leads}, got {tdim}.")

        if sst_pcs is None:
            sst_pcs = global_tokens.new_zeros((bsz, tdim, self.pcs_dim))
        else:
            if sst_pcs.dim() == 2:
                sst_pcs = sst_pcs.unsqueeze(1)
            if sst_pcs.dim() != 3:
                raise ValueError(f"sst_pcs must be [B,T,K], got shape {tuple(sst_pcs.shape)}")
            if sst_pcs.shape[:2] != (bsz, tdim):
                raise ValueError(
                    f"sst_pcs batch/time mismatch, expected {(bsz, tdim)}, got {tuple(sst_pcs.shape[:2])}"
                )
            if sst_pcs.shape[2] != self.pcs_dim:
                raise ValueError(f"sst_pcs feature mismatch: expect {self.pcs_dim}, got {sst_pcs.shape[2]}")

        if lead_ids is None:
            lead_idx = torch.arange(tdim, device=global_tokens.device).unsqueeze(0).expand(bsz, -1)
        else:
            if lead_ids.dim() == 1:
                if lead_ids.shape[0] != tdim:
                    raise ValueError(f"lead_ids length mismatch: expected {tdim}, got {lead_ids.shape[0]}")
                lead_idx = lead_ids.unsqueeze(0).expand(bsz, -1)
            elif lead_ids.dim() == 2:
                if lead_ids.shape[:2] != (bsz, tdim):
                    raise ValueError(
                        f"lead_ids shape mismatch: expected {(bsz, tdim)}, got {tuple(lead_ids.shape[:2])}"
                    )
                lead_idx = lead_ids
            else:
                raise ValueError(f"lead_ids must be [T] or [B,T], got shape {tuple(lead_ids.shape)}")

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
        cond_local_dim: int = 16,
    ):
        super().__init__()
        if len(channels) != 4:
            raise ValueError(f"channels must be length 4, got {channels}")
        c1, c2, c3, c4 = channels

        self.cond_channels = cond_channels
        self.leads = leads
        self.pcs_dim = pcs_dim
        self.out_hw = (120, 140)
        self.cond_local_dim = min(cond_local_dim, c1)

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

        self.cond_local_proj = nn.Conv2d(cond_channels, self.cond_local_dim, kernel_size=1)
        self.cond_local_norm = nn.GroupNorm(_num_groups(self.cond_local_dim), self.cond_local_dim)
        self.cond_local_act = nn.GELU()
        self.cond_scale2 = nn.Conv2d(self.cond_local_dim, c2, kernel_size=1)
        self.cond_scale3 = nn.Conv2d(self.cond_local_dim, c3, kernel_size=1)
        self.cond_scale4 = nn.Conv2d(self.cond_local_dim, c4, kernel_size=1)

        self.enc1 = ConvBlock(2 + lead_embed_dim + self.cond_local_dim, c1)
        self.enc2 = ConvBlock(c1, c2)
        self.enc3 = ConvBlock(c2, c3)
        self.bottleneck = ConvBlock(c3, c4)

        self.temporal_local_bottleneck = TemporalLeadMixer(c4)
        self.temporal_local_mid = TemporalLeadMixer(c3)

        self.up2 = UpBlock(c4, c3, c3)
        self.up1 = UpBlock(c3, c2, c2)
        self.up0 = UpBlock(c2, c1, c1)
        self.head = nn.Conv2d(c1, 1, kernel_size=1)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Conv3d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    @staticmethod
    def _apply_film(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        # Keep FiLM modulation bounded for stability.
        g = 1.0 + 0.1 * torch.tanh(gamma).unsqueeze(-1).unsqueeze(-1)
        b = 0.1 * beta.unsqueeze(-1).unsqueeze(-1)
        return x * g + b

    def forward_step(
        self,
        cond_t: torch.Tensor,
        ec_base_t: Optional[torch.Tensor] = None,
        sst_pcs_t: Optional[torch.Tensor] = None,
        prev_pred: Optional[torch.Tensor] = None,
        lead_id: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # cond_t: [B, C, H, W]
        if cond_t.dim() != 4:
            raise ValueError(f"cond_t must be [B,C,H,W], got shape {tuple(cond_t.shape)}")
        bsz, cdim, hdim, wdim = cond_t.shape
        if cdim != self.cond_channels:
            raise ValueError(f"cond channel mismatch: expected {self.cond_channels}, got {cdim}")

        out_h, out_w = self.out_hw
        if ec_base_t is None:
            ec_base_t = cond_t.new_zeros((bsz, 1, out_h, out_w))
        if ec_base_t.dim() != 4 or ec_base_t.shape[1] != 1:
            raise ValueError(f"ec_base_t must be [B,1,H,W], got shape {tuple(ec_base_t.shape)}")
        out_h, out_w = ec_base_t.shape[-2], ec_base_t.shape[-1]

        if prev_pred is None:
            prev_pred = ec_base_t[:, 0]
        if prev_pred.dim() == 3:
            prev_pred = prev_pred.unsqueeze(1)
        if prev_pred.dim() != 4 or prev_pred.shape[1] != 1:
            raise ValueError(f"prev_pred must be [B,1,H,W], got shape {tuple(prev_pred.shape)}")
        if prev_pred.shape[-2:] != (out_h, out_w):
            raise ValueError(
                f"prev_pred spatial mismatch: expected {(out_h, out_w)}, got {tuple(prev_pred.shape[-2:])}"
            )

        if lead_id is None:
            lead_ids = cond_t.new_zeros((bsz, 1), dtype=torch.long)
        else:
            if isinstance(lead_id, int):
                lead_ids = cond_t.new_full((bsz, 1), int(lead_id), dtype=torch.long)
            elif torch.is_tensor(lead_id):
                if lead_id.dim() == 0:
                    lead_ids = lead_id.view(1, 1).expand(bsz, 1).to(dtype=torch.long)
                elif lead_id.dim() == 1:
                    if lead_id.shape[0] == bsz:
                        lead_ids = lead_id.view(bsz, 1).to(dtype=torch.long)
                    elif lead_id.shape[0] == 1:
                        lead_ids = lead_id.expand(bsz).view(bsz, 1).to(dtype=torch.long)
                    else:
                        raise ValueError(f"lead_id length mismatch: {lead_id.shape[0]} vs {bsz}")
                elif lead_id.dim() == 2 and lead_id.shape == (bsz, 1):
                    lead_ids = lead_id.to(dtype=torch.long)
                else:
                    raise ValueError(f"lead_id must be int or shape [B] / [B,1], got {tuple(lead_id.shape)}")
            else:
                raise ValueError(f"Unsupported lead_id type: {type(lead_id)}")

        # Global branch (single step).
        g = self.global_encoder(cond_t)
        g = g.unsqueeze(1)
        g = self.temporal_fusion(g)
        film_params = self.film_generator(g, sst_pcs_t, lead_ids=lead_ids)

        # Local residual branch.
        cond_in = F.interpolate(cond_t, size=(out_h, out_w), mode="bilinear", align_corners=False)
        cond_in = self.cond_local_act(self.cond_local_norm(self.cond_local_proj(cond_in)))
        lead_embed = self.local_lead_embed(lead_ids[:, 0])
        lead_map = lead_embed.view(bsz, -1, 1, 1).expand(bsz, -1, out_h, out_w)
        x = torch.cat([ec_base_t, prev_pred, lead_map, cond_in], dim=1)

        s1 = self.enc1(x)
        x = self.pool(s1)
        s2 = self.enc2(x)
        cond2 = F.interpolate(cond_in, size=s2.shape[-2:], mode="bilinear", align_corners=False)
        s2 = s2 + self.cond_local_act(self.cond_scale2(cond2))

        x = self.pool(s2)
        s3 = self.enc3(x)
        cond3 = F.interpolate(cond_in, size=s3.shape[-2:], mode="bilinear", align_corners=False)
        s3 = s3 + self.cond_local_act(self.cond_scale3(cond3))

        x = self.pool(s3)
        x = self.bottleneck(x)
        cond4 = F.interpolate(cond_in, size=x.shape[-2:], mode="bilinear", align_corners=False)
        x = x + self.cond_local_act(self.cond_scale4(cond4))

        g0, b0 = film_params[0]
        x = self._apply_film(x, g0[:, 0], b0[:, 0])

        x = self.up2(x, s3)
        g1, b1 = film_params[1]
        x = self._apply_film(x, g1[:, 0], b1[:, 0])

        x = self.up1(x, s2)
        g2, b2 = film_params[2]
        x = self._apply_film(x, g2[:, 0], b2[:, 0])

        x = self.up0(x, s1)
        delta = self.head(x).squeeze(1)
        pred = ec_base_t[:, 0] + delta
        return torch.nan_to_num(pred, nan=0.0, posinf=1e3, neginf=-1e3)

    def forward(
        self,
        cond_global: torch.Tensor,
        ec_base: Optional[torch.Tensor] = None,
        sst_pcs: Optional[torch.Tensor] = None,
        prev_pred: Optional[torch.Tensor] = None,
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

        if prev_pred is None:
            prev_pred = ec_base[:, :, 0]
        if prev_pred.dim() == 4:
            prev_pred = prev_pred.unsqueeze(2)
        if prev_pred.dim() != 5 or prev_pred.shape[:2] != (bsz, tdim) or prev_pred.shape[2] != 1:
            raise ValueError(f"prev_pred must be [B,T,1,H,W], got shape {tuple(prev_pred.shape)}")

        # Global branch.
        g = self.global_encoder(cond_global.reshape(bsz * tdim, cdim, hdim, wdim))
        g = g.reshape(bsz, tdim, -1)
        g = self.temporal_fusion(g)
        film_params = self.film_generator(g, sst_pcs)

        # Local residual branch.
        lead_ids = torch.arange(tdim, device=cond_global.device)
        lead_embed = self.local_lead_embed(lead_ids)  # [T, E]
        out_h, out_w = ec_base.shape[-2], ec_base.shape[-1]

        cond_bt = cond_global.reshape(bsz * tdim, cdim, hdim, wdim)
        cond_in = F.interpolate(cond_bt, size=(out_h, out_w), mode="bilinear", align_corners=False)
        cond_in = self.cond_local_act(self.cond_local_norm(self.cond_local_proj(cond_in)))
        cond_in_bt = cond_in
        cond_in = cond_in.reshape(bsz, tdim, self.cond_local_dim, out_h, out_w)

        lead_map = lead_embed.view(1, tdim, -1, 1, 1).expand(bsz, -1, -1, out_h, out_w)
        x = torch.cat([ec_base, prev_pred, lead_map, cond_in], dim=2)
        x = x.reshape(bsz * tdim, -1, out_h, out_w)

        s1 = self.enc1(x)
        x = self.pool(s1)
        s2 = self.enc2(x)
        cond2 = F.interpolate(cond_in_bt, size=s2.shape[-2:], mode="bilinear", align_corners=False)
        s2 = s2 + self.cond_local_act(self.cond_scale2(cond2))

        x = self.pool(s2)
        s3 = self.enc3(x)
        cond3 = F.interpolate(cond_in_bt, size=s3.shape[-2:], mode="bilinear", align_corners=False)
        s3 = s3 + self.cond_local_act(self.cond_scale3(cond3))

        x = self.pool(s3)
        x = self.bottleneck(x)
        cond4 = F.interpolate(cond_in_bt, size=x.shape[-2:], mode="bilinear", align_corners=False)
        x = x + self.cond_local_act(self.cond_scale4(cond4))

        g0, b0 = film_params[0]
        x = self._apply_film(x, g0.reshape(-1, x.shape[1]), b0.reshape(-1, x.shape[1]))
        if tdim > 1:
            x = x.reshape(bsz, tdim, x.shape[1], x.shape[2], x.shape[3])
            x = self.temporal_local_bottleneck(x)
            x = x.reshape(bsz * tdim, x.shape[2], x.shape[3], x.shape[4])

        x = self.up2(x, s3)
        g1, b1 = film_params[1]
        x = self._apply_film(x, g1.reshape(-1, x.shape[1]), b1.reshape(-1, x.shape[1]))
        if tdim > 1:
            x = x.reshape(bsz, tdim, x.shape[1], x.shape[2], x.shape[3])
            x = self.temporal_local_mid(x)
            x = x.reshape(bsz * tdim, x.shape[2], x.shape[3], x.shape[4])

        x = self.up1(x, s2)
        g2, b2 = film_params[2]
        x = self._apply_film(x, g2.reshape(-1, x.shape[1]), b2.reshape(-1, x.shape[1]))

        x = self.up0(x, s1)
        delta = self.head(x).squeeze(1)
        delta = delta.reshape(bsz, tdim, out_h, out_w)
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

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


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.hidden_dim = hidden_dim
        self.conv = nn.Conv2d(
            input_dim + hidden_dim,
            4 * hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
        )

    def forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
        c: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [B, C, H, W], h/c: [B, hidden, H, W]
        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(gates, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c + i * g
        c_next = torch.clamp(c_next, -1e2, 1e2)
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class ConvLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            self.layers.append(ConvLSTMCell(in_dim, hidden_dim))

    def init_state(
        self,
        batch_size: int,
        spatial: Tuple[int, int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        h, w = spatial
        state: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for _ in range(self.num_layers):
            h_t = torch.zeros((batch_size, self.hidden_dim, h, w), device=device, dtype=dtype)
            c_t = torch.zeros_like(h_t)
            state.append((h_t, c_t))
        return state

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        # x: [B, C, H, W]
        if state is None:
            raise ValueError("ConvLSTM state must be provided (use init_state).")
        new_state: List[Tuple[torch.Tensor, torch.Tensor]] = []
        out = x
        for i, cell in enumerate(self.layers):
            h, c = state[i]
            h, c = cell(out, h, c)
            new_state.append((h, c))
            out = h
        return out, new_state


class SSTHistoryEncoder(nn.Module):
    def __init__(self, in_channels: int = 1, hidden_dim: int = 8, num_layers: int = 1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.convlstm = ConvLSTM(input_dim=in_channels, hidden_dim=hidden_dim, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, H, W] or [B, T, C, H, W]
        if x.dim() == 4:
            x = x.unsqueeze(2)
        if x.dim() != 5:
            raise ValueError(f"sst_hist must be [B,T,H,W] or [B,T,C,H,W], got {tuple(x.shape)}")
        bsz, tdim, _, hdim, wdim = x.shape
        state = self.convlstm.init_state(bsz, (hdim, wdim), x.device, x.dtype)
        out = None
        for t in range(tdim):
            out, state = self.convlstm(x[:, t], state)
        if out is None:
            raise ValueError("sst_hist sequence is empty.")
        return out


class Seas2RainModel(nn.Module):
    def __init__(
        self,
        cond_channels: int = 7,
        hidden_dim: int = 64,
        num_layers: int = 1,
        encoder_channels: int = 64,
        decoder_channels: int = 32,
        ps_scale: int = 2,
        sst_feat_channels: int = 8,
        sst_num_layers: int = 1,
    ):
        super().__init__()
        if ps_scale != 2:
            raise ValueError("Only PixelShuffle scale=2 is supported for 60x70 output.")

        self.cond_channels = cond_channels
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.ps_scale = ps_scale
        self.sst_feat_channels = sst_feat_channels

        self.sst_encoder = SSTHistoryEncoder(
            in_channels=1,
            hidden_dim=sst_feat_channels,
            num_layers=sst_num_layers,
        )

        in_ch = cond_channels + sst_feat_channels + 2  # cond + sst_feat + seas_anom + prev_pred

        self.encoder = nn.Sequential(
            nn.Conv2d(in_ch, encoder_channels, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(_num_groups(encoder_channels), encoder_channels),
            nn.GELU(),
            nn.Conv2d(encoder_channels, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(_num_groups(hidden_dim), hidden_dim),
            nn.GELU(),
        )

        self.convlstm = ConvLSTM(input_dim=hidden_dim, hidden_dim=hidden_dim, num_layers=num_layers)

        self.up_conv = nn.Conv2d(hidden_dim, decoder_channels * (ps_scale ** 2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(ps_scale)
        self.out_conv = nn.Conv2d(decoder_channels, 1, kernel_size=3, padding=1)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def encode_sst(self, sst_hist: torch.Tensor) -> torch.Tensor:
        return self.sst_encoder(sst_hist)

    def init_state(
        self,
        batch_size: int,
        spatial: Tuple[int, int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        return self.convlstm.init_state(batch_size=batch_size, spatial=spatial, device=device, dtype=dtype)

    def forward_step(
        self,
        cond_t: torch.Tensor,
        seas_anom_t: torch.Tensor,
        ec_base_t: torch.Tensor,
        prev_pred: torch.Tensor,
        sst_feat: Optional[torch.Tensor] = None,
        state: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        # cond_t: [B, 7, 60, 70]
        # seas_anom_t/ec_base_t: [B, 1, 60, 70]
        # prev_pred: [B, 60, 70] or [B, 1, 60, 70]
        if prev_pred.dim() == 3:
            prev_pred = prev_pred.unsqueeze(1)

        target_hw = seas_anom_t.shape[-2:]
        if cond_t.shape[-2:] != target_hw:
            cond_t = F.adaptive_avg_pool2d(cond_t, target_hw)

        if sst_feat is None:
            sst_feat_t = cond_t.new_zeros((cond_t.shape[0], self.sst_feat_channels, target_hw[0], target_hw[1]))
        else:
            if sst_feat.dim() == 3:
                sst_feat = sst_feat.unsqueeze(1)
            if sst_feat.dim() != 4:
                raise ValueError(f"sst_feat must be [B,C,H,W], got {tuple(sst_feat.shape)}")
            if sst_feat.shape[-2:] != target_hw:
                sst_feat_t = F.adaptive_avg_pool2d(sst_feat, target_hw)
            else:
                sst_feat_t = sst_feat

        x = torch.cat([cond_t, sst_feat_t, seas_anom_t, prev_pred], dim=1)
        x = self.encoder(x)
        if state is None:
            h_out, w_out = x.shape[-2], x.shape[-1]
            state = self.init_state(x.shape[0], (h_out, w_out), x.device, x.dtype)

        h, state = self.convlstm(x, state)
        y = self.up_conv(h)
        y = self.pixel_shuffle(y)
        y = F.gelu(y)
        delta = self.out_conv(y).squeeze(1)
        pred = ec_base_t[:, 0] + delta
        pred = torch.nan_to_num(pred, nan=0.0, posinf=1e3, neginf=-1e3)
        return pred, state

    def forward(
        self,
        cond: torch.Tensor,
        seas_anom: torch.Tensor,
        ec_base: torch.Tensor,
        sst_feat: Optional[torch.Tensor] = None,
        prev_pred: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # cond: [B, T, 7, 60, 70]
        bsz, tdim = cond.shape[0], cond.shape[1]
        preds: List[torch.Tensor] = []

        prev_seq: Optional[torch.Tensor] = None
        if prev_pred is None:
            prev = ec_base[:, 0, 0]
        else:
            if prev_pred.dim() == 5:
                prev_seq = prev_pred[:, :, 0]
                prev = prev_seq[:, 0]
            elif prev_pred.dim() == 4:
                prev_seq = prev_pred
                prev = prev_seq[:, 0]
            elif prev_pred.dim() == 3:
                prev = prev_pred
            else:
                raise ValueError(f"prev_pred shape not supported: {tuple(prev_pred.shape)}")

        state = None
        for t in range(tdim):
            if prev_seq is not None:
                prev_in = prev_seq[:, t]
            else:
                prev_in = prev
            pred_t, state = self.forward_step(
                cond_t=cond[:, t],
                seas_anom_t=seas_anom[:, t],
                ec_base_t=ec_base[:, t],
                prev_pred=prev_in,
                sst_feat=sst_feat,
                state=state,
            )
            preds.append(pred_t)
            prev = pred_t

        return torch.stack(preds, dim=1)

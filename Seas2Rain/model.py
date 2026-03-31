from __future__ import annotations

from typing import List, Optional, Tuple

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


class SSTPCEncoder(nn.Module):
    def __init__(self, pc_dim: int, hidden_dim: int, num_layers: int = 1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=pc_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.0 if num_layers <= 1 else 0.1,
        )
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, K]
        if x.dim() != 3:
            raise ValueError(f"sst_pcs must be [B,T,K], got {tuple(x.shape)}")
        _, h = self.gru(x)
        vec = h[-1]
        vec = self.out_norm(vec)
        return vec


class FiLM(nn.Module):
    def __init__(self, cond_dim: int, num_features: int, hidden_dim: Optional[int] = None):
        super().__init__()
        if hidden_dim is None:
            self.net = nn.Linear(cond_dim, 2 * num_features)
        else:
            self.net = nn.Sequential(
                nn.Linear(cond_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 2 * num_features),
            )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        params = self.net(cond)
        gamma, beta = params.chunk(2, dim=-1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return x * (1.0 + gamma) + beta


class Seas2RainModel(nn.Module):
    def __init__(
        self,
        cond_channels: int = 7,
        cond_pc_k: int = 8,
        sst_pc_k: int = 8,
        sst_pc_hidden: int = 16,
        hidden_dim: int = 64,
        num_layers: int = 1,
        encoder_channels: int = 64,
        decoder_channels: int = 32,
        ps_scale: int = 2,
        sst_gru_layers: int = 1,
    ):
        super().__init__()
        if ps_scale != 2:
            raise ValueError("Only PixelShuffle scale=2 is supported for 60x70 output.")

        self.cond_channels = cond_channels
        self.cond_pc_k = cond_pc_k
        self.sst_pc_k = sst_pc_k
        self.sst_pc_hidden = sst_pc_hidden
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.ps_scale = ps_scale

        cond_vec_dim = cond_channels * cond_pc_k + sst_pc_hidden

        self.sst_encoder = SSTPCEncoder(pc_dim=sst_pc_k, hidden_dim=sst_pc_hidden, num_layers=sst_gru_layers)

        in_ch = 2  # seas_anom + prev_pred

        self.enc_conv1 = nn.Conv2d(in_ch, encoder_channels, kernel_size=3, stride=2, padding=1)
        self.enc_norm1 = nn.GroupNorm(_num_groups(encoder_channels), encoder_channels)
        self.enc_film1 = FiLM(cond_vec_dim, encoder_channels)

        self.enc_conv2 = nn.Conv2d(encoder_channels, hidden_dim, kernel_size=3, padding=1)
        self.enc_norm2 = nn.GroupNorm(_num_groups(hidden_dim), hidden_dim)
        self.enc_film2 = FiLM(cond_vec_dim, hidden_dim)

        self.convlstm = ConvLSTM(input_dim=hidden_dim, hidden_dim=hidden_dim, num_layers=num_layers)

        self.up_conv = nn.Conv2d(hidden_dim, decoder_channels * (ps_scale ** 2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(ps_scale)
        self.dec_film = FiLM(cond_vec_dim, decoder_channels)
        self.out_conv = nn.Conv2d(decoder_channels, 1, kernel_size=3, padding=1)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def encode_sst_pcs(self, sst_pcs: torch.Tensor) -> torch.Tensor:
        return self.sst_encoder(sst_pcs)

    def init_state(
        self,
        batch_size: int,
        spatial: Tuple[int, int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        return self.convlstm.init_state(batch_size=batch_size, spatial=spatial, device=device, dtype=dtype)

    def _build_cond_vec(self, cond_pcs_t: torch.Tensor, sst_vec: Optional[torch.Tensor]) -> torch.Tensor:
        if cond_pcs_t.dim() == 3:
            cond_flat = cond_pcs_t.reshape(cond_pcs_t.shape[0], -1)
        elif cond_pcs_t.dim() == 2:
            cond_flat = cond_pcs_t
        else:
            raise ValueError(f"cond_pcs_t must be [B,V,K] or [B,V*K], got {tuple(cond_pcs_t.shape)}")

        if sst_vec is None:
            sst_vec = cond_flat.new_zeros((cond_flat.shape[0], self.sst_pc_hidden))
        if sst_vec.dim() != 2:
            raise ValueError(f"sst_vec must be [B, H], got {tuple(sst_vec.shape)}")

        cond_vec = torch.cat([cond_flat, sst_vec], dim=1)
        expected = self.cond_channels * self.cond_pc_k + self.sst_pc_hidden
        if cond_vec.shape[1] != expected:
            raise ValueError(f"cond_vec dim mismatch: {cond_vec.shape[1]} vs {expected}")
        return cond_vec

    def forward_step(
        self,
        cond_pcs_t: torch.Tensor,
        seas_anom_t: torch.Tensor,
        ec_base_t: torch.Tensor,
        prev_pred: torch.Tensor,
        sst_vec: Optional[torch.Tensor] = None,
        state: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        # seas_anom_t/ec_base_t: [B, 1, H, W]
        # prev_pred: [B, H, W] or [B, 1, H, W]
        if prev_pred.dim() == 3:
            prev_pred = prev_pred.unsqueeze(1)

        cond_vec = self._build_cond_vec(cond_pcs_t, sst_vec)

        x = torch.cat([seas_anom_t, prev_pred], dim=1)
        x = self.enc_conv1(x)
        x = self.enc_norm1(x)
        x = self.enc_film1(x, cond_vec)
        x = F.gelu(x)

        x = self.enc_conv2(x)
        x = self.enc_norm2(x)
        x = self.enc_film2(x, cond_vec)
        x = F.gelu(x)

        if state is None:
            h_out, w_out = x.shape[-2], x.shape[-1]
            state = self.init_state(x.shape[0], (h_out, w_out), x.device, x.dtype)

        h, state = self.convlstm(x, state)
        y = self.up_conv(h)
        y = self.pixel_shuffle(y)
        y = self.dec_film(y, cond_vec)
        y = F.gelu(y)
        delta = self.out_conv(y).squeeze(1)
        pred = ec_base_t[:, 0] + delta
        pred = torch.nan_to_num(pred, nan=0.0, posinf=1e3, neginf=-1e3)
        return pred, state

    def forward(
        self,
        cond_pcs: torch.Tensor,
        seas_anom: torch.Tensor,
        ec_base: torch.Tensor,
        sst_pcs: Optional[torch.Tensor] = None,
        sst_vec: Optional[torch.Tensor] = None,
        prev_pred: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # cond_pcs: [B, T, V, K]
        bsz, tdim = cond_pcs.shape[0], cond_pcs.shape[1]
        preds: List[torch.Tensor] = []

        if sst_vec is None:
            if sst_pcs is None:
                raise ValueError("sst_pcs or sst_vec must be provided")
            sst_vec = self.encode_sst_pcs(sst_pcs)

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
                cond_pcs_t=cond_pcs[:, t],
                seas_anom_t=seas_anom[:, t],
                ec_base_t=ec_base[:, t],
                prev_pred=prev_in,
                sst_vec=sst_vec,
                state=state,
            )
            preds.append(pred_t)
            prev = pred_t

        return torch.stack(preds, dim=1)

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class SPADE(nn.Module):
    def __init__(
        self,
        cond_channels: int,
        out_channels: int,
        lead_embed_dim: int,
        hidden_channels: int = 16,
        gate_hidden: int = 32,
        gate_init_bias: float = 4.0,
        cond_dropout: float = 0.0,
    ):
        super().__init__()
        self.cond_channels = cond_channels
        self.lead_embed_dim = lead_embed_dim
        self.bn = nn.BatchNorm2d(out_channels, affine=False)
        self.cond_drop = nn.Dropout2d(cond_dropout) if cond_dropout > 0.0 else nn.Identity()
        self.gate_mlp = nn.Sequential(
            nn.Linear(lead_embed_dim, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, cond_channels),
        )
        self.conv_shared = nn.Conv2d(cond_channels + lead_embed_dim, hidden_channels, kernel_size=3, padding=1)
        self.conv_gamma = nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1)
        self.conv_beta = nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1)
        self._zero_init(gate_init_bias=gate_init_bias)

    def _zero_init(self, gate_init_bias: float = 4.0) -> None:
        gate_out = self.gate_mlp[-1]
        nn.init.zeros_(gate_out.weight)
        nn.init.constant_(gate_out.bias, gate_init_bias)
        nn.init.zeros_(self.conv_gamma.weight)
        nn.init.zeros_(self.conv_gamma.bias)
        nn.init.zeros_(self.conv_beta.weight)
        nn.init.zeros_(self.conv_beta.bias)

    def forward(self, x: torch.Tensor, cond_map: torch.Tensor, lead_emb: torch.Tensor) -> torch.Tensor:
        if cond_map.shape[-2:] != x.shape[-2:]:
            cond_map = F.adaptive_avg_pool2d(cond_map, x.shape[-2:])

        gate = torch.sigmoid(self.gate_mlp(lead_emb)).view(lead_emb.shape[0], self.cond_channels, 1, 1)
        gated_cond = self.cond_drop(cond_map * gate)
        lead_map = lead_emb[:, :, None, None].expand(-1, -1, x.shape[-2], x.shape[-1])
        fused_cond = torch.cat([gated_cond, lead_map], dim=1)

        h = self.conv_shared(fused_cond)
        gamma = self.conv_gamma(h)
        beta = self.conv_beta(h)
        return self.bn(x) * (1.0 + gamma) + beta


class Seas2RainModel(nn.Module):
    def __init__(
        self,
        cond_channels: int = 2,
        sst_hist_channels: int = 12,
        spade_hidden: int = 16,
        lead_embed_dim: int = 8,
        lead_gate_hidden: int = 32,
        lead_gate_init_bias: float = 4.0,
        enc_spade1_hidden: Optional[int] = None,
        enc_spade2_hidden: Optional[int] = None,
        dec_spade_hidden: Optional[int] = None,
        dropout: float = 0.1,
        cond_dropout: float = 0.1,
        hidden_dim: int = 64,
        num_layers: int = 1,
        encoder_channels: int = 64,
        decoder_channels: int = 32,
        ps_scale: int = 2,
    ):
        super().__init__()
        if ps_scale != 2:
            raise ValueError("Only PixelShuffle scale=2 is supported for 60x70 output.")

        self.cond_channels = cond_channels
        self.sst_hist_channels = sst_hist_channels
        self.spade_hidden = spade_hidden
        self.num_leads = 6
        self.lead_embed_dim = lead_embed_dim
        self.lead_gate_hidden = lead_gate_hidden
        self.lead_gate_init_bias = lead_gate_init_bias
        self.dropout = dropout
        self.cond_dropout = cond_dropout
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.ps_scale = ps_scale

        cond_in_ch = cond_channels + sst_hist_channels
        enc_spade1_hidden = spade_hidden if enc_spade1_hidden is None else enc_spade1_hidden
        enc_spade2_hidden = spade_hidden if enc_spade2_hidden is None else enc_spade2_hidden
        dec_spade_hidden = spade_hidden if dec_spade_hidden is None else dec_spade_hidden
        self.enc_spade1_hidden = enc_spade1_hidden
        self.enc_spade2_hidden = enc_spade2_hidden
        self.dec_spade_hidden = dec_spade_hidden
        self.lead_embedding = nn.Embedding(self.num_leads, lead_embed_dim)

        in_ch = 2  # seas_anom + prev_pred

        self.enc_conv1 = nn.Conv2d(in_ch, encoder_channels, kernel_size=3, stride=2, padding=1)
        self.enc_spade1 = SPADE(
            cond_in_ch,
            encoder_channels,
            lead_embed_dim=lead_embed_dim,
            hidden_channels=enc_spade1_hidden,
            gate_hidden=lead_gate_hidden,
            gate_init_bias=lead_gate_init_bias,
            cond_dropout=cond_dropout,
        )
        self.enc_drop1 = nn.Dropout2d(dropout)

        self.enc_conv2 = nn.Conv2d(encoder_channels, hidden_dim, kernel_size=3, padding=1)
        self.enc_spade2 = SPADE(
            cond_in_ch,
            hidden_dim,
            lead_embed_dim=lead_embed_dim,
            hidden_channels=enc_spade2_hidden,
            gate_hidden=lead_gate_hidden,
            gate_init_bias=lead_gate_init_bias,
            cond_dropout=cond_dropout,
        )
        self.enc_drop2 = nn.Dropout2d(dropout)

        self.convlstm = ConvLSTM(input_dim=hidden_dim, hidden_dim=hidden_dim, num_layers=num_layers)

        self.up_conv = nn.Conv2d(hidden_dim, decoder_channels * (ps_scale ** 2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(ps_scale)
        self.dec_spade = SPADE(
            cond_in_ch,
            decoder_channels,
            lead_embed_dim=lead_embed_dim,
            hidden_channels=dec_spade_hidden,
            gate_hidden=lead_gate_hidden,
            gate_init_bias=lead_gate_init_bias,
            cond_dropout=cond_dropout,
        )
        self.dec_drop = nn.Dropout2d(dropout)
        self.out_conv = nn.Conv2d(decoder_channels, 1, kernel_size=3, padding=1)

        self._init_weights()
        self._zero_spade()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _zero_spade(self) -> None:
        for module in self.modules():
            if isinstance(module, SPADE):
                module._zero_init(gate_init_bias=self.lead_gate_init_bias)

    def init_state(
        self,
        batch_size: int,
        spatial: Tuple[int, int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        return self.convlstm.init_state(batch_size=batch_size, spatial=spatial, device=device, dtype=dtype)

    def _build_cond_map(
        self,
        cond_t: torch.Tensor,
        sst_hist: torch.Tensor,
        target_hw: Tuple[int, int],
    ) -> torch.Tensor:
        if cond_t.shape[-2:] != target_hw:
            cond_t = F.adaptive_avg_pool2d(cond_t, target_hw)

        if sst_hist.dim() == 3:
            sst_hist = sst_hist.unsqueeze(1)
        if sst_hist.dim() != 4:
            raise ValueError(f"sst_hist must be [B,T,H,W], got {tuple(sst_hist.shape)}")
        if sst_hist.shape[1] != self.sst_hist_channels:
            raise ValueError(f"sst_hist channels {sst_hist.shape[1]} != expected {self.sst_hist_channels}")
        if sst_hist.shape[-2:] != target_hw:
            sst_hist = F.adaptive_avg_pool2d(sst_hist, target_hw)

        cond_map = torch.cat([cond_t, sst_hist], dim=1)
        return cond_map

    def forward_step(
        self,
        cond_t: torch.Tensor,
        seas_anom_t: torch.Tensor,
        ec_base_t: torch.Tensor,
        prev_pred: torch.Tensor,
        sst_hist: torch.Tensor,
        lead_idx: torch.Tensor,
        state: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        # cond_t: [B, 2, Hc, Wc]
        # seas_anom_t/ec_base_t: [B, 1, 60, 70]
        # prev_pred: [B, 60, 70] or [B, 1, 60, 70]
        if prev_pred.dim() == 3:
            prev_pred = prev_pred.unsqueeze(1)

        target_hw = seas_anom_t.shape[-2:]
        cond_map = self._build_cond_map(cond_t, sst_hist, target_hw)
        lead_emb = self.lead_embedding(lead_idx)

        x = torch.cat([seas_anom_t, prev_pred], dim=1)
        x = self.enc_conv1(x)
        x = self.enc_spade1(x, cond_map, lead_emb)
        x = F.gelu(x)
        x = self.enc_drop1(x)

        x = self.enc_conv2(x)
        x = self.enc_spade2(x, cond_map, lead_emb)
        x = F.gelu(x)
        x = self.enc_drop2(x)

        if state is None:
            h_out, w_out = x.shape[-2], x.shape[-1]
            state = self.init_state(x.shape[0], (h_out, w_out), x.device, x.dtype)

        h, state = self.convlstm(x, state)
        y = self.up_conv(h)
        y = self.pixel_shuffle(y)
        y = self.dec_spade(y, cond_map, lead_emb)
        y = F.gelu(y)
        y = self.dec_drop(y)
        delta = self.out_conv(y).squeeze(1)
        pred = ec_base_t[:, 0] + delta
        pred = torch.nan_to_num(pred, nan=0.0, posinf=1e3, neginf=-1e3)
        return pred, state

    def forward(
        self,
        cond: torch.Tensor,
        seas_anom: torch.Tensor,
        ec_base: torch.Tensor,
        sst_hist: torch.Tensor,
        prev_pred: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # cond: [B, T, 2, Hc, Wc]
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
            lead_idx = torch.full((bsz,), t, device=cond.device, dtype=torch.long)
            pred_t, state = self.forward_step(
                cond_t=cond[:, t],
                seas_anom_t=seas_anom[:, t],
                ec_base_t=ec_base[:, t],
                prev_pred=prev_in,
                sst_hist=sst_hist,
                lead_idx=lead_idx,
                state=state,
            )
            preds.append(pred_t)
            prev = pred_t

        return torch.stack(preds, dim=1)

# model.py
# PyTorch implementation of a spatio-temporal correction network
# Components: UNet encoder/decoder (GroupNorm), simple FNO block, TemporalTransformer, ConvLSTM (optional),
# hierarchical per-channel FiLM (from SST PCs + lead embedding), lead-adapter (MoE), probabilistic head: zero-inflated Gamma.
# Designed for inputs:
#   cond: [B, C, T, H, W]
#   target: [B, 1, T, H, W]
#   sst: either [B, M, H, W] or [B, T, M, H, W]
# Notes: This is a skeleton with practical choices — tune widths/heads/blocks per GPU budget.

import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# --------------------
# Utilities
# --------------------
def exists(x): return x is not None

def default(val, d):
    return val if exists(val) else d

def nanstd(x, dim, keepdim=False):
    """Compute standard deviation ignoring NaN values (compatible with older PyTorch)."""
    x_np = x.cpu().detach().numpy()
    result = np.nanstd(x_np, axis=dim, keepdims=keepdim)
    result = torch.from_numpy(result).to(x.device).to(x.dtype)
    return result

def safe_nanmean(x, dim, keepdim=False):
    """Compute mean ignoring NaN values."""
    x_np = x.cpu().detach().numpy()
    result = np.nanmean(x_np, axis=dim, keepdims=keepdim)
    result = torch.from_numpy(result).to(x.device).to(x.dtype)
    return result

def safe_nanmin(x, dim, keepdim=False):
    """Compute min ignoring NaN values."""
    x_np = x.cpu().detach().numpy()
    result = np.nanmin(x_np, axis=dim, keepdims=keepdim)
    result = torch.from_numpy(result).to(x.device).to(x.dtype)
    return result

def safe_nanmax(x, dim, keepdim=False):
    """Compute max ignoring NaN values."""
    x_np = x.cpu().detach().numpy()
    result = np.nanmax(x_np, axis=dim, keepdims=keepdim)
    result = torch.from_numpy(result).to(x.device).to(x.dtype)
    return result

# small ConvLSTM cell (used optionally)
class ConvLSTMCell(nn.Module):
    def __init__(self, in_ch, hidden_ch, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_ch + hidden_ch, 4 * hidden_ch, kernel_size, padding=padding)
        self.hidden_ch = hidden_ch

    def forward(self, x, h, c):
        # x, h, c: [B, C, H, W]
        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)
        i, f, o, g = gates.chunk(4, dim=1)
        i = torch.sigmoid(i); f = torch.sigmoid(f); o = torch.sigmoid(o); g = torch.tanh(g)
        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)
        return h_new, c_new

class ConvLSTM(nn.Module):
    def __init__(self, in_ch, hidden_ch, kernel_size=3):
        super().__init__()
        self.cell = ConvLSTMCell(in_ch, hidden_ch, kernel_size)

    def forward(self, seq):  # seq: [B, T, C, H, W]
        B, T, C, H, W = seq.shape
        h = torch.zeros(B, self.cell.hidden_ch, H, W, device=seq.device)
        c = torch.zeros_like(h)
        outs = []
        for t in range(T):
            h, c = self.cell(seq[:, t], h, c)
            outs.append(h)
        out = torch.stack(outs, dim=1)  # [B, T, hidden, H, W]
        return out

# --------------------
# FiLM block (per-channel)
# --------------------
class FiLMConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, groups=8):
        super().__init__()
        padding = kernel // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, padding=padding)
        self.norm = nn.GroupNorm(default(groups, 8), out_ch)
        self.act = nn.GELU()

    def forward(self, x, gamma=None, beta=None):
        x = self.conv(x)
        x = self.norm(x)
        # gamma, beta shapes: [B, out_ch] or None
        if exists(gamma) and exists(beta):
            # reshape to [B, out_ch, 1, 1] and broadcast
            g = gamma.unsqueeze(-1).unsqueeze(-1)
            b = beta.unsqueeze(-1).unsqueeze(-1)
            x = g * x + b
        x = self.act(x)
        return x

# --------------------
# Small UNet encoder/decoder blocks (shared across time)
# --------------------
class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            FiLMConvBlock(in_ch, out_ch),
            FiLMConvBlock(out_ch, out_ch)
        )
        self.pool = nn.AvgPool2d(2)

    def forward(self, x, gamma=None, beta=None):
        # gamma/beta can be lists per layer; but for simplicity we accept block-level vectors
        x = self.block[0](x, gamma=default(gamma, None), beta=default(beta, None))
        x = self.block[1](x, gamma=default(gamma, None), beta=default(beta, None))
        x_down = self.pool(x)
        return x, x_down

class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.block = nn.Sequential(
            FiLMConvBlock(in_ch, out_ch),
            FiLMConvBlock(out_ch, out_ch)
        )

    def forward(self, x, skip, gamma=None, beta=None):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.block[0](x, gamma=default(gamma, None), beta=default(beta, None))
        x = self.block[1](x, gamma=default(gamma, None), beta=default(beta, None))
        return x

# --------------------
# Simple FNO-like block (low-mode spectral conv)
# Implementation note: For readability we implement a simple spectral-mixing block using FFT on last 2 dims.
# --------------------
class SimpleFNO2D(nn.Module):
    def __init__(self, channels, modes_height=12, modes_width=12):
        super().__init__()
        self.channels = channels
        self.modes_h = modes_height
        self.modes_w = modes_width
        # complex weights for low modes
        self.scale = 1 / (channels * channels)
        self.weights = nn.Parameter(self.scale * torch.randn(channels, channels, self.modes_h, self.modes_w, 2))

        self.w0 = nn.Conv2d(channels, channels, kernel_size=1)

    def compl_mul2d(self, input, weights):
        # input: (..., 2) real/imag last dim
        # weights: (...,2)
        # complex multiply (a+ib)*(c+id) = (ac-bd) + i(ad+bc)
        r1, i1 = input[..., 0], input[..., 1]
        r2, i2 = weights[..., 0], weights[..., 1]
        return torch.stack([r1 * r2 - i1 * i2, r1 * i2 + i1 * r2], dim=-1)

    def forward(self, x):
        """
        x: [B, C, H, W]
        """
        B, C, H, W = x.shape
        # real fft
        x_ft = torch.fft.rfft2(x, norm='ortho')
        # rfft returns complex dtype; convert to real-imag last-dim tensor
        # select low modes
        out_ft = torch.zeros(B, C, x_ft.size(-2), x_ft.size(-1), device=x.device, dtype=torch.cfloat)
        # apply weights on low modes (careful shapes)
        h_lim = min(self.modes_h, x_ft.size(-2))
        w_lim = min(self.modes_w, x_ft.size(-1))
        # naive loop (could be improved for speed), but clarity prioritized
        # apply weight as linear transform across channels for low modes
        for i in range(h_lim):
            for j in range(w_lim):
                # x_ft[:, :, i, j]: [B, C] complex
                # weights: [C, C] complex
                w = torch.view_as_complex(self.weights[:, :, i, j].to(x.device))
                # matrix multiply across channel dim
                # x_ft at mode: [B, C]
                out_ft[:, :, i, j] = (x_ft[:, :, i, j].unsqueeze(1) * w.unsqueeze(0)).sum(dim=2)
        # inverse fft
        x_ifft = torch.fft.irfft2(out_ft, s=(H, W), norm='ortho')
        x = self.w0(x) + x_ifft
        return x

# --------------------
# Temporal Transformer (time-axis attention)
# We'll take bottleneck spatial features, flatten spatial dims to tokens or pool to patches.
# Simpler: global average pool spatial dims -> sequence of length T with token dim D
# --------------------
class TemporalTransformer(nn.Module):
    def __init__(self, token_dim, n_layers=2, n_heads=4, mlp_dim=512, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=n_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, tokens):  # tokens: [B, T, D]
        return self.transformer(tokens)  # [B, T, D]


class TemporalConvFusion(nn.Module):
    """Lightweight temporal interaction over lead axis using depthwise separable Conv1D."""
    def __init__(self, token_dim, kernel_size=3, dropout=0.1):
        super().__init__()
        padding = kernel_size // 2
        self.dw = nn.Conv1d(token_dim, token_dim, kernel_size, padding=padding, groups=token_dim)
        self.pw = nn.Conv1d(token_dim, token_dim, kernel_size=1)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(token_dim)

    def forward(self, tokens):  # [B,T,D]
        x = tokens.transpose(1, 2)  # [B,D,T]
        x = self.dw(x)
        x = self.act(x)
        x = self.pw(x)
        x = self.drop(x)
        x = x.transpose(1, 2)  # [B,T,D]
        return self.norm(tokens + x)

# --------------------
# IndexEncoder: map SST PCs + lead embedding to per-block per-channel FiLM vectors (gamma/beta)
# Accepts sst either [B, M, H, W] or [B, T, M, H, W]. If only [B, M, H, W], we optionally broadcast.
# We include optional PCA transform inside; here we implement a small MLP that ingests flattened PCs (user can replace with EOF PCs)
# --------------------
class IndexEncoder(nn.Module):
    def __init__(self, in_dim, hidden=256, film_channels_per_block=[32,64,128,128], lead_embed_dim=16):
        """
        in_dim: M * (pc_num) or M*H*W flattened feature count (we assume user passes compact PCs)
        film_channels_per_block: list matches UNet block output channels -> produce gamma/beta per block
        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim + lead_embed_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU()
        )
        self.lead_embed = nn.Embedding(32, lead_embed_dim)  # up to 32 leads
        # per-block projections
        self.gamma_projs = nn.ModuleList([nn.Linear(hidden, ch) for ch in film_channels_per_block])
        self.beta_projs = nn.ModuleList([nn.Linear(hidden, ch) for ch in film_channels_per_block])

    def forward(self, pcs, lead_idx):
        """
        pcs: [B, T, in_dim] OR [B, in_dim] if broadcasted
        lead_idx: integer lead index 0..T-1 OR tensor [B, T] with lead ids
        returns per block gamma/beta lists: each is [B, T, ch]
        """
        B = pcs.shape[0]
        if pcs.dim() == 2:
            pcs = pcs.unsqueeze(1)  # [B,1,in_dim] broadcastable
        B, T, D = pcs.shape
        # lead embedding: form tensor [T, lead_embed_dim] then expand to [B,T,lead_embed_dim]
        if isinstance(lead_idx, int):
            lead_idx = torch.arange(T, device=pcs.device).unsqueeze(0).repeat(B,1)  # [B,T]
        if lead_idx.dim()==1:
            lead_idx = lead_idx.unsqueeze(0).repeat(B,1)
        lead_emb = self.lead_embed(lead_idx)  # [B,T,lead_embed_dim]
        x = torch.cat([pcs, lead_emb], dim=-1)  # [B,T, D+lead_dim]
        # apply mlp per token
        B, T, L = x.shape
        x_flat = x.reshape(B*T, L)
        h = self.mlp(x_flat)  # [B*T, hidden]
        gammas = []
        betas = []
        for gproj, bproj in zip(self.gamma_projs, self.beta_projs):
            g = gproj(h).reshape(B, T, -1)
            b = bproj(h).reshape(B, T, -1)
            gammas.append(g)
            betas.append(b)
        return gammas, betas  # lists of length num_blocks, each [B,T,ch]

# --------------------
# LeadAdapter (Mixture of Experts small conv stacks) — returns correction map
# --------------------
class LeadAdapterMoE(nn.Module):
    def __init__(self, in_ch, out_ch, n_experts=4):
        super().__init__()
        self.experts = nn.ModuleList([nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.GroupNorm(8, out_ch), nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.GroupNorm(8, out_ch), nn.GELU()
        ) for _ in range(n_experts)])
        self.gate = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(in_ch, n_experts))

    def forward(self, x):
        # x: [B, in_ch, H, W]
        B = x.shape[0]
        gate_logits = self.gate(x)  # [B, n_experts]
        gate = torch.softmax(gate_logits, dim=-1)
        outs = []
        for i, expert in enumerate(self.experts):
            outs.append(expert(x))
        # stack experts: [n_experts, B, out_ch, H, W] -> combine
        stacked = torch.stack(outs, dim=0)  # [E, B, C, H, W]
        # weight combine per batch: compute weighted sum
        gate = gate.permute(1,0).unsqueeze(2).unsqueeze(3).unsqueeze(4)  # [E,B,1,1,1]
        combined = (stacked * gate).sum(dim=0)
        return combined

# --------------------
# Probabilistic head: zero-inflated Gamma param output
# Input: decoder feature map [B, ch, H, W] per time slice (we will apply per time slice)
# Output: p0 (prob zero), alpha, beta for Gamma; predictions shape [B, 1, H, W]
# --------------------
class ProbabilisticHead(nn.Module):
    def __init__(self, in_ch, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1),
            nn.GroupNorm(8, hidden),
            nn.GELU()
        )
        # heads
        self.logit_p0 = nn.Conv2d(hidden, 1, 1)  # logits for zero-inflation
        self.log_alpha = nn.Conv2d(hidden, 1, 1)
        self.log_beta = nn.Conv2d(hidden, 1, 1)
        # deterministic mean head (optional)
        self.delta = nn.Conv2d(hidden, 1, 1)

    def forward(self, feat):
        # feat: [B, in_ch, H, W]
        h = self.net(feat)
        p0 = torch.sigmoid(self.logit_p0(h))  # [B,1,H,W] in (0,1)
        alpha = F.softplus(self.log_alpha(h)) + 1e-6
        beta = F.softplus(self.log_beta(h)) + 1e-6
        mu_delta = self.delta(h)  # residual mean
        return {'p0': p0, 'alpha': alpha, 'beta': beta, 'delta': mu_delta}

# --------------------
# Main architecture: SpatioTemporalCorrector
# --------------------
class SpatioTemporalCorrector(nn.Module):
    def __init__(self,
                 in_ch=10,
                 cond_time=6,
                 sst_m=6,
                 pc_dim: Optional[int] = None,
                 base_channels=32,
                 film_channels=None,
                 use_convlstm=True,
                 use_fno=True,
                 temporal_mode='transformer',
                 use_lead_adapter=True,
                 index_hidden=256,
                 fno_modes=(12,12),
                 transformer_dim=256,
                 transformer_layers=2,
                 n_experts=4):
        super().__init__()
        self.T = cond_time
        self.in_ch = in_ch
        self.base_channels = base_channels
        if film_channels is None:
            film_channels = [base_channels, base_channels * 2, base_channels * 4, base_channels * 4]
        if len(film_channels) != 4:
            raise ValueError(f"film_channels must have length 4, got {film_channels}")
        # Shared UNet encoder (weights shared across T)
        chs = film_channels
        self.enc0 = DownBlock(in_ch, chs[0])
        self.enc1 = DownBlock(chs[0], chs[1])
        self.enc2 = DownBlock(chs[1], chs[2])
        self.enc3 = DownBlock(chs[2], chs[3])

        # Bottleneck conv
        self.bottleneck_conv = nn.Conv2d(chs[3], chs[3], 3, padding=1)

        # FNO block (optional due to high compute cost)
        self.use_fno = use_fno
        self.fno = SimpleFNO2D(chs[3], modes_height=fno_modes[0], modes_width=fno_modes[1]) if use_fno else nn.Identity()

        # Temporal modules: pool spatial to token dim and transformer
        self.pool = nn.AdaptiveAvgPool2d(1)
        token_dim = chs[3]
        self.temporal_mode = temporal_mode
        if temporal_mode == 'transformer':
            self.temporal_transformer = TemporalTransformer(token_dim, n_layers=transformer_layers, n_heads=4, mlp_dim=token_dim*2)
        elif temporal_mode == 'conv':
            self.temporal_transformer = TemporalConvFusion(token_dim, kernel_size=3, dropout=0.1)
        elif temporal_mode == 'none':
            self.temporal_transformer = nn.Identity()
        else:
            raise ValueError(f"Unknown temporal_mode: {temporal_mode}")

        # optional ConvLSTM to recover temporal-spatial features
        self.use_convlstm = use_convlstm
        if use_convlstm:
            self.convlstm = ConvLSTM(in_ch=chs[3], hidden_ch=chs[3])

        # IndexEncoder (SST PCs + lead embedding) -> per-channel gammas/betas
        # If pc_dim is not given, fallback to simple SST stats size: sst_m * 4
        if pc_dim is None:
            pc_dim = sst_m * 4
        self.pc_dim = pc_dim
        self.index_encoder = IndexEncoder(
            in_dim=self.pc_dim,
            hidden=index_hidden,
            film_channels_per_block=film_channels,
            lead_embed_dim=16,
        )

        # Decoder (UNet-style) with FiLM injection per block
        self.up3 = UpBlock(chs[3]*2, chs[2])
        self.up2 = UpBlock(chs[2]*2, chs[1])
        self.up1 = UpBlock(chs[1]*2, chs[0])
        self.up0 = UpBlock(chs[0]*2, chs[0])
        self.final_conv = nn.Conv2d(chs[0], chs[0], 3, padding=1)

        # Lead adapter and probabilistic head (apply per time slice)
        self.lead_adapter = LeadAdapterMoE(in_ch=chs[0], out_ch=chs[0], n_experts=n_experts) if use_lead_adapter else nn.Identity()
        self.prob_head = ProbabilisticHead(in_ch=chs[0])

    def forward(self, cond, sst=None, sst_pcs=None):
        """
        cond: [B, C, T, H, W]
        sst: [B, M, H, W] or [B, T, M, H, W]
        sst_pcs: optionally [B, T, pc_dim] precomputed; if None we will pool sst to make a small descriptor
        returns dict with keys:
            'p0','alpha','beta','delta' each [B,1,T,H,W]
        """
        B, C, T, H, W = cond.shape
        assert T == self.T, "cond time mismatch"

        device = cond.device

        # Prepare sst_pcs if not given: simple global pooling + linear projection to pc_dim
        if sst_pcs is None:
            if sst is None:
                raise ValueError("Either sst_pcs or sst must be provided.")
            if sst.dim() == 4:  # [B, M, H, W] -> broadcast
                # compute simple stats per month: global mean + std + min + max => M*4 dims
                sst_stats = []
                for m in range(sst.size(1)):
                    s = sst[:, m:m+1]  # [B,1,H,W]
                    sst_stats.append(safe_nanmean(s.view(B, -1), dim=1, keepdim=True))
                    sst_stats.append(nanstd(s.view(B, -1), dim=1, keepdim=True))
                    sst_stats.append(safe_nanmin(s.view(B, -1), dim=1, keepdim=True))
                    sst_stats.append(safe_nanmax(s.view(B, -1), dim=1, keepdim=True))
                # stats list length M*4 -> stack -> [B, M*4]
                pcs_base = torch.cat(sst_stats, dim=1)  # [B, M*4]
                # broadcast to T tokens
                sst_pcs = pcs_base.unsqueeze(1).repeat(1, T, 1)  # [B,T,pc_dim]
            elif sst.dim() == 5:  # [B, T, M, H, W] -> per lead stats
                sst_stats = []
                B, TT, M, HH, WW = sst.shape
                stats = []
                for t in range(TT):
                    cur = sst[:, t]  # [B, M, H, W]
                    stats_t = []
                    for m in range(M):
                        s = cur[:, m:m+1]
                        stats_t.append(safe_nanmean(s.view(B, -1), dim=1, keepdim=True))
                        stats_t.append(nanstd(s.view(B, -1), dim=1, keepdim=True))
                        stats_t.append(safe_nanmin(s.view(B, -1), dim=1, keepdim=True))
                        stats_t.append(safe_nanmax(s.view(B, -1), dim=1, keepdim=True))
                    stats_t = torch.cat(stats_t, dim=1)  # [B, M*4]
                    sst_stats.append(stats_t.unsqueeze(1))
                sst_pcs = torch.cat(sst_stats, dim=1)  # [B,T,pc_dim]
            else:
                raise ValueError("sst dims not understood")

        # Spatial encoding per time slice (shared weights)
        # cond: [B, C, T, H, W] -> iterate time slices or vectorize
        cond_ts = cond.permute(0, 2, 1, 3, 4)  # [B, T, C, H, W]
        enc_feats = []
        skips0 = []; skips1 = []; skips2 = []; bottlenecks = []
        for t in range(T):
            x = cond_ts[:, t]  # [B, C, H, W]
            s0, x = self.enc0(x)
            s1, x = self.enc1(x)
            s2, x = self.enc2(x)
            s3, x = self.enc3(x)
            bott = self.bottleneck_conv(x)
            # Keep FFT path in FP32 under autocast to avoid half-precision cuFFT issues.
            if self.use_fno and bott.is_cuda and torch.is_autocast_enabled():
                with torch.cuda.amp.autocast(enabled=False):
                    bott = self.fno(bott.float())
                bott = bott.to(x.dtype)
            else:
                bott = self.fno(bott)
            enc_feats.append(bott)
            skips0.append(s0); skips1.append(s1); skips2.append(s2); bottlenecks.append(s3)

        # Temporal fusion
        # pool spatial to tokens
        tokens = torch.stack([self.pool(b).squeeze(-1).squeeze(-1) for b in enc_feats], dim=1)  # [B, T, D]
        tokens_out = self.temporal_transformer(tokens)  # [B,T,D]

        # optionally expand tokens_out back to spatial via broadcasting
        # combine tokens with bottleneck features
        bott_stack = torch.stack(enc_feats, dim=1)  # [B, T, D, H', W']
        # fuse tokens (channel-wise scale)
        D = tokens_out.shape[-1]
        tokens_scale = tokens_out.unsqueeze(-1).unsqueeze(-1)  # [B,T,D,1,1]
        bott_fused = bott_stack * tokens_scale  # broadcasting

        # optional ConvLSTM to add local temporal dynamics
        if self.use_convlstm:
            # shape needed [B, T, C, H, W]
            convlstm_in = bott_fused  # [B,T,C,H',W']
            convlstm_out = self.convlstm(convlstm_in)  # [B,T,C,H',W']
            bott_fused = convlstm_out

        # IndexEncoder -> per-block gamma/beta
        # sst_pcs: [B,T,pc_dim] ; lead_idx: tensor 0..T-1
        if sst_pcs.dim() != 3:
            raise ValueError(f"sst_pcs must be 3D [B,T,pc_dim], got shape {tuple(sst_pcs.shape)}")
        if sst_pcs.shape[1] != T:
            raise ValueError(f"sst_pcs time dimension mismatch: expected T={T}, got {sst_pcs.shape[1]}")
        if sst_pcs.shape[2] != self.pc_dim:
            raise ValueError(
                f"sst_pcs feature dim mismatch: model expects pc_dim={self.pc_dim}, got {sst_pcs.shape[2]}. "
                f"Build model with pc_dim={sst_pcs.shape[2]} or pass matching pcs."
            )
        lead_idx = torch.arange(self.T, device=cond.device)
        gammas, betas = self.index_encoder(sst_pcs, lead_idx)

        # decode per time slice
        p0_list = []; alpha_list = []; beta_list = []; delta_list = []
        for t in range(T):
            b = bott_fused[:, t]  # [B, D, H', W']
            # decoder with FiLM injections using gammas/betas lists
            # Note: gammas/betas are lists corresponding to enc blocks order; pick row t
            # up3 expects input channels D*2 (we will rebuild skip connections)
            s0 = skips0[t]; s1 = skips1[t]; s2 = skips2[t]; s3 = bottlenecks[t]
            # decoder step by step
            up = self.up3(b, s3, gamma=gammas[2][:, t], beta=betas[2][:, t])
            up = self.up2(up, s2, gamma=gammas[1][:, t], beta=betas[1][:, t])
            up = self.up1(up, s1, gamma=gammas[0][:, t], beta=betas[0][:, t])
            up = self.up0(up, s0)
            up = self.final_conv(up)
            # lead adapter
            adapted = self.lead_adapter(up)
            head_out = self.prob_head(adapted)
            # ensure shapes [B,1,H,W]
            p0_list.append(head_out['p0'])
            alpha_list.append(head_out['alpha'])
            beta_list.append(head_out['beta'])
            delta_list.append(head_out['delta'])

        # stack across time -> [B,1,T,H,W]
        p0 = torch.stack(p0_list, dim=2)
        alph = torch.stack(alpha_list, dim=2)
        bet = torch.stack(beta_list, dim=2)
        delt = torch.stack(delta_list, dim=2)
        return {'p0': p0, 'alpha': alph, 'beta': bet, 'delta': delt}

# --------------------
# Losses: zero-inflated gamma negative log-likelihood + approximate CRPS (by sampling)
# target (obs) shape: [B,1,T,H,W]  (anomaly values; note gamma is positive -> if anomalies negative we treat residual correction)
# We assume target is anomaly; for NLL we predict the correction (delta) added to raw model if needed.
# For simplicity: interpret network output as correction to raw forecast; but here we'll just model positive part as Gamma of residuals > 0.
# --------------------
def zero_inflated_gamma_nll(pred, target, eps=1e-6):
    """
    pred: dict with 'p0','alpha','beta','delta' each [B,1,T,H,W]
    target: [B,1,T,H,W] (may contain NaNs)
    """
    p0    = pred['p0'].clamp(1e-6, 1 - 1e-6)
    alpha = pred['alpha'].clamp(min=1e-6)
    beta  = pred['beta'].clamp(min=1e-6)

    # --------
    # NaN mask
    # --------
    valid_mask = ~torch.isnan(target)          # True where valid
    if valid_mask.sum() == 0:
        # 防止 batch 全 NaN
        return torch.tensor(0.0, device=target.device, requires_grad=True)

    # 只在有效点上计算
    y = torch.where(valid_mask, target, torch.zeros_like(target))
    y = y.clamp(min=0.0)

    # --------
    # log Gamma pdf
    # --------
    log_pdf = (
        (alpha - 1) * torch.log(y + eps)
        - y / beta
        - alpha * torch.log(beta + eps)
        - torch.lgamma(alpha)
    )

    ll_nonzero = torch.log(1.0 - p0 + eps) + log_pdf
    ll_zero    = torch.log(p0 + eps)

    is_zero = (y <= eps)
    ll = torch.where(is_zero, ll_zero, ll_nonzero)

    # --------
    # mask & normalize
    # --------
    ll = torch.where(valid_mask, ll, torch.zeros_like(ll))

    nll = -ll.sum() / valid_mask.sum()

    return nll

def approx_crps_by_sampling(pred, target, n_samples=64, eps=1e-6):
    """
    pred: dict with 'p0','alpha','beta'
    target: [B,1,T,H,W] (may contain NaNs)
    """
    p0    = pred['p0'].clamp(1e-6, 1 - 1e-6)
    alpha = pred['alpha'].clamp(min=1e-6)
    beta  = pred['beta'].clamp(min=1e-6)

    valid_mask = ~torch.isnan(target)
    if valid_mask.sum() == 0:
        return torch.tensor(0.0, device=target.device, requires_grad=True)

    y = torch.where(valid_mask, target, torch.zeros_like(target))
    y = y.clamp(min=0.0)

    # --------
    # Monte-Carlo CRPS approximation with O(S) pairing
    # CRPS(F,y)=E|X-y|-0.5E|X-X'|, where X and X' are i.i.d.
    # --------
    gamma_dist = torch.distributions.Gamma(alpha, 1.0 / beta)
    s1 = gamma_dist.sample((n_samples,))  # [S,B,1,T,H,W]
    s2 = gamma_dist.sample((n_samples,))  # [S,B,1,T,H,W]

    z1 = torch.rand_like(s1) < p0.unsqueeze(0)
    z2 = torch.rand_like(s2) < p0.unsqueeze(0)
    s1 = torch.where(z1, torch.zeros_like(s1), s1)
    s2 = torch.where(z2, torch.zeros_like(s2), s2)

    valid = valid_mask.unsqueeze(0)  # [1,B,1,T,H,W]
    n_valid = valid_mask.sum().clamp_min(1)

    ey = torch.where(valid, torch.abs(s1 - y.unsqueeze(0)), torch.zeros_like(s1)).sum()
    epair = torch.where(valid, torch.abs(s1 - s2), torch.zeros_like(s1)).sum()

    ey = ey / (n_samples * n_valid)
    epair = 0.5 * epair / (n_samples * n_valid)
    return ey - epair

# --------------------
# Example training step (sketch)
# --------------------
def training_step(model, batch, optimizer, device='cuda'):
    """
    batch: dict with keys 'cond','target','sst' (torch tensors)
    """
    model.train()
    cond = batch['cond'].to(device)
    target = batch['target'].to(device)
    sst = batch['sst'].to(device)
    pred = model(cond, sst)
    nll = zero_inflated_gamma_nll(pred, target)
    crps = approx_crps_by_sampling(pred, target, n_samples=16)  # smaller S for speed
    # auxiliary MSE on log(1+y) to stabilize training
    mse = F.mse_loss(torch.log1p(pred['delta']), torch.log1p(target))
    loss = nll + 0.1 * crps + 0.5 * mse
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return {'loss': loss.item(), 'nll': nll.item(), 'crps': crps.item(), 'mse': mse.item()}

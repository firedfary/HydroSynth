"""Conditional 2D UNet with Channel Concatenation and Adaptive Group Normalization (AdaGN).

Designed specifically for meteorological & hydrological spatial downscaling:
- Spatial alignment via direct channel concatenation (x_t concatenated with condition y).
- Sinusoidal timestep embedding with MLP projection.
- AdaGN (Scale & Shift) modulation inside ResBlocks for timestep conditioning.
- Residual skip connections with GroupNorm and SiLU.
"""

from __future__ import annotations

import math
from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionalEmbedding(nn.Module):
    """Sinusoidal positional embedding for diffusion timesteps."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timesteps: 1D tensor of timesteps [B].
        Returns:
            Tensor of shape [B, dim].
        """
        device = timesteps.device
        half_dim = self.dim // 2
        emb_scale = math.log(10000) / (half_dim - 1)
        freqs = torch.exp(-emb_scale * torch.arange(half_dim, device=device).float())
        args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class ResBlock(nn.Module):
    """Residual Block with AdaGN (scale and shift modulation from timestep embedding)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embed_dim: int,
        dropout: float = 0.0,
        num_groups: int = 8,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Adjust groups if channels are smaller than num_groups
        groups1 = min(num_groups, in_channels)
        while in_channels % groups1 != 0 and groups1 > 1:
            groups1 -= 1

        groups2 = min(num_groups, out_channels)
        while out_channels % groups2 != 0 and groups2 > 1:
            groups2 -= 1

        self.norm1 = nn.GroupNorm(groups1, in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Time embedding projection to scale and shift
        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embed_dim, out_channels * 2),
        )

        self.norm2 = nn.GroupNorm(groups2, out_channels)
        self.act2 = nn.SiLU()
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Feature map [B, in_channels, H, W]
            time_emb: Time embedding [B, time_embed_dim]
        Returns:
            Output feature map [B, out_channels, H, W]
        """
        h = self.conv1(self.act1(self.norm1(x)))

        # AdaGN modulation: scale and shift
        scale_shift = self.time_proj(time_emb)  # [B, 2 * out_channels]
        scale, shift = scale_shift.chunk(2, dim=1)
        scale = scale.unsqueeze(-1).unsqueeze(-1)
        shift = shift.unsqueeze(-1).unsqueeze(-1)

        h = self.norm2(h) * (1.0 + scale) + shift
        h = self.conv2(self.dropout(self.act2(h)))

        return h + self.shortcut(x)


class Downsample(nn.Module):
    """Spatial downsampling layer using stride-2 convolution."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """Spatial upsampling layer using nearest interpolation followed by 3x3 conv."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, target_shape: Tuple[int, int] | None = None) -> torch.Tensor:
        if target_shape is not None:
            x = F.interpolate(x, size=target_shape, mode="nearest")
        else:
            x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class ConditionalUNet(nn.Module):
    """
    2D Conditional UNet for Meteorological & Hydrological Downscaling.

    Concatenates target noisy image (1 channel) and driving conditions (C_cond channels)
    at the input layer. Timestep is injected via AdaGN in every ResBlock.
    """

    def __init__(
        self,
        in_channels: int = 1,
        condition_channels: int = 10,
        out_channels: int = 1,
        base_channels: int = 64,
        channel_multipliers: Sequence[int] = (1, 2, 4),
        num_res_blocks: int = 2,
        dropout: float = 0.0,
        num_groups: int = 8,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.condition_channels = condition_channels
        self.total_in_channels = in_channels + condition_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.channel_multipliers = list(channel_multipliers)
        self.num_res_blocks = num_res_blocks

        # Timestep embedding MLP
        time_embed_dim = base_channels * 4
        self.time_embedding = nn.Sequential(
            SinusoidalPositionalEmbedding(base_channels),
            nn.Linear(base_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # Initial convolution
        self.head = nn.Conv2d(self.total_in_channels, base_channels, kernel_size=3, padding=1)

        # Encoder (Down path)
        self.down_stages = nn.ModuleList()
        current_ch = base_channels
        channel_history = []

        for stage_idx, mult in enumerate(self.channel_multipliers):
            out_ch = base_channels * mult
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(
                    ResBlock(
                        in_channels=current_ch,
                        out_channels=out_ch,
                        time_embed_dim=time_embed_dim,
                        dropout=dropout,
                        num_groups=num_groups,
                    )
                )
                current_ch = out_ch
                channel_history.append(current_ch)

            downsample = Downsample(current_ch) if stage_idx != len(self.channel_multipliers) - 1 else None
            self.down_stages.append(nn.ModuleDict({"blocks": blocks, "downsample": downsample}))

        # Middle bottleneck
        self.mid_block1 = ResBlock(current_ch, current_ch, time_embed_dim, dropout, num_groups)
        self.mid_block2 = ResBlock(current_ch, current_ch, time_embed_dim, dropout, num_groups)

        # Decoder (Up path)
        self.up_stages = nn.ModuleList()
        for stage_idx, mult in reversed(list(enumerate(self.channel_multipliers))):
            out_ch = base_channels * mult
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                skip_ch = channel_history.pop()
                blocks.append(
                    ResBlock(
                        in_channels=current_ch + skip_ch,
                        out_channels=out_ch,
                        time_embed_dim=time_embed_dim,
                        dropout=dropout,
                        num_groups=num_groups,
                    )
                )
                current_ch = out_ch

            upsample = Upsample(current_ch) if stage_idx != 0 else None
            self.up_stages.append(nn.ModuleDict({"blocks": blocks, "upsample": upsample}))

        assert len(channel_history) == 0, f"channel_history should be empty, but has {len(channel_history)}"

        # Final output projection
        tail_groups = min(num_groups, current_ch)
        while current_ch % tail_groups != 0 and tail_groups > 1:
            tail_groups -= 1

        self.tail = nn.Sequential(
            nn.GroupNorm(tail_groups, current_ch),
            nn.SiLU(),
            nn.Conv2d(current_ch, out_channels, kernel_size=3, padding=1),
        )

        # Zero-initialize the final convolution weights for stable initial training
        nn.init.zeros_(self.tail[-1].weight)
        nn.init.zeros_(self.tail[-1].bias)

    def forward(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x_t: Noisy high-resolution target [B, in_channels, H, W]
            timesteps: Timestep indices [B]
            condition: Low-resolution / multi-channel driving condition [B, condition_channels, H, W]
        Returns:
            Predicted noise epsilon [B, out_channels, H, W]
        """
        # Ensure condition matches spatial resolution of x_t if needed
        if condition.shape[-2:] != x_t.shape[-2:]:
            condition = F.interpolate(condition, size=x_t.shape[-2:], mode="bilinear", align_corners=False)

        # Channel concatenation: [B, in_ch + cond_ch, H, W]
        h = torch.cat([x_t, condition], dim=1)

        # Compute time embedding: [B, time_embed_dim]
        time_emb = self.time_embedding(timesteps)

        h = self.head(h)
        skips = []
        spatial_shapes = []

        # Down path
        for stage in self.down_stages:
            for block in stage["blocks"]:
                h = block(h, time_emb)
                skips.append(h)
            if stage["downsample"] is not None:
                spatial_shapes.append(h.shape[-2:])
                h = stage["downsample"](h)

        # Middle bottleneck
        h = self.mid_block1(h, time_emb)
        h = self.mid_block2(h, time_emb)

        # Up path
        for stage in self.up_stages:
            for block in stage["blocks"]:
                skip = skips.pop()
                # Align spatial shapes if odd dimensions caused rounding mismatch
                if h.shape[-2:] != skip.shape[-2:]:
                    h = F.interpolate(h, size=skip.shape[-2:], mode="nearest")
                h = torch.cat([h, skip], dim=1)
                h = block(h, time_emb)
            if stage["upsample"] is not None:
                target_shape = spatial_shapes.pop() if spatial_shapes else None
                h = stage["upsample"](h, target_shape=target_shape)

        assert len(skips) == 0, f"All skips should be consumed, but {len(skips)} remain"
        return self.tail(h)

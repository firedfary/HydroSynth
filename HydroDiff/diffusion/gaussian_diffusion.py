"""Gaussian Diffusion Engine for Conditional Downscaling.

Implements:
- Cosine beta schedule (Nichol & Dhariwal 2021).
- Closed-form forward diffusion q(x_t | x_0).
- Masked training loss calculation.
- Fast deterministic DDIM reverse sampling (Song et al., 2020) and standard DDPM sampling.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def betas_for_alpha_bar(num_diffusion_timesteps: int, max_beta: float = 0.999) -> torch.Tensor:
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].
    
    Contains an offset to prevent beta from being too small near t = 0.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        alpha_bar_t1 = math.cos((t1 + 0.008) / 1.008 * math.pi / 2) ** 2
        alpha_bar_t2 = math.cos((t2 + 0.008) / 1.008 * math.pi / 2) ** 2
        beta = min(1.0 - alpha_bar_t2 / alpha_bar_t1, max_beta)
        betas.append(beta)
    return torch.tensor(betas, dtype=torch.float32)


def extract(tensor: torch.Tensor, timesteps: torch.Tensor, broadcast_shape: Tuple[int, ...]) -> torch.Tensor:
    """Extract coefficients at specified timesteps and reshape for broadcasting."""
    res = tensor.to(timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res.unsqueeze(-1)
    return res


class GaussianDiffusion(nn.Module):
    """
    Gaussian Diffusion process wrapper for training and sampling.
    """

    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_schedule: str = "cosine",
        prediction_type: str = "epsilon",
    ):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.prediction_type = prediction_type

        if beta_schedule == "cosine":
            betas = betas_for_alpha_bar(num_timesteps)
        elif beta_schedule == "linear":
            betas = torch.linspace(1e-4, 0.02, num_timesteps, dtype=torch.float32)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # Register buffers for forward process
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1.0))

        # Posterior variance for DDPM
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.log(torch.clamp(posterior_variance, min=1e-20)),
        )

    def q_sample(
        self,
        x_0: torch.Tensor,
        timesteps: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward diffusion q(x_t | x_0): diffuse target to timestep t."""
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alpha_bar = extract(self.sqrt_alphas_cumprod, timesteps, x_0.shape)
        sqrt_one_minus_alpha_bar = extract(self.sqrt_one_minus_alphas_cumprod, timesteps, x_0.shape)

        return sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise

    def compute_loss(
        self,
        model: nn.Module,
        x_0: torch.Tensor,
        condition: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute masked diffusion training loss.
        
        Args:
            model: Conditional UNet
            x_0: Clean normalized target [B, 1, H, W]
            condition: Driving condition [B, C_cond, H, W]
            mask: Spatial valid mask [B, 1, H, W] (1 for valid station/land, 0 for invalid)
        """
        batch_size = x_0.shape[0]
        timesteps = torch.randint(0, self.num_timesteps, (batch_size,), device=x_0.device).long()
        noise = torch.randn_like(x_0)

        # Forward diffuse
        x_t = self.q_sample(x_0=x_0, timesteps=timesteps, noise=noise)

        # Predict noise
        pred_noise = model(x_t, timesteps, condition)

        if self.prediction_type == "epsilon":
            target = noise
            loss_raw = (pred_noise - target) ** 2
        elif self.prediction_type == "sample":
            target = x_0
            loss_raw = (pred_noise - target) ** 2
        else:
            raise NotImplementedError(f"Prediction type {self.prediction_type} not supported")

        if mask is not None:
            mask_expanded = mask.float().expand_as(loss_raw)
            loss = (loss_raw * mask_expanded).sum() / (mask_expanded.sum() + 1e-8)
        else:
            loss = loss_raw.mean()

        metrics = {
            "loss": loss.item(),
            "pred_mean": pred_noise.mean().item(),
            "pred_std": pred_noise.std().item(),
        }

        return loss, metrics

    @torch.no_grad()
    def predict_x0_from_eps(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        eps: torch.Tensor,
    ) -> torch.Tensor:
        """Invert diffusion equation: x_0 = (x_t - sqrt(1 - alpha_bar) * eps) / sqrt(alpha_bar)."""
        sqrt_recip_alpha = extract(self.sqrt_recip_alphas_cumprod, timesteps, x_t.shape)
        sqrt_recipm1_alpha = extract(self.sqrt_recipm1_alphas_cumprod, timesteps, x_t.shape)
        return sqrt_recip_alpha * x_t - sqrt_recipm1_alpha * eps

    @torch.no_grad()
    def ddim_sample(
        self,
        model: nn.Module,
        condition: torch.Tensor,
        shape: Tuple[int, ...],
        num_inference_steps: int = 50,
        eta: float = 0.0,
        generator: Optional[torch.Generator] = None,
        return_intermediates: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, list]:
        """
        Fast DDIM reverse sampling.
        
        Args:
            model: Conditional UNet
            condition: Driving condition [B, C_cond, H, W]
            shape: Shape of target [B, 1, H, W]
            num_inference_steps: Number of DDIM steps (e.g., 20~50)
            eta: Stochasticity parameter (0.0 = deterministic DDIM)
            generator: PyTorch random generator for reproducibility
            return_intermediates: Whether to return intermediate trajectory images
        """
        device = condition.device
        batch_size = shape[0]

        # Subsample timesteps uniformly across [0, num_timesteps - 1]
        step_ratio = self.num_timesteps // num_inference_steps
        timesteps = (np.arange(num_inference_steps) * step_ratio).round().astype(np.int64)
        timesteps = list(reversed(timesteps))

        # Initial pure Gaussian noise
        x_t = torch.randn(shape, generator=generator, device=device).float()
        intermediates = [x_t.clone()]

        for i, t_idx in enumerate(timesteps):
            t = torch.full((batch_size,), t_idx, device=device, dtype=torch.long)
            prev_t_idx = timesteps[i + 1] if i + 1 < len(timesteps) else -1
            prev_t = torch.full((batch_size,), prev_t_idx, device=device, dtype=torch.long) if prev_t_idx >= 0 else None

            # Model prediction
            model_out = model(x_t, t, condition)

            alpha_bar_t = extract(self.alphas_cumprod, t, x_t.shape)
            alpha_bar_prev = extract(self.alphas_cumprod, prev_t, x_t.shape) if prev_t is not None else torch.ones_like(alpha_bar_t)

            if self.prediction_type == "epsilon":
                eps_pred = model_out
                pred_x0 = (x_t - torch.sqrt(1.0 - alpha_bar_t) * eps_pred) / torch.sqrt(alpha_bar_t)
            elif self.prediction_type == "sample":
                pred_x0 = model_out
                eps_pred = (x_t - torch.sqrt(alpha_bar_t) * pred_x0) / torch.sqrt(torch.clamp(1.0 - alpha_bar_t, min=1e-8))
            else:
                raise NotImplementedError(f"Prediction type {self.prediction_type} not supported")

            if prev_t_idx < 0:
                # Reached t=0
                x_t = pred_x0
            else:
                # DDIM step equation
                sigma = eta * torch.sqrt(
                    (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t) * (1.0 - alpha_bar_t / alpha_bar_prev)
                )
                pred_dir = torch.sqrt(torch.clamp(1.0 - alpha_bar_prev - sigma**2, min=0.0)) * eps_pred
                noise = torch.randn(x_t.shape, generator=generator, device=device) if eta > 0 else 0.0
                x_t = torch.sqrt(alpha_bar_prev) * pred_x0 + pred_dir + sigma * noise

            if return_intermediates:
                intermediates.append(x_t.clone())

        if return_intermediates:
            return x_t, intermediates
        return x_t

"""Step 1: Sanity Check & Overfitting Verification for HydroDiff.

Verifies:
1. Conditional UNet + Gaussian Diffusion training loop with channel concatenation.
2. Effective masked MSE loss calculation.
3. Stable training convergence on paired real meteorological samples.
4. Fast deterministic DDIM reverse sampling (50 steps).
5. Accurate reconstruction with ACC >= 0.95 and exact physical scale restoration.
6. Clean output saving to <HYDRO_WORKSPACE>/results/HydroDiff/sanity_check/.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add project root to sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from HydroDiff.data.dataset import PairedDownscalingDataset
from HydroDiff.diffusion.gaussian_diffusion import GaussianDiffusion
from HydroDiff.models.conditional_unet import ConditionalUNet
from HydroDiff.project_paths import experiment_dir, paths


def compute_acc(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    """Compute Anomaly Correlation Coefficient (ACC) over valid masked pixels."""
    valid = mask.astype(bool)
    p = pred[valid]
    t = target[valid]

    p_anom = p - np.mean(p)
    t_anom = t - np.mean(t)

    var_p = np.sum(p_anom**2)
    var_t = np.sum(t_anom**2)

    if var_p == 0 or var_t == 0:
        return 0.0

    acc = np.sum(p_anom * t_anom) / np.sqrt(var_p * var_t)
    return float(np.clip(acc, -1.0, 1.0))


def compute_rmse(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    """Compute Root Mean Squared Error (RMSE) over valid masked pixels."""
    valid = mask.astype(bool)
    diff = pred[valid] - target[valid]
    return float(np.sqrt(np.mean(diff**2)))


def run_sanity_check(
    total_steps: int = 2000,
    lr: float = 5e-4,
    device_str: str = "cuda:0" if torch.cuda.is_available() else "cpu",
    spatial_crop: tuple[int, int, int, int] = (16, 80, 32, 96),  # 64x64 dense observation area
    num_samples: int = 2,
    num_timesteps: int = 100,
    ddim_steps: int = 50,
    prediction_type: str = "sample",
) -> dict:
    device = torch.device(device_str)
    print(f"============================================================")
    print(f" HydroDiff Step 1: Sanity Check & Overfitting Verification ")
    print(f" Device: {device} | Total Steps: {total_steps} | LR: {lr}")
    print(f" Prediction Type: {prediction_type} | Timesteps T: {num_timesteps}")
    print(f" Spatial Crop: {spatial_crop} (64x64) | Samples: {num_samples}")
    print(f"============================================================")

    # Resolve managed output directories
    exp_dir = experiment_dir("sanity_check")
    ckpt_dir = exp_dir / "checkpoints"
    fig_dir = exp_dir / "figures"
    log_dir = exp_dir / "logs"

    # Locate source cache data
    u_net_cache = paths.workspace_root / "cache" / "U_Net_3D"
    hr_file = u_net_cache / "station_observations" / "hr_observations_ref1994_2010_aligned.npy"
    lr_file = u_net_cache / "prepared" / "lr_unet" / "lr_data_reconstructed2.npy"

    if not hr_file.exists():
        raise FileNotFoundError(f"High-resolution observation file not found: {hr_file}")
    if not lr_file.exists():
        raise FileNotFoundError(f"Low-resolution condition file not found: {lr_file}")

    print(f"Loading raw cached arrays...")
    hr_raw = np.load(hr_file, mmap_mode="r")[:num_samples]  # [N, 120, 140]
    lr_raw = np.load(lr_file, mmap_mode="r")[:num_samples]  # [N, 10, 120, 140]

    # Initialize paired dataset
    dataset = PairedDownscalingDataset(
        hr_data=hr_raw,
        lr_data=lr_raw,
        spatial_crop=spatial_crop,
    )
    stats = dataset.get_stats()
    print(f"Dataset stats: mean={stats['mean']:.4f}, std={stats['std']:.4f}, valid_ratio={stats['valid_ratio']*100:.1f}%")

    dataloader = DataLoader(dataset, batch_size=num_samples, shuffle=False)

    # Initialize model and diffusion engine
    model = ConditionalUNet(
        in_channels=1,
        condition_channels=10,
        out_channels=1,
        base_channels=64,
        channel_multipliers=(1, 2, 4),
        num_res_blocks=2,
        dropout=0.0,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"ConditionalUNet initialized: {total_params:,} trainable parameters.")

    diffusion = GaussianDiffusion(
        num_timesteps=num_timesteps,
        beta_schedule="cosine",
        prediction_type=prediction_type,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)

    # Training loop for overfitting test
    print(f"\nStarting overfitting training ({total_steps} gradient steps with {prediction_type}-prediction, T={num_timesteps})...")
    loss_history = []

    model.train()
    data_iter = iter(dataloader)

    for step in range(1, total_steps + 1):
        try:
            x_0, cond, mask = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            x_0, cond, mask = next(data_iter)

        x_0 = x_0.to(device)
        cond = cond.to(device)
        mask = mask.to(device)

        optimizer.zero_grad()
        loss, metrics = diffusion.compute_loss(model, x_0, cond, mask=mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        loss_history.append(loss.item())

        if step % 250 == 0 or step == 1 or step == total_steps:
            current_lr = scheduler.get_last_lr()[0]
            print(f"Step [{step:04d}/{total_steps:04d}] - Loss: {loss.item():.6f} | LR: {current_lr:.6e}")

    # Reverse Sampling with DDIM (50 steps)
    print("\nRunning DDIM reverse sampling (50 steps, eta=0.0)...")
    model.eval()

    sample_idx = 0
    x_0_norm, cond, mask = dataset[sample_idx]
    cond = cond.unsqueeze(0).to(device)
    mask_np = mask.squeeze().numpy()

    with torch.no_grad():
        x_recon_norm, intermediates = diffusion.ddim_sample(
            model=model,
            condition=cond,
            shape=(1, 1, 64, 64),
            num_inference_steps=ddim_steps,
            eta=0.0,
            return_intermediates=True,
        )

    # De-normalize to physical precipitation anomaly units
    x_recon_physical = dataset.denormalize(x_recon_norm.squeeze().cpu().numpy())
    x_gt_physical = dataset.denormalize(x_0_norm.squeeze().numpy())

    # Quantitative metrics
    final_acc = compute_acc(x_recon_physical, x_gt_physical, mask_np)
    final_rmse = compute_rmse(x_recon_physical, x_gt_physical, mask_np)

    gt_mean = float(np.mean(x_gt_physical[mask_np > 0]))
    gt_std = float(np.std(x_gt_physical[mask_np > 0]))
    recon_mean = float(np.mean(x_recon_physical[mask_np > 0]))
    recon_std = float(np.std(x_recon_physical[mask_np > 0]))

    print(f"\n============================================================")
    print(f" Overfitting & Reconstruction Verification Results ")
    print(f"============================================================")
    print(f" Final Loss (step {total_steps}) : {loss_history[-1]:.6f}")
    print(f" Reconstruction ACC          : {final_acc:.4f}  (Target: >= 0.95)")
    print(f" Reconstruction RMSE         : {final_rmse:.4f}")
    print(f" Ground Truth Range          : Mean={gt_mean:.4f}, Std={gt_std:.4f}")
    print(f" Reconstructed Range         : Mean={recon_mean:.4f}, Std={recon_std:.4f}")
    print(f" Amplitude Ratio (Recon/GT)  : {recon_std / gt_std:.4f}  (Target: ~1.00)")
    print(f"============================================================")

    # Save Checkpoint
    ckpt_path = ckpt_dir / "sanity_check_model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "total_steps": total_steps,
            "loss": loss_history[-1],
            "acc": final_acc,
            "rmse": final_rmse,
            "stats": stats,
        },
        ckpt_path,
    )
    print(f"Saved checkpoint to: {ckpt_path}")

    # Generate Visualization Figure
    fig, axs = plt.subplots(1, 5, figsize=(20, 4), dpi=150)

    # 1. Ground Truth
    gt_plot = np.where(mask_np > 0, x_gt_physical, np.nan)
    im0 = axs[0].imshow(gt_plot, cmap="RdBu_r")
    axs[0].set_title(f"Ground Truth Observation\n(Mean={gt_mean:.2f}, Std={gt_std:.2f})")
    plt.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

    # 2. Condition Channel 0 (Coarse Base Precip)
    cond_ch0 = cond[0, 0].cpu().numpy()
    im1 = axs[1].imshow(cond_ch0, cmap="Blues")
    axs[1].set_title("Driving Condition (Ch 0)\n(Coarse Base Field)")
    plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

    # 3. Intermediate state at t=T (Initial Gaussian Noise)
    init_noise = intermediates[0].squeeze().cpu().numpy()
    im2 = axs[2].imshow(init_noise, cmap="coolwarm")
    axs[2].set_title("Initial Pure Noise $x_T$\n(Standard Gaussian)")
    plt.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)

    # 4. DDIM Reconstructed
    recon_plot = np.where(mask_np > 0, x_recon_physical, np.nan)
    im3 = axs[3].imshow(recon_plot, cmap="RdBu_r")
    axs[3].set_title(f"DDIM Reconstructed $\\hat{{x}}_0$\n(ACC={final_acc:.4f}, RMSE={final_rmse:.4f})")
    plt.colorbar(im3, ax=axs[3], fraction=0.046, pad=0.04)

    # 5. Absolute Error Map
    error_map = np.where(mask_np > 0, np.abs(x_recon_physical - x_gt_physical), np.nan)
    im4 = axs[4].imshow(error_map, cmap="Reds")
    axs[4].set_title(f"Absolute Error Map\n(Max Err={np.nanmax(error_map):.4f})")
    plt.colorbar(im4, ax=axs[4], fraction=0.046, pad=0.04)

    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    fig_path = fig_dir / "sanity_check_reconstruction.png"
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()
    print(f"Saved reconstruction comparison figure to: {fig_path}")

    # Save Metrics JSON
    results = {
        "status": "PASSED" if final_acc >= 0.95 else "NEEDS_TUNING",
        "final_loss": loss_history[-1],
        "acc": final_acc,
        "rmse": final_rmse,
        "gt_mean": gt_mean,
        "gt_std": gt_std,
        "recon_mean": recon_mean,
        "recon_std": recon_std,
        "amplitude_ratio": recon_std / gt_std,
        "num_samples": num_samples,
        "total_steps": total_steps,
        "ddim_steps": ddim_steps,
        "num_timesteps": num_timesteps,
        "prediction_type": prediction_type,
        "ckpt_path": str(ckpt_path),
        "figure_path": str(fig_path),
    }

    log_path = log_dir / "sanity_check_metrics.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    print(f"Saved metrics summary to: {log_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HydroDiff Step 1 Sanity Check")
    parser.add_argument("--total_steps", type=int, default=2000, help="Number of overfitting gradient steps")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Device")
    parser.add_argument("--num_samples", type=int, default=2, help="Number of overfitting samples")
    parser.add_argument("--num_timesteps", type=int, default=100, help="Number of diffusion timesteps")
    parser.add_argument("--ddim_steps", type=int, default=50, help="Number of DDIM sampling steps")
    parser.add_argument("--prediction_type", type=str, default="sample", choices=["sample", "epsilon"], help="Prediction type")

    args = parser.parse_args()
    run_sanity_check(
        total_steps=args.total_steps,
        lr=args.lr,
        device_str=args.device,
        num_samples=args.num_samples,
        num_timesteps=args.num_timesteps,
        ddim_steps=args.ddim_steps,
        prediction_type=args.prediction_type,
    )

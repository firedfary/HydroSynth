"""Full-Domain Training and Validation Monitoring for HydroDiff.

Features:
1. Full-dataset training across 292 historical months (1994-2018).
2. Leakage-free normalization strictly calibrated on training set.
3. Full-domain padded grid (128x144) training with masked loss.
4. Periodic validation evaluation (every N epochs) using fast DDIM reverse sampling.
5. Tracking of best validation ACC and automatic early stopping.
6. Checkpoints and training curves saved to <HYDRO_WORKSPACE>/results/HydroDiff/full_run/.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from HydroDiff.data.dataset import create_spatiotemporal_splits
from HydroDiff.data.metrics import compute_spatial_metrics
from HydroDiff.diffusion.gaussian_diffusion import GaussianDiffusion
from HydroDiff.models.conditional_unet import ConditionalUNet
from HydroDiff.project_paths import experiment_dir, paths


def evaluate_validation(
    model: nn.Module,
    diffusion: GaussianDiffusion,
    val_loader: DataLoader,
    val_ds,
    transform,
    ddim_steps: int = 20,
    device: torch.device = torch.device("cuda:0"),
) -> dict:
    """Run DDIM reverse sampling across the validation split and compute spatial metrics."""
    model.eval()
    all_metrics = []

    with torch.no_grad():
        for b_x0, b_cond, b_mask in val_loader:
            b_size = b_x0.shape[0]
            b_cond = b_cond.to(device)

            recon_norm = diffusion.ddim_sample(
                model=model,
                condition=b_cond,
                shape=(b_size, 1, 128, 144),
                num_inference_steps=ddim_steps,
                eta=0.0,
            )

            recon_phys = val_ds.denormalize(recon_norm)  # [B, 1, 120, 140]
            gt_phys = val_ds.denormalize(b_x0)            # [B, 1, 120, 140]
            eval_mask = transform.unpad_tensor(b_mask).cpu().numpy()  # [B, 1, 120, 140]

            for i in range(b_size):
                m = compute_spatial_metrics(
                    pred=recon_phys[i, 0],
                    target=gt_phys[i, 0],
                    mask=eval_mask[i, 0],
                )
                all_metrics.append(m)

    # Average metrics over all validation samples
    keys = all_metrics[0].keys()
    avg_metrics = {k: float(np.mean([m[k] for m in all_metrics])) for k in keys}
    avg_metrics["acc_std"] = float(np.std([m["acc"] for m in all_metrics]))
    avg_metrics["num_val_samples"] = len(all_metrics)

    return avg_metrics


def train_full_model(
    epochs: int = 100,
    batch_size: int = 16,
    lr: float = 3e-4,
    base_channels: int = 32,
    num_timesteps: int = 100,
    ddim_val_steps: int = 20,
    val_every: int = 5,
    patience: int = 8,  # Stop if no val improvement for 8 validation checks (40 epochs)
    prediction_type: str = "sample",
    device_str: str = "cuda:0" if torch.cuda.is_available() else "cpu",
) -> dict:
    device = torch.device(device_str)
    print("============================================================")
    print(" HydroDiff: Full-Dataset Training & Validation Monitoring   ")
    print(f" Device: {device} | Epochs: {epochs} | Batch Size: {batch_size}")
    print(f" Base Channels: {base_channels} | Diffusion Steps: {num_timesteps}")
    print(f" Val Every: {val_every} epochs | DDIM Val Steps: {ddim_val_steps}")
    print("============================================================")

    # Setup managed output directories
    exp_dir = experiment_dir("full_run")
    ckpt_dir = exp_dir / "checkpoints"
    fig_dir = exp_dir / "figures"
    log_dir = exp_dir / "logs"

    # 1. Load data and create temporal splits
    u_net_cache = paths.workspace_root / "cache" / "U_Net_3D"
    hr_file = u_net_cache / "station_observations" / "hr_observations_ref1994_2010_aligned.npy"
    lr_file = u_net_cache / "prepared" / "lr_unet" / "lr_data_reconstructed2.npy"

    print("\n[1/4] Preparing temporal splits (80% Train, 10% Val, 10% Test)...")
    splits = create_spatiotemporal_splits(
        hr_file=hr_file,
        lr_file=lr_file,
        split_ratios=(0.8, 0.1, 0.1),
        pad_to=(128, 144),
        spatial_crop=None,
    )

    train_ds = splits["train"]
    val_ds = splits["val"]
    test_ds = splits["test"]
    stats = splits["stats"]
    transform = splits["transform"]

    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)} | Test samples: {len(test_ds)}")
    print(f"Normalizer (Calibrated strictly on Train): Mean={stats['train_mean']:.4f}, Std={stats['train_std']:.4f}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=True)

    # 2. Initialize Model and Diffusion
    print("\n[2/4] Initializing ConditionalUNet & Diffusion Engine...")
    model = ConditionalUNet(
        in_channels=1,
        condition_channels=10,
        out_channels=1,
        base_channels=base_channels,
        channel_multipliers=(1, 2, 4),
        num_res_blocks=2,
        dropout=0.05,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"ConditionalUNet parameters: {total_params:,}")

    diffusion = GaussianDiffusion(
        num_timesteps=num_timesteps,
        beta_schedule="cosine",
        prediction_type=prediction_type,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    total_opt_steps = epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_opt_steps, eta_min=1e-6)

    # 3. Training & Validation Loop
    print("\n[3/4] Launching training and validation loop...")
    history = {
        "train_loss": [],
        "val_epochs": [],
        "val_acc": [],
        "val_rmse": [],
        "val_mae": [],
        "val_bias": [],
        "val_amplitude_ratio": [],
        "lr": [],
    }

    best_val_acc = -1.0
    best_epoch = 0
    patience_counter = 0
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses = []

        for b_x0, b_cond, b_mask in train_loader:
            b_x0 = b_x0.to(device, non_blocking=True)
            b_cond = b_cond.to(device, non_blocking=True)
            b_mask = b_mask.to(device, non_blocking=True)

            optimizer.zero_grad()
            loss, _ = diffusion.compute_loss(model, b_x0, b_cond, mask=b_mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_losses.append(loss.item())

        avg_train_loss = float(np.mean(epoch_losses))
        current_lr = scheduler.get_last_lr()[0]
        history["train_loss"].append(avg_train_loss)
        history["lr"].append(current_lr)

        # Periodic Validation
        if epoch % val_every == 0 or epoch == epochs or epoch == 1:
            val_metrics = evaluate_validation(
                model=model,
                diffusion=diffusion,
                val_loader=val_loader,
                val_ds=val_ds,
                transform=transform,
                ddim_steps=ddim_val_steps,
                device=device,
            )

            history["val_epochs"].append(epoch)
            history["val_acc"].append(val_metrics["acc"])
            history["val_rmse"].append(val_metrics["rmse"])
            history["val_mae"].append(val_metrics["mae"])
            history["val_bias"].append(val_metrics["bias"])
            history["val_amplitude_ratio"].append(val_metrics["amplitude_ratio"])

            is_best = val_metrics["acc"] > best_val_acc
            best_tag = ""
            if is_best:
                best_val_acc = val_metrics["acc"]
                best_epoch = epoch
                patience_counter = 0
                best_tag = " -> [BEST MODEL SAVED]"

                # Save best checkpoint
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_metrics": val_metrics,
                        "stats": stats,
                        "config": {
                            "base_channels": base_channels,
                            "prediction_type": prediction_type,
                            "num_timesteps": num_timesteps,
                        },
                    },
                    ckpt_dir / "best_model.pt",
                )
            else:
                patience_counter += 1

            print(
                f"Epoch [{epoch:03d}/{epochs:03d}] - Train Loss: {avg_train_loss:.5f} | "
                f"Val ACC: {val_metrics['acc']:.4f} (±{val_metrics['acc_std']:.3f}) | "
                f"Val RMSE: {val_metrics['rmse']:.4f} | Amp Ratio: {val_metrics['amplitude_ratio']:.3f}"
                f"{best_tag}"
            )

            # Early Stopping Check
            if patience_counter >= patience:
                print(f"\n[Early Stopping triggered] Val ACC did not improve for {patience} checks. Best Epoch: {best_epoch} (ACC: {best_val_acc:.4f})")
                break
        else:
            if epoch % 2 == 0:
                print(f"Epoch [{epoch:03d}/{epochs:03d}] - Train Loss: {avg_train_loss:.5f} | LR: {current_lr:.6e}")

    elapsed_time = time.time() - start_time
    print(f"\nTraining completed in {elapsed_time:.1f}s. Best Epoch: {best_epoch} with Val ACC: {best_val_acc:.4f}")

    # Save latest checkpoint
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "stats": stats,
            "history": history,
        },
        ckpt_dir / "latest_model.pt",
    )

    # 4. Save Logs and Training Curves
    print("\n[4/4] Generating training curves and saving metrics...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

    # Loss curve
    ax1.plot(range(1, len(history["train_loss"]) + 1), history["train_loss"], label="Train Loss (Masked MSE)", color="tab:blue", lw=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss Convergence")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend()

    # Validation ACC & RMSE
    ax2.plot(history["val_epochs"], history["val_acc"], label="Val ACC", color="tab:red", marker="o", lw=2)
    ax2.axhline(y=best_val_acc, color="tab:red", linestyle=":", label=f"Best ACC ({best_val_acc:.4f} @ Ep {best_epoch})")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("ACC", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    ax2.set_title("Validation Generalization Dynamics")
    ax2.grid(True, linestyle="--", alpha=0.6)

    ax2_right = ax2.twinx()
    ax2_right.plot(history["val_epochs"], history["val_rmse"], label="Val RMSE", color="tab:green", marker="s", linestyle="--")
    ax2_right.set_ylabel("RMSE", color="tab:green")
    ax2_right.tick_params(axis="y", labelcolor="tab:green")

    plt.tight_layout()
    curve_path = fig_dir / "training_curves.png"
    plt.savefig(curve_path, bbox_inches="tight")
    plt.close()
    print(f"Saved training curves to: {curve_path}")

    # Save complete JSON history
    summary_report = {
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "epochs_trained": len(history["train_loss"]),
        "elapsed_seconds": elapsed_time,
        "stats": stats,
        "history": history,
        "best_checkpoint": str(ckpt_dir / "best_model.pt"),
        "training_curves": str(curve_path),
    }

    summary_path = log_dir / "training_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=4)
    print(f"Saved training summary to: {summary_path}")

    return summary_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HydroDiff Full Training")
    parser.add_argument("--epochs", type=int, default=100, help="Maximum number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--base_channels", type=int, default=32, help="UNet base channels")
    parser.add_argument("--num_timesteps", type=int, default=100, help="Diffusion timesteps T")
    parser.add_argument("--ddim_val_steps", type=int, default=20, help="DDIM sampling steps during validation")
    parser.add_argument("--val_every", type=int, default=5, help="Validation frequency (epochs)")
    parser.add_argument("--patience", type=int, default=8, help="Early stopping patience (number of val checks)")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Device")

    args = parser.parse_args()
    train_full_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        base_channels=args.base_channels,
        num_timesteps=args.num_timesteps,
        ddim_val_steps=args.ddim_val_steps,
        val_every=args.val_every,
        patience=args.patience,
        device_str=args.device,
    )

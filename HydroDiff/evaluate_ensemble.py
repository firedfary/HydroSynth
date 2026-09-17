"""Multi-Member Ensemble Evaluation for HydroDiff on the Independent Test Split.

Evaluates:
1. Deterministic skill of the Ensemble Mean (ACC, RMSE, MAE, Bias).
2. Distribution and skill of individual stochastic members.
3. Spatial Ensemble Spread (Uncertainty Quantification) and Spread-Skill Ratio (SSR).
4. 2-row multi-panel visualization: Ground Truth, Condition, Ensemble Mean, Error, Spread, and 5 Realizations.
5. Structured JSON metrics saved to <HYDRO_WORKSPACE>/results/HydroDiff/full_run/.
"""

from __future__ import annotations

import argparse
import json
import sys
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
from HydroDiff.data.metrics import compute_acc, compute_mae, compute_rmse, compute_spatial_metrics
from HydroDiff.diffusion.gaussian_diffusion import GaussianDiffusion
from HydroDiff.models.conditional_unet import ConditionalUNet
from HydroDiff.project_paths import experiment_dir, paths


def evaluate_ensemble_pipeline(
    checkpoint_path: str | Path | None = None,
    num_members: int = 10,
    ddim_steps: int = 25,
    eta: float = 0.0,
    batch_size: int = 8,
    device_str: str = "cuda:0" if torch.cuda.is_available() else "cpu",
) -> dict:
    device = torch.device(device_str)
    print("============================================================")
    print(" HydroDiff Step 3: Multi-Member Ensemble Test Evaluation   ")
    print(f" Device: {device} | Ensemble Members K: {num_members}")
    print(f" DDIM Steps: {ddim_steps} | Stochasticity Eta: {eta}")
    print("============================================================")

    exp_dir = experiment_dir("full_run")
    ckpt_dir = exp_dir / "checkpoints"
    fig_dir = exp_dir / "figures"
    log_dir = exp_dir / "logs"

    if checkpoint_path is None:
        checkpoint_path = ckpt_dir / "best_model.pt"
    else:
        checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    cfg = checkpoint.get("config", {})
    stats = checkpoint.get("stats", {})
    base_channels = cfg.get("base_channels", 32)
    prediction_type = cfg.get("prediction_type", "sample")
    num_timesteps = cfg.get("num_timesteps", 100)

    # 1. Load Data Splits
    u_net_cache = paths.workspace_root / "cache" / "U_Net_3D"
    hr_file = u_net_cache / "station_observations" / "hr_observations_ref1994_2010_aligned.npy"
    lr_file = u_net_cache / "prepared" / "lr_unet" / "lr_data_reconstructed2.npy"

    splits = create_spatiotemporal_splits(
        hr_file=hr_file,
        lr_file=lr_file,
        split_ratios=(0.8, 0.1, 0.1),
        pad_to=(128, 144),
        spatial_crop=None,
    )

    test_ds = splits["test"]
    transform = splits["transform"]
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    print(f"Test samples: {len(test_ds)} (Independent future evaluation split)")

    # 2. Reconstruct Model & Diffusion
    model = ConditionalUNet(
        in_channels=1,
        condition_channels=10,
        out_channels=1,
        base_channels=base_channels,
        channel_multipliers=(1, 2, 4),
        num_res_blocks=2,
        dropout=0.0,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    diffusion = GaussianDiffusion(
        num_timesteps=num_timesteps,
        beta_schedule="cosine",
        prediction_type=prediction_type,
    ).to(device)

    # 3. Generate Ensemble Members
    print(f"\nGenerating {num_members} ensemble members per test sample via stochastic initial noise...")
    all_gt_phys = []
    all_cond_first_ch = []
    all_masks = []
    all_member_phys = [[] for _ in range(num_members)]  # K lists of arrays

    with torch.no_grad():
        for b_x0, b_cond, b_mask in test_loader:
            b_size = b_x0.shape[0]
            b_cond_dev = b_cond.to(device)

            gt_p = test_ds.denormalize(b_x0)                         # [B, 1, 120, 140]
            m_np = transform.unpad_tensor(b_mask).cpu().numpy()      # [B, 1, 120, 140]
            c_np = transform.unpad_tensor(b_cond)[:, :1].cpu().numpy()  # [B, 1, 120, 140]

            all_gt_phys.append(gt_p)
            all_masks.append(m_np)
            all_cond_first_ch.append(c_np)

            # Sample K members with different random initial noise
            for k in range(num_members):
                gen = torch.Generator(device=device).manual_seed(42 + k * 1000)
                recon_norm = diffusion.ddim_sample(
                    model=model,
                    condition=b_cond_dev,
                    shape=(b_size, 1, 128, 144),
                    num_inference_steps=ddim_steps,
                    eta=eta,
                    generator=gen,
                )
                recon_phys = test_ds.denormalize(recon_norm)  # [B, 1, 120, 140]
                all_member_phys[k].append(recon_phys)

    # Concatenate all batches along N dimension
    gt_all = np.concatenate(all_gt_phys, axis=0)                # [N_test, 1, 120, 140]
    masks_all = np.concatenate(all_masks, axis=0)              # [N_test, 1, 120, 140]
    cond_all = np.concatenate(all_cond_first_ch, axis=0)        # [N_test, 1, 120, 140]
    N_test = gt_all.shape[0]

    # Stack ensemble members: [K, N_test, 1, 120, 140]
    members_stacked = np.stack([np.concatenate(m_list, axis=0) for m_list in all_member_phys], axis=0)

    # 4. Compute Ensemble Statistics
    print("\nComputing ensemble statistical metrics...")
    # Ensemble Mean: [N_test, 1, 120, 140]
    ens_mean = np.mean(members_stacked, axis=0)

    # Ensemble Spread: [N_test, 1, 120, 140]
    ens_spread = np.std(members_stacked, axis=0, ddof=1) if num_members > 1 else np.zeros_like(ens_mean)

    # Evaluation per test sample
    ens_mean_accs, ens_mean_rmses, ens_mean_maes = [], [], []
    member_accs = [[] for _ in range(num_members)]
    spread_list = []

    for i in range(N_test):
        m = masks_all[i, 0].astype(bool)
        t = gt_all[i, 0]
        p_mean = ens_mean[i, 0]
        s = ens_spread[i, 0]

        # Ensemble Mean skill
        ens_mean_accs.append(compute_acc(p_mean, t, m))
        ens_mean_rmses.append(compute_rmse(p_mean, t, m))
        ens_mean_maes.append(compute_mae(p_mean, t, m))
        spread_list.append(float(np.mean(s[m])))

        # Individual member skills
        for k in range(num_members):
            p_k = members_stacked[k, i, 0]
            member_accs[k].append(compute_acc(p_k, t, m))

    # Overall Summary
    mean_acc = float(np.mean(ens_mean_accs))
    mean_rmse = float(np.mean(ens_mean_rmses))
    mean_mae = float(np.mean(ens_mean_maes))
    mean_spread = float(np.mean(spread_list))
    spread_skill_ratio = float(mean_spread / mean_rmse) if mean_rmse > 1e-6 else 0.0

    member_mean_accs = [float(np.mean(accs)) for accs in member_accs]
    avg_individual_acc = float(np.mean(member_mean_accs))

    print("============================================================")
    print(" Independent Test Split Performance (38 Months: 2021-2024) ")
    print("============================================================")
    print(f" Ensemble Mean ACC          : {mean_acc:.4f}  (Std: {np.std(ens_mean_accs):.4f})")
    print(f" Ensemble Mean RMSE         : {mean_rmse:.4f}")
    print(f" Ensemble Mean MAE          : {mean_mae:.4f}")
    print(f" Average Member ACC         : {avg_individual_acc:.4f}  (Range: {min(member_mean_accs):.4f} - {max(member_mean_accs):.4f})")
    print(f" Average Ensemble Spread    : {mean_spread:.4f}")
    print(f" Spread-Skill Ratio (SSR)   : {spread_skill_ratio:.4f}  (Spread / RMSE)")
    print(f" Ensemble Gain over Members : +{(mean_acc - avg_individual_acc):.4f} ACC")
    print("============================================================")

    # 5. High-Resolution Visual Comparison
    print("\nGenerating multi-panel ensemble visualizations...")
    
    # 1. Best Case (Highest Ensemble Mean ACC)
    best_acc_idx = int(np.argmax(ens_mean_accs))
    # 2. Typical Case (Closest to Median Ensemble Mean ACC)
    median_acc = float(np.median(ens_mean_accs))
    median_acc_idx = int(np.argmin(np.abs(np.array(ens_mean_accs) - median_acc)))

    vis_cases = [
        ("ensemble_evaluation.png", best_acc_idx, f"High-Skill Case (Sample {best_acc_idx})"),
        ("ensemble_evaluation_typical.png", median_acc_idx, f"Typical Median-Skill Case (Sample {median_acc_idx})"),
    ]

    for filename, vis_idx, case_title in vis_cases:
        t_vis = np.where(masks_all[vis_idx, 0] > 0, gt_all[vis_idx, 0], np.nan)
        c_vis = cond_all[vis_idx, 0]
        m_vis = np.where(masks_all[vis_idx, 0] > 0, ens_mean[vis_idx, 0], np.nan)
        err_vis = np.where(masks_all[vis_idx, 0] > 0, np.abs(ens_mean[vis_idx, 0] - gt_all[vis_idx, 0]), np.nan)
        spd_vis = np.where(masks_all[vis_idx, 0] > 0, ens_spread[vis_idx, 0], np.nan)

        fig, axs = plt.subplots(2, 5, figsize=(22, 9), dpi=160)
        fig.suptitle(f"HydroDiff Test Set Ensemble Evaluation - {case_title}", fontsize=16, y=0.98)

        # Row 1: Deterministic Overview & Uncertainty
        im0 = axs[0, 0].imshow(t_vis, cmap="RdBu_r")
        axs[0, 0].set_title(f"Ground Truth Observation\n(Test Sample {vis_idx})", fontsize=12)
        plt.colorbar(im0, ax=axs[0, 0], fraction=0.046, pad=0.04)

        im1 = axs[0, 1].imshow(c_vis, cmap="Blues")
        axs[0, 1].set_title("Driving Condition (Ch 0)\n(Coarse Base Field)", fontsize=12)
        plt.colorbar(im1, ax=axs[0, 1], fraction=0.046, pad=0.04)

        im2 = axs[0, 2].imshow(m_vis, cmap="RdBu_r")
        s_acc = ens_mean_accs[vis_idx]
        s_rmse = ens_mean_rmses[vis_idx]
        axs[0, 2].set_title(f"Ensemble Mean (K={num_members})\n(ACC={s_acc:.4f}, RMSE={s_rmse:.4f})", fontsize=12)
        plt.colorbar(im2, ax=axs[0, 2], fraction=0.046, pad=0.04)

        im3 = axs[0, 3].imshow(err_vis, cmap="YlOrRd")
        axs[0, 3].set_title("Absolute Error Map\n|Ens Mean - GT|", fontsize=12)
        plt.colorbar(im3, ax=axs[0, 3], fraction=0.046, pad=0.04)

        im4 = axs[0, 4].imshow(spd_vis, cmap="Purples")
        axs[0, 4].set_title(f"Ensemble Spread $\\sigma_K$\n(Uncertainty)", fontsize=12)
        plt.colorbar(im4, ax=axs[0, 4], fraction=0.046, pad=0.04)

        # Row 2: 5 Individual Ensemble Members
        for k in range(5):
            m_k = np.where(masks_all[vis_idx, 0] > 0, members_stacked[k, vis_idx, 0], np.nan)
            im_k = axs[1, k].imshow(m_k, cmap="RdBu_r")
            k_acc = member_accs[k][vis_idx]
            axs[1, k].set_title(f"Ensemble Member {k+1}\n(ACC={k_acc:.4f})", fontsize=12)
            plt.colorbar(im_k, ax=axs[1, k], fraction=0.046, pad=0.04)

        for row in axs:
            for ax in row:
                ax.set_xticks([])
                ax.set_yticks([])

        plt.tight_layout()
        fig_path = fig_dir / filename
        plt.savefig(fig_path, bbox_inches="tight")
        plt.close()
        print(f"Saved ensemble visualization figure to: {fig_path}")

    # 6. Save JSON Results
    results = {
        "status": "COMPLETED",
        "num_test_samples": N_test,
        "num_members": num_members,
        "ddim_steps": ddim_steps,
        "eta": eta,
        "ensemble_mean_acc": mean_acc,
        "ensemble_mean_rmse": mean_rmse,
        "ensemble_mean_mae": mean_mae,
        "average_member_acc": avg_individual_acc,
        "ensemble_gain_acc": mean_acc - avg_individual_acc,
        "ensemble_spread_mean": mean_spread,
        "spread_skill_ratio": spread_skill_ratio,
        "figure_path": str(fig_path),
        "checkpoint_evaluated": str(checkpoint_path),
    }

    json_path = log_dir / "test_ensemble_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    print(f"Saved test ensemble metrics to: {json_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HydroDiff Multi-Member Ensemble Test Evaluation")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint (default: best_model.pt)")
    parser.add_argument("--members", type=int, default=10, help="Number of ensemble members K")
    parser.add_argument("--ddim_steps", type=int, default=25, help="Number of DDIM steps")
    parser.add_argument("--eta", type=float, default=0.0, help="DDIM stochasticity parameter")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Device")

    args = parser.parse_args()
    evaluate_ensemble_pipeline(
        checkpoint_path=args.checkpoint,
        num_members=args.members,
        ddim_steps=args.ddim_steps,
        eta=args.eta,
        batch_size=args.batch_size,
        device_str=args.device,
    )

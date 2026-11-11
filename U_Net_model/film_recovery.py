"""Validation-led, isolated FiLM capacity and conditioning-depth experiment."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from utils.paths import SubprojectPaths
from unetlitefilm import UNetLiteFiLM
from run_unified_eval import (
    build_date_index,
    compute_strict_sst_pcs,
    compute_suite_metrics,
    set_seed,
)


CANDIDATES = {
    "full32": (0, 1, 2, 3, 4),
    "deep32": (2, 3, 4),
    "bottleneck32": (4,),
}
OBJECTIVE_CANDIDATES = {
    "huber1": "huber1",
    "huber1_acc1": "huber1_acc1",
    "mse_acc2": "mse_acc2",
}


def spatial_acc(pred, target, mask):
    good = ~mask
    count = good.flatten(1).sum(1).clamp_min(1)
    p_mean = torch.where(good, pred, 0).flatten(1).sum(1) / count
    y_mean = torch.where(good, target, 0).flatten(1).sum(1) / count
    p_centered = torch.where(good, pred - p_mean[:, None, None, None], 0).flatten(1)
    y_centered = torch.where(good, target - y_mean[:, None, None, None], 0).flatten(1)
    numerator = (p_centered * y_centered).sum(1)
    denominator = (p_centered.square().sum(1) * y_centered.square().sum(1)).sqrt().clamp_min(1e-8)
    return (numerator / denominator).mean()


def predict(model, cond, pcs, device):
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(cond), 8):
            out.append(model(cond[start:start + 8].to(device), pcs[start:start + 8].to(device))[:, 0].cpu().numpy())
    return np.concatenate(out)


def matched_metrics(pred, obs, valid, p90):
    common = valid[None] & np.isfinite(obs) & np.isfinite(pred)
    pred = np.where(common, pred, np.nan)
    obs = np.where(common, obs, np.nan)
    return compute_suite_metrics(pred, obs, valid, p90)


def train_candidate(name, layers, train_ds, val_ds, device, epochs, objective="mse", select_acc=False):
    set_seed(42)
    model = UNetLiteFiLM(
        n_channels=10, index_dim=5, base_filters=32, dropout=0.1,
        film_layers=layers, identity_init=True,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)
    best_loss = float("inf")
    best_acc = -float("inf")
    best_epoch = 0
    best_weights = None

    for epoch in range(1, epochs + 1):
        model.train()
        for y, x, mask, pcs in train_loader:
            y, x, mask, pcs = (item.to(device) for item in (y, x, mask, pcs))
            optimizer.zero_grad(set_to_none=True)
            pred = model(x, pcs)
            if objective.startswith("huber1"):
                loss = nn.functional.huber_loss(pred[~mask], y[~mask], delta=1.0)
            else:
                loss = nn.functional.mse_loss(pred[~mask], y[~mask])
            if objective.endswith("acc1"):
                loss = loss + (1.0 - spatial_acc(pred, y, mask))
            elif objective.endswith("acc2"):
                loss = loss + 2.0 * (1.0 - spatial_acc(pred, y, mask))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        val_losses = []
        val_acc_total = 0.0
        val_count = 0
        with torch.no_grad():
            for y, x, mask, pcs in val_loader:
                y, x, mask, pcs = (item.to(device) for item in (y, x, mask, pcs))
                pred = model(x, pcs)
                val_losses.append(nn.functional.mse_loss(pred[~mask], y[~mask]).item())
                val_acc_total += spatial_acc(pred, y, mask).item() * len(y)
                val_count += len(y)
        val_loss = float(np.mean(val_losses))
        val_acc = val_acc_total / val_count
        improved = val_acc > best_acc if select_acc else val_loss < best_loss
        if improved:
            best_loss = val_loss
            best_acc = val_acc
            best_epoch = epoch
            best_weights = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if epoch % 10 == 0 or epoch == 1:
            print(f"{name}: epoch={epoch} val_mse={val_loss:.5f} val_acc={val_acc:.4f} best_epoch={best_epoch}", flush=True)

    model.load_state_dict(best_weights)
    model.eval()
    return model, {"val_mse": best_loss, "val_acc_at_selection": best_acc, "best_epoch": best_epoch,
                   "parameters": sum(p.numel() for p in model.parameters())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--diagnose-only", action="store_true")
    parser.add_argument("--objective-sweep", action="store_true")
    args = parser.parse_args()
    if args.epochs <= 0:
        parser.error("--epochs must be positive")

    set_seed(42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    paths = SubprojectPaths(__file__)
    exp_dir = Path(paths.get_exp_dir("film_objective" if args.objective_sweep else "film_recovery"))
    old_dir = Path(paths.get_exp_dir("unified_eval"))
    dates = build_date_index()
    train_mask = (dates <= "2017-12-01")
    val_mask = (dates >= "2018-01-01") & (dates <= "2022-12-01")
    test_mask = dates >= "2023-01-01"
    target = np.load(Path(config.modelconfig["hr_path"]) / "hr_data.npy").astype(np.float32)
    cond = np.load(Path(config.modelconfig["lr_path"]) / "lr_data_reconstructed2.npy").astype(np.float32)
    sst = np.load(config.modelconfig["sst_file"]).astype(np.float32)
    assert target.shape == (366, 120, 140) and cond.shape == (366, 10, 120, 140)
    assert sst.shape == (366, 6, 89, 180)
    pcs, _, _ = compute_strict_sst_pcs(sst, train_mask, 5)
    valid = np.isfinite(target[0])
    p90 = float(np.percentile(target[train_mask][:, valid], 90))

    y_t = torch.from_numpy(target[:, None])
    x_t = torch.from_numpy(cond)
    m_t = torch.isnan(y_t) | torch.from_numpy(~valid)[None, None]
    p_t = torch.from_numpy(pcs)
    train_ds = TensorDataset(y_t[train_mask], x_t[train_mask], m_t[train_mask], p_t[train_mask])
    val_ds = TensorDataset(y_t[val_mask], x_t[val_mask], m_t[val_mask], p_t[val_mask])

    old_metrics = json.loads((old_dir / "unified_metrics.json").read_text(encoding="utf-8"))
    old_film = UNetLiteFiLM(n_channels=10, index_dim=5, base_filters=16, dropout=0.1).to(device)
    old_film.load_state_dict(torch.load(old_dir / "checkpoints" / "film_best.pt", map_location=device, weights_only=True))
    pc_scale = {}
    for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
        pred = predict(old_film, x_t[val_mask], p_t[val_mask] * alpha, device)
        pc_scale[str(alpha)] = {key: value for key, value in
            matched_metrics(pred, target[val_mask], valid, p90).items()
            if key in ("mean_acc", "mean_tcc", "rmse")}
    print("Original FiLM PC scaling on validation:", pc_scale, flush=True)

    if args.diagnose_only:
        return

    results = {}
    models = {}
    candidates = (OBJECTIVE_CANDIDATES if args.objective_sweep else CANDIDATES)
    for name, candidate in candidates.items():
        layers = (0, 1, 2, 3, 4) if args.objective_sweep else candidate
        objective = candidate if args.objective_sweep else "mse"
        model, info = train_candidate(name, layers, train_ds, val_ds, device, args.epochs,
                                      objective=objective, select_acc=args.objective_sweep)
        pred_val = predict(model, x_t[val_mask], p_t[val_mask], device)
        pred_zero = predict(model, x_t[val_mask], torch.zeros_like(p_t[val_mask]), device)
        info["val"] = {key: value for key, value in
            matched_metrics(pred_val, target[val_mask], valid, p90).items()
            if key in ("mean_acc", "mean_tcc", "rmse")}
        info["val_pc_zero"] = {key: value for key, value in
            matched_metrics(pred_zero, target[val_mask], valid, p90).items()
            if key in ("mean_acc", "mean_tcc", "rmse")}
        info["film_layers"] = list(layers)
        info["objective"] = objective
        results[name] = info
        models[name] = model
        torch.save(model.state_dict(), exp_dir / "checkpoints" / f"{name}.pt")
        print(f"{name}: selected epoch {info['best_epoch']}, validation {info['val']}", flush=True)

    selected = (max(results, key=lambda name: results[name]["val"]["mean_acc"])
                if args.objective_sweep else min(results, key=lambda name: results[name]["val_mse"]))
    selected_model = models[selected]
    pred_test = predict(selected_model, x_t[test_mask], p_t[test_mask], device)
    pred_test_zero = predict(selected_model, x_t[test_mask], torch.zeros_like(p_t[test_mask]), device)
    selected_test = {}
    for label, pred in (("full", pred_test), ("pc_zero", pred_test_zero)):
        metrics = matched_metrics(pred, target[test_mask], valid, p90)
        selected_test[label] = {key: metrics[key] for key in
                                ("mean_acc", "mean_tcc", "rmse", "sc_pct", "csi")}

    with np.load(old_dir / "predictions_all.npz") as prior_predictions:
        prior_corrected = {
            name: {
                key: value for key, value in matched_metrics(
                    prior_predictions[field][test_mask], target[test_mask], valid, p90
                ).items() if key in ("mean_acc", "mean_tcc", "rmse", "sc_pct", "csi")
            }
            for name, field in (("Baseline", "pred_base"), ("EOF-cat", "pred_cat"),
                                ("FiLM-U-Net", "pred_film"))
        }

    payload = {
        "protocol": {"train": "1994-01/2017-12", "val": "2018-01/2022-12",
                     "test": "2023-01/2024-09", "epochs": args.epochs,
                     "selection": ("maximum validation ACC" if args.objective_sweep
                                   else "minimum validation MSE") + "; test evaluated once after selection",
                     "seed": 42, "lr": 2e-4, "dropout": 0.1, "batch_size": 8},
        "original_pc_scale_validation": pc_scale,
        "prior_models": {name: old_metrics["table1_metrics"][name] for name in ("Baseline", "EOF-cat", "FiLM-U-Net")},
        "prior_models_matched_mask": prior_corrected,
        "candidates": results,
        "selected": selected,
        "selected_test": selected_test,
    }
    result_path = exp_dir / ("film_objective.json" if args.objective_sweep else "film_recovery.json")
    result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Selected {selected} by validation {'ACC' if args.objective_sweep else 'MSE'}; independent test {selected_test}", flush=True)
    print(f"Saved {result_path}", flush=True)


if __name__ == "__main__":
    main()

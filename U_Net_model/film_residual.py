"""Test whether FiLM adds skill to a frozen, trained local U-Net."""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from utils.paths import SubprojectPaths
from unetlite import UNetLite
from unetlitefilm import UNetLiteFiLM
from film_recovery import matched_metrics, predict, spatial_acc
from run_unified_eval import build_date_index, compute_strict_sst_pcs, set_seed


VARIANTS = {
    "all_mse": ((0, 1, 2, 3, 4), 0.0),
    "all_acc": ((0, 1, 2, 3, 4), 1.0),
    "deep_acc": ((2, 3, 4), 1.0),
}


def train_variant(name, layers, acc_weight, base_weights, train_ds, val_ds,
                  baseline_val_acc, device):
    set_seed(42)
    model = UNetLiteFiLM(n_channels=10, index_dim=5, base_filters=32, dropout=0.1,
                         film_layers=layers, identity_init=True).to(device)
    missing, unexpected = model.load_state_dict(base_weights, strict=False)
    if unexpected or any(not key.startswith("index_encoder.") for key in missing):
        raise RuntimeError(f"Unexpected weight mismatch: {missing}, {unexpected}")
    for key, parameter in model.named_parameters():
        parameter.requires_grad_(key.startswith("index_encoder."))
    model.eval()
    optimizer = torch.optim.AdamW(model.index_encoder.parameters(), lr=2e-4, weight_decay=1e-4)
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)
    best_acc = baseline_val_acc
    best_epoch = 0
    best_weights = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    for epoch in range(1, 101):
        model.eval()
        for y, x, mask, pcs in train_loader:
            y, x, mask, pcs = (item.to(device) for item in (y, x, mask, pcs))
            optimizer.zero_grad(set_to_none=True)
            pred = model(x, pcs)
            loss = F.mse_loss(pred[~mask], y[~mask])
            if acc_weight:
                loss = loss + acc_weight * (1 - spatial_acc(pred, y, mask))
            gamma, beta = model.index_encoder(pcs)
            penalty = sum(gamma[level].square().mean() + beta[level].square().mean()
                          for level in layers) / len(layers)
            loss = loss + 0.1 * penalty
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.index_encoder.parameters(), 2.0)
            optimizer.step()

        val_sum = 0.0
        val_count = 0
        with torch.no_grad():
            for y, x, mask, pcs in val_loader:
                y, x, mask, pcs = (item.to(device) for item in (y, x, mask, pcs))
                val_sum += spatial_acc(model(x, pcs), y, mask).item() * len(y)
                val_count += len(y)
        val_acc = val_sum / val_count
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            best_weights = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if epoch % 10 == 0 or epoch == 1:
            print(f"{name}: epoch={epoch} val_acc={val_acc:.4f} best={best_acc:.4f} at {best_epoch}", flush=True)

    model.load_state_dict(best_weights)
    model.eval()
    return model, {"best_epoch": best_epoch, "val_acc": best_acc,
                   "film_layers": list(layers), "acc_weight": acc_weight}


def main():
    paths = SubprojectPaths(__file__)
    exp_dir = Path(paths.get_exp_dir("film_residual"))
    old_dir = Path(paths.get_exp_dir("unified_eval"))
    dates = build_date_index()
    train_mask = dates <= "2017-12-01"
    val_mask = (dates >= "2018-01-01") & (dates <= "2022-12-01")
    test_mask = dates >= "2023-01-01"
    target = np.load(Path(config.modelconfig["hr_path"]) / "hr_data.npy").astype(np.float32)
    cond = np.load(Path(config.modelconfig["lr_path"]) / "lr_data_reconstructed2.npy").astype(np.float32)
    sst = np.load(config.modelconfig["sst_file"]).astype(np.float32)
    pcs, _, _ = compute_strict_sst_pcs(sst, train_mask, 5)
    valid = np.isfinite(target[0])
    p90 = float(np.percentile(target[train_mask][:, valid], 90))
    y_t = torch.from_numpy(target[:, None])
    x_t = torch.from_numpy(cond)
    m_t = torch.isnan(y_t) | torch.from_numpy(~valid)[None, None]
    p_t = torch.from_numpy(pcs)
    train_ds = TensorDataset(y_t[train_mask], x_t[train_mask], m_t[train_mask], p_t[train_mask])
    val_ds = TensorDataset(y_t[val_mask], x_t[val_mask], m_t[val_mask], p_t[val_mask])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    base_weights = torch.load(old_dir / "checkpoints" / "baseline_best.pt",
                              map_location=device, weights_only=True)
    baseline = UNetLite(n_channels=10, base_filters=32, dropout=0.1).to(device)
    baseline.load_state_dict(base_weights)
    baseline.eval()
    with torch.no_grad():
        reference = baseline(x_t[val_mask][:2].to(device))
    identity = UNetLiteFiLM(n_channels=10, index_dim=5, base_filters=32, dropout=0.1,
                            identity_init=True).to(device)
    identity.load_state_dict(base_weights, strict=False)
    identity.eval()
    with torch.no_grad():
        transfer_diff = float((identity(x_t[val_mask][:2].to(device),
                                        p_t[val_mask][:2].to(device)) - reference).abs().max().item())
    if transfer_diff > 1e-5:
        raise RuntimeError(f"FiLM identity does not match baseline: {transfer_diff}")

    base_val_pred = predict(identity, x_t[val_mask], p_t[val_mask], device)
    base_val = matched_metrics(base_val_pred, target[val_mask], valid, p90)
    baseline_val_acc = base_val["mean_acc"]
    results = {}
    models = {}
    for name, (layers, acc_weight) in VARIANTS.items():
        model, info = train_variant(name, layers, acc_weight, base_weights, train_ds,
                                    val_ds, baseline_val_acc, device)
        val_pred = predict(model, x_t[val_mask], p_t[val_mask], device)
        val_zero = predict(model, x_t[val_mask], torch.zeros_like(p_t[val_mask]), device)
        info["val"] = {key: value for key, value in
                       matched_metrics(val_pred, target[val_mask], valid, p90).items()
                       if key in ("mean_acc", "mean_tcc", "rmse")}
        info["val_pc_zero"] = {key: value for key, value in
                               matched_metrics(val_zero, target[val_mask], valid, p90).items()
                               if key in ("mean_acc", "mean_tcc", "rmse")}
        results[name] = info
        models[name] = model
        torch.save(model.state_dict(), exp_dir / "checkpoints" / f"{name}.pt")
        print(f"{name}: selected validation {info['val']}", flush=True)

    selected = max(results, key=lambda name: results[name]["val"]["mean_acc"])
    if results[selected]["val"]["mean_acc"] <= baseline_val_acc:
        selected = "baseline_identity"
        selected_test = None
        print("No FiLM branch improved validation ACC over baseline.", flush=True)
    else:
        pred_test = predict(models[selected], x_t[test_mask], p_t[test_mask], device)
        metrics = matched_metrics(pred_test, target[test_mask], valid, p90)
        selected_test = {key: metrics[key] for key in ("mean_acc", "mean_tcc", "rmse", "sc_pct", "csi")}
        print(f"Selected {selected} on validation; exploratory test {selected_test}", flush=True)

    payload = {"protocol": "frozen baseline, train FiLM branch only on 1994-2017; "
                           "validation 2018-2022, test 2023-2024, seed 42",
               "transfer_max_abs_diff": transfer_diff, "baseline_val_acc": baseline_val_acc,
               "candidates": results, "selected": selected, "selected_test": selected_test}
    path = exp_dir / "film_residual.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved {path}", flush=True)


if __name__ == "__main__":
    main()

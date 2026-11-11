"""Matched U-Net and FiLM comparison on train-fitted meteorological inputs."""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from utils.paths import SubprojectPaths
from unetlite import UNetLite
from unetlitefilm import UNetLiteFiLM
from film_recovery import matched_metrics, spatial_acc
from run_unified_eval import build_date_index, compute_strict_sst_pcs, paired_stat_tests, set_seed


MODELS = ("baseline32", "eof_cat32", "film16_original", "film32_identity")
METRICS = ("mean_acc", "mean_tcc", "rmse", "sc_pct", "csi", "sig_pct")


def make_model(name, dropout, index_dim=5):
    if name == "baseline32":
        return UNetLite(n_channels=10, base_filters=32, dropout=dropout)
    if name == "eof_cat32":
        return UNetLite(n_channels=10 + index_dim, base_filters=32, dropout=dropout)
    return UNetLiteFiLM(
        n_channels=10, index_dim=5,
        base_filters=16 if name == "film16_original" else 32,
        dropout=dropout, identity_init=name == "film32_identity",
    )


def forward(model, name, x, pcs):
    if name == "eof_cat32":
        pcs_map = pcs[:, :, None, None].expand(-1, -1, x.shape[2], x.shape[3])
        return model(torch.cat((x, pcs_map), dim=1))
    if name.startswith("film"):
        return model(x, pcs)
    return model(x)


def predict(model, name, x, pcs, device):
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(x), 8):
            outputs.append(forward(model, name, x[start:start + 8].to(device),
                                   pcs[start:start + 8].to(device))[:, 0].cpu().numpy())
    return np.concatenate(outputs)


def train(name, train_ds, val_ds, device, epochs, lr, dropout, seed, index_dim=5):
    set_seed(seed)
    model = make_model(name, dropout, index_dim=index_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)
    best_acc = -float("inf")
    best_epoch = 0
    best_state = None
    history = []
    started = time.monotonic()

    for epoch in range(1, epochs + 1):
        model.train()
        train_sum = 0.0
        train_count = 0
        for y, x, mask, pcs in train_loader:
            y, x, mask, pcs = (item.to(device) for item in (y, x, mask, pcs))
            optimizer.zero_grad(set_to_none=True)
            pred = forward(model, name, x, pcs)
            loss = F.mse_loss(pred[~mask], y[~mask])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            train_sum += float(loss.item()) * len(y)
            train_count += len(y)
        scheduler.step()

        model.eval()
        val_sum = 0.0
        val_acc_sum = 0.0
        val_count = 0
        with torch.no_grad():
            for y, x, mask, pcs in val_loader:
                y, x, mask, pcs = (item.to(device) for item in (y, x, mask, pcs))
                pred = forward(model, name, x, pcs)
                val_sum += float(F.mse_loss(pred[~mask], y[~mask]).item()) * len(y)
                val_acc_sum += float(spatial_acc(pred, y, mask).item()) * len(y)
                val_count += len(y)
        val_acc = val_acc_sum / val_count
        val_mse = val_sum / val_count
        history.append({"epoch": epoch, "train_mse": train_sum / train_count,
                        "val_mse": val_mse, "val_acc": val_acc})
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone()
                          for key, value in model.state_dict().items()}
        if epoch == 1 or epoch % 10 == 0:
            print(f"{name} epoch={epoch} val_acc={val_acc:.4f} val_mse={val_mse:.4f} "
                  f"best_epoch={best_epoch} elapsed={time.monotonic() - started:.0f}s", flush=True)

    model.load_state_dict(best_state)
    model.eval()
    info = {"best_epoch": best_epoch, "best_val_acc": best_acc,
            "parameters": sum(p.numel() for p in model.parameters()),
            "elapsed_seconds": time.monotonic() - started, "history": history}
    return model, info


def summary(pred, obs, valid, p90):
    result = matched_metrics(pred, obs, valid, p90)
    return {key: float(result[key]) for key in METRICS}, result["acc_series"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--experiment", default="strict_fair_v1")
    args = parser.parse_args()
    if args.epochs <= 0 or not args.experiment or "/" in args.experiment or "\\" in args.experiment:
        parser.error("epochs must be positive and experiment must be a directory name")

    set_seed(args.seed)
    paths = SubprojectPaths(__file__)
    input_dir = paths.cache_dir / "strict_inputs"
    input_file = input_dir / "cond_trainfit.npy"
    input_manifest = json.loads((input_dir / "manifest.json").read_text(encoding="utf-8"))
    if hashlib.sha256(input_file.read_bytes()).hexdigest() != input_manifest["sha256"]:
        raise ValueError("preprocessed input hash differs from manifest")
    exp_dir = Path(paths.get_exp_dir(args.experiment))
    result_file = exp_dir / "strict_fair_metrics.json"
    if result_file.exists() or any((exp_dir / "checkpoints").glob("*.pt")):
        raise FileExistsError(f"experiment already contains results: {exp_dir}")

    dates = build_date_index()
    train_mask = np.asarray(dates <= "2017-12-01")
    val_mask = np.asarray((dates >= "2018-01-01") & (dates <= "2022-12-01"))
    test_mask = np.asarray(dates >= "2023-01-01")
    x = np.load(input_file).astype(np.float32)
    y = np.load(Path(config.modelconfig["hr_path"]) / "hr_data.npy").astype(np.float32)
    sst = np.load(config.modelconfig["sst_file"]).astype(np.float32)
    if x.shape != (366, 10, 120, 140) or y.shape != (366, 120, 140) or sst.shape != (366, 6, 89, 180):
        raise ValueError("input, target, or SST shape differs from expected chronology")
    valid = np.isfinite(y).all(axis=0)
    p90 = float(np.percentile(y[train_mask][:, valid], 90))
    pcs, _, explained = compute_strict_sst_pcs(sst, train_mask, 5)
    if not np.isfinite(x).all() or not np.isfinite(pcs).all():
        raise ValueError("input or SST PCs contain non-finite values")

    x_t = torch.from_numpy(x)
    y_t = torch.from_numpy(y[:, None])
    mask_t = torch.isnan(y_t) | torch.from_numpy(~valid)[None, None]
    pcs_t = torch.from_numpy(pcs)
    dataset = TensorDataset(y_t, x_t, mask_t, pcs_t)
    train_ds = torch.utils.data.Subset(dataset, np.flatnonzero(train_mask))
    val_ds = torch.utils.data.Subset(dataset, np.flatnonzero(val_mask))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device={device} train={len(train_ds)} val={len(val_ds)} test={int(test_mask.sum())}", flush=True)

    lr, dropout = 2e-4, 0.1
    models = {}
    validation = {}
    for name in MODELS:
        model, info = train(name, train_ds, val_ds, device, args.epochs, lr, dropout, args.seed)
        torch.save(model.state_dict(), exp_dir / "checkpoints" / f"{name}.pt")
        pred_val = predict(model, name, x_t[val_mask], pcs_t[val_mask], device)
        info["validation"], _ = summary(pred_val, y[val_mask], valid, p90)
        validation[name] = info
        models[name] = model.cpu()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"{name} validation={info['validation']}", flush=True)

    selected_film = max(("film16_original", "film32_identity"),
                        key=lambda name: validation[name]["validation"]["mean_acc"])
    print(f"Selected FiLM on validation: {selected_film}; evaluating test", flush=True)
    predictions = {}
    test_metrics = {}
    test_acc = {}
    for name in MODELS:
        model = models[name].to(device)
        pred = predict(model, name, x_t[test_mask], pcs_t[test_mask], device)
        predictions[name] = pred
        test_metrics[name], test_acc[name] = summary(pred, y[test_mask], valid, p90)
        models[name] = model.cpu()
        print(f"{name} test={test_metrics[name]}", flush=True)

    paired = {other: paired_stat_tests(test_acc[selected_film], test_acc[other],
                                       dates[test_mask].year.values)
              for other in ("baseline32", "eof_cat32")}
    np.savez_compressed(exp_dir / "test_predictions.npz",
                        dates=dates[test_mask].strftime("%Y-%m-%d").values,
                        target=y[test_mask], valid=valid, pcs=pcs[test_mask], **predictions)
    payload = {
        "protocol": {"train": "1994-01/2017-12", "validation": "2018-01/2022-12",
                     "test": "2023-01/2024-09", "epochs": args.epochs,
                     "seed": args.seed, "batch_size": 8, "lr": lr, "dropout": dropout,
                     "loss": "masked MSE", "checkpoint_selection": "validation mean ACC",
                     "film_selection": "validation mean ACC among original-16 and identity-32",
                     "test_status": "exploratory: earlier experiments already inspected test period"},
        "input_sha256": input_manifest["sha256"],
        "sst_pca_explained_variance": [float(value) for value in explained],
        "target_provenance": "existing hr_data.npy is elementwise identical to the 1994-2010 "
                             "reference-climatology reconstruction from station observations",
        "train_p90": p90, "validation": validation, "selected_film": selected_film,
        "test": test_metrics, "paired_acc_selected_film": paired,
    }
    result_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved {result_file}", flush=True)


if __name__ == "__main__":
    main()

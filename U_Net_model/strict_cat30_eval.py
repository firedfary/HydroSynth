"""Evaluate input concatenation with the same 30 lagged SST PCs as FiLM."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from utils.paths import SubprojectPaths
from unetlite import UNetLite
from run_unified_eval import build_date_index, compute_strict_sst_pcs, paired_stat_tests, set_seed
from strict_fair_eval import predict, summary, train
from strict_sst_window_eval import FOLDS, SEEDS, baseline_weights, fold_data


def checked_prediction(file, key, expected_dates, expected_target):
    with np.load(file, allow_pickle=True) as source:
        if (not np.array_equal(source["dates"], expected_dates) or
                source[key].shape != expected_target.shape or
                not np.allclose(source["target"], expected_target, equal_nan=True)):
            raise ValueError(f"source prediction chronology or target mismatch: {file}")
        return source[key].copy()


def metric_row(pred, target, valid, p90):
    metrics, acc = summary(pred, target, valid, p90)
    return metrics, acc


class FrozenInputCat(nn.Module):
    def __init__(self, base_weights):
        super().__init__()
        self.baseline = UNetLite(n_channels=10, base_filters=32, dropout=0.1)
        self.baseline.load_state_dict(base_weights)
        for parameter in self.baseline.parameters():
            parameter.requires_grad_(False)
        self.pc_conv = nn.Conv2d(30, 32, kernel_size=3, padding=1, bias=False)
        nn.init.zeros_(self.pc_conv.weight)

    def forward(self, cat_input):
        x, pc_map = cat_input[:, :10], cat_input[:, 10:]
        base = self.baseline
        x1 = base.inc.double_conv[0](x) + self.pc_conv(pc_map)
        for layer in list(base.inc.double_conv.children())[1:]:
            x1 = layer(x1)
        x2 = base.down1(x1)
        x3 = base.down2(x2)
        x4 = base.down3(x3)
        x5 = base.down4(x4)
        out = base.up1(x5, x4)
        out = base.up2(out, x3)
        out = base.up3(out, x2)
        out = base.up4(out, x1)
        return base.outc(out)


def train_frozen_cat(train_ds, val_arrays, base_weights, device, epochs, seed, lr):
    set_seed(seed)
    model = FrozenInputCat(base_weights).to(device)
    optimizer = torch.optim.AdamW(model.pc_conv.parameters(), lr=lr, weight_decay=1e-4)
    loader = DataLoader(train_ds, batch_size=8, shuffle=True, drop_last=True)
    x_val, pcs_val, y_val, valid, p90 = val_arrays
    initial_pred = predict(model, "eof_cat32", x_val, pcs_val, device)
    initial, _ = summary(initial_pred, y_val, valid, p90)
    best_acc, best_epoch = initial["mean_acc"], 0
    best_state = {key: value.detach().cpu().clone()
                  for key, value in model.state_dict().items()}
    history = []
    for epoch in range(1, epochs + 1):
        model.eval()
        for target, x, mask, pcs in loader:
            target, x, mask, pcs = (item.to(device) for item in (target, x, mask, pcs))
            optimizer.zero_grad(set_to_none=True)
            pcs_map = pcs[:, :, None, None].expand(-1, -1, x.shape[2], x.shape[3])
            pred = model(torch.cat((x, pcs_map), dim=1))
            loss = F.mse_loss(pred[~mask], target[~mask])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.pc_conv.parameters(), 2.0)
            optimizer.step()
        val_pred = predict(model, "eof_cat32", x_val, pcs_val, device)
        val, _ = summary(val_pred, y_val, valid, p90)
        val_acc = val["mean_acc"]
        history.append({"epoch": epoch, "val_acc": val_acc, "val_rmse": val["rmse"]})
        if val_acc > best_acc:
            best_acc, best_epoch = val_acc, epoch
            best_state = {key: value.detach().cpu().clone()
                          for key, value in model.state_dict().items()}
        if epoch == 1 or epoch % 20 == 0:
            print(f"frozen_cat seed={seed} epoch={epoch} val_acc={val_acc:.4f} "
                  f"best={best_acc:.4f}", flush=True)
    model.load_state_dict(best_state)
    return model, {"best_epoch": best_epoch, "best_val_acc": best_acc,
                   "baseline_val_acc": initial["mean_acc"],
                   "trainable_pc_weights": 30 * 32 * 3 * 3, "history": history}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--experiment", default="strict_cat30_v1")
    parser.add_argument("--initialization", choices=("scratch", "frozen"), default="scratch")
    parser.add_argument("--frozen-lr", type=float, default=2e-4)
    args = parser.parse_args()
    if (args.epochs <= 0 or not args.experiment or "/" in args.experiment or
            "\\" in args.experiment or not np.isfinite(args.frozen_lr) or
            args.frozen_lr <= 0):
        parser.error("epochs must be positive and experiment must be a directory name")

    paths = SubprojectPaths(__file__)
    if (paths.results_dir / args.experiment).exists():
        raise FileExistsError(paths.results_dir / args.experiment)
    dates = build_date_index()
    sst = np.load(config.modelconfig["sst_file"]).astype(np.float32)
    y = np.load(Path(config.modelconfig["hr_path"]) / "hr_data.npy").astype(np.float32)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    exp_dir = Path(paths.get_exp_dir(args.experiment))
    result_file = exp_dir / "cat30_metrics.json"
    payload = {
        "protocol": {
            "conditioning": "six oldest-to-newest monthly vectors of five train-fitted SST EOF PCs",
            "architecture": "32-channel U-Net with 30 input-broadcast PC maps",
            "initialization": args.initialization,
            "frozen_mode": "matched Baseline backbone; only new first-layer PC weights trained"
            if args.initialization == "frozen" else None,
            "epochs": args.epochs, "seeds": list(SEEDS), "batch_size": 8,
            "lr": args.frozen_lr if args.initialization == "frozen" else 2e-4,
            "dropout": 0.1, "loss": "masked MSE",
            "checkpoint_selection": "validation spatial ACC",
            "comparison": "fixed three-seed ensemble of early FiLM and matched Baseline",
            "evidence_status": "all holdout periods were previously inspected; exploratory only",
        }, "folds": {},
    }

    for fold, spec in FOLDS.items():
        x, digest, train_mask, val_mask, holdout_mask, valid, p90, _ = fold_data(
            paths, spec, sst, y, dates, windows=(6,),
        )
        pcs, _, explained = compute_strict_sst_pcs(sst, train_mask, 5, 6, keep_lags=True)
        if pcs.shape != (len(dates), 30) or not np.isfinite(pcs).all():
            raise ValueError(f"invalid 30-PC features for {fold}")
        source_metrics = json.loads((paths.results_dir / "film_early_v1" /
                                     "trajectory_metrics.json").read_text(encoding="utf-8"))
        source_fold = source_metrics["folds"][fold]
        pcs_hash = hashlib.sha256(pcs.tobytes()).hexdigest()
        if (source_fold["input_sha256"] != digest or
                source_fold["trajectory_sha256"] != pcs_hash):
            raise ValueError(f"FiLM source input or SST PC mismatch for {fold}")
        x_t, pcs_t = torch.from_numpy(x), torch.from_numpy(pcs)
        y_t = torch.from_numpy(y[:, None])
        mask_t = torch.isnan(y_t) | torch.from_numpy(~valid)[None, None]
        train_ds = TensorDataset(y_t[train_mask], x_t[train_mask],
                                 mask_t[train_mask], pcs_t[train_mask])
        val_ds = TensorDataset(y_t[val_mask], x_t[val_mask],
                               mask_t[val_mask], pcs_t[val_mask])
        val_dates = dates[val_mask].strftime("%Y-%m-%d").values
        holdout_dates = dates[holdout_mask].strftime("%Y-%m-%d").values
        fold_result = {"input_sha256": digest, "trajectory_sha256": pcs_hash,
                       "sst_pca_explained_variance": [float(v) for v in explained],
                       "seeds": {}}
        payload["folds"][fold] = fold_result
        ensemble = {period: {name: [] for name in ("cat30", "film", "baseline")}
                    for period in ("validation", "holdout")}
        for seed in SEEDS:
            base_weights = baseline_weights(paths, fold, seed)
            if args.initialization == "frozen":
                val_arrays = (x_t[val_mask], pcs_t[val_mask], y[val_mask], valid, p90)
                model, info = train_frozen_cat(train_ds, val_arrays, base_weights,
                                               device, args.epochs, seed, args.frozen_lr)
            else:
                model, info = train("eof_cat32", train_ds, val_ds, device,
                                    args.epochs, 2e-4, 0.1, seed, index_dim=30)
            torch.save(model.state_dict(), exp_dir / "checkpoints" / f"{fold}_s{seed}.pt")
            cat_val = predict(model, "eof_cat32", x_t[val_mask], pcs_t[val_mask], device)
            cat_holdout = predict(model, "eof_cat32", x_t[holdout_mask],
                                  pcs_t[holdout_mask], device)
            film_val_file = paths.results_dir / "film_early_v1" / f"{fold}_s{seed}_validation.npz"
            with np.load(film_val_file, allow_pickle=True) as source:
                if (not np.array_equal(source["dates"], val_dates) or
                        source["prediction"].shape != cat_val.shape):
                    raise ValueError(f"FiLM validation mismatch: {film_val_file}")
                film_val = source["prediction"].copy()
            film_holdout_file = paths.results_dir / "film_early_v1" / f"{fold}_s{seed}_holdout.npz"
            film_holdout = checked_prediction(film_holdout_file, "trajectory",
                                              holdout_dates, y[holdout_mask])
            base_holdout = checked_prediction(film_holdout_file, "baseline",
                                              holdout_dates, y[holdout_mask])
            baseline = UNetLite(n_channels=10, base_filters=32, dropout=0.1).to(device)
            baseline.load_state_dict(base_weights)
            base_val = predict(baseline, "baseline32", x_t[val_mask], pcs_t[val_mask], device)
            if args.initialization == "frozen":
                zero_pcs = torch.zeros_like(pcs_t[val_mask])
                zero_pred = predict(model, "eof_cat32", x_t[val_mask], zero_pcs, device)
                transfer_diff = float(np.max(np.abs(zero_pred - base_val)))
                if transfer_diff > 1e-4:
                    raise ValueError(f"zero-PC frozen cat differs from Baseline: {transfer_diff}")
                info["zero_pc_transfer_max_abs_diff"] = transfer_diff
            del baseline
            for period, arrays in (("validation", (cat_val, film_val, base_val)),
                                   ("holdout", (cat_holdout, film_holdout, base_holdout))):
                for name, arr in zip(("cat30", "film", "baseline"), arrays):
                    if arr is not None:
                        ensemble[period][name].append(arr)
            fold_result["seeds"][str(seed)] = {
                "training": info,
                "validation": {
                    "cat30": metric_row(cat_val, y[val_mask], valid, p90)[0],
                    "film": metric_row(film_val, y[val_mask], valid, p90)[0],
                    "baseline": metric_row(base_val, y[val_mask], valid, p90)[0],
                },
                "exploratory_holdout": {
                    "cat30": metric_row(cat_holdout, y[holdout_mask], valid, p90)[0],
                    "film": metric_row(film_holdout, y[holdout_mask], valid, p90)[0],
                    "baseline": metric_row(base_holdout, y[holdout_mask], valid, p90)[0],
                },
            }
            np.savez_compressed(exp_dir / f"{fold}_s{seed}_predictions.npz",
                                validation_dates=val_dates, holdout_dates=holdout_dates,
                                cat30_validation=cat_val, cat30_holdout=cat_holdout)
            result_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(f"{fold} seed={seed} validation ACC cat30="
                  f"{fold_result['seeds'][str(seed)]['validation']['cat30']['mean_acc']:.5f}",
                  flush=True)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        for period, mask in (("validation", val_mask), ("holdout", holdout_mask)):
            target = y[mask]
            metrics = {}
            acc_series = {}
            for name, predictions in ensemble[period].items():
                if not predictions:
                    continue
                metrics[name], acc_series[name] = metric_row(
                    np.mean(predictions, axis=0), target, valid, p90,
                )
            paired = {other: paired_stat_tests(acc_series["film"], acc_series[other],
                                               dates[mask].year.values)
                      for other in metrics if other != "film"}
            fold_result["ensemble_" + period] = {"metrics": metrics,
                                                  "paired_film_minus": paired}
            print(f"{fold} {period} ensemble ACC " +
                  " ".join(f"{name}={row['mean_acc']:.5f}"
                           for name, row in metrics.items()), flush=True)
        result_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved {result_file}", flush=True)


if __name__ == "__main__":
    main()

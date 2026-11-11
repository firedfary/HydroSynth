"""Matched deep-layer continuation of Baseline and lagged-EOF FiLM."""

import argparse
import hashlib
import json
import sys
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
from run_unified_eval import build_date_index, compute_strict_sst_pcs, paired_stat_tests, set_seed
from strict_fair_eval import predict, summary
from strict_sst_window_eval import FOLDS, SEEDS, baseline_weights, fold_data


DEEP_MODULES = ("down3.", "down4.", "up1.", "up2.")


def make_model(name, paths, fold, seed, device):
    if name == "baseline":
        model = UNetLite(n_channels=10, base_filters=32, dropout=0.1)
        model.load_state_dict(baseline_weights(paths, fold, seed))
    else:
        model = UNetLiteFiLM(
            n_channels=10, index_dim=30, base_filters=32, index_hidden=16,
            dropout=0.1, film_layers=(3, 4), identity_init=True,
            zero_pc_identity=True,
        )
        model.load_state_dict(torch.load(
            paths.results_dir / "strict_trajectory_v1" / "checkpoints" /
            f"{fold}_s{seed}.pt", map_location="cpu", weights_only=True,
        ))
    return model.to(device)


def continue_training(name, model, train_ds, val_arrays, device, epochs, seed):
    set_seed(seed)
    deep, head = [], []
    for key, parameter in model.named_parameters():
        is_deep = key.startswith(DEEP_MODULES)
        is_head = name == "film" and key.startswith("index_encoder.")
        parameter.requires_grad_(is_deep or is_head)
        if is_deep:
            deep.append(parameter)
        elif is_head:
            head.append(parameter)
    if not deep or (name == "film" and not head):
        raise RuntimeError("expected trainable deep layers and FiLM head")
    groups = [{"params": deep, "lr": 2e-5}]
    if head:
        groups.append({"params": head, "lr": 1e-4})
    optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
    loader = DataLoader(train_ds, batch_size=8, shuffle=True, drop_last=True)
    xx, pp, yy, valid, p90 = val_arrays
    initial_pred = predict(model, "film32_identity" if name == "film" else "baseline32",
                           xx, pp, device)
    initial, _ = summary(initial_pred, yy, valid, p90)
    best_acc, best_epoch = initial["mean_acc"], 0
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    history = [{"epoch": 0, "val_acc": best_acc, "val_rmse": initial["rmse"]}]
    for epoch in range(1, epochs + 1):
        model.eval()
        for target, cond, mask, indices in loader:
            target, cond, mask, indices = (item.to(device) for item in
                                           (target, cond, mask, indices))
            optimizer.zero_grad(set_to_none=True)
            pred = model(cond, indices) if name == "film" else model(cond)
            loss = F.mse_loss(pred[~mask], target[~mask])
            if name == "film":
                gammas, betas = model.index_encoder(indices)
                zero_gammas, zero_betas = model.index_encoder(torch.zeros_like(indices))
                penalty = sum(
                    (gammas[level] - zero_gammas[level]).square().mean() +
                    (betas[level] - zero_betas[level]).square().mean()
                    for level in (3, 4)
                ) / 2
                loss = loss + 0.1 * penalty
            loss.backward()
            torch.nn.utils.clip_grad_norm_(deep + head, 2.0)
            optimizer.step()
        val_pred = predict(model, "film32_identity" if name == "film" else "baseline32",
                           xx, pp, device)
        val, _ = summary(val_pred, yy, valid, p90)
        history.append({"epoch": epoch, "val_acc": val["mean_acc"],
                        "val_rmse": val["rmse"]})
        if val["mean_acc"] > best_acc:
            best_acc, best_epoch = val["mean_acc"], epoch
            best_state = {key: value.detach().cpu().clone()
                          for key, value in model.state_dict().items()}
        if epoch == 1 or epoch % 10 == 0:
            print(f"seed={seed} {name} epoch={epoch} val_acc={val['mean_acc']:.4f} "
                  f"best={best_acc:.4f} at {best_epoch}", flush=True)
    model.load_state_dict(best_state)
    val_pred = predict(model, "film32_identity" if name == "film" else "baseline32",
                       xx, pp, device)
    val, _ = summary(val_pred, yy, valid, p90)
    return model, {"initial_validation": initial, "validation": val,
                   "best_epoch": best_epoch, "history": history}, val_pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--experiment", default="strict_finetune_v1")
    args = parser.parse_args()
    if args.epochs <= 0 or not args.experiment or "/" in args.experiment or "\\" in args.experiment:
        parser.error("epochs must be positive and experiment must be a directory name")
    paths = SubprojectPaths(__file__)
    exp_dir = paths.results_dir / args.experiment
    if exp_dir.exists():
        raise FileExistsError(exp_dir)
    exp_dir = Path(paths.get_exp_dir(args.experiment))
    result_file = exp_dir / "finetune_metrics.json"
    dates = build_date_index()
    sst = np.load(config.modelconfig["sst_file"]).astype(np.float32)
    y = np.load(Path(config.modelconfig["hr_path"]) / "hr_data.npy").astype(np.float32)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    payload = {
        "protocol": {"candidate": "joint deep-layer and 30-PC trajectory head fine-tuning",
                     "control": "same deep-layer continuation of Baseline",
                     "trainable_layers": list(DEEP_MODULES), "epochs": args.epochs,
                     "backbone_lr": 2e-5, "head_lr": 1e-4, "loss": "masked MSE",
                     "checkpoint_selection": "validation ACC including epoch zero",
                     "holdout_gate": "FiLM exceeds matched continued Baseline and frozen FiLM on both fold mean validation ACC",
                     "holdout_status": "previously inspected periods are exploratory"},
        "folds": {},
    }
    for fold, spec in FOLDS.items():
        x, digest, train_mask, val_mask, _, valid, p90, _ = fold_data(
            paths, spec, sst, y, dates, windows=(6,),
        )
        pcs, _, _ = compute_strict_sst_pcs(sst, train_mask, 5, 6, keep_lags=True)
        x_t, pcs_t = torch.from_numpy(x), torch.from_numpy(pcs)
        y_t = torch.from_numpy(y[:, None])
        mask_t = torch.isnan(y_t) | torch.from_numpy(~valid)[None, None]
        train_ds = TensorDataset(y_t[train_mask], x_t[train_mask],
                                 mask_t[train_mask], pcs_t[train_mask])
        val_arrays = (x_t[val_mask], pcs_t[val_mask], y[val_mask], valid, p90)
        fold_result = {"input_sha256": digest,
                       "trajectory_sha256": hashlib.sha256(pcs.tobytes()).hexdigest(),
                       "seeds": {}}
        for seed in SEEDS:
            rows = {}
            for name in ("baseline", "film"):
                model, info, val_pred = continue_training(
                    name, make_model(name, paths, fold, seed, device),
                    train_ds, val_arrays, device, args.epochs, seed,
                )
                torch.save(model.state_dict(),
                           exp_dir / "checkpoints" / f"{fold}_s{seed}_{name}.pt")
                np.savez_compressed(exp_dir / f"{fold}_s{seed}_{name}_validation.npz",
                                    dates=dates[val_mask].strftime("%Y-%m-%d").values,
                                    prediction=val_pred)
                rows[name] = info
                del model
            fold_result["seeds"][str(seed)] = rows
            payload["folds"][fold] = fold_result
            result_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    fold_means = {}
    for fold in FOLDS:
        rows = payload["folds"][fold]["seeds"]
        fold_means[fold] = {
            name: float(np.mean([rows[str(seed)][name]["validation"]["mean_acc"]
                                 for seed in SEEDS])) for name in ("baseline", "film")
        }
        fold_means[fold]["frozen_film"] = float(np.mean([
            rows[str(seed)]["film"]["initial_validation"]["mean_acc"] for seed in SEEDS
        ]))
    passed = all(row["film"] > max(row["baseline"], row["frozen_film"])
                 for row in fold_means.values())
    payload["selection"] = {"fold_mean_val_acc": fold_means,
                            "passed_two_fold_gate": passed}
    result_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"fine-tuning selection: {payload['selection']}", flush=True)
    if not passed:
        print("Fine-tuning did not pass the two-fold validation gate.", flush=True)
        return

    for fold, spec in FOLDS.items():
        x, _, train_mask, _, holdout_mask, valid, p90, _ = fold_data(
            paths, spec, sst, y, dates, windows=(6,),
        )
        pcs, _, _ = compute_strict_sst_pcs(sst, train_mask, 5, 6, keep_lags=True)
        xx = torch.from_numpy(x[holdout_mask])
        pp = torch.from_numpy(pcs[holdout_mask])
        for seed in SEEDS:
            predictions, metrics, series = {}, {}, {}
            for name in ("baseline", "film"):
                model = make_model(name, paths, fold, seed, device)
                model.load_state_dict(torch.load(
                    exp_dir / "checkpoints" / f"{fold}_s{seed}_{name}.pt",
                    map_location=device, weights_only=True,
                ))
                predictions[name] = predict(model, "film32_identity" if name == "film" else
                                            "baseline32", xx, pp, device)
                metrics[name], series[name] = summary(predictions[name],
                                                      y[holdout_mask], valid, p90)
                del model
            payload["folds"][fold]["seeds"][str(seed)]["exploratory_holdout"] = {
                "baseline": metrics["baseline"], "film": metrics["film"],
                "paired_acc": paired_stat_tests(series["film"], series["baseline"],
                                                 dates[holdout_mask].year.values),
            }
            np.savez_compressed(exp_dir / f"{fold}_s{seed}_holdout.npz",
                                dates=dates[holdout_mask].strftime("%Y-%m-%d").values,
                                target=y[holdout_mask], **predictions)
            print(f"{fold} seed={seed} exploratory holdout ACC "
                  f"FiLM={metrics['film']['mean_acc']:.4f} "
                  f"Baseline={metrics['baseline']['mean_acc']:.4f}", flush=True)
    result_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved {result_file}", flush=True)


if __name__ == "__main__":
    main()

"""Validation-gated FiLM adaptation of a Baseline trained on strict inputs."""

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--experiment", default="strict_residual_v1")
    parser.add_argument("--fair-experiment", default="strict_fair_v1")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.epochs <= 0 or not args.experiment or "/" in args.experiment or "\\" in args.experiment:
        parser.error("epochs must be positive and experiment must be a directory name")

    set_seed(args.seed)
    paths = SubprojectPaths(__file__)
    fair_dir = paths.results_dir / args.fair_experiment
    fair = json.loads((fair_dir / "strict_fair_metrics.json").read_text(encoding="utf-8"))
    if fair["protocol"]["seed"] != args.seed:
        raise ValueError("residual seed must match the Baseline training seed")
    input_file = paths.cache_dir / "strict_inputs" / "cond_trainfit.npy"
    if hashlib.sha256(input_file.read_bytes()).hexdigest() != fair["input_sha256"]:
        raise ValueError("strict input differs from the Baseline training input")
    exp_dir = Path(paths.get_exp_dir(args.experiment))
    result_file = exp_dir / "strict_residual_metrics.json"
    if result_file.exists() or any((exp_dir / "checkpoints").glob("*.pt")):
        raise FileExistsError(f"experiment already contains results: {exp_dir}")

    dates = build_date_index()
    train_mask = np.asarray(dates <= "2017-12-01")
    val_mask = np.asarray((dates >= "2018-01-01") & (dates <= "2022-12-01"))
    test_mask = np.asarray(dates >= "2023-01-01")
    x = np.load(input_file).astype(np.float32)
    y = np.load(Path(config.modelconfig["hr_path"]) / "hr_data.npy").astype(np.float32)
    sst = np.load(config.modelconfig["sst_file"]).astype(np.float32)
    pcs, _, _ = compute_strict_sst_pcs(sst, train_mask, 5)
    valid = np.isfinite(y).all(axis=0)
    p90 = float(np.percentile(y[train_mask][:, valid], 90))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    x_t, pcs_t = torch.from_numpy(x), torch.from_numpy(pcs)
    y_t = torch.from_numpy(y[:, None])
    mask_t = torch.isnan(y_t) | torch.from_numpy(~valid)[None, None]
    ds = TensorDataset(y_t[train_mask], x_t[train_mask], mask_t[train_mask], pcs_t[train_mask])
    loader = DataLoader(ds, batch_size=8, shuffle=True, drop_last=True)

    base_weights = torch.load(fair_dir / "checkpoints" / "baseline32.pt",
                              map_location="cpu", weights_only=True)
    baseline = UNetLite(n_channels=10, base_filters=32, dropout=0.1).to(device)
    baseline.load_state_dict(base_weights)
    baseline.eval()
    film = UNetLiteFiLM(n_channels=10, index_dim=5, base_filters=32, dropout=0.1,
                        identity_init=True).to(device)
    missing, unexpected = film.load_state_dict(base_weights, strict=False)
    if unexpected or any(not key.startswith("index_encoder.") for key in missing):
        raise RuntimeError(f"unexpected transfer mismatch: {missing}, {unexpected}")
    film.eval()
    with torch.no_grad():
        transfer_diff = float((baseline(x_t[val_mask][:2].to(device)) -
                               film(x_t[val_mask][:2].to(device), pcs_t[val_mask][:2].to(device))).abs().max())
    if transfer_diff > 1e-5:
        raise RuntimeError(f"identity transfer differs from baseline: {transfer_diff}")
    for name, parameter in film.named_parameters():
        parameter.requires_grad_(name.startswith("index_encoder."))

    optimizer = torch.optim.AdamW(film.index_encoder.parameters(), lr=2e-4, weight_decay=1e-4)
    baseline_val_acc = fair["validation"]["baseline32"]["validation"]["mean_acc"]
    best_acc, best_epoch = baseline_val_acc, 0
    best_state = {key: value.detach().cpu().clone() for key, value in film.state_dict().items()}
    history = []
    for epoch in range(1, args.epochs + 1):
        film.eval()
        for target, cond, mask, indices in loader:
            target, cond, mask, indices = (item.to(device) for item in (target, cond, mask, indices))
            optimizer.zero_grad(set_to_none=True)
            pred = film(cond, indices)
            gamma, beta = film.index_encoder(indices)
            penalty = sum(g.square().mean() + b.square().mean() for g, b in zip(gamma, beta)) / 5
            loss = F.mse_loss(pred[~mask], target[~mask]) + 0.1 * penalty
            loss.backward()
            torch.nn.utils.clip_grad_norm_(film.index_encoder.parameters(), 2.0)
            optimizer.step()
        pred_val = predict(film, "film32_identity", x_t[val_mask], pcs_t[val_mask], device)
        val_metrics, _ = summary(pred_val, y[val_mask], valid, p90)
        history.append({"epoch": epoch, "val_acc": val_metrics["mean_acc"],
                        "val_rmse": val_metrics["rmse"]})
        if val_metrics["mean_acc"] > best_acc:
            best_acc = val_metrics["mean_acc"]
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in film.state_dict().items()}
        if epoch == 1 or epoch % 10 == 0:
            print(f"residual epoch={epoch} val_acc={val_metrics['mean_acc']:.4f} "
                  f"best={best_acc:.4f} at {best_epoch}", flush=True)

    film.load_state_dict(best_state)
    torch.save(film.state_dict(), exp_dir / "checkpoints" / "film_residual.pt")
    val_pred = predict(film, "film32_identity", x_t[val_mask], pcs_t[val_mask], device)
    val_zero = predict(film, "film32_identity", x_t[val_mask], torch.zeros_like(pcs_t[val_mask]), device)
    val, _ = summary(val_pred, y[val_mask], valid, p90)
    zero_val, _ = summary(val_zero, y[val_mask], valid, p90)
    payload = {
        "protocol": f"frozen strict-input Baseline; identity FiLM; train index encoder "
                    f"{args.epochs} epochs; select epoch by validation ACC; "
                    "evaluate test only if validation improves Baseline",
        "input_sha256": fair["input_sha256"], "transfer_max_abs_diff": transfer_diff,
        "seed": args.seed, "fair_experiment": args.fair_experiment,
        "baseline_val_acc": baseline_val_acc, "best_epoch": best_epoch,
        "validation": val, "validation_pc_zero": zero_val, "history": history,
        "test_status": "not evaluated: no validation ACC improvement" if best_epoch == 0 else
                       "exploratory: earlier experiments inspected test period",
    }
    if best_epoch > 0:
        pred_test = predict(film, "film32_identity", x_t[test_mask], pcs_t[test_mask], device)
        test, acc = summary(pred_test, y[test_mask], valid, p90)
        with np.load(fair_dir / "test_predictions.npz") as prior:
            base_pred = prior["baseline32"]
            cat_pred = prior["eof_cat32"]
        base, base_acc = summary(base_pred, y[test_mask], valid, p90)
        cat, cat_acc = summary(cat_pred, y[test_mask], valid, p90)
        payload["test"] = test
        payload["test_comparators"] = {"baseline32": base, "eof_cat32": cat}
        payload["paired_acc"] = {
            "baseline32": paired_stat_tests(acc, base_acc, dates[test_mask].year.values),
            "eof_cat32": paired_stat_tests(acc, cat_acc, dates[test_mask].year.values),
        }
        np.savez_compressed(exp_dir / "test_predictions.npz",
                            dates=dates[test_mask].strftime("%Y-%m-%d").values,
                            target=y[test_mask], film_residual=pred_test,
                            baseline32=base_pred, eof_cat32=cat_pred)
        print(f"residual exploratory test={test}", flush=True)
    result_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved {result_file}", flush=True)


if __name__ == "__main__":
    main()

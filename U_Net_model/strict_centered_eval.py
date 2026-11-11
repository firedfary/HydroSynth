"""Validation-first test of zero-PC-identity FiLM on frozen strict-input U-Nets."""

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
from film_recovery import spatial_acc
from run_unified_eval import build_date_index, compute_strict_sst_pcs, paired_stat_tests, set_seed
from strict_fair_eval import predict, summary


CANDIDATES = {
    "deep_mse": ((3, 4), 0.0),
    "deep_acc": ((3, 4), 1.0),
    "bottle_mse": ((4,), 0.0),
    "bottle_acc": ((4,), 1.0),
}
FAIR_EXPERIMENTS = {42: "strict_fair_v1", 43: "strict_fair_s43", 44: "strict_fair_s44"}


def train_candidate(name, layers, acc_weight, base_weights, dataset, val_arrays,
                    baseline_val_acc, device, epochs, seed, condition_dropout=0.0,
                    loss_name="mse", reference_zero_dims=None):
    set_seed(seed)
    model = UNetLiteFiLM(
        n_channels=10, index_dim=val_arrays[1].shape[1], base_filters=32, index_hidden=16,
        dropout=0.1, film_layers=layers, identity_init=True, zero_pc_identity=True,
        reference_zero_dims=reference_zero_dims,
    ).to(device)
    missing, unexpected = model.load_state_dict(base_weights, strict=False)
    if unexpected or any(not key.startswith("index_encoder.") for key in missing):
        raise RuntimeError(f"baseline transfer mismatch: {missing}, {unexpected}")
    for key, parameter in model.named_parameters():
        parameter.requires_grad_(key.startswith("index_encoder."))
    model.eval()
    x_val, pcs_val, y_val, valid, p90 = val_arrays
    with torch.no_grad():
        zero_pred = predict(model, "film32_identity", x_val[:2],
                            model.reference_indices(pcs_val[:2]), device)
    reference = UNetLite(n_channels=10, base_filters=32, dropout=0.1).to(device)
    reference.load_state_dict(base_weights)
    base_pred = predict(reference, "baseline32", x_val[:2], pcs_val[:2], device)
    transfer_diff = float(np.max(np.abs(zero_pred - base_pred)))
    if transfer_diff > 1e-5:
        raise RuntimeError(f"zero-PC FiLM differs from baseline: {transfer_diff}")
    del reference

    loader = DataLoader(dataset, batch_size=8, shuffle=True, drop_last=True)
    if not 0.0 <= condition_dropout < 1.0:
        raise ValueError("condition_dropout must be in [0, 1)")
    if loss_name not in ("mse", "huber1"):
        raise ValueError("loss_name must be mse or huber1")
    optimizer = torch.optim.AdamW(model.index_encoder.parameters(), lr=2e-4,
                                  weight_decay=1e-4)
    best_acc, best_epoch, best_state = -float("inf"), 0, None
    history = []
    for epoch in range(1, epochs + 1):
        model.eval()
        for target, cond, mask, pcs in loader:
            target, cond, mask, pcs = (item.to(device) for item in (target, cond, mask, pcs))
            optimizer.zero_grad(set_to_none=True)
            train_pcs = pcs
            if condition_dropout:
                keep = (torch.rand(len(pcs), 1, device=device) >= condition_dropout)
                train_pcs = torch.where(keep, pcs, model.reference_indices(pcs))
            pred = model(cond, train_pcs)
            gammas, betas = model.index_encoder(pcs)
            zero_gammas, zero_betas = model.index_encoder(model.reference_indices(pcs))
            penalty = sum(
                (gammas[level] - zero_gammas[level]).square().mean() +
                (betas[level] - zero_betas[level]).square().mean()
                for level in layers
            ) / len(layers)
            data_loss = (F.huber_loss(pred[~mask], target[~mask], delta=1.0)
                         if loss_name == "huber1" else
                         F.mse_loss(pred[~mask], target[~mask]))
            loss = (data_loss +
                    acc_weight * (1.0 - spatial_acc(pred, target, mask)) +
                    0.1 * penalty)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.index_encoder.parameters(), 2.0)
            optimizer.step()
        val_pred = predict(model, "film32_identity", x_val, pcs_val, device)
        val, _ = summary(val_pred, y_val, valid, p90)
        val_acc = val["mean_acc"]
        history.append({"epoch": epoch, "val_acc": val_acc, "val_rmse": val["rmse"]})
        if val_acc > best_acc:
            best_acc, best_epoch = val_acc, epoch
            best_state = {key: value.detach().cpu().clone()
                          for key, value in model.state_dict().items()}
        if epoch == 1 or epoch % 20 == 0:
            print(f"seed={seed} {name} epoch={epoch} val_acc={val_acc:.4f} "
                  f"best={best_acc:.4f} baseline={baseline_val_acc:.4f}", flush=True)

    model.load_state_dict(best_state)
    val_pred = predict(model, "film32_identity", x_val, pcs_val, device)
    val_zero = predict(model, "film32_identity", x_val,
                       model.reference_indices(pcs_val), device)
    val, _ = summary(val_pred, y_val, valid, p90)
    zero, _ = summary(val_zero, y_val, valid, p90)
    info = {"best_epoch": best_epoch, "validation": val, "validation_pc_zero": zero,
            "transfer_max_abs_diff": transfer_diff, "history": history}
    return model, info, val_pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--experiment", default="strict_centered_v1")
    args = parser.parse_args()
    if args.epochs <= 0 or not args.experiment or "/" in args.experiment or "\\" in args.experiment:
        parser.error("epochs must be positive and experiment must be a directory name")

    paths = SubprojectPaths(__file__)
    input_file = paths.cache_dir / "strict_inputs" / "cond_trainfit.npy"
    input_hash = hashlib.sha256(input_file.read_bytes()).hexdigest()
    fair = {}
    for seed, label in FAIR_EXPERIMENTS.items():
        fair_dir = paths.results_dir / label
        fair[seed] = json.loads((fair_dir / "strict_fair_metrics.json").read_text(encoding="utf-8"))
        if fair[seed]["input_sha256"] != input_hash or fair[seed]["protocol"]["seed"] != seed:
            raise ValueError(f"strict input or seed differs from Baseline run {label}")
    exp_dir = paths.results_dir / args.experiment
    if exp_dir.exists():
        raise FileExistsError(exp_dir)
    exp_dir = Path(paths.get_exp_dir(args.experiment))

    dates = build_date_index()
    train_mask = np.asarray(dates <= "2017-12-01")
    val_mask = np.asarray((dates >= "2018-01-01") & (dates <= "2022-12-01"))
    test_mask = np.asarray(dates >= "2023-01-01")
    x = np.load(input_file).astype(np.float32)
    y = np.load(Path(config.modelconfig["hr_path"]) / "hr_data.npy").astype(np.float32)
    sst = np.load(config.modelconfig["sst_file"]).astype(np.float32)
    pcs, _, explained = compute_strict_sst_pcs(sst, train_mask, 5)
    valid = np.isfinite(y).all(axis=0)
    p90 = float(np.percentile(y[train_mask][:, valid], 90))
    x_t, pcs_t = torch.from_numpy(x), torch.from_numpy(pcs)
    y_t = torch.from_numpy(y[:, None])
    mask_t = torch.isnan(y_t) | torch.from_numpy(~valid)[None, None]
    train_ds = TensorDataset(y_t[train_mask], x_t[train_mask],
                             mask_t[train_mask], pcs_t[train_mask])
    val_arrays = (x_t[val_mask], pcs_t[val_mask], y[val_mask], valid, p90)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    payload = {
        "protocol": {"train": "1994-01/2017-12", "validation": "2018-01/2022-12",
                     "test": "2023-01/2024-09", "seeds": list(FAIR_EXPERIMENTS),
                     "epochs": args.epochs, "batch_size": 8, "head_lr": 2e-4,
                     "head_weight_decay": 1e-4, "index_hidden": 16,
                     "penalty_weight": 0.1, "checkpoint_selection": "validation ACC",
                     "candidate_selection": "mean validation ACC across three seeds",
                     "test_status": "exploratory: period already inspected in earlier work"},
        "input_sha256": input_hash,
        "sst_pca_explained_variance": [float(value) for value in explained],
        "candidates": {name: {"layers": list(layers), "acc_weight": weight}
                       for name, (layers, weight) in CANDIDATES.items()},
        "seeds": {},
    }
    for seed, label in FAIR_EXPERIMENTS.items():
        fair_dir = paths.results_dir / label
        base_weights = torch.load(fair_dir / "checkpoints" / "baseline32.pt",
                                  map_location="cpu", weights_only=True)
        base_val_acc = fair[seed]["validation"]["baseline32"]["validation"]["mean_acc"]
        seed_info = {"fair_experiment": label, "baseline_val_acc": base_val_acc,
                     "candidates": {}}
        for name, (layers, acc_weight) in CANDIDATES.items():
            model, info, val_pred = train_candidate(
                name, layers, acc_weight, base_weights, train_ds, val_arrays,
                base_val_acc, device, args.epochs, seed,
            )
            torch.save(model.state_dict(), exp_dir / "checkpoints" / f"s{seed}_{name}.pt")
            np.savez_compressed(exp_dir / f"s{seed}_{name}_validation.npz",
                                dates=dates[val_mask].strftime("%Y-%m-%d").values,
                                prediction=val_pred)
            seed_info["candidates"][name] = info
            del model
        payload["seeds"][str(seed)] = seed_info
        (exp_dir / "strict_centered_metrics.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8")

    candidate_means = {
        name: float(np.mean([payload["seeds"][str(seed)]["candidates"][name]
                             ["validation"]["mean_acc"] for seed in FAIR_EXPERIMENTS]))
        for name in CANDIDATES
    }
    baseline_mean = float(np.mean([payload["seeds"][str(seed)]["baseline_val_acc"]
                                   for seed in FAIR_EXPERIMENTS]))
    selected = max(CANDIDATES, key=candidate_means.get)
    payload["selection"] = {"candidate_mean_val_acc": candidate_means,
                            "baseline_mean_val_acc": baseline_mean,
                            "selected": selected if candidate_means[selected] > baseline_mean else None}
    print(f"validation selection: {payload['selection']}", flush=True)
    if payload["selection"]["selected"] is not None:
        layers, _ = CANDIDATES[selected]
        for seed, label in FAIR_EXPERIMENTS.items():
            model = UNetLiteFiLM(
                n_channels=10, index_dim=5, base_filters=32, index_hidden=16,
                dropout=0.1, film_layers=layers, identity_init=True,
                zero_pc_identity=True,
            ).to(device)
            model.load_state_dict(torch.load(exp_dir / "checkpoints" / f"s{seed}_{selected}.pt",
                                             map_location=device, weights_only=True))
            pred = predict(model, "film32_identity", x_t[test_mask], pcs_t[test_mask], device)
            zero_pred = predict(model, "film32_identity", x_t[test_mask],
                                torch.zeros_like(pcs_t[test_mask]), device)
            test, acc = summary(pred, y[test_mask], valid, p90)
            zero_test, _ = summary(zero_pred, y[test_mask], valid, p90)
            with np.load(paths.results_dir / label / "test_predictions.npz") as prior:
                base_pred = prior["baseline32"]
                cat_pred = prior["eof_cat32"]
            base, base_acc = summary(base_pred, y[test_mask], valid, p90)
            cat, cat_acc = summary(cat_pred, y[test_mask], valid, p90)
            payload["seeds"][str(seed)]["exploratory_test"] = {
                "film": test, "film_pc_zero": zero_test, "baseline": base,
                "eof_cat": cat,
                "paired_acc_baseline": paired_stat_tests(acc, base_acc, dates[test_mask].year.values),
                "paired_acc_eof_cat": paired_stat_tests(acc, cat_acc, dates[test_mask].year.values),
            }
            np.savez_compressed(exp_dir / f"s{seed}_{selected}_test.npz",
                                dates=dates[test_mask].strftime("%Y-%m-%d").values,
                                target=y[test_mask], film=pred, film_pc_zero=zero_pred,
                                baseline=base_pred, eof_cat=cat_pred)
            print(f"seed={seed} exploratory test ACC={test['mean_acc']:.4f} "
                  f"baseline={base['mean_acc']:.4f}", flush=True)
            del model
    result_file = exp_dir / "strict_centered_metrics.json"
    result_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved {result_file}", flush=True)


if __name__ == "__main__":
    main()

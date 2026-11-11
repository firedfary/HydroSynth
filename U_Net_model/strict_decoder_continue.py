"""Paired decoder continuation of Baseline and shallow SST FiLM."""

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
from run_unified_eval import build_date_index, compute_strict_sst_pcs, set_seed
from strict_fair_eval import predict, summary
from strict_sst_window_eval import FOLDS, SEEDS, baseline_weights, fold_data


FOLDS = {**FOLDS, "early": {"input": "strict_pre2005", "train_end": "2004-12-01",
                            "val_end": "2007-12-01", "holdout_end": "2010-12-01"}}
SOURCE = {"recent": "film_early_v1", "historical": "film_early_v1",
          "early": "film_early_origin_v1"}
DECODER = ("up1.", "up2.", "up3.", "up4.", "outc.")
FILM_LAYERS = (1, 2, 3, 4)


def source_info(paths, fold, digest, pc_digest):
    source = paths.results_dir / SOURCE[fold]
    if fold == "early":
        prior = json.loads((source / "early_origin_metrics.json").read_text(encoding="utf-8"))
        expected_input, expected_pc = prior["input_sha256"], prior["trajectory_sha256"]
    else:
        prior = json.loads((source / "trajectory_metrics.json").read_text(encoding="utf-8"))
        if prior["protocol"]["film_layers"] != list(FILM_LAYERS):
            raise ValueError(f"unexpected FiLM layers for {fold}")
        expected_input = prior["folds"][fold]["input_sha256"]
        expected_pc = prior["folds"][fold]["trajectory_sha256"]
    if (digest, pc_digest) != (expected_input, expected_pc):
        raise ValueError(f"input or SST trajectory changed for {fold}")
    return source


def make_model(paths, fold, seed, name, source, device):
    if name == "baseline":
        model = UNetLite(n_channels=10, base_filters=32, dropout=0.1)
        if fold == "early":
            weights = torch.load(source / "checkpoints" / f"baseline_s{seed}.pt",
                                 map_location="cpu", weights_only=True)
        else:
            weights = baseline_weights(paths, fold, seed)
    else:
        model = UNetLiteFiLM(
            n_channels=10, index_dim=30, base_filters=32, index_hidden=16,
            dropout=0.1, film_layers=FILM_LAYERS, identity_init=True,
            zero_pc_identity=True,
        )
        filename = (f"film_s{seed}.pt" if fold == "early" else
                    f"{fold}_s{seed}.pt")
        weights = torch.load(source / "checkpoints" / filename,
                             map_location="cpu", weights_only=True)
    model.load_state_dict(weights)
    return model.to(device)


def continue_model(model, name, dataset, val_arrays, device, epochs, seed):
    set_seed(seed)
    decoder, head = [], []
    for key, parameter in model.named_parameters():
        is_decoder = key.startswith(DECODER)
        is_head = name == "film" and key.startswith("index_encoder.")
        parameter.requires_grad_(is_decoder or is_head)
        if is_decoder:
            decoder.append(parameter)
        elif is_head:
            head.append(parameter)
    if not decoder or (name == "film" and not head):
        raise RuntimeError("decoder or FiLM head was not selected")
    groups = [{"params": decoder, "lr": 2e-5}]
    if head:
        groups.append({"params": head, "lr": 1e-4})
    optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
    loader = DataLoader(dataset, batch_size=8, shuffle=True, drop_last=True)
    xx, pp, yy, valid, p90 = val_arrays
    mode = "film32_identity" if name == "film" else "baseline32"
    model.eval()
    initial_pred = predict(model, mode, xx, pp, device)
    initial, _ = summary(initial_pred, yy, valid, p90)
    best_acc, best_epoch = initial["mean_acc"], 0
    best_state = {key: value.detach().cpu().clone()
                  for key, value in model.state_dict().items()}
    history = [{"epoch": 0, "val_acc": best_acc, "val_rmse": initial["rmse"]}]
    for epoch in range(1, epochs + 1):
        model.eval()
        for target, cond, mask, pcs in loader:
            target, cond, mask, pcs = (item.to(device) for item in
                                       (target, cond, mask, pcs))
            optimizer.zero_grad(set_to_none=True)
            pred = model(cond, pcs) if name == "film" else model(cond)
            loss = F.mse_loss(pred[~mask], target[~mask])
            if name == "film":
                gammas, betas = model.index_encoder(pcs)
                zero_gammas, zero_betas = model.index_encoder(torch.zeros_like(pcs))
                penalty = sum(
                    (gammas[level] - zero_gammas[level]).square().mean() +
                    (betas[level] - zero_betas[level]).square().mean()
                    for level in FILM_LAYERS
                ) / len(FILM_LAYERS)
                loss = loss + 0.1 * penalty
            loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder + head, 2.0)
            optimizer.step()
        val_pred = predict(model, mode, xx, pp, device)
        val, _ = summary(val_pred, yy, valid, p90)
        acc = val["mean_acc"]
        history.append({"epoch": epoch, "val_acc": acc, "val_rmse": val["rmse"]})
        if acc > best_acc:
            best_acc, best_epoch = acc, epoch
            best_state = {key: value.detach().cpu().clone()
                          for key, value in model.state_dict().items()}
        if epoch == 1 or epoch % 10 == 0:
            print(f"seed={seed} {name} epoch={epoch} val_acc={acc:.5f} "
                  f"best={best_acc:.5f} at {best_epoch}", flush=True)
    model.load_state_dict(best_state)
    return model, {"initial": initial, "best_epoch": best_epoch,
                   "validation": summary(predict(model, mode, xx, pp, device),
                                         yy, valid, p90)[0], "history": history}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--experiment", default="film_dec_cont_v1")
    args = parser.parse_args()
    if (args.epochs <= 0 or not args.experiment or "/" in args.experiment or
            "\\" in args.experiment):
        parser.error("invalid epochs or experiment name")
    paths = SubprojectPaths(__file__)
    if (paths.results_dir / args.experiment).exists():
        raise FileExistsError(paths.results_dir / args.experiment)
    exp_dir = Path(paths.get_exp_dir(args.experiment))
    result_file = exp_dir / "decoder_metrics.json"
    dates = build_date_index()
    sst = np.load(config.modelconfig["sst_file"]).astype(np.float32)
    y = np.load(Path(config.modelconfig["hr_path"]) / "hr_data.npy").astype(np.float32)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    payload = {
        "protocol": {
            "candidate": "shallow FiLM plus trainable decoder, frozen encoder",
            "control": "matched Baseline plus trainable decoder, frozen encoder",
            "sources": SOURCE, "seeds": list(SEEDS), "epochs": args.epochs,
            "film_layers": list(FILM_LAYERS), "trainable_decoder": list(DECODER),
            "decoder_lr": 2e-5, "film_head_lr": 1e-4, "batch_size": 8,
            "loss": "masked MSE, plus 0.1 FiLM penalty for candidate",
            "checkpoint_selection": "validation ACC, including epoch 0",
            "selection_gate": "candidate beats both matched continued Baseline and initial shallow FiLM in all three fold mean validation ACC values",
            "holdout_status": "not accessed by this experiment",
        }, "folds": {},
    }
    for fold, spec in FOLDS.items():
        x, digest, train_mask, val_mask, _, valid, p90, _ = fold_data(
            paths, spec, sst, y, dates, windows=(6,),
        )
        pcs, _, _ = compute_strict_sst_pcs(sst, train_mask, 5, 6, keep_lags=True)
        pc_digest = hashlib.sha256(pcs.tobytes()).hexdigest()
        source = source_info(paths, fold, digest, pc_digest)
        x_t, pcs_t = torch.from_numpy(x), torch.from_numpy(pcs)
        y_t = torch.from_numpy(y[:, None])
        mask_t = torch.isnan(y_t) | torch.from_numpy(~valid)[None, None]
        dataset = TensorDataset(y_t[train_mask], x_t[train_mask],
                                mask_t[train_mask], pcs_t[train_mask])
        val_arrays = (x_t[val_mask], pcs_t[val_mask], y[val_mask], valid, p90)
        row = {"input_sha256": digest, "trajectory_sha256": pc_digest, "seeds": {}}
        for seed in SEEDS:
            row["seeds"][str(seed)] = {}
            for name in ("baseline", "film"):
                model, info = continue_model(
                    make_model(paths, fold, seed, name, source, device),
                    name, dataset, val_arrays, device, args.epochs, seed,
                )
                torch.save(model.state_dict(), exp_dir / "checkpoints" /
                           f"{fold}_s{seed}_{name}.pt")
                row["seeds"][str(seed)][name] = info
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            payload["folds"][fold] = row
            result_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"completed {fold}", flush=True)
    means = {}
    for fold, row in payload["folds"].items():
        means[fold] = {
            "baseline": float(np.mean([row["seeds"][str(seed)]["baseline"]
                                       ["validation"]["mean_acc"] for seed in SEEDS])),
            "film": float(np.mean([row["seeds"][str(seed)]["film"]
                                   ["validation"]["mean_acc"] for seed in SEEDS])),
            "initial_film": float(np.mean([row["seeds"][str(seed)]["film"]
                                           ["initial"]["mean_acc"] for seed in SEEDS])),
        }
    payload["selection"] = {
        "fold_mean_val_acc": means,
        "passed_three_fold_gate": all(
            value["film"] > max(value["baseline"], value["initial_film"])
            for value in means.values()
        ),
    }
    result_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"selection: {payload['selection']}", flush=True)


if __name__ == "__main__":
    main()

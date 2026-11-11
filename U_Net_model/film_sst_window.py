"""Compare recent SST windows with the existing six-month mean PC input."""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import TensorDataset

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from utils.paths import SubprojectPaths
from film_recovery import matched_metrics, predict, train_candidate
from run_unified_eval import build_date_index, compute_strict_sst_pcs


def main():
    paths = SubprojectPaths(__file__)
    exp_dir = Path(paths.get_exp_dir("film_sst_window"))
    objective_dir = Path(paths.get_exp_dir("film_objective"))
    previous = json.loads((objective_dir / "film_objective.json").read_text(encoding="utf-8"))
    incumbent_acc = previous["candidates"]["huber1_acc1"]["val"]["mean_acc"]
    dates = build_date_index()
    train_mask = dates <= "2017-12-01"
    val_mask = (dates >= "2018-01-01") & (dates <= "2022-12-01")
    test_mask = dates >= "2023-01-01"
    target = np.load(Path(config.modelconfig["hr_path"]) / "hr_data.npy").astype(np.float32)
    cond = np.load(Path(config.modelconfig["lr_path"]) / "lr_data_reconstructed2.npy").astype(np.float32)
    sst = np.load(config.modelconfig["sst_file"]).astype(np.float32)
    valid = np.isfinite(target[0])
    p90 = float(np.percentile(target[train_mask][:, valid], 90))
    y_t = torch.from_numpy(target[:, None])
    x_t = torch.from_numpy(cond)
    m_t = torch.isnan(y_t) | torch.from_numpy(~valid)[None, None]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    candidates = {}
    models = {}
    pcs_by_name = {}
    for name, n_months in (("last1", 1), ("last3", 3)):
        sst_window = np.nanmean(sst[:, -n_months:], axis=1)
        pcs, _, variance = compute_strict_sst_pcs(sst_window, train_mask, 5)
        pcs_t = torch.from_numpy(pcs)
        train_ds = TensorDataset(y_t[train_mask], x_t[train_mask], m_t[train_mask], pcs_t[train_mask])
        val_ds = TensorDataset(y_t[val_mask], x_t[val_mask], m_t[val_mask], pcs_t[val_mask])
        model, info = train_candidate(name, (0, 1, 2, 3, 4), train_ds, val_ds, device, 100,
                                      objective="huber1_acc1", select_acc=True)
        val_pred = predict(model, x_t[val_mask], pcs_t[val_mask], device)
        val_zero = predict(model, x_t[val_mask], torch.zeros_like(pcs_t[val_mask]), device)
        info["val"] = {key: value for key, value in
                       matched_metrics(val_pred, target[val_mask], valid, p90).items()
                       if key in ("mean_acc", "mean_tcc", "rmse")}
        info["val_pc_zero"] = {key: value for key, value in
                               matched_metrics(val_zero, target[val_mask], valid, p90).items()
                               if key in ("mean_acc", "mean_tcc", "rmse")}
        info["sst_months"] = n_months
        info["pca_explained_variance"] = float(variance.sum())
        candidates[name] = info
        models[name] = model
        pcs_by_name[name] = pcs_t
        torch.save(model.state_dict(), exp_dir / "checkpoints" / f"{name}.pt")
        print(f"{name}: validation {info['val']}", flush=True)

    selected = max(candidates, key=lambda name: candidates[name]["val"]["mean_acc"])
    if candidates[selected]["val"]["mean_acc"] <= incumbent_acc:
        selected = "mean6"
        test = None
        print("Recent-window candidates did not beat mean6 on validation ACC; test not evaluated.", flush=True)
    else:
        pred = predict(models[selected], x_t[test_mask], pcs_by_name[selected][test_mask], device)
        metrics = matched_metrics(pred, target[test_mask], valid, p90)
        test = {key: metrics[key] for key in ("mean_acc", "mean_tcc", "rmse", "sc_pct", "csi")}
        print(f"Selected {selected} on validation; exploratory test {test}", flush=True)

    payload = {"protocol": "train 1994-2017, validation 2018-2022; Huber(1)+ACC, "
                           "width 32, all encoder layers, identity init, seed 42",
               "mean6_incumbent_val_acc": incumbent_acc, "candidates": candidates,
               "selected": selected, "selected_test": test}
    result_path = exp_dir / "film_sst_window.json"
    result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved {result_path}", flush=True)


if __name__ == "__main__":
    main()

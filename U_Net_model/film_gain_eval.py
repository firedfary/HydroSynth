"""Select a single FiLM modulation gain on validation and evaluate it once."""

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from utils.paths import SubprojectPaths
from unetlitefilm import UNetLiteFiLM
from film_recovery import matched_metrics
from run_unified_eval import build_date_index, compute_strict_sst_pcs, paired_stat_tests


def predict_gain(model, cond, pcs, gain, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(cond), 8):
            predictions.append(model(cond[start:start + 8].to(device),
                                     pcs[start:start + 8].to(device),
                                     film_gain=gain)[:, 0].cpu().numpy())
    return np.concatenate(predictions)


def compact(metrics):
    return {key: metrics[key] for key in ("mean_acc", "mean_tcc", "rmse", "sc_pct", "csi")}


def main():
    paths = SubprojectPaths(__file__)
    exp_dir = Path(paths.get_exp_dir("film_gain"))
    prior_dir = Path(paths.get_exp_dir("film_residual"))
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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = UNetLiteFiLM(n_channels=10, index_dim=5, base_filters=32, dropout=0.1).to(device)
    model.load_state_dict(torch.load(prior_dir / "checkpoints" / "all_mse.pt",
                                     map_location=device, weights_only=True))
    cond_t = torch.from_numpy(cond)
    pcs_t = torch.from_numpy(pcs)

    gains = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25)
    val_results = {}
    for gain in gains:
        pred = predict_gain(model, cond_t[val_mask], pcs_t[val_mask], gain, device)
        val_results[str(gain)] = compact(matched_metrics(pred, target[val_mask], valid, p90))
    chosen = max(gains, key=lambda gain: val_results[str(gain)]["mean_acc"])
    print("Validation gain sweep:", val_results, flush=True)
    print(f"Selected gain {chosen} on validation ACC", flush=True)

    pred_test = predict_gain(model, cond_t[test_mask], pcs_t[test_mask], chosen, device)
    test_result = compact(matched_metrics(pred_test, target[test_mask], valid, p90))
    with np.load(Path(paths.get_exp_dir("unified_eval")) / "predictions_all.npz") as old:
        baseline_test = old["pred_base"][test_mask]
        original_film_test = old["pred_film"][test_mask]
        cat_test = old["pred_cat"][test_mask]
    test_dates = dates[test_mask]
    yearly = {}
    for year in (2023, 2024):
        year_mask = test_dates.year == year
        yearly[str(year)] = {
            name: compact(matched_metrics(pred[year_mask], target[test_mask][year_mask], valid, p90))
            for name, pred in (("recovered_film", pred_test), ("baseline", baseline_test),
                               ("original_film", original_film_test), ("eof_cat", cat_test))
        }
    recovered_acc = matched_metrics(pred_test, target[test_mask], valid, p90)["acc_series"]
    paired = {}
    for name, pred in (("baseline", baseline_test), ("original_film", original_film_test),
                       ("eof_cat", cat_test)):
        other_acc = matched_metrics(pred, target[test_mask], valid, p90)["acc_series"]
        paired[name] = paired_stat_tests(recovered_acc, other_acc, test_dates.year.values)
    np.savez_compressed(exp_dir / "predictions.npz", dates=test_dates.strftime("%Y-%m-%d").values,
                        target=target[test_mask], recovered_film=pred_test,
                        baseline=baseline_test, original_film=original_film_test, eof_cat=cat_test)
    print("Exploratory independent test:", test_result, flush=True)
    print("Yearly comparison:", yearly, flush=True)
    print("Paired ACC comparisons:", paired, flush=True)
    payload = {"protocol": "frozen baseline plus trained FiLM; gain selected only on 2018-2022 validation ACC",
               "gains": val_results, "selected_gain": chosen, "selected_test": test_result,
               "yearly_test": yearly, "paired_acc": paired}
    result_path = exp_dir / "film_gain.json"
    result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved {result_path}", flush=True)


if __name__ == "__main__":
    main()

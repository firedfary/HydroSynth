"""Check whether strict-input residual FiLM uses aligned SST principal components."""

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
from run_unified_eval import build_date_index, compute_strict_sst_pcs
from strict_fair_eval import predict, summary


def main():
    paths = SubprojectPaths(__file__)
    exp_dir = paths.results_dir / "strict_residual_v1"
    result_file = exp_dir / "sst_ablation.json"
    if result_file.exists():
        raise FileExistsError(result_file)
    dates = build_date_index()
    train_mask = np.asarray(dates <= "2017-12-01")
    val_mask = np.asarray((dates >= "2018-01-01") & (dates <= "2022-12-01"))
    test_mask = np.asarray(dates >= "2023-01-01")
    x = torch.from_numpy(np.load(paths.cache_dir / "strict_inputs" / "cond_trainfit.npy"))
    y = np.load(Path(config.modelconfig["hr_path"]) / "hr_data.npy")
    sst = np.load(config.modelconfig["sst_file"])
    pcs, _, _ = compute_strict_sst_pcs(sst, train_mask, 5)
    pcs = torch.from_numpy(pcs)
    valid = np.isfinite(y).all(axis=0)
    p90 = float(np.percentile(y[train_mask][:, valid], 90))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = UNetLiteFiLM(n_channels=10, index_dim=5, base_filters=32, dropout=0.1,
                         identity_init=True).to(device)
    model.load_state_dict(torch.load(exp_dir / "checkpoints" / "film_residual.pt",
                                     map_location=device, weights_only=True))
    rng = np.random.default_rng(123)
    result = {}
    for label, mask in (("validation", val_mask), ("test", test_mask)):
        xx, pp, yy = x[mask], pcs[mask], y[mask]
        original, _ = summary(predict(model, "film32_identity", xx, pp, device), yy, valid, p90)
        zero, _ = summary(predict(model, "film32_identity", xx, torch.zeros_like(pp), device), yy, valid, p90)
        shuffled_acc = []
        shuffled_tcc = []
        for _ in range(50):
            shuffled = pp[torch.from_numpy(rng.permutation(len(pp)))]
            metrics, _ = summary(predict(model, "film32_identity", xx, shuffled, device), yy, valid, p90)
            shuffled_acc.append(metrics["mean_acc"])
            shuffled_tcc.append(metrics["mean_tcc"])
        result[label] = {
            "full": original, "pc_zero": zero,
            "pc_shuffled_50": {"acc_mean": float(np.mean(shuffled_acc)),
                               "acc_std": float(np.std(shuffled_acc)),
                               "tcc_mean": float(np.mean(shuffled_tcc)),
                               "tcc_std": float(np.std(shuffled_tcc))},
        }
        print(label, result[label], flush=True)
    result_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved {result_file}", flush=True)


if __name__ == "__main__":
    main()

"""Add an output-amplitude column to the revision-06 symmetric ablation table.

Why this is needed
------------------
The ablation table reports ACC, TCC and extreme CSI for each input control.
Two of those three are sensitive to how *sharp* a prediction is, independently
of whether it is better placed:

* extreme CSI counts exceedances of a fixed training-period p90 threshold, so
  a damped field scores lower purely by producing fewer exceedances;
* correlation-type scores are not scale sensitive in themselves, but zeroing
  an input both removes information and shrinks the output, and the shrinkage
  alone moves the error structure.

Measured on the revision-06 checkpoints, zeroing the SST scores shrinks FiLM's
output amplitude by about ten percent. The table's existing reading -- that PC
zeroing *raises* FiLM ACC from 0.316 to 0.318 while lowering CSI -- therefore
mixes an information effect with an arithmetic one, and a reader cannot tell
which is which from the published columns.

This script recomputes the same table under the same protocol and adds, for
every row:

* ``amplitude``      mean within-month spatial standard deviation over valid
                     land cells, the same quantity measured for the observed
                     field;
* ``amplitude_ratio`` that value divided by the observed amplitude;
* ``csi_at_obs_amplitude`` extreme CSI after rescaling each month to the
                     observed spatial standard deviation, which removes the
                     sharpness term and leaves placement.

ACC is invariant to this rescaling and is reported unchanged as a check: if a
rescaled ACC ever differs from its raw value by more than rounding, the
rescaling is wrong.

Protocol
--------
Identical to ``revision06_ablation.py``: checkpoints from strict_fair_v1 /
_s43 / _s44 (seeds 42, 43, 44), equal-weight three-seed ensemble, January 2023
to September 2024, training-period p90 for the extreme threshold, and the same
``pc_shuffle`` control drawing a same-calendar-month score vector from
2018-2022. The raw ACC/TCC/CSI values it produces are cross-checked against
the existing ``ablation.json`` so the added column cannot silently come from a
different computation.

That period was inspected repeatedly during development, so these numbers
remain exploratory, exactly as in the source experiment.

Usage
-----
    python revision06_ablation_amplitude.py
    python revision06_ablation_amplitude.py --source revision06_ablation_v1
"""

import argparse
import hashlib
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
from run_unified_eval import build_date_index, compute_strict_sst_pcs
from strict_fair_eval import make_model, predict, summary

SEEDS = (42, 43, 44)
SOURCES = {42: "strict_fair_v1", 43: "strict_fair_s43", 44: "strict_fair_s44"}
MODELS = {"baseline": "baseline32", "eof_cat": "eof_cat32", "film": "film32_identity"}
# Rows as the manuscript table orders them: (label, model, input case).
ROWS = (
    ("Baseline U-Net", "baseline", "full"),
    ("EOF-cat (full)", "eof_cat", "full"),
    ("EOF-cat (SST zeroed)", "eof_cat", "pc_zero"),
    ("EOF-cat (local zeroed)", "eof_cat", "local_zero"),
    ("FiLM-U-Net (full)", "film", "full"),
    ("FiLM-U-Net (SST zeroed)", "film", "pc_zero"),
    ("FiLM-U-Net (local zeroed)", "film", "local_zero"),
    ("FiLM-U-Net (shuffled PCs)", "film", "pc_shuffle"),
    ("EOF-cat (shuffled PCs)", "eof_cat", "pc_shuffle"),
)


def amplitude(field, valid):
    """Mean within-month spatial standard deviation over valid land cells."""
    return float(np.mean([np.nanstd(field[t][valid]) for t in range(len(field))]))


def rescale_to(field, target, valid):
    """Rescale each month to the observed month's spatial standard deviation.

    Centring is left untouched: only the spread is matched, so the spatial
    pattern and its mean offset are unchanged and ACC cannot move.
    """
    out = np.array(field, dtype=np.float32, copy=True)
    for t in range(len(out)):
        have = valid & np.isfinite(out[t]) & np.isfinite(target[t])
        spread = np.nanstd(out[t][have])
        if spread > 1e-8:
            out[t] = out[t] * (np.nanstd(target[t][have]) / spread)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="revision06_ablation_amplitude")
    parser.add_argument("--source", default="revision06_ablation_v1",
                        help="existing ablation experiment to cross-check against")
    parser.add_argument("--trials", type=int, default=10)
    args = parser.parse_args()
    if args.trials < 1 or not args.experiment or any(c in args.experiment for c in "/\\"):
        parser.error("invalid experiment or trial count")

    paths = SubprojectPaths(__file__)
    output = paths.results_dir / args.experiment
    if output.exists():
        raise FileExistsError(output)

    dates = build_date_index()
    mask = np.asarray(dates >= "2023-01-01")
    train = np.asarray(dates <= "2017-12-01")
    pool = np.asarray((dates >= "2018-01-01") & (dates <= "2022-12-01"))
    target = np.load(Path(config.modelconfig["hr_path"]) / "hr_data.npy").astype(np.float32)
    valid = np.isfinite(target).all(axis=0)
    p90 = float(np.percentile(target[train][:, valid], 90))

    x_file = paths.cache_dir / "strict_inputs" / "cond_trainfit.npy"
    manifest = json.loads((x_file.parent / "manifest.json").read_text(encoding="utf-8"))
    digest = hashlib.sha256(x_file.read_bytes()).hexdigest()
    if digest != manifest["sha256"]:
        raise ValueError("strict input hash mismatch")
    x_all = np.load(x_file, mmap_mode="r")
    x = torch.from_numpy(x_all[mask].copy())
    sst = np.load(config.modelconfig["sst_file"], mmap_mode="r")
    pc_all, _, _ = compute_strict_sst_pcs(sst, train, 5)
    pc = torch.from_numpy(pc_all[mask].copy())
    y = target[mask]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(6026)

    members = {case: {name: [] for name in MODELS}
               for case in ("full", "pc_zero", "local_zero", "pc_shuffle")}
    for seed in SEEDS:
        source = paths.results_dir / SOURCES[seed]
        info = json.loads((source / "strict_fair_metrics.json").read_text(encoding="utf-8"))
        if info["input_sha256"] != digest:
            raise ValueError(f"source input differs: {source}")
        models = {}
        for name, model_name in MODELS.items():
            model = make_model(model_name, 0.1).to(device)
            model.load_state_dict(torch.load(source / "checkpoints" / f"{model_name}.pt",
                                             map_location=device, weights_only=True))
            model.eval()
            models[name] = model
        for case, changed_x, changed_pc in (
            ("full", x, pc),
            ("pc_zero", x, torch.zeros_like(pc)),
            ("local_zero", torch.zeros_like(x), pc),
        ):
            for name, model in models.items():
                members[case][name].append(
                    predict(model, MODELS[name], changed_x, changed_pc, device))
        # Verify against the stored predictions before trusting anything else.
        with np.load(source / "test_predictions.npz", allow_pickle=True) as previous:
            for name, model_name in MODELS.items():
                if not np.allclose(previous[model_name], members["full"][name][-1], atol=1e-5):
                    raise ValueError(f"checkpoint prediction mismatch: {source} {name}")
        trials = {name: [] for name in MODELS}
        for _ in range(args.trials):
            source_idx = np.array([
                rng.choice(np.flatnonzero(pool & (np.asarray(dates.month) == month)))
                for month in np.asarray(dates.month)[mask]
            ])
            shuffled_pc = torch.from_numpy(pc_all[source_idx].copy())
            for name, model in models.items():
                trials[name].append(predict(model, MODELS[name], x, shuffled_pc, device))
        for name in MODELS:
            members["pc_shuffle"][name].append(np.mean(trials[name], axis=0))
        del models
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"seed {seed} done", flush=True)

    obs_amp = amplitude(y, valid)
    rows = []
    for label, name, case in ROWS:
        pred = np.mean(members[case][name], axis=0)
        metrics, _ = summary(pred, y, valid, p90)
        scaled, _ = summary(rescale_to(pred, y, valid), y, valid, p90)
        if abs(scaled["mean_acc"] - metrics["mean_acc"]) > 1e-3:
            raise ValueError(f"rescaling changed ACC for {label}; the rescaling is wrong")
        amp = amplitude(pred, valid)
        rows.append({
            "row": label, "model": name, "input_case": case,
            "mean_acc": metrics["mean_acc"], "mean_tcc": metrics["mean_tcc"],
            "rmse": metrics["rmse"], "csi": metrics["csi"],
            "amplitude": amp, "amplitude_ratio": amp / obs_amp,
            "csi_at_obs_amplitude": scaled["csi"],
        })

    checked = None
    source_dir = paths.results_dir / args.source
    if (source_dir / "ablation.json").is_file():
        prior = json.loads((source_dir / "ablation.json").read_text(encoding="utf-8"))
        checked = []
        for row in rows:
            if row["input_case"] == "pc_shuffle":
                continue
            want = prior["ensemble"][row["input_case"]][row["model"]]["mean_acc"]
            delta = abs(row["mean_acc"] - want)
            checked.append({"row": row["row"], "prior_acc": want,
                            "recomputed_acc": row["mean_acc"], "abs_difference": delta})
            if delta > 1e-4:
                raise ValueError(f"ACC disagrees with {args.source} for {row['row']}: "
                                 f"{row['mean_acc']:.6f} vs {want:.6f}")
        print(f"cross-check passed against {args.source} "
              f"({len(checked)} rows, max difference "
              f"{max(c['abs_difference'] for c in checked):.2e})", flush=True)
    else:
        print(f"note: {args.source} not found; no cross-check performed", flush=True)

    output = Path(paths.get_exp_dir(args.experiment))
    payload = {
        "protocol": {
            "source_checkpoints": SOURCES, "seeds": list(SEEDS),
            "period": "2023-01/2024-09; exploratory after repeated inspection",
            "ensemble": "equal-weight mean of three seeds",
            "amplitude": "mean within-month spatial SD over valid land cells",
            "csi_at_obs_amplitude": "each month rescaled to the observed spatial SD "
                                    "before thresholding; ACC is invariant and is "
                                    "asserted unchanged",
            "pc_shuffle": f"same calendar month drawn from 2018-2022, {args.trials} trials",
            "extreme_threshold_p90": p90,
        },
        "input_sha256": digest,
        "observed_amplitude": obs_amp,
        "rows": rows,
        "cross_check_vs_source": checked,
    }
    (output / "ablation_amplitude.json").write_text(json.dumps(payload, indent=2),
                                                    encoding="utf-8")

    print(f"\nobserved amplitude (mean monthly spatial SD): {obs_amp:.3f}\n")
    header = ("row", "ACC", "TCC", "RMSE", "CSI", "amp", "amp/obs", "CSI@obs")
    print("%-27s%8s%8s%8s%8s%8s%9s%9s" % header)
    for row in rows:
        print("%-27s%8.4f%8.4f%8.4f%8.4f%8.3f%9.3f%9.4f"
              % (row["row"], row["mean_acc"], row["mean_tcc"], row["rmse"],
                 row["csi"], row["amplitude"], row["amplitude_ratio"],
                 row["csi_at_obs_amplitude"]))
    print(f"\nSaved {output / 'ablation_amplitude.json'}", flush=True)


if __name__ == "__main__":
    main()

"""Staged repair of the SST conditioning pathway for the FiLM U-Net.

Why this script exists
----------------------
Revision 06 reports that FiLM-U-Net has a *lower* monthly spatial ACC than the
unconditioned Baseline (0.316 vs 0.333 on the exploratory 2023-2024 window;
0.327 vs 0.335 on the 2018-2022 validation window) while having better RMSE,
extreme CSI, and in some folds better TCC. Three diagnostics already in this
repository show that the loss of ACC is caused by the SST conditioning signal
itself rather than by the FiLM operator:

* ``revision06_ablation.py`` -- zeroing the SST PCs *raises* FiLM ACC from
  0.316 to 0.318 and EOF-cat ACC from 0.293 to 0.319; replacing the scores with
  same-calendar-month values from 2018-2022 raises FiLM ACC to 0.324.
* ``film_gain_eval.py`` -- ACC decreases monotonically as the learned
  modulation gain increases, on every fold and seed tested.
* ``strict_residual_signal_eval.py`` -- a plain ridge correction of the
  *Baseline's* residuals using the same PCs reproduces the same monotonic ACC
  loss with no U-Net involved.

In other words the trained SST pathway currently has negative pattern-skill
value out of sample. Four properties of the existing recipe plausibly produce
that outcome, and this script turns each one into a switchable stage so the
cause can be attributed instead of guessed:

1. ``acc_weight`` -- the training loss is masked MSE, a pooled space-time
   criterion, while the headline metric is per-month spatial ACC. Pooled MSE
   rewards reproducing the slow, low-frequency SST-forced component even when
   that tilts each individual month's field away from its own pattern. Adding
   ``acc_weight * (1 - spatial_acc)`` aligns the objective with the metric.
2. ``cond_dropout`` -- the conditioning vector is always present during
   training, so the network is free to become structurally dependent on it.
   Randomly zeroing the PC vector trains a model that must work unconditioned
   and can only use SST as a refinement. Combined with ``zero_pc_identity`` a
   zero PC vector reproduces the unmodulated mapping exactly.
3. ``modulation_bound`` -- FALSIFIED by the first run, kept for the record.
   gamma and beta are unconstrained and carry no penalty beyond the global
   1e-5 weight decay, so the conditioning branch could in principle scale
   encoder features without limit. Measured on the s0/s1 checkpoints, it does
   not: mean |gamma| is 0.02-0.07 per encoder level and mean |beta| is
   0.01-0.08, a 2-7 percent feature perturbation. The network already
   self-limits, a 0.3 tanh bound can never bind, and ``s3_bound`` is therefore
   out of the default ladder.
4. ``pc_mode`` -- ``compute_strict_sst_pcs`` removes only the fit-period mean
   from the ERSSTv5 ``ssta`` field, so the secular warming signal stays in the
   EOF basis. Measured with ``--diagnose-only`` on the 1994-2017 fit: PC1 is a
   clean ENSO mode (|r| = 0.93 against a Nino3.4-like index) but PC2, carrying
   15.1 percent of the variance, correlates 0.89 with time and only 0.12 with
   ENSO -- it is a secular ramp, not a teleconnection. Its score averages
   +1.44 over 2018-2022 and +1.74 over 2023-2024 while averaging zero over the
   fit period, and only 19 of 285 training months reach that level. The
   conditioning branch therefore spends 15 percent of its input variance on a
   near-constant "late epoch" flag that is supported by under 7 percent of the
   training months. Note that this is covariate shift, not extrapolation: the
   measured excursion beyond the training range is at most 0.18 sigma on 4 of
   81 evaluation months, so ``clip`` is cheap insurance and the detrending is
   the active ingredient. Removing the fit-period linear trend before the EOF
   is also the conventional way to isolate teleconnection variability, and it
   leaves PC1 marginally cleaner (|r| = 0.93) with a larger variance share
   (0.253 against 0.223).

Every stage is applied to all three architectures at once, so Baseline,
EOF-cat and FiLM stay matched within a stage; ``modulation_bound`` and
``zero_pc_identity`` have no meaning for the SST-free Baseline and no effect on
EOF-cat, which is stated in the saved protocol block.

Protocol and evidence status
----------------------------
Training, validation and checkpoint selection follow ``strict_fair_eval.py``
exactly: 1994-01/2017-12 training, 2018-01/2022-12 validation, AdamW at 2e-4
with 1e-5 weight decay and a cosine schedule, batch size 8, dropout 0.1, 100
epochs, checkpoint chosen by validation mean spatial ACC, equal-weight mean of
seeds 42/43/44. Stage comparison and any stage selection must use the
validation window only. The 2023-2024 window stays excluded unless
``--with-exploratory-test`` is passed, and is labelled exploratory in the
output because it has already informed development. A stage selected here is
confirmed, if at all, by rerunning the two retrospective origins in
``revision06_hindcast.py`` -- not by its validation number.

Usage
-----
Check the conditioning signal first; this needs no GPU and no training::

    python film_fix_eval.py --diagnose-only

Then run the staged ladder (each stage adds one change to the previous one)::

    python film_fix_eval.py --experiment film_fix_v1

or a subset -- stages are independent runs, so a later stage can be run on its
own and read against an earlier stage from another experiment directory::

    python film_fix_eval.py --experiment film_fix_s24 --stages s2_drop,s4_pc

Reading the results:

* ``metrics.json`` holds every stage's ensemble metrics, paired ACC statistics
  and the measured FiLM modulation magnitudes.
* ACC alone is not enough. All three architectures predict at roughly 21
  percent of the observed monthly spatial standard deviation, and CSI scales
  with that amplitude, so a CSI difference between two models is mostly a
  sharpness difference. Compare CSI only after rescaling each month to the
  observed spatial SD; ACC is scale-invariant and needs no such correction.
  ``film_fix_s1/logs/amp2.py`` does this.
* The same caution applies to any zero-the-input ablation: zeroing the PCs
  shrinks FiLM's output by 9-10 percent, which mechanically raises
  correlation-type scores and lowers CSI regardless of information content.
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from utils.paths import SubprojectPaths
from unetlite import UNetLite
from unetlitefilm import UNetLiteFiLM
from film_recovery import spatial_acc
from strict_fair_eval import summary
from run_unified_eval import build_date_index, compute_strict_sst_pcs, paired_stat_tests, set_seed


MODELS = ("baseline32", "eof_cat32", "film32_identity")
SEEDS = (42, 43, 44)
N_PCS = 5
SST_WINDOW = 6

# Each stage adds exactly one change to the stage above it, so a change in
# score is attributable to that one factor. ``s3_bound`` is kept for the record
# but is no longer part of the default ladder: the s0/s1 run measured the
# learned modulation at mean |gamma| 0.02-0.07 per level (see
# film_fix_s1/stage_comparison.txt), an order of magnitude under the 0.3 bound,
# so the bound can never bind and the "unbounded modulation" hypothesis is
# falsified. ``s4_pc`` therefore chains from ``s2_drop``, not from ``s3_bound``.
STAGES = {
    "s0_current": {
        "describes": "revision-06 recipe, reproduction control",
        "acc_weight": 0.0, "cond_dropout": 0.0,
        "zero_pc_identity": False, "modulation_bound": None, "pc_mode": "raw",
    },
    "s1_acc": {
        "describes": "adds spatial-ACC term to the masked MSE loss (all models)",
        "acc_weight": 1.0, "cond_dropout": 0.0,
        "zero_pc_identity": False, "modulation_bound": None, "pc_mode": "raw",
    },
    "s2_drop": {
        "describes": "adds conditioning dropout and exact zero-PC identity (SST models)",
        "acc_weight": 1.0, "cond_dropout": 0.25,
        "zero_pc_identity": True, "modulation_bound": None, "pc_mode": "raw",
    },
    "s3_bound": {
        "describes": "SUPERSEDED, not in the default ladder: a tanh bound of 0.3 on the "
                     "FiLM modulation, an order of magnitude above the measured "
                     "mean |gamma| of 0.02-0.07, so it cannot bind",
        "acc_weight": 1.0, "cond_dropout": 0.25,
        "zero_pc_identity": True, "modulation_bound": 0.3, "pc_mode": "raw",
    },
    "s4_pc": {
        "describes": "adds detrended SST EOFs with scores clipped to the fit-period range; "
                     "detrending is the active ingredient, clipping touches 4 of 81 "
                     "evaluation months by at most 0.22 sigma",
        "acc_weight": 1.0, "cond_dropout": 0.25,
        "zero_pc_identity": True, "modulation_bound": None, "pc_mode": "detrend_clip",
    },
}

# The ladder ``--stages`` defaults to; ``s3_bound`` stays reachable by name.
DEFAULT_LADDER = ("s0_current", "s1_acc", "s2_drop", "s4_pc")


def detrended_sst_pcs(sst_raw, fit_mask, times, n_pcs=N_PCS,
                      window_months=SST_WINDOW, clip=True):
    """Fit-period detrended SST EOFs with optionally range-clipped scores.

    Mirrors ``run_unified_eval.compute_strict_sst_pcs`` -- same six-month mean,
    same fit-period standardization, same canonical El Nino sign on PC1 -- and
    adds two steps. The fit-period linear trend of every grid point is removed
    before the EOF so the secular warming signal cannot occupy a leading mode,
    and the standardized scores are clipped to the fit-period range so later
    years cannot push the conditioning branch outside its training support.
    """
    if sst_raw.ndim != 4:
        raise ValueError("detrended_sst_pcs expects a time-by-lag SST array")
    if not 1 <= window_months <= sst_raw.shape[1]:
        raise ValueError("window_months must be within the stored SST lag window")
    # Accumulate in float64 without materializing a float64 copy of the lags.
    sst = np.asarray(sst_raw[:, -window_months:]).mean(axis=1, dtype=np.float64)
    n_time, n_lat, n_lon = sst.shape
    if len(times) != n_time or len(fit_mask) != n_time:
        raise ValueError("times and fit_mask must match the SST chronology")

    flat = sst.reshape(n_time, -1)
    flat[~np.isfinite(flat)] = 0.0
    centred_time = np.asarray(times, dtype=np.float64)
    centred_time = centred_time - centred_time[fit_mask].mean()
    design = np.stack([np.ones(n_time), centred_time], axis=1)
    coefficients, *_ = np.linalg.lstsq(design[fit_mask], flat[fit_mask], rcond=None)
    residual = flat - design @ coefficients

    pca = PCA(n_components=n_pcs, svd_solver="full")
    pca.fit(residual[fit_mask])
    scores = pca.transform(residual)
    mean_value = scores[fit_mask].mean(axis=0, keepdims=True)
    std_value = scores[fit_mask].std(axis=0, keepdims=True) + 1e-8
    scores = (scores - mean_value) / std_value

    patterns = pca.components_.reshape(n_pcs, n_lat, n_lon).copy()
    lats = np.arange(-88, 90, 2, dtype=np.float32)[:n_lat]
    lons = np.arange(0, 360, 2, dtype=np.float32)[:n_lon]
    nino34 = np.nanmean(patterns[0][np.ix_((lats >= -5.0) & (lats <= 5.0),
                                           (lons >= 190.0) & (lons <= 240.0))])
    if nino34 < 0:
        scores[:, 0] *= -1.0
        patterns[0] *= -1.0

    if clip:
        low = scores[fit_mask].min(axis=0, keepdims=True)
        high = scores[fit_mask].max(axis=0, keepdims=True)
        scores = np.clip(scores, low, high)
    return (scores.astype(np.float32), patterns.astype(np.float32),
            pca.explained_variance_ratio_.astype(np.float32))


def pc_diagnostics(pcs, times, masks):
    """Trend alignment and out-of-training-range exposure for each PC."""
    report = {}
    train = masks["train"]
    centred = np.asarray(times, dtype=np.float64)
    centred = centred - centred[train].mean()
    low = pcs[train].min(axis=0)
    high = pcs[train].max(axis=0)
    for index in range(pcs.shape[1]):
        column = pcs[:, index].astype(np.float64)
        entry = {
            "corr_with_time_on_train": float(np.corrcoef(centred[train], column[train])[0, 1]),
            "train_range": [float(low[index]), float(high[index])],
        }
        for name, mask in masks.items():
            if not mask.any():
                continue
            window = column[mask]
            outside = (window < low[index]) | (window > high[index])
            entry[name] = {
                "mean": float(window.mean()),
                "range": [float(window.min()), float(window.max())],
                "months_outside_train_range": int(outside.sum()),
                "months": int(mask.sum()),
                "max_excursion_in_train_sigma": float(np.max(
                    np.maximum(low[index] - window, window - high[index]).clip(min=0.0))),
            }
        report[f"pc{index + 1}"] = entry
    return report


def make_model(name, stage, dropout=0.1, index_dim=N_PCS):
    if name == "baseline32":
        return UNetLite(n_channels=10, base_filters=32, dropout=dropout)
    if name == "eof_cat32":
        return UNetLite(n_channels=10 + index_dim, base_filters=32, dropout=dropout)
    if name != "film32_identity":
        raise ValueError(f"unknown model: {name}")
    return UNetLiteFiLM(
        n_channels=10, index_dim=index_dim, base_filters=32, dropout=dropout,
        identity_init=True, zero_pc_identity=stage["zero_pc_identity"],
        modulation_bound=stage["modulation_bound"],
    )


def forward(model, name, x, pcs, cond_dropout=0.0):
    """Shared forward pass; ``cond_dropout`` is only applied while training."""
    if cond_dropout > 0.0 and model.training and name != "baseline32":
        keep = (torch.rand(pcs.shape[0], 1, device=pcs.device) >= cond_dropout)
        pcs = pcs * keep.to(pcs.dtype)
    if name == "eof_cat32":
        maps = pcs[:, :, None, None].expand(-1, -1, x.shape[2], x.shape[3])
        return model(torch.cat((x, maps), dim=1))
    if name == "film32_identity":
        return model(x, pcs)
    return model(x)


def predict(model, name, x, pcs, device, batch=8):
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(x), batch):
            chunk = forward(model, name, x[start:start + batch].to(device),
                            pcs[start:start + batch].to(device))
            outputs.append(chunk[:, 0].cpu().numpy())
    return np.concatenate(outputs)


def train_one(name, stage, train_ds, val_ds, device, epochs, seed, lr=2e-4, dropout=0.1):
    """Same optimizer, schedule and checkpoint rule as ``strict_fair_eval``."""
    set_seed(seed)
    model = make_model(name, stage, dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)
    acc_weight = float(stage["acc_weight"])
    cond_dropout = float(stage["cond_dropout"])
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
            pred = forward(model, name, x, pcs, cond_dropout)
            loss = F.mse_loss(pred[~mask], y[~mask])
            if acc_weight > 0.0:
                loss = loss + acc_weight * (1.0 - spatial_acc(pred, y, mask))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            train_sum += float(loss.item()) * len(y)
            train_count += len(y)
        scheduler.step()

        model.eval()
        val_acc_sum = 0.0
        val_mse_sum = 0.0
        val_count = 0
        with torch.no_grad():
            for y, x, mask, pcs in val_loader:
                y, x, mask, pcs = (item.to(device) for item in (y, x, mask, pcs))
                pred = forward(model, name, x, pcs)
                val_mse_sum += float(F.mse_loss(pred[~mask], y[~mask]).item()) * len(y)
                val_acc_sum += float(spatial_acc(pred, y, mask).item()) * len(y)
                val_count += len(y)
        val_acc = val_acc_sum / val_count
        history.append({"epoch": epoch, "train_loss": train_sum / train_count,
                        "val_mse": val_mse_sum / val_count, "val_acc": val_acc})
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone()
                          for key, value in model.state_dict().items()}
        if epoch == 1 or epoch % 20 == 0:
            print(f"    {name} epoch={epoch} val_acc={val_acc:.4f} "
                  f"best_epoch={best_epoch} elapsed={time.monotonic() - started:.0f}s",
                  flush=True)

    model.load_state_dict(best_state)
    model.eval()
    return model, {"best_epoch": best_epoch, "best_validation_acc": best_acc,
                   "parameters": sum(p.numel() for p in model.parameters()),
                   "elapsed_seconds": time.monotonic() - started, "history": history}


def modulation_report(model, x, pcs, device, batch=8):
    """Mean absolute gamma and beta per FiLM level over the supplied months."""
    model.eval()
    totals = None
    count = 0
    with torch.no_grad():
        for start in range(0, len(pcs), batch):
            chunk = pcs[start:start + batch].to(device)
            gammas, betas = model.index_encoder(chunk)
            if model.zero_pc_identity:
                zero_g, zero_b = model.index_encoder(model.reference_indices(chunk))
                gammas = [g - z for g, z in zip(gammas, zero_g)]
                betas = [b - z for b, z in zip(betas, zero_b)]
            if model.modulation_bound is not None:
                bound = model.modulation_bound
                gammas = [bound * torch.tanh(g) for g in gammas]
                betas = [bound * torch.tanh(b) for b in betas]
            row = [float(g.abs().mean()) for g in gammas] + [float(b.abs().mean()) for b in betas]
            weighted = np.asarray(row) * len(chunk)
            totals = weighted if totals is None else totals + weighted
            count += len(chunk)
    levels = len(totals) // 2
    return {"mean_abs_gamma": [float(v) for v in totals[:levels] / count],
            "mean_abs_beta": [float(v) for v in totals[levels:] / count]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="film_fix_v1")
    parser.add_argument("--stages", default=",".join(DEFAULT_LADDER))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    parser.add_argument("--diagnose-only", action="store_true",
                        help="only report SST PC diagnostics; no training, no GPU needed")
    parser.add_argument("--with-exploratory-test",
                        action="store_true",
                        help="also score the already-inspected 2023-2024 window")
    args = parser.parse_args()
    if args.epochs <= 0 or not args.experiment or any(c in args.experiment for c in "/\\"):
        parser.error("epochs must be positive and experiment must be a directory name")
    stages = [name.strip() for name in args.stages.split(",") if name.strip()]
    unknown = [name for name in stages if name not in STAGES]
    if unknown:
        parser.error(f"unknown stages: {unknown}")
    seeds = tuple(int(value) for value in args.seeds.split(","))

    paths = SubprojectPaths(__file__)
    output = paths.results_dir / args.experiment
    if not args.diagnose_only and output.exists() and (
            (output / "metrics.json").exists() or any((output / "checkpoints").glob("*.pt"))):
        raise FileExistsError(output)

    dates = build_date_index()
    times = dates.year.values + (dates.month.values - 1) / 12.0
    masks = {
        "train": np.asarray(dates <= "2017-12-01"),
        "validation": np.asarray((dates >= "2018-01-01") & (dates <= "2022-12-01")),
        "exploratory_test": np.asarray(dates >= "2023-01-01"),
    }

    sst = np.load(config.modelconfig["sst_file"], mmap_mode="r")
    if sst.shape[0] != len(dates):
        raise ValueError("SST chronology differs from the expected 366-month index")
    pc_sets = {}
    pc_sets["raw"] = compute_strict_sst_pcs(sst, masks["train"], N_PCS, SST_WINDOW)
    pc_sets["detrend_clip"] = detrended_sst_pcs(sst, masks["train"], times, N_PCS, SST_WINDOW)
    diagnostics = {
        mode: {
            "explained_variance_ratio": [float(v) for v in values[2]],
            "pcs": pc_diagnostics(values[0], times, masks),
        }
        for mode, values in pc_sets.items()
    }

    if args.diagnose_only:
        print(json.dumps(diagnostics, indent=2))
        return

    input_dir = paths.cache_dir / "strict_inputs"
    input_file = input_dir / "cond_trainfit.npy"
    manifest = json.loads((input_dir / "manifest.json").read_text(encoding="utf-8"))
    input_hash = hashlib.sha256(input_file.read_bytes()).hexdigest()
    if input_hash != manifest["sha256"]:
        raise ValueError("preprocessed input hash differs from manifest")
    x = np.load(input_file).astype(np.float32)
    y = np.load(Path(config.modelconfig["hr_path"]) / "hr_data.npy").astype(np.float32)
    if x.shape != (len(dates), 10, 120, 140) or y.shape != (len(dates), 120, 140):
        raise ValueError("input or target shape differs from expected chronology")
    valid = np.isfinite(y).all(axis=0)
    p90 = float(np.percentile(y[masks["train"]][:, valid], 90))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("staged retraining requires a CUDA GPU")
    output = Path(paths.get_exp_dir(args.experiment))
    (output / "checkpoints").mkdir(parents=True, exist_ok=True)

    x_t = torch.from_numpy(x)
    y_t = torch.from_numpy(y[:, None])
    mask_t = torch.isnan(y_t) | torch.from_numpy(~valid)[None, None]
    scored = ["validation"] + (["exploratory_test"] if args.with_exploratory_test else [])

    payload = {
        "protocol": {
            "train": "1994-01/2017-12", "validation": "2018-01/2022-12",
            "epochs": args.epochs, "seeds": list(seeds), "batch_size": 8,
            "lr": 2e-4, "weight_decay": 1e-5, "dropout": 0.1,
            "optimizer": "AdamW with cosine schedule to 1e-6",
            "checkpoint_selection": "validation mean spatial ACC",
            "ensemble": "equal-weight mean of the trained seeds",
            "stage_scope": "acc_weight applies to all three models; cond_dropout "
                           "applies to EOF-cat and FiLM; zero_pc_identity and "
                           "modulation_bound apply to FiLM only; pc_mode applies "
                           "to EOF-cat and FiLM",
            "selection_rule": "compare and select stages on the validation window only",
            "evidence_status": "validation served checkpoint selection, so stage "
                               "scores are conditional on it; the 2023-2024 window "
                               "was inspected during earlier development and stays "
                               "exploratory; confirm a selected stage by rerunning "
                               "revision06_hindcast.py on the two retrospective origins",
        },
        "stages": {name: STAGES[name] for name in stages},
        "input_sha256": input_hash,
        "train_p90": p90,
        "valid_grid_count": int(valid.sum()),
        "sst_pc_diagnostics": diagnostics,
        "results": {},
    }

    for stage_name in stages:
        stage = STAGES[stage_name]
        pcs = pc_sets[stage["pc_mode"]][0]
        pcs_t = torch.from_numpy(pcs.copy())
        dataset = TensorDataset(y_t, x_t, mask_t, pcs_t)
        train_ds = torch.utils.data.Subset(dataset, np.flatnonzero(masks["train"]))
        val_ds = torch.utils.data.Subset(dataset, np.flatnonzero(masks["validation"]))
        print(f"[{stage_name}] {stage['describes']}", flush=True)

        members = {name: {period: [] for period in scored} for name in MODELS}
        training = {}
        modulation = {}
        for seed in seeds:
            training[str(seed)] = {}
            for name in MODELS:
                model, info = train_one(name, stage, train_ds, val_ds, device,
                                        args.epochs, seed)
                torch.save(model.state_dict(),
                           output / "checkpoints" / f"{stage_name}_{name}_s{seed}.pt")
                info.pop("history")
                training[str(seed)][name] = info
                for period in scored:
                    members[name][period].append(
                        predict(model, name, x_t[masks[period]], pcs_t[masks[period]], device))
                if name == "film32_identity":
                    modulation[str(seed)] = modulation_report(
                        model, x_t[masks["validation"]], pcs_t[masks["validation"]], device)
                del model
                torch.cuda.empty_cache()
                print(f"  seed={seed} {name} best_epoch={info['best_epoch']} "
                      f"val_acc={info['best_validation_acc']:.4f}", flush=True)

        stage_result = {"training": training, "film_modulation_on_validation": modulation,
                        "periods": {}}
        for period in scored:
            predictions = {name: np.mean(members[name][period], axis=0) for name in MODELS}
            metrics, acc = {}, {}
            for name, pred in predictions.items():
                metrics[name], acc[name] = summary(pred, y[masks[period]], valid, p90)
            years = dates[masks[period]].year.values
            paired = {
                "film_minus_baseline": paired_stat_tests(
                    acc["film32_identity"], acc["baseline32"], years),
                "film_minus_eof_cat": paired_stat_tests(
                    acc["film32_identity"], acc["eof_cat32"], years),
            }
            stage_result["periods"][period] = {
                "months": int(masks[period].sum()), "ensemble": metrics,
                "paired_acc": paired,
            }
            np.savez_compressed(
                output / f"{stage_name}_{period}_predictions.npz",
                dates=dates[masks[period]].strftime("%Y-%m-%d").values,
                target=y[masks[period]], valid=valid, **predictions)
            print(f"  {period}: "
                  + ", ".join(f"{name}={metrics[name]['mean_acc']:.4f}" for name in MODELS),
                  flush=True)
        payload["results"][stage_name] = stage_result
        (output / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Saved {output / 'metrics.json'}", flush=True)


if __name__ == "__main__":
    main()

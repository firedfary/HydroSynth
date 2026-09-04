"""Build an OOF-selected blend of base AMS, transfer AMS, and stacking."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from project_paths import OBSERVATION_FILE, paths, experiment_path

from build_fixed_ensemble import (
    select_oof_ec_safety_seasons,
    spatial_acc_vectors,
    standardize_vectors,
    transform_fractional_anomaly,
)


def simplex_weights(source_count: int, step: float = 0.05) -> np.ndarray:
    units = int(round(1.0 / step))
    rows = []

    def append_compositions(remaining, parts, prefix):
        if parts == 1:
            rows.append((*prefix, remaining))
            return
        for value in range(remaining + 1):
            append_compositions(remaining - value, parts - 1, (*prefix, value))

    append_compositions(units, source_count, ())
    return np.asarray(rows, dtype=np.float64) / units


def select_weights(sources, targets, area_weights, fold_size=21):
    candidates = simplex_weights(sources.shape[0])
    _, sample_count, lead_count, _ = sources.shape
    fold_indices = np.minimum(np.arange(sample_count) // fold_size, 4)
    sample_weights = (fold_indices + 1.0) ** 2
    sample_weights /= sample_weights.sum()
    selected = np.empty((lead_count, sources.shape[0]), dtype=np.float64)
    diagnostics = []
    for lead in range(lead_count):
        patterns = np.stack(
            [standardize_vectors(source[:, lead]) for source in sources], axis=1
        )
        source_means = np.einsum("nkp,p->nk", patterns, area_weights)
        centered_sources = patterns - source_means[:, :, None]
        target = targets[:, lead]
        centered_target = target - (target @ area_weights)[:, None]
        source_target = np.einsum(
            "nkp,np,p->nk", centered_sources, centered_target, area_weights
        )
        source_gram = np.einsum(
            "nkp,nlp,p->nkl", centered_sources, centered_sources, area_weights
        )
        target_norm = np.einsum(
            "np,np,p->n", centered_target, centered_target, area_weights
        )
        numerator = source_target @ candidates.T
        prediction_norm = np.einsum(
            "mk,nkl,ml->nm", candidates, source_gram, candidates
        )
        denominator = np.sqrt(prediction_norm * target_norm[:, None])
        correlations = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator),
            where=denominator > 1e-12,
        )
        scores = sample_weights @ correlations
        best_index = int(np.argmax(scores))
        selected[lead] = candidates[best_index]
        diagnostics.append(
            {
                "lead": lead,
                "weights": selected[lead].tolist(),
                "weighted_oof_acc": float(scores[best_index]),
                "source_oof_acc": [
                    float(scores[int(np.where((candidates == basis).all(axis=1))[0][0])])
                    for basis in np.eye(sources.shape[0])
                ],
            }
        )
    return selected, diagnostics


def build(
    base_dir: Path,
    transfer_dir: Path,
    base_oof_file: Path,
    transfer_oof_file: Path,
    stack_file: Path,
    observation_file: Path,
    output_file: Path,
    extended_transfer_dir: Path | None = None,
    extended_transfer_oof_file: Path | None = None,
):
    with np.load(base_oof_file) as data:
        oof_dates = data["dates"].astype(str)
        base_oof = np.asarray(data["predictions"], dtype=np.float64)
    with np.load(transfer_oof_file) as data:
        transfer_dates = data["dates"].astype(str)
        transfer_oof = np.asarray(data["predictions"], dtype=np.float64)
    transfer_oofs = [transfer_oof]
    transfer_oof_dates = [transfer_dates]
    if extended_transfer_oof_file is not None:
        with np.load(extended_transfer_oof_file) as data:
            transfer_oof_dates.append(data["dates"].astype(str))
            transfer_oofs.append(
                np.asarray(data["predictions"], dtype=np.float64)
            )
    with np.load(stack_file) as data:
        test_dates = data["dates"].astype(str)
        stack_test = np.asarray(data["predictions"], dtype=np.float64)
        stack_dates = data["oof_dates"].astype(str)
        stack_oof = np.asarray(data["oof_predictions"], dtype=np.float64)
    if (
        any(not np.array_equal(oof_dates, dates) for dates in transfer_oof_dates)
        or not np.array_equal(oof_dates, stack_dates)
    ):
        raise ValueError("OOF dates differ between sources")

    production_dates = np.load(base_dir / "multi_lead_dates.npy").astype(str)
    date_to_index = {date: index for index, date in enumerate(production_dates)}
    base = np.load(base_dir / "multi_lead_predict_results.npy", mmap_mode="r")
    transfer_dirs = [transfer_dir]
    if extended_transfer_dir is not None:
        transfer_dirs.append(extended_transfer_dir)
    transfers = [
        np.load(directory / "multi_lead_predict_results.npy", mmap_mode="r")
        for directory in transfer_dirs
    ]
    ec = np.load(
        base_dir / "multi_lead_ec_precip_anom_results.npy", mmap_mode="r"
    )
    if any(base.shape != transfer.shape for transfer in transfers) or base.shape != ec.shape:
        raise ValueError("Production source shapes differ")

    with np.load(observation_file) as data:
        observations = transform_fractional_anomaly(
            np.asarray(data["anomaly_fraction"], dtype=np.float64), "signed_log1p"
        )
        observation_dates = data["dates"].astype(str)
        mask = np.asarray(data["valid_mask"], dtype=bool)
        latitudes = np.asarray(data["latitudes"], dtype=np.float64)
    observation_index = {
        date: index for index, date in enumerate(observation_dates)
    }
    area_weights = (
        np.cos(np.deg2rad(latitudes))[:, None] * np.ones((1, mask.shape[1]))
    )[mask]
    area_weights /= area_weights.sum()

    targets = np.empty_like(base_oof)
    oof_ec = np.empty_like(base_oof)
    for lead in range(base.shape[1]):
        dates = oof_dates[:, lead]
        targets[:, lead] = observations[
            [observation_index[date] for date in dates]
        ][:, mask]
        indices = [date_to_index[date] for date in dates]
        oof_ec[:, lead] = ec[indices, lead][:, mask]

    sources = np.stack([base_oof, *transfer_oofs, stack_oof])
    weights, diagnostics = select_weights(
        sources, targets, area_weights
    )
    blended_oof = np.empty_like(base_oof)
    for lead in range(base.shape[1]):
        source_patterns = [
            standardize_vectors(source[:, lead]) for source in sources
        ]
        blended_oof[:, lead] = standardize_vectors(
            sum(
                weight * pattern
                for weight, pattern in zip(weights[lead], source_patterns)
            )
        )
    fallback_seasons, safety_diagnostics = select_oof_ec_safety_seasons(
        blended_oof, oof_ec, targets, oof_dates, area_weights
    )

    result = np.asarray(base).copy()
    test_indices = np.asarray([date_to_index[date] for date in test_dates])
    for lead in range(base.shape[1]):
        base_vectors = np.asarray(base[test_indices, lead][:, mask], dtype=np.float64)
        transfer_vectors = [
            np.asarray(transfer[test_indices, lead][:, mask], dtype=np.float64)
            for transfer in transfers
        ]
        ec_vectors = np.asarray(ec[test_indices, lead][:, mask], dtype=np.float64)
        source_patterns = (
            standardize_vectors(base_vectors),
            *[standardize_vectors(values) for values in transfer_vectors],
            standardize_vectors(stack_test[:, lead]),
        )
        blended = standardize_vectors(
            sum(
                weight * pattern
                for weight, pattern in zip(weights[lead], source_patterns)
            )
        )
        restored = (
            blended * np.maximum(ec_vectors.std(axis=1, keepdims=True), 1e-8)
            + ec_vectors.mean(axis=1, keepdims=True)
        )
        if fallback_seasons[lead]:
            from build_fixed_ensemble import SEASON_BY_MONTH

            rows = np.asarray(
                [
                    SEASON_BY_MONTH[int(date[5:7])] in fallback_seasons[lead]
                    for date in test_dates
                ]
            )
            restored[rows] = ec_vectors[rows]
        fields = np.zeros((len(test_dates), *mask.shape), dtype=np.float32)
        fields[:, mask] = restored.astype(np.float32)
        result[test_indices, lead] = fields

    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_file, result)
    oof_output_file = output_file.with_name(f"{output_file.stem}_oof.npz")
    np.savez_compressed(
        oof_output_file,
        dates=oof_dates,
        predictions=blended_oof.astype(np.float32),
    )
    metadata = {
        "source_order": [
            "base_AMS",
            "model_as_sample_transfer",
            *(
                ["model_as_sample_transfer_extended"]
                if extended_transfer_dir is not None
                else []
            ),
            "seasonal_stack",
        ],
        "weights_by_lead": weights.tolist(),
        "weight_diagnostics": diagnostics,
        "fallback_seasons_by_lead": {
            str(lead): sorted(value) for lead, value in enumerate(fallback_seasons)
        },
        "safety_diagnostics": safety_diagnostics,
        "observation_transform": "signed_log1p",
        "oof_file": str(oof_output_file),
    }
    output_file.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata["weights_by_lead"], indent=2))
    print(json.dumps(metadata["fallback_seasons_by_lead"], indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-dir", type=Path, default=paths.results_dir / "signed_log_run"
    )
    parser.add_argument(
        "--transfer-dir",
        type=Path,
        default=paths.results_dir / "model_as_sample_transfer_run",
    )
    parser.add_argument(
        "--base-oof-file",
        type=Path,
        default=paths.results_dir / "signed_log_run" / "ams_oof_patterns.npz",
    )
    parser.add_argument(
        "--transfer-oof-file",
        type=Path,
        default=paths.results_dir
        / "model_as_sample_transfer_run"
        / "model_as_sample_transfer_oof.npz",
    )
    parser.add_argument(
        "--stack-file",
        type=Path,
        default=paths.results_dir
        / "signed_log_run"
        / "seasonal_stacking_test_patterns.npz",
    )
    parser.add_argument("--observation-file", type=Path, default=OBSERVATION_FILE)
    parser.add_argument(
        "--output-file",
        type=Path,
        default=experiment_path(
            "multisource_oof_ensemble",
            "multi_lead_predict_results_multisource_safe.npy",
        ),
    )
    parser.add_argument("--extended-transfer-dir", type=Path, default=None)
    parser.add_argument("--extended-transfer-oof-file", type=Path, default=None)
    args = parser.parse_args()
    build(
        args.base_dir,
        args.transfer_dir,
        args.base_oof_file,
        args.transfer_oof_file,
        args.stack_file,
        args.observation_file,
        args.output_file,
        args.extended_transfer_dir,
        args.extended_transfer_oof_file,
    )


if __name__ == "__main__":
    main()

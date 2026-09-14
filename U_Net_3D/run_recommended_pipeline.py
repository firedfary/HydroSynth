"""Run the complete currently recommended multi-lead precipitation pipeline.

This module is the reproducible entry point for the production candidate.  It
orchestrates the existing, independently testable stage scripts instead of
duplicating their scientific implementations:

1. rebuild leakage-free station observations with the 1994-2010 reference;
2. train the signed-log AMS baseline and produce rolling OOF predictions;
3. train signed-log seasonal stacking and produce its rolling OOF predictions;
4. train ECMWF/NCEP/JMA model-as-sample Ridge transfer candidates with
   fold-local climatology/PCA and lead-dependent recency selection;
5. select a nonnegative transfer/stacking blend from historical OOF data and
   apply the ECMWF seasonal safety anchor;
6. evaluate national metrics with block bootstrap and stratified skill.

All generated artifacts are placed below ``paths.get_exp_dir()`` or the
managed observation cache.  Raw data are read-only.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from project_paths import (
    OBSERVATION_FILE,
    STATION_TABLE_FILE,
    experiment_dir,
)


SCRIPT_DIR = Path(__file__).resolve().parent
STAGE_ORDER = (
    "observations",
    "base_ams",
    "seasonal_stacking",
    "transfer",
    "ensemble",
    "metrics",
    "stratified",
)


@dataclass(frozen=True)
class PipelineConfig:
    run_dir: Path
    observation_file: Path = OBSERVATION_FILE
    station_csv: Path = STATION_TABLE_FILE
    reference_start: int = 1994
    reference_end: int = 2010
    minimum_reference_years: int = 10
    climatology_floor_mm: float = 0.1
    observation_start: str = "1994-01-01"
    observation_end: str = "2024-12-01"
    transfer_sources: tuple[str, ...] = ("ECMWF", "NCEP", "JMA")
    selection_fold_count: int = 5
    test_months: int = 21
    bootstrap: int = 10_000
    seed: int = 42
    ams_candidates: tuple[str, ...] = ()

    @property
    def aligned_observation_file(self) -> Path:
        return self.observation_file.with_name(
            self.observation_file.stem + "_aligned.npy"
        )

    @property
    def base_dir(self) -> Path:
        return self.run_dir / "base_ams"

    @property
    def base_oof(self) -> Path:
        return self.base_dir / "ams_oof_patterns.npz"

    @property
    def stacking_dir(self) -> Path:
        return self.run_dir / "seasonal_stacking"

    @property
    def stacking_file(self) -> Path:
        return self.stacking_dir / "seasonal_stacking_test_patterns.npz"

    @property
    def transfer_dir(self) -> Path:
        return self.run_dir / "model_as_sample_transfer"

    @property
    def transfer_metrics(self) -> Path:
        return self.transfer_dir / "model_as_sample_transfer_metrics.json"

    @property
    def transfer_oof(self) -> Path:
        return self.transfer_dir / "model_as_sample_transfer_oof.npz"

    @property
    def final_dir(self) -> Path:
        return self.run_dir / "final"

    @property
    def prediction_file(self) -> Path:
        return self.final_dir / "multi_lead_predict_results_ensemble_safe.npy"

    @property
    def evaluation_dir(self) -> Path:
        return self.run_dir / "evaluation"

    @property
    def metrics_prefix(self) -> Path:
        return self.evaluation_dir / "multilead_evaluation_metrics"

    @property
    def stratified_file(self) -> Path:
        return self.evaluation_dir / "stratified_metrics.csv"


@dataclass(frozen=True)
class Stage:
    name: str
    command: tuple[str, ...]
    outputs: tuple[Path, ...]
    validator: Callable[[PipelineConfig], None]
    environment: dict[str, str] = field(default_factory=dict)


def _script(name: str) -> str:
    return str(SCRIPT_DIR / name)


def _base_product_files(directory: Path) -> tuple[Path, ...]:
    return tuple(
        directory / name
        for name in (
            "multi_lead_dates.npy",
            "multi_lead_obs_results.npy",
            "multi_lead_ec_precip_anom_results.npy",
            "multi_lead_predict_results.npy",
        )
    )


def build_stages(config: PipelineConfig) -> list[Stage]:
    python = sys.executable
    observation_outputs = (
        config.observation_file,
        config.aligned_observation_file,
        config.observation_file.with_suffix(".json"),
    )
    base_outputs = _base_product_files(config.base_dir) + (config.base_oof,)
    transfer_outputs = _base_product_files(config.transfer_dir) + (
        config.transfer_metrics,
        config.transfer_oof,
    )
    ams_env = {
        "AMS_OUTPUT_DIR": str(config.base_dir),
        "AMS_OBS_PATH": str(config.aligned_observation_file),
        "AMS_OOF_PATH": str(config.base_oof),
        "AMS_TARGET_TRANSFORM": "signed_log1p",
        "AMS_USE_FORECAST_SST": "0",
        "AMS_USE_FORECAST_SST_FIELD": "0",
        "AMS_USE_PHYSICAL_INDICES": "0",
        "AMS_USE_SEASON_INTERACTIONS": "0",
        "AMS_ENABLE_SAFETY_FALLBACK": "0",
    }
    if config.ams_candidates:
        ams_env["AMS_CANDIDATES"] = ",".join(config.ams_candidates)

    transfer_env = {
        "AMS_TRANSFER_SOURCES": ",".join(config.transfer_sources),
        "AMS_SELECTION_FOLD_COUNT": str(config.selection_fold_count),
    }

    return [
        Stage(
            "observations",
            (
                python,
                _script("rebuild_station_observations.py"),
                "--csv", str(config.station_csv),
                "--output", str(config.observation_file),
                "--reference-start", str(config.reference_start),
                "--reference-end", str(config.reference_end),
                "--minimum-reference-years", str(config.minimum_reference_years),
                "--climatology-floor-mm", str(config.climatology_floor_mm),
                "--start", config.observation_start,
                "--end", config.observation_end,
            ),
            observation_outputs,
            validate_observations,
        ),
        Stage(
            "base_ams",
            (python, _script("train_pcr_multilead.py")),
            base_outputs,
            validate_base_ams,
            ams_env,
        ),
        Stage(
            "seasonal_stacking",
            (
                python,
                _script("experiment_seasonal_stacking.py"),
                "--observation-file", str(config.observation_file),
                "--observation-transform", "signed_log1p",
                "--output-file", str(config.stacking_file),
            ),
            (config.stacking_file,),
            validate_stacking,
        ),
        Stage(
            "transfer",
            (
                python,
                _script("experiment_model_as_sample_transfer.py"),
                "--observation-file", str(config.observation_file),
                "--base-data-dir", str(config.base_dir),
                "--output-dir", str(config.transfer_dir),
                "--output-file", str(config.transfer_metrics),
                "--oof-file", str(config.transfer_oof),
            ),
            transfer_outputs,
            validate_transfer,
            transfer_env,
        ),
        Stage(
            "ensemble",
            (
                python,
                _script("build_fixed_ensemble.py"),
                "--data-dir", str(config.transfer_dir),
                "--stack-file", str(config.stacking_file),
                "--output-file", str(config.prediction_file),
                "--ams-oof-file", str(config.transfer_oof),
                "--observation-file", str(config.observation_file),
                "--ams-weight", "0.5",
                "--select-weights-from-oof",
                "--seasonal-ec-safety",
                "--no-calibrate-amplitude",
                "--observation-transform", "signed_log1p",
            ),
            (config.prediction_file, config.prediction_file.with_suffix(".json")),
            validate_ensemble,
        ),
        Stage(
            "metrics",
            (
                python,
                _script("evaluate_multilead_metrics.py"),
                "--data-dir", str(config.transfer_dir),
                "--prediction-file", str(config.prediction_file),
                "--observation-file", str(config.observation_file),
                "--test-months", str(config.test_months),
                "--bootstrap", str(config.bootstrap),
                "--seed", str(config.seed),
                "--observation-transform", "signed_log1p",
                "--output-prefix", str(config.metrics_prefix),
            ),
            (
                config.metrics_prefix.with_suffix(".csv"),
                config.metrics_prefix.with_suffix(".json"),
            ),
            validate_metrics,
        ),
        Stage(
            "stratified",
            (
                python,
                _script("evaluate_stratified_skill.py"),
                "--data-dir", str(config.transfer_dir),
                "--prediction-file", str(config.prediction_file),
                "--observation-file", str(config.observation_file),
                "--output-file", str(config.stratified_file),
                "--test-months", str(config.test_months),
            ),
            (config.stratified_file,),
            validate_stratified,
        ),
    ]


def _require_files(paths: tuple[Path, ...]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing pipeline artifact(s): " + ", ".join(missing))


def _validate_product_directory(directory: Path) -> None:
    files = _base_product_files(directory)
    _require_files(files)
    dates = np.load(files[0], mmap_mode="r")
    obs = np.load(files[1], mmap_mode="r")
    ec = np.load(files[2], mmap_mode="r")
    prediction = np.load(files[3], mmap_mode="r")
    if obs.shape != ec.shape or obs.shape != prediction.shape:
        raise ValueError(f"Inconsistent product shapes in {directory}")
    if obs.ndim != 4 or obs.shape[1:] != (6, 120, 140):
        raise ValueError(f"Unexpected product shape {obs.shape} in {directory}")
    if len(dates) != obs.shape[0]:
        raise ValueError(f"Date/product length mismatch in {directory}")


def validate_observations(config: PipelineConfig) -> None:
    _require_files(
        (
            config.observation_file,
            config.aligned_observation_file,
            config.observation_file.with_suffix(".json"),
        )
    )
    with np.load(config.observation_file) as data:
        required = {
            "dates", "latitudes", "longitudes", "precipitation_mm",
            "anomaly_fraction", "valid_mask", "station_counts",
        }
        if not required.issubset(data.files):
            raise ValueError("Observation NPZ is missing required arrays")
        anomaly = data["anomaly_fraction"]
        mask = data["valid_mask"]
        dates = data["dates"]
        if anomaly.ndim != 3 or anomaly.shape[1:] != (120, 140):
            raise ValueError(f"Unexpected observation shape {anomaly.shape}")
        if mask.shape != (120, 140) or len(dates) != len(anomaly):
            raise ValueError("Observation dates or mask are inconsistent")
    aligned = np.load(config.aligned_observation_file, mmap_mode="r")
    if aligned.shape[1:] != (120, 140):
        raise ValueError(f"Unexpected aligned observation shape {aligned.shape}")


def validate_base_ams(config: PipelineConfig) -> None:
    _validate_product_directory(config.base_dir)
    _require_files((config.base_oof,))
    with np.load(config.base_oof) as data:
        if not {"dates", "predictions"}.issubset(data.files):
            raise ValueError("AMS OOF file is missing dates or predictions")
        if data["predictions"].ndim != 3 or data["predictions"].shape[1] != 6:
            raise ValueError("AMS OOF prediction shape is invalid")


def validate_stacking(config: PipelineConfig) -> None:
    _require_files((config.stacking_file,))
    with np.load(config.stacking_file) as data:
        required = {"dates", "predictions", "oof_dates", "oof_predictions", "valid_mask"}
        if not required.issubset(data.files):
            raise ValueError("Seasonal stacking file is incomplete")
        if data["predictions"].ndim != 3 or data["predictions"].shape[1] != 6:
            raise ValueError("Seasonal stacking prediction shape is invalid")


def validate_transfer(config: PipelineConfig) -> None:
    _validate_product_directory(config.transfer_dir)
    _require_files((config.transfer_metrics, config.transfer_oof))
    metrics = json.loads(config.transfer_metrics.read_text(encoding="utf-8"))
    if len(metrics) != 6 or {row.get("lead") for row in metrics} != set(range(6)):
        raise ValueError("Transfer metrics must contain Lead 0-5")
    with np.load(config.transfer_oof) as data:
        if not {"dates", "predictions"}.issubset(data.files):
            raise ValueError("Transfer OOF file is incomplete")
        if data["predictions"].ndim != 3 or data["predictions"].shape[1] != 6:
            raise ValueError("Transfer OOF prediction shape is invalid")


def validate_ensemble(config: PipelineConfig) -> None:
    _require_files((config.prediction_file, config.prediction_file.with_suffix(".json")))
    prediction = np.load(config.prediction_file, mmap_mode="r")
    reference = np.load(
        config.transfer_dir / "multi_lead_ec_precip_anom_results.npy",
        mmap_mode="r",
    )
    if prediction.shape != reference.shape:
        raise ValueError("Final ensemble and ECMWF arrays have different shapes")
    metadata = json.loads(
        config.prediction_file.with_suffix(".json").read_text(encoding="utf-8")
    )
    expected = {
        "weights_selected_from_oof": True,
        "seasonal_ec_safety": True,
        "observation_transform": "signed_log1p",
        "amplitude_calibrated_from_oof": False,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"Final ensemble metadata mismatch: {key}")


def validate_metrics(config: PipelineConfig) -> None:
    csv_path = config.metrics_prefix.with_suffix(".csv")
    json_path = config.metrics_prefix.with_suffix(".json")
    _require_files((csv_path, json_path))
    table = pd.read_csv(csv_path)
    required = {"lead", "spatial_acc", "ec_spatial_acc", "pooled_rmse", "ec_pooled_rmse"}
    if not required.issubset(table.columns) or set(table["lead"]) != set(range(6)):
        raise ValueError("National evaluation table is incomplete")
    json.loads(json_path.read_text(encoding="utf-8"))


def validate_stratified(config: PipelineConfig) -> None:
    _require_files((config.stratified_file,))
    table = pd.read_csv(config.stratified_file)
    required = {"lead", "scope_type", "period", "acc", "ec_acc", "rmse", "ec_rmse"}
    if not required.issubset(table.columns):
        raise ValueError("Stratified evaluation table is incomplete")
    national_jja = table[(table["scope_type"] == "national") & (table["period"] == "JJA")]
    if set(national_jja["lead"]) != set(range(6)):
        raise ValueError("Stratified evaluation is missing national JJA Lead 0-5")


def validate_experiment_name(name: str) -> str:
    if not name or Path(name).name != name or name in {".", ".."}:
        raise argparse.ArgumentTypeError("experiment name must be one directory name")
    return name


def select_stages(stages: list[Stage], start: str, end: str) -> list[Stage]:
    start_index = STAGE_ORDER.index(start)
    end_index = STAGE_ORDER.index(end)
    if start_index > end_index:
        raise ValueError("--from-stage must not follow --through-stage")
    return stages[start_index : end_index + 1]


def _command_text(command: tuple[str, ...]) -> str:
    return subprocess.list2cmdline(list(command))


def _write_manifest(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def _artifact_records(outputs: tuple[Path, ...]) -> list[dict]:
    return [
        {
            "path": str(path),
            "exists": path.is_file(),
            "size_bytes": path.stat().st_size if path.is_file() else None,
        }
        for path in outputs
    ]


def run_stage(stage: Stage, config: PipelineConfig, log_dir: Path) -> dict:
    for output in stage.outputs:
        output.parent.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{stage.name}.log"
    env = os.environ.copy()
    env.update(stage.environment)
    env["PYTHONUTF8"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"command={_command_text(stage.command)}\n")
        if stage.environment:
            log.write("stage_environment=" + json.dumps(stage.environment, ensure_ascii=False) + "\n")
        log.flush()
        process = subprocess.Popen(
            stage.command,
            cwd=SCRIPT_DIR,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for output_line in process.stdout:
            print(output_line, end="")
            log.write(output_line)
        return_code = process.wait()
    duration = time.monotonic() - started
    if return_code != 0:
        raise RuntimeError(
            f"Stage {stage.name} failed with exit code {return_code}; see {log_path}"
        )
    stage.validator(config)
    return {
        "status": "completed",
        "duration_seconds": duration,
        "command": list(stage.command),
        "environment": stage.environment,
        "log": str(log_path),
        "artifacts": _artifact_records(stage.outputs),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the complete current recommended Lead 0-5 model pipeline."
    )
    parser.add_argument(
        "--experiment-name",
        type=validate_experiment_name,
        default="current_recommended_pipeline",
        help="Managed results namespace below results/U_Net_3D/.",
    )
    parser.add_argument("--station-csv", type=Path, default=STATION_TABLE_FILE)
    parser.add_argument("--observation-file", type=Path, default=OBSERVATION_FILE)
    parser.add_argument("--reference-start", type=int, default=1994)
    parser.add_argument("--reference-end", type=int, default=2010)
    parser.add_argument("--minimum-reference-years", type=int, default=10)
    parser.add_argument("--climatology-floor-mm", type=float, default=0.1)
    parser.add_argument("--observation-start", default="1994-01-01")
    parser.add_argument("--observation-end", default="2024-12-01")
    parser.add_argument(
        "--transfer-sources",
        default="ECMWF,NCEP,JMA",
        help="Comma-separated model-as-sample sources; ECMWF must be first.",
    )
    parser.add_argument("--selection-fold-count", type=int, choices=range(1, 6), default=5)
    parser.add_argument("--test-months", type=int, default=21)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--ams-candidates",
        default="",
        help="Optional comma-separated AMS candidate subset; blank preserves the full search.",
    )
    parser.add_argument("--from-stage", choices=STAGE_ORDER, default=STAGE_ORDER[0])
    parser.add_argument("--through-stage", choices=STAGE_ORDER, default=STAGE_ORDER[-1])
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Validate and skip complete stages (default: enabled).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run selected stages even when their validated outputs already exist.",
    )
    parser.add_argument("--plan", action="store_true", help="Print the resolved workflow only.")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate selected existing artifacts without running stages.",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> PipelineConfig:
    sources = tuple(item.strip().upper() for item in args.transfer_sources.split(",") if item.strip())
    if not sources or sources[0] != "ECMWF":
        raise ValueError("--transfer-sources must start with ECMWF")
    if args.reference_start > args.reference_end:
        raise ValueError("reference start must not follow reference end")
    if args.test_months <= 0 or args.bootstrap <= 0:
        raise ValueError("test months and bootstrap iterations must be positive")
    candidates = tuple(item.strip() for item in args.ams_candidates.split(",") if item.strip())
    return PipelineConfig(
        run_dir=experiment_dir(args.experiment_name),
        observation_file=args.observation_file.resolve(),
        station_csv=args.station_csv.resolve(),
        reference_start=args.reference_start,
        reference_end=args.reference_end,
        minimum_reference_years=args.minimum_reference_years,
        climatology_floor_mm=args.climatology_floor_mm,
        observation_start=args.observation_start,
        observation_end=args.observation_end,
        transfer_sources=sources,
        selection_fold_count=args.selection_fold_count,
        test_months=args.test_months,
        bootstrap=args.bootstrap,
        seed=args.seed,
        ams_candidates=candidates,
    )


def main() -> None:
    args = build_parser().parse_args()
    config = config_from_args(args)
    stages = select_stages(
        build_stages(config), args.from_stage, args.through_stage
    )
    print("Current recommended model pipeline")
    print(f"Python: {sys.executable}")
    print(f"Run directory: {config.run_dir}")
    print(
        "Scientific contract: ref=1994-2010; transform=signed_log1p; "
        f"sources={','.join(config.transfer_sources)}; recency=0/5/10 years; "
        "fold-local PCA + Ridge; OOF blend + ECMWF safety"
    )
    for index, stage in enumerate(stages, start=1):
        print(f"[{index}/{len(stages)}] {stage.name}: {_command_text(stage.command)}")
        if stage.environment:
            print("  env: " + json.dumps(stage.environment, ensure_ascii=False))
    if args.plan:
        return

    manifest_path = config.run_dir / "pipeline_manifest.json"
    manifest = {
        "pipeline": "current_recommended_multilead_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.executable,
        "run_dir": str(config.run_dir),
        "configuration": {
            "observation_file": str(config.observation_file),
            "station_csv": str(config.station_csv),
            "reference_period": [config.reference_start, config.reference_end],
            "minimum_reference_years": config.minimum_reference_years,
            "climatology_floor_mm": config.climatology_floor_mm,
            "observation_period": [config.observation_start, config.observation_end],
            "target_transform": "signed_log1p",
            "transfer_sources": list(config.transfer_sources),
            "selection_fold_count": config.selection_fold_count,
            "recency_halflives_months": [0, 60, 120],
            "test_months": config.test_months,
            "bootstrap": config.bootstrap,
            "seed": config.seed,
            "ams_candidates": list(config.ams_candidates) or "all",
            "amplitude_calibration": False,
            "oof_weight_selection": True,
            "seasonal_ec_safety": True,
        },
        "stages": {},
    }
    _write_manifest(manifest_path, manifest)

    for index, stage in enumerate(stages, start=1):
        print(f"\n=== Stage {index}/{len(stages)}: {stage.name} ===")
        try:
            if args.validate_only:
                stage.validator(config)
                result = {
                    "status": "validated",
                    "artifacts": _artifact_records(stage.outputs),
                }
            elif args.resume and not args.force:
                try:
                    stage.validator(config)
                except (OSError, ValueError, KeyError, json.JSONDecodeError):
                    result = run_stage(stage, config, config.run_dir / "logs")
                else:
                    print("Validated existing outputs; stage skipped.")
                    result = {
                        "status": "skipped_valid",
                        "artifacts": _artifact_records(stage.outputs),
                    }
            else:
                result = run_stage(stage, config, config.run_dir / "logs")
        except Exception as error:
            manifest["status"] = "failed"
            manifest["failed_stage"] = stage.name
            manifest["error"] = f"{type(error).__name__}: {error}"
            manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
            _write_manifest(manifest_path, manifest)
            raise
        manifest["stages"][stage.name] = result
        manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
        _write_manifest(manifest_path, manifest)

    final_artifacts = (
        config.prediction_file,
        config.metrics_prefix.with_suffix(".csv"),
        config.metrics_prefix.with_suffix(".json"),
        config.stratified_file,
    )
    pipeline_complete = all(path.is_file() for path in final_artifacts)
    manifest["status"] = "completed" if pipeline_complete else "partial_completed"
    if pipeline_complete:
        manifest["final_prediction"] = str(config.prediction_file)
        manifest["national_metrics"] = str(config.metrics_prefix.with_suffix(".csv"))
        manifest["stratified_metrics"] = str(config.stratified_file)
    manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
    _write_manifest(manifest_path, manifest)
    if pipeline_complete:
        print(f"\nPipeline complete. Final prediction: {config.prediction_file}")
    else:
        print("\nSelected stages complete; final model artifacts are not complete yet.")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()

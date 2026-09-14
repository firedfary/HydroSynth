"""Focused regression tests for the recommended-pipeline orchestrator."""

from pathlib import Path

from run_recommended_pipeline import (
    PipelineConfig,
    STAGE_ORDER,
    build_stages,
    select_stages,
    validate_experiment_name,
)


def test_stage_order_and_current_scientific_contract(tmp_path: Path) -> None:
    config = PipelineConfig(run_dir=tmp_path / "run")
    stages = build_stages(config)
    assert tuple(stage.name for stage in stages) == STAGE_ORDER

    transfer = next(stage for stage in stages if stage.name == "transfer")
    assert transfer.environment["AMS_TRANSFER_SOURCES"] == "ECMWF,NCEP,JMA"
    assert transfer.environment["AMS_SELECTION_FOLD_COUNT"] == "5"

    ensemble = next(stage for stage in stages if stage.name == "ensemble")
    command = list(ensemble.command)
    assert "--select-weights-from-oof" in command
    assert "--seasonal-ec-safety" in command
    assert "--no-calibrate-amplitude" in command
    assert command[command.index("--observation-transform") + 1] == "signed_log1p"
    assert command[command.index("--data-dir") + 1] == str(config.transfer_dir)
    assert command[command.index("--ams-oof-file") + 1] == str(config.transfer_oof)


def test_stage_slice_is_inclusive(tmp_path: Path) -> None:
    stages = build_stages(PipelineConfig(run_dir=tmp_path / "run"))
    selected = select_stages(stages, "seasonal_stacking", "ensemble")
    assert [stage.name for stage in selected] == [
        "seasonal_stacking", "transfer", "ensemble"
    ]


def test_experiment_name_rejects_path_traversal() -> None:
    assert validate_experiment_name("recommended-v1") == "recommended-v1"
    for invalid in ("", "..", "nested/run", r"nested\run"):
        try:
            validate_experiment_name(invalid)
        except Exception:
            pass
        else:
            raise AssertionError(f"Expected invalid experiment name: {invalid}")

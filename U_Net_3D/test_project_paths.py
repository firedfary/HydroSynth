from pathlib import Path

from project_paths import (
    ALIGNED_OBSERVATION_FILE,
    MODEL_DATA_DIR,
    OBSERVATION_FILE,
    paths,
)

def test_subproject_paths_are_external_and_namespaced():
    repo_root = Path(__file__).resolve().parents[1]

    assert paths.subproject_name == "U_Net_3D"
    assert paths.cache_dir == paths.workspace_root / "cache" / "U_Net_3D"
    assert paths.results_dir == paths.workspace_root / "results" / "U_Net_3D"
    assert not paths.cache_dir.is_relative_to(repo_root)
    assert not paths.results_dir.is_relative_to(repo_root)


def test_raw_and_cached_inputs_have_expected_ownership():
    assert MODEL_DATA_DIR == paths.raw_data_dir / "model_data"
    assert OBSERVATION_FILE.parent == paths.cache_dir / "station_observations"
    assert ALIGNED_OBSERVATION_FILE.parent == OBSERVATION_FILE.parent


def test_raw_data_paths_cannot_escape_the_configured_root():
    try:
        paths.get_raw_data("../outside")
    except ValueError:
        pass
    else:
        raise AssertionError("parent traversal must be rejected")

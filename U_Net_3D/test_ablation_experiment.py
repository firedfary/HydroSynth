"""Unit tests for ReMAP Ablation & Sensitivity Analysis Experiment Suite.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from project_paths import OBSERVATION_FILE, paths
from experiment_ablation_and_sensitivity import (
    FACTOR_METADATA,
    PhysicalIndicesEngine,
    load_observations,
    get_area_weights,
    weighted_row_acc,
)


def test_observation_loading():
    """Verify reconstructed observation dataset integrity and dimensions."""
    observations, date_to_idx, mask = load_observations(OBSERVATION_FILE, "signed_log1p")
    assert observations.ndim == 3
    assert mask.ndim == 2
    assert len(date_to_idx) == len(observations)
    assert mask.sum() > 4000  # 5243 valid China land points


def test_area_weights():
    """Verify area weights normalize to 1.0 and match grid cells."""
    _, _, mask = load_observations(OBSERVATION_FILE, "signed_log1p")
    weights = get_area_weights(mask)
    assert len(weights) == mask.sum()
    assert np.isclose(weights.sum(), 1.0)
    assert np.all(weights > 0.0)


def test_physical_indices_cache():
    """Verify physical indices cache exists, has 11 dimensions, and proper shape."""
    cache_file = paths.cache_dir / "physical_indices_cache.npz"
    assert cache_file.exists(), f"Cache file missing: {cache_file}"

    with np.load(cache_file, allow_pickle=True) as data:
        indices = data["indices"]
        factor_names = data["factor_names"]
        assert indices.shape[-1] == 11
        assert len(factor_names) == 11
        assert len(FACTOR_METADATA) == 11
        # No NaNs or Infs
        assert not np.isnan(indices).any()
        assert not np.isinf(indices).any()


def test_factor_contribution_rates_computation():
    """Verify contribution rates are positive and sum to 100%."""
    observations, date_to_idx, mask = load_observations(OBSERVATION_FILE, "signed_log1p")
    weights = get_area_weights(mask)
    obs_dates = sorted(list(date_to_idx.keys()))

    engine = PhysicalIndicesEngine(obs_dates, paths.cache_dir)
    contributions = engine.compute_contribution_rates(
        observations, date_to_idx, mask, weights
    )

    assert len(contributions) == 11
    mean_pcts = [item["mean_contribution_pct"] for item in contributions]
    assert np.isclose(sum(mean_pcts), 100.0, atol=1.0)
    for item in contributions:
        assert item["mean_contribution_pct"] >= 0.0
        assert len(item["lead_contributions_pct"]) == 6


if __name__ == "__main__":
    print("Running test_observation_loading()...")
    test_observation_loading()
    print("Running test_area_weights()...")
    test_area_weights()
    print("Running test_physical_indices_cache()...")
    test_physical_indices_cache()
    print("Running test_factor_contribution_rates_computation()...")
    test_factor_contribution_rates_computation()
    print("\nAll unit tests passed successfully!")


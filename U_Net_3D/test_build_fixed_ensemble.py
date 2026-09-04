import numpy as np

from build_fixed_ensemble import (
    fit_oof_amplitude_calibration,
    restore_calibrated_patterns,
    select_oof_ec_safety_seasons,
    select_oof_ams_weights,
    standardize_vectors,
)


def test_standardize_vectors_is_per_map():
    values = np.asarray([[1.0, 2.0, 3.0], [10.0, 14.0, 18.0]])
    standardized = standardize_vectors(values)
    np.testing.assert_allclose(standardized.mean(axis=1), 0.0, atol=1e-12)
    np.testing.assert_allclose(standardized.std(axis=1), 1.0, atol=1e-12)


def test_oof_calibration_recovers_mean_and_spatial_coefficients():
    weights = np.asarray([0.2, 0.3, 0.5])
    patterns = standardize_vectors(
        np.asarray([[1.0, -0.5, 0.2], [-0.2, 1.2, -0.4], [0.5, -1.0, 0.7]])
    )
    ec = np.asarray([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [-1.0, -2.0, -3.0]])
    expected_mean_coefficient = 0.8
    expected_spatial_coefficient = 0.35
    target = restore_calibrated_patterns(
        patterns,
        ec,
        weights,
        expected_mean_coefficient,
        expected_spatial_coefficient,
    )
    actual = fit_oof_amplitude_calibration(patterns, ec, target, weights)
    np.testing.assert_allclose(
        actual,
        (expected_mean_coefficient, expected_spatial_coefficient),
        atol=1e-12,
    )


def test_oof_weight_selection_prefers_the_skillful_source_per_lead():
    area_weights = np.full(4, 0.25)
    target_pattern = np.asarray([[-1.0, -0.5, 0.5, 1.0]])
    targets = np.stack(
        [
            np.repeat(target_pattern, 21, axis=0),
            np.repeat(-target_pattern, 21, axis=0),
        ],
        axis=1,
    )
    ams = np.stack(
        [
            np.repeat(target_pattern, 21, axis=0),
            np.repeat(target_pattern, 21, axis=0),
        ],
        axis=1,
    )
    stack = np.stack(
        [
            np.repeat(-target_pattern, 21, axis=0),
            np.repeat(-target_pattern, 21, axis=0),
        ],
        axis=1,
    )

    selected, diagnostics = select_oof_ams_weights(
        ams, stack, targets, area_weights
    )

    assert selected[0] > 0.5
    assert selected[1] < 0.5
    np.testing.assert_allclose(
        [item["weighted_oof_acc"] for item in diagnostics], [1.0, 1.0]
    )


def test_ec_safety_flags_only_unanimously_bad_seasons():
    base = np.asarray([[-1.0, -0.5, 0.5, 1.0]])
    targets = np.repeat(base[:, None, :], 12, axis=0)
    ec = targets.copy()
    blend = targets.copy()
    blend[[0, 1, 11]] *= -1.0
    dates = np.asarray(
        [[f"2020-{month:02d}-01"] for month in range(1, 13)]
    )

    fallback, diagnostics = select_oof_ec_safety_seasons(
        blend,
        ec,
        targets,
        dates,
        np.full(4, 0.25),
        fold_size=12,
    )

    assert fallback == [{"DJF"}]
    assert sum(item["fallback_to_ecmwf"] for item in diagnostics) == 1

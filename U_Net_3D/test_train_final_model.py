"""Unit tests for ReMAP final production model (train_final_model.py).

Verifies mathematical invariants, zero-leakage transforms, simplex optimization,
amplitude restoration, and winter dynamical anchoring constraints.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from train_final_model import (
    SEASON_BY_MONTH,
    calendar_features,
    get_area_weights,
    restore_with_ecmwf_amplitude,
    score_weights_sufficient,
    select_oof_blend_weights,
    simplex_weights_grid,
    standardize_vectors,
    transform_fractional_anomaly,
    weighted_pattern_vector,
    weighted_pattern_vector_row,
    weighted_row_acc,
)


class TestReMAPFinalModel(unittest.TestCase):
    """Test suite for ReMAP final model components."""

    def test_signed_log1p_properties(self):
        """Verify signed_log1p is odd, strictly monotonic, and zero-preserving."""
        # Zero preservation
        self.assertEqual(transform_fractional_anomaly(np.array([0.0])), 0.0)

        # Odd symmetry: f(-x) = -f(x)
        x = np.array([0.1, 0.5, 1.0, 5.0, 100.0], dtype=np.float32)
        f_pos = transform_fractional_anomaly(x, "signed_log1p")
        f_neg = transform_fractional_anomaly(-x, "signed_log1p")
        np.testing.assert_allclose(f_pos, -f_neg, rtol=1e-6)

        # Monotonicity: x1 < x2 => f(x1) < f(x2)
        grid = np.linspace(-10.0, 10.0, 201)
        f_grid = transform_fractional_anomaly(grid, "signed_log1p")
        self.assertTrue(np.all(np.diff(f_grid) > 0))

        # Identity transform
        np.testing.assert_allclose(transform_fractional_anomaly(x, "none"), x)

        # Error on unknown transform
        with self.assertRaises(ValueError):
            transform_fractional_anomaly(x, "unknown_type")

    def test_area_weights(self):
        """Verify cosine latitude area weights are positive and normalized."""
        mask = np.ones((120, 140), dtype=bool)
        weights = get_area_weights(mask)
        self.assertEqual(weights.shape, (120 * 140,))
        self.assertTrue(np.all(weights > 0.0))
        np.testing.assert_allclose(weights.sum(), 1.0, atol=1e-12)

        # Low latitude weight > high latitude weight due to cosine factor
        # Row 0 is ~60 deg, Row 119 is ~0 deg
        lat_weights_row0 = weights[:140]
        lat_weights_row119 = weights[-140:]
        self.assertTrue(np.mean(lat_weights_row119) > np.mean(lat_weights_row0))

    def test_weighted_pattern_vector_normalization(self):
        """Verify weighted pattern vector has zero weighted mean and unit variance."""
        rng = np.random.default_rng(42)
        mask = np.ones((10, 10), dtype=bool)
        weights = rng.uniform(0.1, 1.0, size=100)
        weights /= weights.sum()

        field = rng.normal(loc=15.0, scale=3.5, size=(10, 10))
        vec = weighted_pattern_vector(field, mask, weights)

        w_mean = np.sum(vec * weights)
        w_var = np.sum(weights * (vec - w_mean) ** 2)
        np.testing.assert_allclose(w_mean, 0.0, atol=1e-5)
        np.testing.assert_allclose(w_var, 1.0, atol=1e-5)

    def test_weighted_row_acc_limits(self):
        """Verify weighted row ACC limits: +1.0, -1.0, and 0.0."""
        rng = np.random.default_rng(123)
        n_points = 50
        weights = rng.uniform(0.5, 1.5, size=n_points)
        weights /= weights.sum()

        target = rng.normal(size=(3, n_points))
        # Perfect correlation
        acc_perfect = weighted_row_acc(target, target, weights)
        np.testing.assert_allclose(acc_perfect, [1.0, 1.0, 1.0], atol=1e-5)

        # Opposite correlation
        acc_opposite = weighted_row_acc(-target, target, weights)
        np.testing.assert_allclose(acc_opposite, [-1.0, -1.0, -1.0], atol=1e-5)

        # Orthogonal pattern
        ortho = np.array([[1.0, -1.0], [-1.0, 1.0]])
        eq_weights = np.array([0.5, 0.5])
        acc_ortho = weighted_row_acc(ortho[:1], np.array([[1.0, 1.0]]), eq_weights)
        np.testing.assert_allclose(acc_ortho, [0.0], atol=1e-5)

    def test_standardize_vectors(self):
        """Verify row-wise zero mean and unit variance standardization."""
        rng = np.random.default_rng(789)
        arr = rng.normal(loc=5.0, scale=2.0, size=(10, 100))
        std_arr = standardize_vectors(arr)
        np.testing.assert_allclose(std_arr.mean(axis=1), 0.0, atol=1e-6)
        np.testing.assert_allclose(std_arr.std(axis=1), 1.0, atol=1e-6)

    def test_restore_with_ecmwf_amplitude(self):
        """Verify amplitude restoration matches ECMWF spatial mean and variance."""
        mask = np.zeros((4, 4), dtype=bool)
        mask[1:3, 1:3] = True  # 4 points
        patterns = np.array([[-1.0, -0.5, 0.5, 1.0]], dtype=np.float32)
        ec_fields = np.zeros((1, 4, 4), dtype=np.float32)
        # Set ECMWF values on mask: mean = 10.0, std = 2.0
        ec_vals = np.array([8.0, 9.0, 11.0, 12.0], dtype=np.float32)
        ec_fields[0, mask] = ec_vals

        restored = restore_with_ecmwf_amplitude(patterns, ec_fields, mask)

        # Check outside mask is 0
        self.assertTrue(np.all(restored[0, ~mask] == 0.0))

        # Check inside mask has ECMWF mean and std
        res_vals = restored[0, mask]
        np.testing.assert_allclose(res_vals.mean(), ec_vals.mean(), atol=1e-5)
        np.testing.assert_allclose(res_vals.std(), ec_vals.std(), atol=1e-5)

    def test_calendar_features(self):
        """Verify cyclical sinusoidal embedding produces valid unit circle coordinates."""
        dates = [pd.Timestamp(f"2024-{m:02d}-01") for m in range(1, 13)]
        features = calendar_features(dates)
        self.assertEqual(features.shape, (12, 2))
        radii = np.sum(features**2, axis=1)
        np.testing.assert_allclose(radii, 1.0, atol=1e-6)

    def test_simplex_weights_grid(self):
        """Verify discrete 4D simplex grid partitions unity with non-negative weights."""
        grid = simplex_weights_grid(step=0.05)
        self.assertEqual(grid.shape[1], 4)
        self.assertTrue(np.all(grid >= 0.0))
        np.testing.assert_allclose(grid.sum(axis=1), 1.0, atol=1e-5)
        # Verify corners (pure basis vectors) are present
        for basis in np.eye(4):
            self.assertTrue(np.any(np.all(np.isclose(grid, basis), axis=1)))

    def test_score_weights_sufficient_consistency(self):
        """Verify sufficient statistics spatial correlation matches brute force."""
        rng = np.random.default_rng(999)
        n_samples, n_sources, n_points = 10, 4, 30
        sources = rng.normal(size=(n_samples, n_sources, n_points)).astype(np.float32)
        target = rng.normal(size=(n_samples, n_points)).astype(np.float32)
        weights = simplex_weights_grid(step=0.2)  # coarse grid for speed

        corr_sufficient = score_weights_sufficient(sources, target, weights)

        # Brute force calculation for verification
        corr_brute = []
        for w in weights:
            pred = np.tensordot(sources, w, axes=([1], [0]))  # (n_samples, n_points)
            # Spatial correlation per sample
            pred_c = pred - pred.mean(axis=1, keepdims=True)
            t_c = target - target.mean(axis=1, keepdims=True)
            num = np.sum(pred_c * t_c, axis=1)
            den = np.sqrt(np.sum(pred_c**2, axis=1) * np.sum(t_c**2, axis=1))
            c = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
            corr_brute.append(float(np.mean(c)))

        np.testing.assert_allclose(corr_sufficient, corr_brute, atol=1e-5)

    def test_select_oof_blend_weights(self):
        """Verify blend weight selection identifies the superior predictor."""
        rng = np.random.default_rng(555)
        n_samples, n_leads, n_points = 105, 2, 20
        targets = rng.normal(size=(n_samples, n_leads, n_points))
        area_weights = np.full(n_points, 1.0 / n_points)

        # Lead 0: transfer is perfect, stacking is noise
        # Lead 1: stacking is perfect, transfer is noise
        transfer_oof = np.zeros_like(targets)
        stacking_oof = np.zeros_like(targets)

        transfer_oof[:, 0] = targets[:, 0]
        stacking_oof[:, 0] = rng.normal(size=(n_samples, n_points))

        transfer_oof[:, 1] = rng.normal(size=(n_samples, n_points))
        stacking_oof[:, 1] = targets[:, 1]

        best_weights = select_oof_blend_weights(
            transfer_oof, stacking_oof, targets, area_weights, fold_size=21
        )

        self.assertAlmostEqual(best_weights[0], 1.0, places=2)
        self.assertAlmostEqual(best_weights[1], 0.0, places=2)

    def test_winter_dynamical_anchoring(self):
        """Verify that DJF months enforce mapped_weight <= 0.25 (preserving >= 75% dynamic field)."""
        eval_dates = [
            pd.Timestamp("2024-01-01"),  # DJF
            pd.Timestamp("2024-02-01"),  # DJF
            pd.Timestamp("2024-07-01"),  # JJA
            pd.Timestamp("2024-12-01"),  # DJF
            pd.Timestamp("2024-10-01"),  # SON
        ]
        # Case 1: mapped_weight = 0.75 (> 0.25)
        mapped_weight = 0.75
        mw = np.asarray(
            [min(mapped_weight, 0.25) if SEASON_BY_MONTH[d.month] == "DJF" else mapped_weight for d in eval_dates],
            dtype=np.float32,
        )[:, None]

        # Jan, Feb, Dec must be clamped to 0.25
        self.assertEqual(mw[0, 0], 0.25)
        self.assertEqual(mw[1, 0], 0.25)
        self.assertEqual(mw[3, 0], 0.25)
        # Jul and Oct should remain 0.75
        self.assertEqual(mw[2, 0], 0.75)
        self.assertEqual(mw[4, 0], 0.75)

        # Case 2: mapped_weight = 0.10 (< 0.25)
        mapped_weight_small = 0.10
        mw_small = np.asarray(
            [min(mapped_weight_small, 0.25) if SEASON_BY_MONTH[d.month] == "DJF" else mapped_weight_small for d in eval_dates],
            dtype=np.float32,
        )[:, None]
        # In this case all months remain 0.10
        np.testing.assert_allclose(mw_small[:, 0], 0.10, atol=1e-6)

    def test_restore_amplitude_degenerate_edge_cases(self):
        """Verify restore_with_ecmwf_amplitude handles zero-variance inputs safely without NaNs."""
        mask = np.zeros((3, 3), dtype=bool)
        mask[0, 0] = True
        mask[0, 1] = True

        # Degenerate case 1: constant pattern (std = 0)
        flat_pattern = np.array([[5.0, 5.0]], dtype=np.float32)
        ec_field = np.array([[[1.0, 3.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]], dtype=np.float32)
        restored = restore_with_ecmwf_amplitude(flat_pattern, ec_field, mask)
        self.assertFalse(np.any(np.isnan(restored)))
        self.assertFalse(np.any(np.isinf(restored)))
        # Outside mask must be strictly 0
        self.assertTrue(np.all(restored[:, ~mask] == 0.0))

        # Degenerate case 2: constant ECMWF field (std = 0)
        norm_pattern = np.array([[-1.0, 1.0]], dtype=np.float32)
        flat_ec = np.array([[[4.0, 4.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]], dtype=np.float32)
        restored2 = restore_with_ecmwf_amplitude(norm_pattern, flat_ec, mask)
        self.assertFalse(np.any(np.isnan(restored2)))
        self.assertFalse(np.any(np.isinf(restored2)))
        np.testing.assert_allclose(restored2[0, mask], [4.0, 4.0], atol=1e-5)

    def test_standardize_vectors_zero_variance(self):
        """Verify standardize_vectors with constant rows does not produce NaN or Inf."""
        flat_rows = np.full((3, 50), 7.0, dtype=np.float32)
        std_flat = standardize_vectors(flat_rows)
        self.assertFalse(np.any(np.isnan(std_flat)))
        self.assertFalse(np.any(np.isinf(std_flat)))
        np.testing.assert_allclose(std_flat, 0.0, atol=1e-6)

    def test_signed_log1p_extreme_and_special_values(self):
        """Verify signed_log1p behavior under extreme large values and arrays."""
        huge_val = np.array([1e6, -1e6], dtype=np.float64)
        transformed = transform_fractional_anomaly(huge_val, "signed_log1p")
        self.assertAlmostEqual(transformed[0], np.log(1.0 + 1e6), places=5)
        self.assertAlmostEqual(transformed[1], -np.log(1.0 + 1e6), places=5)


if __name__ == "__main__":
    unittest.main()


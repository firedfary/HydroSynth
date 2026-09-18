"""Unit and integration test suite for the eval_precip verification toolset."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from utils.eval_precip import (
    PrecipitationBenchmark,
    cma_ps_score,
    contingency_table,
    critical_success_index,
    equitable_threat_score,
    evaluate_event_metrics,
    fractions_skill_score_2d,
    grid_point_significance,
    heidke_skill_score,
    kge,
    multiscale_fss,
    nse,
    pooled_rmse,
    radial_power_spectral_density_2d,
    rmse_skill_pct,
    spatial_acc,
    temporal_correlation,
    willmott_index,
)


class TestContinuousMetrics(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.t, self.n = 10, 50
        self.obs = np.random.randn(self.t, self.n)
        self.weights = np.ones(self.n) / self.n

    def test_perfect_forecast(self):
        pred = self.obs.copy()
        acc = spatial_acc(pred, self.obs, self.weights)
        self.assertTrue(np.allclose(acc, 1.0, atol=1e-6))
        self.assertAlmostEqual(pooled_rmse(pred, self.obs, self.weights), 0.0, places=6)
        self.assertAlmostEqual(willmott_index(pred, self.obs, self.weights), 1.0, places=6)
        self.assertAlmostEqual(nse(pred, self.obs, self.weights), 1.0, places=6)
        kge_val, r, alpha, beta = kge(pred, self.obs, self.weights)
        self.assertAlmostEqual(kge_val, 1.0, places=5)

    def test_opposite_forecast(self):
        pred = -self.obs.copy()
        acc = spatial_acc(pred, self.obs, self.weights)
        self.assertTrue(np.allclose(acc, -1.0, atol=1e-6))

    def test_temporal_correlation(self):
        pred = self.obs * 0.8 + np.random.randn(self.t, self.n) * 0.2
        grid_tcc, f_mean, med = temporal_correlation(pred, self.obs, self.weights)
        self.assertEqual(len(grid_tcc), self.n)
        self.assertTrue(f_mean > 0.5)


class TestCategoricalMetrics(unittest.TestCase):
    def test_perfect_contingency(self):
        pred_e = np.array([True, True, False, False])
        obs_e = np.array([True, True, False, False])
        h, m, f, cn = contingency_table(pred_e, obs_e)
        self.assertEqual(h, 2)
        self.assertEqual(m, 0)
        self.assertEqual(f, 0)
        self.assertEqual(cn, 2)

        self.assertAlmostEqual(critical_success_index(h, m, f), 1.0)
        self.assertAlmostEqual(equitable_threat_score(h, m, f, cn), 1.0)
        self.assertAlmostEqual(heidke_skill_score(h, m, f, cn), 1.0)

    def test_cma_ps_score(self):
        # Perfect matching anomalies
        pred = np.array([-0.6, -0.3, 0.0, 0.3, 0.6])
        obs = np.array([-0.6, -0.3, 0.0, 0.3, 0.6])
        res = cma_ps_score(pred, obs)
        self.assertAlmostEqual(res["cma_ps"], 100.0, places=2)
        self.assertAlmostEqual(res["trend_accuracy_pct"], 100.0, places=2)


class TestSpatialMetrics(unittest.TestCase):
    def test_fss_identical(self):
        arr = np.zeros((30, 30))
        arr[10:20, 10:20] = 1.0
        fss = fractions_skill_score_2d(arr, arr, threshold=0.5, scale=5)
        self.assertAlmostEqual(fss, 1.0, places=5)

    def test_radial_psd(self):
        arr = np.random.randn(32, 32)
        wn, wl, psd = radial_power_spectral_density_2d(arr, dx_km=50.0)
        self.assertTrue(len(wn) > 0)
        self.assertTrue(np.all(psd >= 0.0))


class TestBenchmarkEndToEnd(unittest.TestCase):
    def test_full_pipeline_synthetic(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            t, leads, h, w = 6, 2, 20, 20
            lats = np.linspace(50, 20, h)
            lons = np.linspace(80, 120, w)
            mask = np.ones((h, w), dtype=bool)
            dates = pd.date_range("2023-01-01", periods=t, freq="MS")

            np.random.seed(123)
            obs = np.random.randn(t, leads, h, w) * 0.5
            ec = obs * 0.4 + np.random.randn(t, leads, h, w) * 0.6
            model = obs * 0.7 + np.random.randn(t, leads, h, w) * 0.3

            bench = PrecipitationBenchmark(
                name="TestBench",
                latitudes=lats,
                longitudes=lons,
                valid_mask=mask,
                dates=dates,
                lead_times=[0, 1],
                output_dir=tmp_path,
            )
            bench.set_observations(obs)
            bench.set_baseline("ECMWF", ec)
            bench.add_model("OurModel", model)

            # Continuous metrics
            df_cont = bench.evaluate_continuous_by_lead()
            self.assertFalse(df_cont.empty)
            self.assertIn("spatial_acc", df_cont.columns)
            self.assertIn("msess", df_cont.columns)

            # Categorical metrics
            df_cat = bench.evaluate_categorical_by_lead()
            self.assertFalse(df_cat.empty)
            self.assertIn("cma_ps", df_cat.columns)

            # Export tables
            tbls = bench.export_tables(tmp_path)
            self.assertTrue(tbls["csv_continuous"].exists())
            self.assertTrue(tbls["latex_table"].exists())

            # Export figures
            figs = bench.generate_all_figures(tmp_path, dpi=100)
            self.assertTrue(figs["fig1_acc"].exists())
            self.assertTrue(figs["fig2_taylor"].exists())
            self.assertTrue(figs["fig3_categorical"].exists())


if __name__ == "__main__":
    unittest.main()

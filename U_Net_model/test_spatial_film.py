"""Invariants for the optional spatial FiLM modulation path."""

import unittest

import torch

from unetlite import UNetLite
from unetlitefilm import UNetLiteFiLM


class SpatialFiLMTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.baseline = UNetLite(n_channels=10, base_filters=32, dropout=0.1).eval()
        self.legacy = UNetLiteFiLM(
            n_channels=10, index_dim=30, base_filters=32, index_hidden=16,
            dropout=0.1, film_layers=(1, 2, 3, 4), identity_init=True,
            zero_pc_identity=True,
        ).eval()
        missing, unexpected = self.legacy.load_state_dict(
            self.baseline.state_dict(), strict=False,
        )
        self.assertFalse(unexpected)
        self.assertTrue(all(key.startswith("index_encoder.") for key in missing))
        self.spatial = UNetLiteFiLM(
            n_channels=10, index_dim=30, base_filters=32, index_hidden=16,
            dropout=0.1, film_layers=(1, 2, 3, 4), identity_init=True,
            zero_pc_identity=True, spatial_film=True,
        ).eval()
        missing, unexpected = self.spatial.load_state_dict(
            self.legacy.state_dict(), strict=False,
        )
        self.assertEqual(missing, ["spatial_coeffs"])
        self.assertFalse(unexpected)
        self.x = torch.randn(2, 10, 32, 32)
        self.pcs = torch.randn(2, 30)

    def test_zero_pc_matches_baseline(self):
        with torch.no_grad():
            expected = self.baseline(self.x)
            actual = self.spatial(self.x, torch.zeros_like(self.pcs))
        self.assertTrue(torch.equal(actual, expected))

    def test_zero_spatial_coefficients_match_legacy(self):
        with torch.no_grad():
            self.legacy.index_encoder.gammas[1].weight.fill_(0.02)
            self.spatial.index_encoder.gammas[1].weight.fill_(0.02)
            old = self.legacy(self.x, self.pcs)
            new = self.spatial(self.x, self.pcs)
        torch.testing.assert_close(new, old, atol=1e-6, rtol=0)

    def test_spatial_coefficients_change_output(self):
        with torch.no_grad():
            self.spatial.index_encoder.gammas[1].weight.fill_(0.02)
            before = self.spatial(self.x, self.pcs)
            self.spatial.spatial_coeffs[1, 0] = 1.0
            after = self.spatial(self.x, self.pcs)
        self.assertGreater(float((before - after).abs().max()), 1e-6)

    def test_channel_spatial_film_identity_and_local_effect(self):
        channel = UNetLiteFiLM(
            n_channels=10, index_dim=30, base_filters=32, index_hidden=16,
            dropout=0.1, film_layers=(1, 2, 3, 4), identity_init=True,
            zero_pc_identity=True, spatial_film=True, spatial_per_channel=True,
        ).eval()
        missing, unexpected = channel.load_state_dict(self.legacy.state_dict(), strict=False)
        self.assertFalse(unexpected)
        self.assertTrue(all(key.startswith("spatial_coeffs.") for key in missing))
        with torch.no_grad():
            expected = self.baseline(self.x)
            zero = channel(self.x, torch.zeros_like(self.pcs))
            channel.index_encoder.gammas[1].weight.fill_(0.2)
            before = channel(self.x, self.pcs)
            channel.spatial_coeffs[1][:, 0] = 1.0
            after = channel(self.x, self.pcs)
        self.assertTrue(torch.equal(zero, expected))
        self.assertGreater(float((before - after).abs().max()), 1e-6)


if __name__ == "__main__":
    unittest.main()

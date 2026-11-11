"""Checks for the minibatch temporal correlation objective."""

import unittest

import torch

from strict_centered_eval import temporal_corr


class TemporalCorrTest(unittest.TestCase):
    def test_perfect_and_inverse(self):
        target = torch.arange(8, dtype=torch.float32).reshape(8, 1, 1, 1)
        mask = torch.zeros_like(target, dtype=torch.bool)
        self.assertAlmostEqual(float(temporal_corr(target, target, mask)), 1.0, places=6)
        self.assertAlmostEqual(float(temporal_corr(-target, target, mask)), -1.0, places=6)

    def test_masked_and_constant_cells_have_finite_gradient(self):
        values = torch.arange(8, dtype=torch.float32).reshape(8, 1, 1, 1)
        target = torch.cat((values, torch.ones_like(values), values), dim=-1)
        target[0, :, :, 2] = float("nan")
        mask = torch.zeros_like(target, dtype=torch.bool)
        mask[0, :, :, 2] = True
        pred = torch.nan_to_num(target, nan=0.0).clone().requires_grad_()
        corr = temporal_corr(pred, target, mask)
        self.assertAlmostEqual(float(corr.detach()), 1.0, places=6)
        (1.0 - corr).backward()
        self.assertTrue(torch.isfinite(pred.grad).all())


if __name__ == "__main__":
    unittest.main()

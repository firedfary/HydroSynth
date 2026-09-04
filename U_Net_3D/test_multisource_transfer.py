import numpy as np

from build_multisource_oof_ensemble import select_weights, simplex_weights
from experiment_model_as_sample_transfer import restore_with_ecmwf_amplitude


def test_four_source_simplex_is_complete_and_normalized():
    weights = simplex_weights(4, step=0.05)
    assert weights.shape == (1771, 4)
    assert np.all(weights >= 0.0)
    np.testing.assert_allclose(weights.sum(axis=1), 1.0)
    for basis in np.eye(4):
        assert np.any(np.all(weights == basis, axis=1))


def test_oof_selection_identifies_perfect_source():
    rng = np.random.default_rng(42)
    samples, leads, points = 105, 2, 30
    targets = rng.normal(size=(samples, leads, points))
    sources = np.stack(
        [
            rng.normal(size=targets.shape),
            targets.copy(),
            rng.normal(size=targets.shape),
        ]
    )
    selected, diagnostics = select_weights(
        sources, targets, np.full(points, 1.0 / points)
    )
    np.testing.assert_allclose(selected, np.asarray([[0.0, 1.0, 0.0]] * leads))
    assert all(item["weighted_oof_acc"] > 0.999 for item in diagnostics)


def test_amplitude_restoration_uses_ecmwf_mean_and_scale():
    mask = np.asarray([[True, True], [True, False]])
    patterns = np.asarray([[-1.0, 0.0, 1.0]])
    ec_fields = np.asarray([[[2.0, 4.0], [6.0, 0.0]]])
    restored = restore_with_ecmwf_amplitude(patterns, ec_fields, mask)
    np.testing.assert_allclose(restored[:, mask].mean(axis=1), [4.0])
    np.testing.assert_allclose(
        restored[:, mask].std(axis=1), ec_fields[:, mask].std(axis=1)
    )
    assert restored[0, 1, 1] == 0.0

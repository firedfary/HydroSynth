import numpy as np

from train_pcr_multilead import build_physical_indices, regional_mean_np


def test_regional_mean_respects_constant_samples():
    latitudes = np.arange(-20.0, 41.0, 5.0)
    longitudes = np.arange(0.0, 360.0, 10.0)
    fields = np.stack(
        [
            np.full((len(latitudes), len(longitudes)), 2.0),
            np.full((len(latitudes), len(longitudes)), -3.0),
        ]
    )
    actual = regional_mean_np(
        fields, latitudes, longitudes, (-5, 5), (190, 240)
    )
    np.testing.assert_allclose(actual, [2.0, -3.0])


def test_forecast_sst_adds_five_physical_indices():
    forecast_latitudes = np.arange(-30.0, 51.0, 5.0)
    forecast_longitudes = np.arange(0.0, 360.0, 10.0)
    sst_latitudes = np.arange(-30.0, 31.0, 5.0)
    sst_longitudes = forecast_longitudes
    forecast = np.ones(
        (2, 11, len(forecast_latitudes), len(forecast_longitudes)),
        dtype=np.float32,
    )
    observed_sst = np.ones(
        (2, 6, len(sst_latitudes), len(sst_longitudes)), dtype=np.float32
    )

    without_forecast_sst = build_physical_indices(
        forecast[:, :10],
        observed_sst,
        forecast_latitudes,
        forecast_longitudes,
        sst_latitudes,
        sst_longitudes,
    )
    with_forecast_sst = build_physical_indices(
        forecast,
        observed_sst,
        forecast_latitudes,
        forecast_longitudes,
        sst_latitudes,
        sst_longitudes,
    )

    assert without_forecast_sst.shape == (2, 11)
    assert with_forecast_sst.shape == (2, 16)
    assert np.all(np.isfinite(with_forecast_sst))

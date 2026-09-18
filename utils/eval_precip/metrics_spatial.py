"""Spatial multiscale verification and spectral analysis for precipitation fields.

Implements:
- Fractions Skill Score (FSS) across multiscale spatial neighborhood windows (Roberts & Lean, 2008).
- 2D Radial Power Spectral Density (PSD) analysis to detect smoothing and spectral energy retention.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.ndimage import uniform_filter


def fractions_skill_score_2d(
    pred_field: np.ndarray,
    obs_field: np.ndarray,
    threshold: float,
    scale: int,
    mask: Optional[np.ndarray] = None,
    operator: str = ">",
) -> float:
    """Compute Fractions Skill Score (FSS) for a single 2D field at a given spatial scale.

    Parameters:
    -----------
    pred_field : np.ndarray, shape (H, W)
        Forecast 2D grid.
    obs_field : np.ndarray, shape (H, W)
        Observed 2D grid.
    threshold : float
        Event binary threshold.
    scale : int
        Window size in grid points (e.g. 1, 3, 5, 9, 15). Must be odd or will be made odd.
    mask : np.ndarray, shape (H, W), optional
        Valid evaluation domain mask.
    operator : str
        '>' or '<'.

    Returns:
    --------
    fss : float, range [0.0, 1.0]
    """
    scale = max(1, int(scale))
    if scale % 2 == 0:
        scale += 1

    if operator == ">":
        i_pred = (pred_field > threshold).astype(np.float64)
        i_obs = (obs_field > threshold).astype(np.float64)
    else:
        i_pred = (pred_field < threshold).astype(np.float64)
        i_obs = (obs_field < threshold).astype(np.float64)

    # Neighborhood fractional coverage via uniform filter
    f_s = uniform_filter(i_pred, size=scale, mode="constant", cval=0.0)
    o_s = uniform_filter(i_obs, size=scale, mode="constant", cval=0.0)

    if mask is not None:
        valid = mask.astype(bool)
        f_s = f_s[valid]
        o_s = o_s[valid]
    else:
        f_s = f_s.ravel()
        o_s = o_s.ravel()

    mse_s = float(np.mean((f_s - o_s) ** 2))
    mse_ref = float(np.mean(f_s**2 + o_s**2))

    if mse_ref <= 1e-20:
        return 1.0 if mse_s <= 1e-20 else 0.0

    return float(np.clip(1.0 - mse_s / mse_ref, 0.0, 1.0))


def multiscale_fss(
    pred: np.ndarray,
    obs: np.ndarray,
    threshold: float,
    scales: Tuple[int, ...] = (1, 3, 5, 9, 15, 25),
    mask: Optional[np.ndarray] = None,
    operator: str = ">",
) -> Dict[int, float]:
    """Compute FSS across a range of spatial neighborhood scales.

    Parameters:
    -----------
    pred : np.ndarray, shape (..., H, W)
    obs : np.ndarray, shape (..., H, W)
    threshold : float
    scales : tuple of int
    mask : np.ndarray, shape (H, W), optional

    Returns:
    --------
    dict: {scale: mean_fss_score}
    """
    pred = np.asarray(pred, dtype=np.float64)
    obs = np.asarray(obs, dtype=np.float64)

    # Flatten leading time/sample dimensions
    orig_shape = pred.shape
    h, w = orig_shape[-2], orig_shape[-1]
    n_samples = int(np.prod(orig_shape[:-2])) if len(orig_shape) > 2 else 1
    pred_2d_list = pred.reshape(n_samples, h, w)
    obs_2d_list = obs.reshape(n_samples, h, w)

    fss_by_scale = {}
    for s in scales:
        scores = []
        for i in range(n_samples):
            score = fractions_skill_score_2d(
                pred_2d_list[i], obs_2d_list[i], threshold, s, mask=mask, operator=operator
            )
            scores.append(score)
        fss_by_scale[s] = float(np.mean(scores))

    return fss_by_scale


def radial_power_spectral_density_2d(
    field_2d: np.ndarray,
    dx_km: float = 50.0,
    window: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the radially averaged 2D Power Spectral Density (PSD) of a field.

    Parameters:
    -----------
    field_2d : np.ndarray, shape (H, W)
        2D precipitation field. NaNs are automatically filled with domain mean.
    dx_km : float, default 50.0
        Grid spacing in kilometers.
    window : bool, default True
        Whether to apply a 2D Hann window to taper boundaries.

    Returns:
    --------
    wavenumbers : np.ndarray
        Radial wavenumber (1/km).
    wavelengths_km : np.ndarray
        Corresponding physical spatial scale (km).
    radial_psd : np.ndarray
        Radially averaged power spectral density.
    """
    arr = np.asarray(field_2d, dtype=np.float64).copy()
    if np.isnan(arr).any():
        mean_val = float(np.nanmean(arr))
        arr[np.isnan(arr)] = mean_val

    # Remove domain mean
    arr = arr - np.mean(arr)
    ny, nx = arr.shape

    if window:
        wy = np.hanning(ny)
        wx = np.hanning(nx)
        win2d = np.outer(wy, wx)
        # Preserve energy
        win_norm = np.sqrt(np.mean(win2d**2))
        arr = (arr * win2d) / max(win_norm, 1e-12)

    # 2D FFT and power spectrum
    fft2d = np.fft.fft2(arr)
    fft_shifted = np.fft.fftshift(fft2d)
    power = (np.abs(fft_shifted) ** 2) / (nx * ny)

    # Wavenumber coordinates centered at 0
    kx = np.fft.fftshift(np.fft.fftfreq(nx, d=dx_km))
    ky = np.fft.fftshift(np.fft.fftfreq(ny, d=dx_km))
    kx_grid, ky_grid = np.meshgrid(kx, ky)
    k_radial = np.sqrt(kx_grid**2 + ky_grid**2)

    # Radial binning
    k_max = min(np.max(kx), np.max(ky))
    dk = min(kx[1] - kx[0], ky[1] - ky[0]) if nx > 1 and ny > 1 else 1.0 / (max(nx, ny) * dx_km)
    k_bins = np.arange(dk, k_max, dk)
    
    bin_centers = []
    radial_p = []
    for i in range(len(k_bins) - 1):
        k_low = k_bins[i]
        k_high = k_bins[i + 1]
        mask_k = (k_radial >= k_low) & (k_radial < k_high)
        if np.any(mask_k):
            radial_p.append(float(np.mean(power[mask_k])))
            bin_centers.append(float((k_low + k_high) / 2.0))

    wavenumbers = np.asarray(bin_centers)
    radial_psd = np.asarray(radial_p)
    wavelengths_km = 1.0 / np.maximum(wavenumbers, 1e-12)

    return wavenumbers, wavelengths_km, radial_psd


def average_psd_across_samples(
    fields: np.ndarray,
    dx_km: float = 50.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute average 2D radial PSD across multiple temporal/lead samples."""
    fields = np.asarray(fields, dtype=np.float64)
    orig_shape = fields.shape
    h, w = orig_shape[-2], orig_shape[-1]
    n_samples = int(np.prod(orig_shape[:-2])) if len(orig_shape) > 2 else 1
    fields_2d = fields.reshape(n_samples, h, w)

    psd_list = []
    wavenumbers = None
    wavelengths_km = None

    for i in range(n_samples):
        wn, wl, psd = radial_power_spectral_density_2d(fields_2d[i], dx_km=dx_km)
        if wavenumbers is None:
            wavenumbers = wn
            wavelengths_km = wl
        psd_list.append(psd)

    mean_psd = np.mean(psd_list, axis=0)
    return wavenumbers, wavelengths_km, mean_psd

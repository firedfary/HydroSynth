"""Publication-ready academic visualization suite for precipitation forecasting.

Strictly adheres to top-tier journal aesthetics (Nature, Science, AMS, GRL):
- Clean minimalist white background (#FFFFFF).
- High-contrast, colorblind-safe palettes (diverging for anomalies, sequential for intensities).
- Geographic maps with accurate China borders, nine-dash line, and South China Sea inset.
- Vector output (SVG/PDF) and 300 DPI publication-grade raster output (PNG).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

from .config import PLOT_CONFIG

# Locate GIS shapefiles and GMT borders in utils/
UTILS_DIR = Path(__file__).resolve().parents[1]
CHINA_SHP = UTILS_DIR / "china0.shp"
CN_BORDER_GMT = UTILS_DIR / "CN-border-La.gmt"

try:
    from .. import maskout
except Exception:
    try:
        import utils.maskout as maskout
    except Exception:
        maskout = None


class MidpointNormalize(mcolors.Normalize):
    """Normalize colormap centering around a specific midpoint value (e.g. 0.0)."""

    def __init__(self, vmin=None, vmax=None, midpoint=0.0, clip=False):
        self.midpoint = midpoint
        super().__init__(vmin, vmax, clip)

    def __call__(self, value, clip=None):
        if self.vmin == self.vmax:
            return np.ma.masked_array(np.zeros_like(value, dtype=float))
        x = [self.vmin, self.midpoint, self.vmax]
        y = [0.0, 0.5, 1.0]
        return np.ma.masked_array(np.interp(value, x, y))


def _load_borders() -> List[np.ndarray]:
    """Load China coastlines and borders from CN-border-La.gmt."""
    if not CN_BORDER_GMT.exists():
        return []
    with open(CN_BORDER_GMT, "r", encoding="utf-8") as f:
        content = f.read()
    blocks = [cnt for cnt in content.split(">") if len(cnt) > 0]
    return [np.fromstring(b, dtype=float, sep=" ") for b in blocks]


def plot_spatial_comparison(
    fields: Sequence[np.ndarray],
    titles: Sequence[str],
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    ncols: int = 3,
    vmins: Optional[Union[float, Sequence[Optional[float]]]] = None,
    vmaxs: Optional[Union[float, Sequence[Optional[float]]]] = None,
    color_modes: Union[str, Sequence[str]] = "diverging",
    save_path: Optional[Union[str, Path]] = None,
    figsize_per_panel: Tuple[float, float] = (5.5, 4.5),
    dpi: int = 300,
    show: bool = False,
) -> plt.Figure:
    """Multi-panel spatial field comparison with China borders and South China Sea inset.

    Parameters:
    -----------
    fields : list of 2D np.ndarray (H, W)
    titles : list of titles for each panel (e.g. ['(a) Observation', '(b) ECMWF', ...])
    latitudes : 1D array (H,)
    longitudes : 1D array (W,)
    ncols : int, default 3
    vmins, vmaxs : float or list of floats
    color_modes : 'diverging' (blue-red) or 'sequential'
    save_path : path to save figure
    """
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter

    n_fields = len(fields)
    nrows = int(np.ceil(n_fields / ncols))
    proj = ccrs.PlateCarree(central_longitude=0.0)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(ncols * figsize_per_panel[0], nrows * figsize_per_panel[1]),
        subplot_kw={"projection": proj},
        facecolor="white",
        squeeze=False,
    )

    borders = _load_borders()

    # Normalize limits and color modes
    if vmins is None or isinstance(vmins, (int, float)):
        vmins_list = [vmins] * n_fields
    else:
        vmins_list = list(vmins)

    if vmaxs is None or isinstance(vmaxs, (int, float)):
        vmaxs_list = [vmaxs] * n_fields
    else:
        vmaxs_list = list(vmaxs)

    if isinstance(color_modes, str):
        cmodes_list = [color_modes] * n_fields
    else:
        cmodes_list = list(color_modes)

    # Standard diverging colormap for precipitation anomalies (Red=dry, Blue=wet)
    base_colors = [
        "#8c510a",
        "#bf812d",
        "#dfc27d",
        "#f6e8c3",
        "#f5f5f5",
        "#c7eae5",
        "#80cdc1",
        "#35978f",
        "#01665e",
    ]
    cmap_diverging = mcolors.LinearSegmentedColormap.from_list("precip_anom", base_colors, N=11)
    if hasattr(cmap_diverging, "with_extremes"):
        cmap_diverging = cmap_diverging.with_extremes(over="#003c30", under="#543005")
    else:
        cmap_diverging.set_over("#003c30")
        cmap_diverging.set_under("#543005")

    for idx in range(nrows * ncols):
        r = idx // ncols
        c = idx % ncols
        ax = axes[r, c]

        if idx >= n_fields:
            ax.axis("off")
            continue

        arr = fields[idx]
        title = titles[idx] if idx < len(titles) else f"Field {idx+1}"
        vmin = vmins_list[idx] if vmins_list[idx] is not None else float(np.nanmin(arr))
        vmax = vmaxs_list[idx] if vmaxs_list[idx] is not None else float(np.nanmax(arr))
        cm_mode = cmodes_list[idx]

        ax.add_feature(cfeature.LAND.with_scale("110m"), facecolor="#F8F8F8")
        for line in borders:
            ax.plot(line[0::2], line[1::2], "-", lw=0.45, color="black", transform=ccrs.Geodetic())

        ax.set_extent([70, 140, 15, 55], crs=proj)
        ax.set_xticks(range(70, 141, 20), crs=proj)
        ax.set_yticks(range(20, 56, 10), crs=proj)
        ax.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=False))
        ax.yaxis.set_major_formatter(LatitudeFormatter())
        ax.tick_params(labelsize=9)

        if cm_mode == "diverging":
            abs_max = max(abs(vmin), abs(vmax))
            norm = MidpointNormalize(vmin=-abs_max, vmax=abs_max, midpoint=0.0)
            levels = np.linspace(-abs_max, abs_max, 11)
            cf = ax.contourf(
                longitudes,
                latitudes,
                arr,
                levels=levels,
                cmap=cmap_diverging,
                norm=norm,
                extend="both",
                transform=proj,
            )
        else:
            cf = ax.contourf(
                longitudes,
                latitudes,
                arr,
                levels=np.linspace(vmin, vmax, 10),
                cmap="viridis",
                extend="both",
                transform=proj,
            )

        if maskout and CHINA_SHP.exists():
            try:
                maskout.shp2clip(cf, ax, str(CHINA_SHP))
            except Exception:
                pass

        ax.set_title(title, fontsize=PLOT_CONFIG["font_size_title"], pad=6, loc="left")

        # Colorbar
        cbar = fig.colorbar(cf, ax=ax, orientation="horizontal", fraction=0.046, pad=0.08)
        cbar.ax.tick_params(labelsize=8)
        cbar.ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

        # Inset for South China Sea Islands
        fig.canvas.draw()
        pos = ax.get_position()
        sub_ax = fig.add_axes(
            [pos.x0 + 0.81 * pos.width, pos.y0 + 0.10 * pos.height, 0.17 * pos.width, 0.22 * pos.height],
            projection=proj,
        )
        sub_ax.add_feature(cfeature.LAND.with_scale("110m"), facecolor="#F8F8F8")
        for line in borders:
            sub_ax.plot(line[0::2], line[1::2], "-", lw=0.4, color="black", transform=ccrs.Geodetic())
        sub_ax.set_extent([105, 125, 3, 25], crs=proj)

    plt.subplots_adjust(wspace=0.18, hspace=0.25)

    if save_path:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_spatial_skill_stippling(
    skill_grid: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    sig_mask: Optional[np.ndarray] = None,
    title: str = "Spatial Skill with Significance (p < 0.05)",
    metric_label: str = "TCC",
    vmin: float = -0.4,
    vmax: float = 0.8,
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 300,
    show: bool = False,
) -> plt.Figure:
    """Plot spatial skill map with black stippling for statistically significant grid cells."""
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter

    proj = ccrs.PlateCarree()
    fig, ax = plt.subplots(figsize=(7.5, 6), subplot_kw={"projection": proj}, facecolor="white")

    borders = _load_borders()
    ax.add_feature(cfeature.LAND.with_scale("110m"), facecolor="#F8F8F8")
    for line in borders:
        ax.plot(line[0::2], line[1::2], "-", lw=0.5, color="black", transform=ccrs.Geodetic())

    ax.set_extent([70, 140, 15, 55], crs=proj)
    ax.set_xticks(range(70, 141, 15), crs=proj)
    ax.set_yticks(range(20, 56, 10), crs=proj)
    ax.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=False))
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.tick_params(labelsize=10)

    cmap = plt.cm.RdYlBu_r
    norm = MidpointNormalize(vmin=vmin, vmax=vmax, midpoint=0.0)
    cf = ax.contourf(
        longitudes,
        latitudes,
        skill_grid,
        levels=np.linspace(vmin, vmax, 13),
        cmap=cmap,
        norm=norm,
        extend="both",
        transform=proj,
    )

    if maskout and CHINA_SHP.exists():
        try:
            maskout.shp2clip(cf, ax, str(CHINA_SHP))
        except Exception:
            pass

    # Stippling for significance
    if sig_mask is not None and np.any(sig_mask):
        lon_grid, lat_grid = np.meshgrid(longitudes, latitudes)
        # Subsample stippling points to avoid cluttering
        step = max(1, len(latitudes) // 40)
        sub_sig = sig_mask[::step, ::step]
        sub_lons = lon_grid[::step, ::step][sub_sig]
        sub_lats = lat_grid[::step, ::step][sub_sig]
        ax.plot(
            sub_lons,
            sub_lats,
            "o",
            color="black",
            markersize=1.8,
            alpha=0.85,
            transform=proj,
            label="p < 0.05",
        )
        ax.legend(loc="lower left", fontsize=9, framealpha=0.8)

    ax.set_title(title, fontsize=PLOT_CONFIG["font_size_title"], loc="left", pad=8)
    cbar = fig.colorbar(cf, ax=ax, orientation="horizontal", fraction=0.046, pad=0.07)
    cbar.set_label(metric_label, fontsize=PLOT_CONFIG["font_size_label"])
    cbar.ax.tick_params(labelsize=9)

    # South China Sea inset
    fig.canvas.draw()
    pos = ax.get_position()
    sub_ax = fig.add_axes(
        [pos.x0 + 0.81 * pos.width, pos.y0 + 0.08 * pos.height, 0.17 * pos.width, 0.22 * pos.height],
        projection=proj,
    )
    sub_ax.add_feature(cfeature.LAND.with_scale("110m"), facecolor="#F8F8F8")
    for line in borders:
        sub_ax.plot(line[0::2], line[1::2], "-", lw=0.4, color="black", transform=ccrs.Geodetic())
    sub_ax.set_extent([105, 125, 3, 25], crs=proj)

    if save_path:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_lead_decay(
    leads: Sequence[int],
    models_data: Dict[str, Sequence[float]],
    ci_bands: Optional[Dict[str, Tuple[Sequence[float], Sequence[float]]]] = None,
    metric_name: str = "Spatial ACC",
    ylabel: str = "Anomaly Correlation Coefficient (ACC)",
    title: str = "Multi-Lead Forecast Skill Decay",
    colors: Optional[Dict[str, str]] = None,
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 300,
    show: bool = False,
) -> plt.Figure:
    """Plot forecast skill decay across lead times with shaded bootstrap 95% confidence bands."""
    fig, ax = plt.subplots(figsize=(7.5, 4.8), facecolor="white")

    default_colors = {
        "ECMWF": "#1f77b4",
        "ECMWF SEAS5": "#1f77b4",
        "Persistence": "#7f7f7f",
        "Climatology": "#bcbd22",
        "PCR_Ridge": "#2ca02c",
        "U_Net_3D": "#9467bd",
        "Ensemble_Safe": "#d62728",
        "Ours": "#d62728",
    }
    if colors:
        default_colors.update(colors)

    leads_arr = np.asarray(leads)

    for m_name, vals in models_data.items():
        v_arr = np.asarray(vals)
        c = default_colors.get(m_name, "#333333")
        is_ref = "ECMWF" in m_name or "Persistence" in m_name
        ls = "--" if is_ref else "-"
        marker = "s" if is_ref else "o"
        lw = PLOT_CONFIG["line_width_ref"] if is_ref else PLOT_CONFIG["line_width_main"]

        mean_val = float(np.mean(v_arr))
        lbl = f"{m_name} (mean: {mean_val:.3f})"
        ax.plot(leads_arr, v_arr, label=lbl, color=c, linestyle=ls, linewidth=lw, marker=marker, markersize=5)

        if ci_bands and m_name in ci_bands:
            low, high = ci_bands[m_name]
            ax.fill_between(leads_arr, low, high, color=c, alpha=0.15)

    ax.set_xticks(leads_arr)
    ax.set_xticklabels([f"Lead {ld}" for ld in leads_arr], fontsize=PLOT_CONFIG["font_size_tick"])
    ax.set_xlabel("Forecast Lead Time", fontsize=PLOT_CONFIG["font_size_label"])
    ax.set_ylabel(ylabel, fontsize=PLOT_CONFIG["font_size_label"])
    ax.set_title(title, fontsize=PLOT_CONFIG["font_size_title"], loc="left", pad=8)

    ax.grid(True, linestyle=PLOT_CONFIG["grid_linestyle"], color=PLOT_CONFIG["grid_color"], alpha=PLOT_CONFIG["grid_alpha"])
    ax.legend(frameon=True, facecolor="white", edgecolor="#D0D0D0", fontsize=PLOT_CONFIG["font_size_legend"])

    if save_path:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_taylor_diagram(
    models_stats: Dict[str, Tuple[float, float, float]],
    ref_label: str = "Observation",
    title: str = "Taylor Diagram (Pattern Correlation & Normalized Amplitude)",
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 300,
    show: bool = False,
) -> plt.Figure:
    """Standard Taylor Diagram displaying Correlation, Normalized STD, and Centered RMSE.

    Parameters:
    -----------
    models_stats : dict of {model_name: (std_ratio, correlation, centered_rmse)}
        std_ratio = std_model / std_ref
        correlation = spatial ACC
        centered_rmse = CRMSE / std_ref
    """
    fig = plt.figure(figsize=(7.0, 6.5), facecolor="white")
    ax = fig.add_subplot(111, polar=True)

    # Polar range: correlation in [0, 1], corresponding to theta in [0, pi/2]
    ax.set_thetamax(90)
    ax.set_thetamin(0)

    # Correlation ticks
    corr_ticks = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])
    theta_ticks = np.arccos(corr_ticks)
    ax.set_xticks(theta_ticks)
    ax.set_xticklabels([f"{c}" for c in corr_ticks], fontsize=8)

    # Label azimuthal axis
    ax.text(np.pi / 4, 1.25, "Correlation Coefficient", ha="center", va="bottom", rotation=-45, fontsize=10)

    # Radial limit
    ax.set_ylim(0, 1.6)
    ax.set_yticks([0.5, 1.0, 1.5])
    ax.set_yticklabels(["0.5", "1.0", "1.5"], fontsize=8)
    ax.text(0, 1.62, "Normalized Standard Deviation", ha="left", va="center", fontsize=9)

    # Reference point at (1.0, 0 rad)
    ax.plot(0, 1.0, "k*", markersize=12, label=f"{ref_label} (Ref)", zorder=10)

    # Arc for reference std = 1.0
    t = np.linspace(0, np.pi / 2, 100)
    ax.plot(t, np.ones_like(t), "k--", lw=0.9, alpha=0.6)

    # CRMSE circles centered at (1, 0)
    for crmse in [0.25, 0.5, 0.75, 1.0, 1.25]:
        phi = np.linspace(0, np.pi, 200)
        x = 1.0 + crmse * np.cos(phi)
        y = crmse * np.sin(phi)
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        valid = (theta >= 0) & (theta <= np.pi / 2) & (r <= 1.6)
        if np.any(valid):
            ax.plot(theta[valid], r[valid], ":", color="#999999", lw=0.8)

    markers = ["o", "s", "^", "D", "v", "P", "X"]
    palette = plt.cm.tab10.colors

    for i, (m_name, (std_r, r_val, _)) in enumerate(models_stats.items()):
        theta = np.arccos(np.clip(r_val, -1.0, 1.0))
        m = markers[i % len(markers)]
        col = palette[i % len(palette)]
        ax.plot(theta, std_r, marker=m, color=col, markersize=8, label=m_name, zorder=5)

    ax.set_title(title, fontsize=PLOT_CONFIG["font_size_title"], pad=18)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.15), frameon=True, fontsize=9)

    if save_path:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_power_spectrum(
    wavelengths_km: np.ndarray,
    psd_models: Dict[str, np.ndarray],
    title: str = "2D Radial Power Spectral Density (PSD)",
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 300,
    show: bool = False,
) -> plt.Figure:
    """Plot 2D radially averaged Power Spectral Density log-log curve."""
    fig, ax = plt.subplots(figsize=(7.0, 5.0), facecolor="white")

    palette = {"Observation": "black", "ECMWF": "#1f77b4", "Ours": "#d62728"}
    default_colors = list(plt.cm.tab10.colors)

    for i, (m_name, psd) in enumerate(psd_models.items()):
        c = palette.get(m_name, default_colors[i % len(default_colors)])
        lw = 2.0 if m_name in ("Observation", "Ours", "Ensemble_Safe") else 1.4
        ls = "-" if m_name != "ECMWF" else "--"
        ax.loglog(wavelengths_km, psd, label=m_name, color=c, linewidth=lw, linestyle=ls)

    ax.set_xlabel("Wavelength / Spatial Scale (km)", fontsize=PLOT_CONFIG["font_size_label"])
    ax.set_ylabel("Power Spectral Density (PSD)", fontsize=PLOT_CONFIG["font_size_label"])
    ax.set_title(title, fontsize=PLOT_CONFIG["font_size_title"], loc="left", pad=8)
    ax.invert_xaxis()  # Large wavelengths on left, fine scale on right

    ax.grid(True, which="both", linestyle="--", color="#E0E0E0", alpha=0.7)
    ax.legend(frameon=True, fontsize=PLOT_CONFIG["font_size_legend"])

    if save_path:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_categorical_skill_bars(
    threshold_labels: Sequence[str],
    model_scores: Dict[str, Sequence[float]],
    metric_name: str = "Equitable Threat Score (ETS)",
    title: str = "Categorical Event Skill Across Thresholds",
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 300,
    show: bool = False,
) -> plt.Figure:
    """Grouped bar chart comparing models across event anomaly thresholds."""
    fig, ax = plt.subplots(figsize=(8.5, 4.8), facecolor="white")

    n_thresholds = len(threshold_labels)
    n_models = len(model_scores)
    x = np.arange(n_thresholds)
    width = 0.8 / max(n_models, 1)

    colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728", "#9467bd"]

    for i, (m_name, scores) in enumerate(model_scores.items()):
        offset = (i - n_models / 2 + 0.5) * width
        c = colors[i % len(colors)]
        rects = ax.bar(x + offset, scores, width, label=m_name, color=c, alpha=0.88, edgecolor="none")

        # Value annotation
        for rect in rects:
            h = rect.get_height()
            if abs(h) > 0.02:
                va = "bottom" if h >= 0 else "top"
                ax.annotate(
                    f"{h:.2f}",
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 2 if h >= 0 else -8),
                    textcoords="offset points",
                    ha="center",
                    va=va,
                    fontsize=7.5,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(threshold_labels, fontsize=PLOT_CONFIG["font_size_tick"])
    ax.set_ylabel(metric_name, fontsize=PLOT_CONFIG["font_size_label"])
    ax.set_title(title, fontsize=PLOT_CONFIG["font_size_title"], loc="left", pad=8)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="-")

    ax.grid(True, axis="y", linestyle="--", color="#E5E5E5", alpha=0.7)
    ax.legend(frameon=True, fontsize=PLOT_CONFIG["font_size_legend"])

    if save_path:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_stratified_heatmap(
    matrix_df: pd.DataFrame,
    title: str = "Regional & Seasonal Skill Gain (ΔACC)",
    metric_label: str = "ΔACC",
    cmap: str = "RdBu_r",
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 300,
    show: bool = False,
) -> plt.Figure:
    """Heatmap matrix of performance skill across geographic regions and seasons."""
    fig, ax = plt.subplots(figsize=(7.5, 5.0), facecolor="white")

    data = matrix_df.values.astype(float)
    vmax = float(np.nanmax(np.abs(data))) if np.any(np.isfinite(data)) else 0.2
    vmax = max(vmax, 0.05)

    im = ax.imshow(data, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_xticks(np.arange(len(matrix_df.columns)))
    ax.set_xticklabels(matrix_df.columns, fontsize=PLOT_CONFIG["font_size_tick"])
    ax.set_yticks(np.arange(len(matrix_df.index)))
    ax.set_yticklabels(matrix_df.index, fontsize=PLOT_CONFIG["font_size_tick"])

    # Cell annotations
    for i in range(len(matrix_df.index)):
        for j in range(len(matrix_df.columns)):
            val = data[i, j]
            if np.isfinite(val):
                text_color = "white" if abs(val) > 0.5 * vmax else "black"
                bold = "bold" if val > 0 else "normal"
                ax.text(j, i, f"{val:+.3f}", ha="center", va="center", color=text_color, fontsize=9, fontweight=bold)

    ax.set_title(title, fontsize=PLOT_CONFIG["font_size_title"], loc="left", pad=10)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(metric_label, fontsize=PLOT_CONFIG["font_size_label"])

    if save_path:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_fss_curves(
    scales: Sequence[int],
    fss_by_model: Dict[str, Sequence[float]],
    threshold_desc: str = "Precip Anomaly > +20%",
    title: str = "Multiscale Fractions Skill Score (FSS)",
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 300,
    show: bool = False,
) -> plt.Figure:
    """Plot Fractions Skill Score (FSS) curves across neighborhood scales."""
    fig, ax = plt.subplots(figsize=(7.5, 4.8), facecolor="white")

    scales_arr = np.asarray(scales)
    colors = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#ff7f0e"]

    for i, (m_name, scores) in enumerate(fss_by_model.items()):
        c = colors[i % len(colors)]
        ax.plot(scales_arr, scores, marker="o", markersize=5, lw=1.8, label=m_name, color=c)

    # Reference target line for useful skill
    ax.axhline(0.5, color="gray", linestyle="--", lw=1.0, label="FSS Useful Target (0.5)")

    ax.set_xlabel("Spatial Neighborhood Scale (Grid Cells)", fontsize=PLOT_CONFIG["font_size_label"])
    ax.set_ylabel("Fractions Skill Score (FSS)", fontsize=PLOT_CONFIG["font_size_label"])
    ax.set_title(f"{title} [{threshold_desc}]", fontsize=PLOT_CONFIG["font_size_title"], loc="left", pad=8)
    ax.set_ylim(-0.05, 1.05)

    ax.grid(True, linestyle="--", color="#E5E5E5", alpha=0.7)
    ax.legend(frameon=True, fontsize=PLOT_CONFIG["font_size_legend"])

    if save_path:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig

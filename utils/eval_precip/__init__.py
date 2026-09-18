"""HydroSynth Precipitation Forecast Universal Evaluation & Academic Visualization Suite (eval_precip).

Comprehensive, model-agnostic benchmarking framework implementing cutting-edge
meteorological and AI verification standards (Nature, Science, AMS, GRL, CMA).
"""

from .config import (
    CMA_ANOMALY_THRESHOLDS,
    DEFAULT_CATEGORICAL_THRESHOLDS,
    REGIONS,
    SEASONS,
    PLOT_CONFIG,
)
from .metrics_continuous import (
    spatial_acc,
    temporal_correlation,
    pooled_rmse,
    centered_rmse,
    mae,
    mean_bias,
    msess,
    rmse_skill_pct,
    willmott_index,
    kge,
    nse,
    spatial_spread,
    compute_all_continuous_metrics,
)
from .metrics_categorical import (
    contingency_table,
    critical_success_index,
    equitable_threat_score,
    heidke_skill_score,
    probability_of_detection,
    false_alarm_ratio,
    probability_of_false_detection,
    frequency_bias,
    evaluate_event_metrics,
    cma_ps_score,
)
from .metrics_spatial import (
    fractions_skill_score_2d,
    multiscale_fss,
    radial_power_spectral_density_2d,
    average_psd_across_samples,
)
from .diagnostics import (
    moving_block_indices,
    bootstrap_metric_intervals,
    grid_point_significance,
    evaluate_stratified_by_season_and_region,
)
from .visualizer import (
    plot_spatial_comparison,
    plot_spatial_skill_stippling,
    plot_lead_decay,
    plot_taylor_diagram,
    plot_power_spectrum,
    plot_categorical_skill_bars,
    plot_stratified_heatmap,
    plot_fss_curves,
)
from .benchmark import PrecipitationBenchmark

__all__ = [
    # Benchmark class
    "PrecipitationBenchmark",
    # Continuous metrics
    "spatial_acc",
    "temporal_correlation",
    "pooled_rmse",
    "centered_rmse",
    "mae",
    "mean_bias",
    "msess",
    "rmse_skill_pct",
    "willmott_index",
    "kge",
    "nse",
    "spatial_spread",
    "compute_all_continuous_metrics",
    # Categorical metrics
    "contingency_table",
    "critical_success_index",
    "equitable_threat_score",
    "heidke_skill_score",
    "probability_of_detection",
    "false_alarm_ratio",
    "probability_of_false_detection",
    "frequency_bias",
    "evaluate_event_metrics",
    "cma_ps_score",
    # Spatial & spectral
    "fractions_skill_score_2d",
    "multiscale_fss",
    "radial_power_spectral_density_2d",
    "average_psd_across_samples",
    # Diagnostics & Significance
    "moving_block_indices",
    "bootstrap_metric_intervals",
    "grid_point_significance",
    "evaluate_stratified_by_season_and_region",
    # Publication visualizer
    "plot_spatial_comparison",
    "plot_spatial_skill_stippling",
    "plot_lead_decay",
    "plot_taylor_diagram",
    "plot_power_spectrum",
    "plot_categorical_skill_bars",
    "plot_stratified_heatmap",
    "plot_fss_curves",
    # Config
    "CMA_ANOMALY_THRESHOLDS",
    "DEFAULT_CATEGORICAL_THRESHOLDS",
    "REGIONS",
    "SEASONS",
    "PLOT_CONFIG",
]

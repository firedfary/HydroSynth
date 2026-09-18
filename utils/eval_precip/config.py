"""Configuration and standard definitions for precipitation evaluation.

Includes:
- Standard Chinese meteorological & geographical diagnostic regions.
- Meteorological seasons (DJF, MAM, JJA, SON).
- CMA (China Meteorological Administration) operational precipitation anomaly categories.
- Default plotting parameters conforming to academic journal standards.
"""

from __future__ import annotations

from typing import Dict, Tuple

# Meteorological seasons (months 1-12)
SEASONS: Dict[str, Tuple[int, ...]] = {
    "DJF": (12, 1, 2),
    "MAM": (3, 4, 5),
    "JJA": (6, 7, 8),
    "SON": (9, 10, 11),
}

# Standard Chinese diagnostic climate regions: (lat_min, lat_max, lon_min, lon_max)
# Bounded within 10-60°N, 70-140°E
REGIONS: Dict[str, Tuple[float, float, float, float]] = {
    "Northwest": (35.0, 50.0, 73.0, 105.0),
    "North": (35.0, 42.0, 105.0, 120.0),
    "Northeast": (40.0, 54.0, 120.0, 135.0),
    "Yangtze": (27.0, 35.0, 105.0, 122.0),
    "South": (20.0, 27.0, 105.0, 123.0),
    "Southwest": (22.0, 35.0, 97.0, 105.0),
}

# CMA standard monthly/seasonal precipitation anomaly percentage categories (fractional: % / 100)
# Class 1: Severe Drought (Pa <= -50%)
# Class 2: Moderate Drought (-50% < Pa <= -20%)
# Class 3: Normal (-20% < Pa < +20%)
# Class 4: Moderate Wet (+20% <= Pa < +50%)
# Class 5: Severe Wet (Pa >= +50%)
CMA_ANOMALY_THRESHOLDS = {
    "severe_dry": -0.50,
    "moderate_dry": -0.20,
    "moderate_wet": 0.20,
    "severe_wet": 0.50,
}

# Standard categorical thresholds for TS/ETS/POD/FAR evaluation
DEFAULT_CATEGORICAL_THRESHOLDS = {
    "Dry_Severe (<-0.50)": ("<", -0.50),
    "Dry_Moderate (<-0.20)": ("<", -0.20),
    "Wet_Any (>0.00)": (">", 0.00),
    "Wet_Moderate (>0.20)": (">", 0.20),
    "Wet_Severe (>0.50)": (">", 0.50),
    "Wet_Extreme (>1.00)": (">", 1.00),
}

# Academic plotting aesthetic defaults
PLOT_CONFIG = {
    "font_family": "sans-serif",
    "font_sans_serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font_size_title": 13,
    "font_size_label": 11,
    "font_size_tick": 10,
    "font_size_legend": 10,
    "line_width_main": 1.8,
    "line_width_ref": 1.4,
    "dpi": 300,
    "figure_facecolor": "#FFFFFF",
    "axes_facecolor": "#FFFFFF",
    "grid_color": "#E5E5E5",
    "grid_linestyle": "--",
    "grid_alpha": 0.7,
}

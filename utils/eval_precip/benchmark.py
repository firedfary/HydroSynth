"""Universal high-level benchmark suite for multi-model precipitation forecasting.

PrecipitationBenchmark:
- Model-agnostic: accepts predictions from any model (UNet, Diffusion, FNO, PCR, Tree models, ECMWF).
- Supports both 2D gridded fields (T, L, H, W) and 1D station vectors (T, L, N).
- Computes comprehensive continuous, categorical, multiscale, and stratified metrics.
- Automatically exports tidy CSV summaries, LaTeX paper tables, and publication-ready figures.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .config import CMA_ANOMALY_THRESHOLDS, DEFAULT_CATEGORICAL_THRESHOLDS, REGIONS, SEASONS
from .diagnostics import (
    bootstrap_metric_intervals,
    evaluate_stratified_by_season_and_region,
    grid_point_significance,
)
from .metrics_categorical import (
    cma_ps_score,
    evaluate_event_metrics,
)
from .metrics_continuous import (
    centered_rmse,
    compute_all_continuous_metrics,
    kge,
    mae,
    mean_bias,
    msess,
    normalize_weights,
    nse,
    pooled_rmse,
    rmse_skill_pct,
    spatial_acc,
    spatial_spread,
    temporal_correlation,
    willmott_index,
)
from .metrics_spatial import (
    average_psd_across_samples,
    multiscale_fss,
)
from .visualizer import (
    plot_categorical_skill_bars,
    plot_fss_curves,
    plot_lead_decay,
    plot_power_spectrum,
    plot_spatial_comparison,
    plot_spatial_skill_stippling,
    plot_stratified_heatmap,
    plot_taylor_diagram,
)


class PrecipitationBenchmark:
    """Universal academic benchmark suite for precipitation prediction models."""

    def __init__(
        self,
        name: str = "Precipitation_Benchmark",
        latitudes: Optional[np.ndarray] = None,
        longitudes: Optional[np.ndarray] = None,
        valid_mask: Optional[np.ndarray] = None,
        dates: Optional[Sequence[Union[str, pd.Timestamp]]] = None,
        lead_times: Optional[Sequence[int]] = None,
        output_dir: Optional[Union[str, Path]] = None,
    ):
        self.name = name
        self.latitudes = np.asarray(latitudes, dtype=np.float64) if latitudes is not None else None
        self.longitudes = np.asarray(longitudes, dtype=np.float64) if longitudes is not None else None
        self.valid_mask = np.asarray(valid_mask, dtype=bool) if valid_mask is not None else None
        self.dates = pd.to_datetime(dates) if dates is not None else None
        self.lead_times = list(lead_times) if lead_times is not None else None
        self.output_dir = Path(output_dir) if output_dir is not None else None

        self.obs: Optional[np.ndarray] = None
        self.baseline_name: Optional[str] = None
        self.baseline_data: Optional[np.ndarray] = None
        self.models: Dict[str, np.ndarray] = {}

        # Area weights derivation
        self.weights_2d: Optional[np.ndarray] = None
        self.weights_1d: Optional[np.ndarray] = None
        if self.latitudes is not None and self.longitudes is not None:
            lat_grid, _ = np.meshgrid(self.latitudes, self.longitudes, indexing="ij")
            self.weights_2d = np.cos(np.deg2rad(lat_grid))
            if self.valid_mask is not None:
                self.weights_1d = self.weights_2d[self.valid_mask]
                self.weights_1d = self.weights_1d / np.sum(self.weights_1d)

    def set_observations(self, obs_data: np.ndarray) -> PrecipitationBenchmark:
        """Register observed ground truth fields."""
        self.obs = np.asarray(obs_data, dtype=np.float64)
        return self

    def set_baseline(self, name: str, data: np.ndarray) -> PrecipitationBenchmark:
        """Register the primary reference baseline (e.g. ECMWF SEAS5)."""
        self.baseline_name = name
        self.baseline_data = np.asarray(data, dtype=np.float64)
        return self

    def add_model(self, name: str, data: np.ndarray) -> PrecipitationBenchmark:
        """Register a candidate model's predictions."""
        self.models[name] = np.asarray(data, dtype=np.float64)
        return self

    def _get_eval_arrays(
        self, arr: np.ndarray, lead: Optional[int] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Slice specified lead time and apply valid_mask if 2D grid."""
        if lead is not None and arr.ndim >= 3:
            # (T, L, H, W) or (T, L, N)
            sub = arr[:, lead]
        else:
            sub = arr

        # sub is now (T, H, W) or (T, N)
        if sub.ndim == 3 and self.valid_mask is not None:
            flat = sub[:, self.valid_mask]
            w = self.weights_1d
            return flat, w
        elif sub.ndim == 3:
            flat = sub.reshape(len(sub), -1)
            w = self.weights_2d.ravel() if self.weights_2d is not None else None
            return flat, w
        else:
            # Already 1D spatial points (T, N)
            return sub, self.weights_1d

    def evaluate_continuous_by_lead(self) -> pd.DataFrame:
        """Evaluate continuous metrics across all models and lead times."""
        if self.obs is None:
            raise ValueError("Observations must be set before evaluation.")

        n_leads = self.obs.shape[1] if self.obs.ndim >= 3 and self.lead_times is not None else 1
        leads_range = self.lead_times if self.lead_times is not None else list(range(n_leads))

        all_models = {}
        if self.baseline_name and self.baseline_data is not None:
            all_models[self.baseline_name] = self.baseline_data
        all_models.update(self.models)

        rows = []
        for lead_idx, lead_val in enumerate(leads_range):
            obs_flat, w = self._get_eval_arrays(self.obs, lead=lead_idx)
            ref_flat = (
                self._get_eval_arrays(self.baseline_data, lead=lead_idx)[0]
                if self.baseline_data is not None
                else None
            )

            for m_name, m_data in all_models.items():
                m_flat, _ = self._get_eval_arrays(m_data, lead=lead_idx)
                # Compute all continuous metrics
                m_metrics = compute_all_continuous_metrics(m_flat, obs_flat, ref=ref_flat, weights=w)
                m_metrics["model"] = m_name
                m_metrics["lead"] = lead_val
                rows.append(m_metrics)

        df = pd.DataFrame(rows)
        # Reorder columns logically
        cols_order = ["model", "lead", "spatial_acc", "acc_gain", "pooled_rmse", "rmse_skill_pct", "msess", "tcc_fisher_mean", "willmott_index", "kge", "nse", "mae", "mean_bias"]
        available_cols = [c for c in cols_order if c in df.columns] + [c for c in df.columns if c not in cols_order]
        return df[available_cols]

    def evaluate_categorical_by_lead(
        self,
        thresholds: Optional[Dict[str, Tuple[str, float]]] = None,
    ) -> pd.DataFrame:
        """Evaluate categorical and extreme precipitation skills across models and leads."""
        if self.obs is None:
            raise ValueError("Observations must be set before evaluation.")
        if thresholds is None:
            thresholds = DEFAULT_CATEGORICAL_THRESHOLDS

        n_leads = self.obs.shape[1] if self.obs.ndim >= 3 and self.lead_times is not None else 1
        leads_range = self.lead_times if self.lead_times is not None else list(range(n_leads))

        all_models = {}
        if self.baseline_name and self.baseline_data is not None:
            all_models[self.baseline_name] = self.baseline_data
        all_models.update(self.models)

        rows = []
        for lead_idx, lead_val in enumerate(leads_range):
            obs_flat, w = self._get_eval_arrays(self.obs, lead=lead_idx)

            for m_name, m_data in all_models.items():
                m_flat, _ = self._get_eval_arrays(m_data, lead=lead_idx)

                # CMA PS score
                ps_res = cma_ps_score(m_flat, obs_flat, weights=w)

                for th_name, (op, th_val) in thresholds.items():
                    ev_res = evaluate_event_metrics(m_flat, obs_flat, op, th_val, weights=w)
                    row = {
                        "model": m_name,
                        "lead": lead_val,
                        "threshold_name": th_name,
                        "threshold_val": th_val,
                        "operator": op,
                        "csi": ev_res["csi"],
                        "ets": ev_res["ets"],
                        "hss": ev_res["hss"],
                        "pod": ev_res["pod"],
                        "far": ev_res["far"],
                        "bias": ev_res["bias"],
                        "cma_ps": ps_res["cma_ps"],
                        "obs_freq": ev_res["obs_freq"],
                    }
                    rows.append(row)

        return pd.DataFrame(rows)

    def evaluate_bootstrap_ci(
        self,
        iterations: int = 1000,
        block_length: int = 3,
        seed: int = 42,
    ) -> Dict[str, Dict[int, Dict[str, Any]]]:
        """Compute moving block bootstrap confidence intervals for all models and leads."""
        if self.obs is None:
            raise ValueError("Observations must be set.")

        n_leads = self.obs.shape[1] if self.obs.ndim >= 3 and self.lead_times is not None else 1
        leads_range = self.lead_times if self.lead_times is not None else list(range(n_leads))

        ci_results = {}
        for m_name, m_data in self.models.items():
            ci_results[m_name] = {}
            for lead_idx, lead_val in enumerate(leads_range):
                obs_flat, w = self._get_eval_arrays(self.obs, lead=lead_idx)
                m_flat, _ = self._get_eval_arrays(m_data, lead=lead_idx)
                ref_flat = (
                    self._get_eval_arrays(self.baseline_data, lead=lead_idx)[0]
                    if self.baseline_data is not None
                    else None
                )

                ci_res = bootstrap_metric_intervals(
                    m_flat,
                    obs_flat,
                    ref_series=ref_flat,
                    weights=w,
                    iterations=iterations,
                    block_length=block_length,
                    seed=seed,
                )
                ci_results[m_name][lead_val] = ci_res

        return ci_results

    def generate_all_figures(
        self,
        output_dir: Optional[Union[str, Path]] = None,
        dpi: int = 300,
    ) -> Dict[str, Path]:
        """Generate full suite of publication-ready academic figures and save to output_dir."""
        out_dir = Path(output_dir) if output_dir is not None else self.output_dir
        if out_dir is None:
            raise ValueError("output_dir must be specified.")
        out_dir.mkdir(parents=True, exist_ok=True)

        generated_figures = {}
        cont_df = self.evaluate_continuous_by_lead()

        # 1. Lead-time decay curves (ACC and RMSE)
        leads = self.lead_times if self.lead_times is not None else sorted(cont_df["lead"].unique().tolist())
        acc_by_model = {}
        rmse_by_model = {}
        for m_name in cont_df["model"].unique():
            sub = cont_df[cont_df["model"] == m_name].sort_values("lead")
            acc_by_model[m_name] = sub["spatial_acc"].tolist()
            rmse_by_model[m_name] = sub["pooled_rmse"].tolist()

        fig1_path = out_dir / "fig1_lead_decay_acc.png"
        plot_lead_decay(
            leads,
            acc_by_model,
            metric_name="Spatial ACC",
            ylabel="Spatial Anomaly Correlation Coefficient (ACC)",
            title="Multi-Lead Precipitation Forecast Skill (Spatial ACC)",
            save_path=fig1_path,
            dpi=dpi,
        )
        generated_figures["fig1_acc"] = fig1_path

        fig1b_path = out_dir / "fig1b_lead_decay_rmse.png"
        plot_lead_decay(
            leads,
            rmse_by_model,
            metric_name="Pooled RMSE",
            ylabel="Dimensionless Fractional Error (RMSE)",
            title="Multi-Lead Precipitation Error (Pooled RMSE)",
            save_path=fig1b_path,
            dpi=dpi,
        )
        generated_figures["fig1_rmse"] = fig1b_path

        # 2. Taylor diagram (Lead 0 and Lead 1)
        taylor_stats = {}
        target_lead = 0
        obs_flat, w = self._get_eval_arrays(self.obs, lead=target_lead)
        obs_std = spatial_spread(obs_flat, w)

        all_models = {}
        if self.baseline_name and self.baseline_data is not None:
            all_models[self.baseline_name] = self.baseline_data
        all_models.update(self.models)

        for m_name, m_data in all_models.items():
            m_flat, _ = self._get_eval_arrays(m_data, lead=target_lead)
            m_std = spatial_spread(m_flat, w)
            std_ratio = m_std / max(obs_std, 1e-12)
            corr = float(np.mean(spatial_acc(m_flat, obs_flat, w)))
            crmse_val = centered_rmse(m_flat, obs_flat, w) / max(obs_std, 1e-12)
            taylor_stats[m_name] = (std_ratio, corr, crmse_val)

        fig2_path = out_dir / "fig2_taylor_diagram_lead0.png"
        plot_taylor_diagram(
            taylor_stats,
            title=f"Taylor Diagram (Lead {target_lead})",
            save_path=fig2_path,
            dpi=dpi,
        )
        generated_figures["fig2_taylor"] = fig2_path

        # 3. Categorical skill bars (ETS and CMA PS)
        cat_df = self.evaluate_categorical_by_lead()
        sub_cat = cat_df[cat_df["lead"] == target_lead]
        threshold_names = sub_cat["threshold_name"].unique().tolist()

        ets_by_model = {}
        for m_name in sub_cat["model"].unique():
            m_sub = sub_cat[sub_cat["model"] == m_name]
            ets_by_model[m_name] = [
                float(m_sub[m_sub["threshold_name"] == th]["ets"].iloc[0])
                for th in threshold_names
            ]

        fig3_path = out_dir / "fig3_categorical_ets.png"
        plot_categorical_skill_bars(
            threshold_names,
            ets_by_model,
            metric_name="Equitable Threat Score (ETS)",
            title=f"Categorical Event Skill Across Anomaly Thresholds (Lead {target_lead})",
            save_path=fig3_path,
            dpi=dpi,
        )
        generated_figures["fig3_categorical"] = fig3_path

        # 4. Spatial comparison map (if 2D grid data is available)
        if self.obs.ndim == 4 and self.latitudes is not None and self.longitudes is not None:
            # Pick a sample test time step (e.g. index 0)
            t_idx = 0
            date_str = str(self.dates[t_idx])[:7] if self.dates is not None else f"Month {t_idx+1}"
            fields = [self.obs[t_idx, target_lead]]
            titles = [f"(a) Observation ({date_str})"]

            if self.baseline_name and self.baseline_data is not None:
                fields.append(self.baseline_data[t_idx, target_lead])
                titles.append(f"(b) {self.baseline_name} (Lead {target_lead})")

            # Add primary candidate model
            primary_model_name = list(self.models.keys())[0] if self.models else None
            if primary_model_name:
                m_field = self.models[primary_model_name][t_idx, target_lead]
                fields.append(m_field)
                titles.append(f"(c) {primary_model_name} (Lead {target_lead})")

                # Add residual error map
                res_field = m_field - self.obs[t_idx, target_lead]
                fields.append(res_field)
                titles.append("(d) Residual Error (Model - Obs)")

            fig4_path = out_dir / "fig4_spatial_comparison_case.png"
            plot_spatial_comparison(
                fields,
                titles,
                self.latitudes,
                self.longitudes,
                ncols=2 if len(fields) <= 4 else 3,
                color_modes=["diverging"] * len(fields),
                save_path=fig4_path,
                dpi=dpi,
            )
            generated_figures["fig4_spatial_map"] = fig4_path

            # 5. 2D Power Spectral Density
            psd_dict = {"Observation": average_psd_across_samples(self.obs[:, target_lead])[2]}
            if self.baseline_name and self.baseline_data is not None:
                psd_dict[self.baseline_name] = average_psd_across_samples(self.baseline_data[:, target_lead])[2]
            if primary_model_name:
                psd_dict[primary_model_name] = average_psd_across_samples(self.models[primary_model_name][:, target_lead])[2]

            _, wavelengths_km, _ = average_psd_across_samples(self.obs[:, target_lead])
            fig5_path = out_dir / "fig5_power_spectral_density.png"
            plot_power_spectrum(
                wavelengths_km,
                psd_dict,
                title=f"2D Radial Power Spectral Density (Lead {target_lead})",
                save_path=fig5_path,
                dpi=dpi,
            )
            generated_figures["fig5_psd"] = fig5_path

            # 6. Spatial skill stippling map
            if primary_model_name and self.baseline_data is not None:
                diff_mean, p_vals, sig_mask = grid_point_significance(
                    self.models[primary_model_name][:, target_lead],
                    self.baseline_data[:, target_lead],
                    self.obs[:, target_lead],
                    metric="absolute_error",
                )
                fig6_path = out_dir / "fig6_spatial_significance_stippling.png"
                plot_spatial_skill_stippling(
                    diff_mean,
                    self.latitudes,
                    self.longitudes,
                    sig_mask=sig_mask,
                    title=f"MAE Reduction ({primary_model_name} vs. {self.baseline_name}) [Lead {target_lead}]",
                    metric_label="MAE Reduction (Higher = Better)",
                    save_path=fig6_path,
                    dpi=dpi,
                )
                generated_figures["fig6_stippling"] = fig6_path

            # 7. Fractions Skill Score (FSS) curves
            fss_dict = {}
            if self.baseline_name and self.baseline_data is not None:
                fss_res = multiscale_fss(self.baseline_data[:, target_lead], self.obs[:, target_lead], threshold=0.20, mask=self.valid_mask)
                fss_dict[self.baseline_name] = list(fss_res.values())
            if primary_model_name:
                fss_res = multiscale_fss(self.models[primary_model_name][:, target_lead], self.obs[:, target_lead], threshold=0.20, mask=self.valid_mask)
                fss_dict[primary_model_name] = list(fss_res.values())

            fig7_path = out_dir / "fig7_fss_multiscale.png"
            plot_fss_curves(
                scales=[1, 3, 5, 9, 15, 25],
                fss_by_model=fss_dict,
                threshold_desc="Precip Anomaly > +20%",
                save_path=fig7_path,
                dpi=dpi,
            )
            generated_figures["fig7_fss"] = fig7_path

        # 8. Seasonal & Regional Heatmap (if dates available)
        if self.dates is not None and primary_model_name and self.baseline_data is not None:
            strat_df = evaluate_stratified_by_season_and_region(
                self.models[primary_model_name][:, target_lead],
                self.obs[:, target_lead],
                self.dates,
                ref_fields=self.baseline_data[:, target_lead],
                latitudes=self.latitudes,
                longitudes=self.longitudes,
                valid_mask=self.valid_mask,
            )
            # Pivot into matrix: index = region, columns = period
            if "acc_gain" in strat_df.columns:
                pivot_df = strat_df.pivot(index="region", columns="period", values="acc_gain")
                fig8_path = out_dir / "fig8_seasonal_regional_heatmap.png"
                plot_stratified_heatmap(
                    pivot_df,
                    title=f"Regional & Seasonal ACC Gain ({primary_model_name} vs. {self.baseline_name})",
                    metric_label="ΔACC",
                    save_path=fig8_path,
                    dpi=dpi,
                )
                generated_figures["fig8_heatmap"] = fig8_path

        return generated_figures

    def export_tables(
        self,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Path]:
        """Export comprehensive evaluation metrics as CSV and academic LaTeX tables."""
        out_dir = Path(output_dir) if output_dir is not None else self.output_dir
        if out_dir is None:
            raise ValueError("output_dir must be specified.")
        out_dir.mkdir(parents=True, exist_ok=True)

        tables = {}
        cont_df = self.evaluate_continuous_by_lead()
        cat_df = self.evaluate_categorical_by_lead()

        # Export CSVs
        csv_cont = out_dir / "metrics_continuous_by_lead.csv"
        cont_df.to_csv(csv_cont, index=False)
        tables["csv_continuous"] = csv_cont

        csv_cat = out_dir / "metrics_categorical_by_lead.csv"
        cat_df.to_csv(csv_cat, index=False)
        tables["csv_categorical"] = csv_cat

        # Export academic LaTeX table for paper publication
        latex_path = out_dir / "table_model_evaluation.tex"
        # Pivot table: rows = leads, columns = models (ACC, RMSE, MSESS)
        pivot_tex = cont_df.pivot(index="lead", columns="model", values=["spatial_acc", "pooled_rmse", "msess"])
        with open(latex_path, "w", encoding="utf-8") as f:
            f.write("% Academic LaTeX Table generated by HydroSynth PrecipitationBenchmark\n")
            f.write(pivot_tex.to_latex(float_format="%.3f"))
        tables["latex_table"] = latex_path

        return tables

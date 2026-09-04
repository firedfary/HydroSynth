"""Audit available seasonal-model precipitation and leak-free blend skill."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import zoom
from scipy.interpolate import RegularGridInterpolator
from sklearn.model_selection import TimeSeriesSplit

from project_paths import ALIGNED_OBSERVATION_FILE, MODEL_DATA_DIR

DATA_ROOT = MODEL_DATA_DIR
OBS_PATH = ALIGNED_OBSERVATION_FILE
START = pd.Timestamp("1994-01-01")
TRAIN_END = pd.Timestamp("2022-12-01")
TEST_START = pd.Timestamp("2023-01-01")
TEST_END = pd.Timestamp("2024-09-01")
EXCLUDED = {
    pd.Timestamp("2011-09-01"),
    pd.Timestamp("2011-10-01"),
    pd.Timestamp("2017-01-01"),
}


def issue_month(path: Path) -> pd.Timestamp:
    match = re.search(r"((?:19|20)\d{4})", path.name)
    if match is None:
        raise ValueError(f"No YYYYMM issue date in {path.name}")
    return pd.Timestamp(f"{match.group(1)[:4]}-{match.group(1)[4:]}-01")


def collect_files(model: str) -> dict[pd.Timestamp, Path]:
    if model == "ECMWF":
        paths = sorted((DATA_ROOT / "MODESv21_ecmwf_seas51").glob("*.nc"))
        return {issue_month(path): path for path in paths}

    if model == "NCEP":
        old = sorted((DATA_ROOT / "MODESv2_ncep_cfs2").glob("*.nc"))
        new = sorted((DATA_ROOT / "MODESv21_ncep_cfs2").glob("*.nc"))
        result = {issue_month(path): path for path in old}
        result.update({issue_month(path): path for path in new})
        return result

    if model == "JMA":
        old = sorted((DATA_ROOT / "MODESv2_jma_cps3").glob("*.nc"))
        new = sorted((DATA_ROOT / "补全").glob("MODESv21_jma_cps3_*.nc"))
        result = {issue_month(path): path for path in old}
        result.update({issue_month(path): path for path in new})
        return result

    if model == "NCC":
        paths = sorted((DATA_ROOT / "MODESv2_ncc_csm11").glob("*.nc"))
        return {issue_month(path): path for path in paths}

    if model == "UKMO":
        # Each delivery directory repeats a same-calendar-month hindcast set.
        # Later snapshots supersede earlier copies of the same issue date.
        paths = sorted(
            (DATA_ROOT / "UKMO_GLOSEA5").glob("*/*.nc"),
            key=lambda path: (path.parent.name, path.name),
        )
        result = {}
        for path in paths:
            result[issue_month(path)] = path
        return result

    if model == "SYSTEM4":
        paths = sorted((DATA_ROOT / "MODES_ECMWF_SYSTEM4").glob("*.nc"))
        return {issue_month(path): path for path in paths}

    raise ValueError(model)


def read_precip(path: Path) -> np.ndarray:
    with xr.open_dataset(path) as dataset:
        if "tp" in dataset:
            var_name = "tp"
            rate_factor = 86400.0 * 1000.0  # m/s to mm/day
        elif "precsfc" in dataset:
            var_name = "precsfc"
            rate_factor = 1.0  # mm/day
        else:
            raise KeyError(f"No precipitation variable in {path}")

        lat_name = "latitude" if "latitude" in dataset else "lat"
        lon_name = "longitude" if "longitude" in dataset else "lon"
        lats = np.asarray(dataset[lat_name].values)
        lons = np.asarray(dataset[lon_name].values)
        coarse_grid = np.median(np.abs(np.diff(lats))) > 1.1
        if coarse_grid:
            lat_idx = np.where((lats >= -3.0) & (lats <= 63.0))[0]
            lon_idx = np.where((lons >= 67.0) & (lons <= 143.0))[0]
        else:
            lat_idx = np.where((lats > 0.0) & (lats <= 60.0))[0]
            lon_idx = np.where((lons >= 70.0) & (lons < 140.0))[0]
        if not np.all(np.diff(lat_idx) == 1) or not np.all(np.diff(lon_idx) == 1):
            raise ValueError(f"Non-contiguous East Asia grid in {path}")

        data = dataset[var_name]
        lead_dims = [dim for dim in data.dims if dim not in (lat_name, lon_name)]
        if len(lead_dims) != 1:
            raise ValueError(f"Unexpected precipitation dimensions in {path}: {data.dims}")
        lead_dim = lead_dims[0]
        values = np.asarray(
            data.isel(
                {
                    lead_dim: slice(0, 6),
                    lat_name: slice(lat_idx[0], lat_idx[-1] + 1),
                    lon_name: slice(lon_idx[0], lon_idx[-1] + 1),
                }
            )
            .transpose(lead_dim, lat_name, lon_name)
            .values,
            dtype=np.float32,
        )
        values *= rate_factor

        regional_lats = lats[lat_idx]
        regional_lons = lons[lon_idx]

    if values.shape[1:] != (60, 70):
        target_lats = np.arange(59.5, -0.5, -1.0)
        target_lons = np.arange(70.5, 140.5, 1.0)
        if regional_lats[0] > regional_lats[-1]:
            regional_lats = regional_lats[::-1]
            values = values[:, ::-1]
        latitude_grid, longitude_grid = np.meshgrid(
            target_lats, target_lons, indexing="ij"
        )
        points = np.column_stack([latitude_grid.ravel(), longitude_grid.ravel()])
        interpolated = []
        for field in values:
            interpolator = RegularGridInterpolator(
                (regional_lats, regional_lons),
                field,
                bounds_error=False,
                fill_value=None,
            )
            interpolated.append(interpolator(points).reshape(60, 70))
        values = np.asarray(interpolated, dtype=np.float32)

    return values


def build_model_fields(model: str):
    raw_by_issue: dict[pd.Timestamp, np.ndarray] = {}
    failures = []

    files = collect_files(model)
    for issue, path in sorted(files.items()):
        if issue < START or issue > TEST_END or issue in EXCLUDED:
            continue
        try:
            raw_by_issue[issue] = read_precip(path)
        except Exception as exc:
            failures.append((path.name, repr(exc)))

    valid_dates = [
        date for date in pd.date_range(START, TEST_END, freq="MS")
        if date not in EXCLUDED
    ]
    valid_date_set = set(valid_dates)
    counts = np.zeros((6, 12), dtype=np.int32)
    test_anom = {}
    for lead in range(6):
        aligned = []
        for target in valid_dates:
            issue = target - pd.DateOffset(months=lead)
            lag_dates = [
                target - pd.DateOffset(months=lead + 1),
                target - pd.DateOffset(months=lead + 2),
                target - pd.DateOffset(months=lead + 3),
                target - pd.DateOffset(months=12),
            ]
            if (
                issue in raw_by_issue
                and lead < raw_by_issue[issue].shape[0]
                and all(d in valid_date_set for d in lag_dates)
            ):
                aligned.append(target)

        train_dates = aligned[:-21]
        sums = np.zeros((12, 60, 70), dtype=np.float64)
        for target in train_dates:
            issue = target - pd.DateOffset(months=lead)
            field = raw_by_issue[issue][lead] * 31.0
            sums[target.month - 1] += field
            counts[lead, target.month - 1] += 1
        climatology = sums / np.maximum(counts[lead, :, None, None], 1)
        for target in aligned[-21:]:
            issue = target - pd.DateOffset(months=lead)
            field = raw_by_issue[issue][lead] * 31.0
            clim = climatology[target.month - 1]
            anomaly = (field - clim) / (clim + 1e-6)
            test_anom[(lead, target)] = zoom(
                anomaly, (2, 2), order=3, grid_mode=True, mode="nearest"
            ).astype(np.float32)
    return test_anom, counts, failures, raw_by_issue


def aligned_dates(raw_by_issue: dict[pd.Timestamp, np.ndarray], lead: int):
    valid_dates = [
        date for date in pd.date_range(START, TEST_END, freq="MS")
        if date not in EXCLUDED
    ]
    valid_date_set = set(valid_dates)
    result = []
    for target in valid_dates:
        issue = target - pd.DateOffset(months=lead)
        lag_dates = [
            target - pd.DateOffset(months=lead + 1),
            target - pd.DateOffset(months=lead + 2),
            target - pd.DateOffset(months=lead + 3),
            target - pd.DateOffset(months=12),
        ]
        if (
            issue in raw_by_issue
            and lead < raw_by_issue[issue].shape[0]
            and all(date in valid_date_set for date in lag_dates)
        ):
            result.append(target)
    return result


def anomalies_for_dates(raw_by_issue, lead, train_dates, eval_dates):
    sums = np.zeros((12, 60, 70), dtype=np.float64)
    counts = np.zeros(12, dtype=np.int32)
    for target in train_dates:
        issue = target - pd.DateOffset(months=lead)
        if issue not in raw_by_issue or lead >= raw_by_issue[issue].shape[0]:
            continue
        sums[target.month - 1] += raw_by_issue[issue][lead] * 31.0
        counts[target.month - 1] += 1
    climatology = sums / np.maximum(counts[:, None, None], 1)
    result = {}
    for target in eval_dates:
        issue = target - pd.DateOffset(months=lead)
        if issue not in raw_by_issue or lead >= raw_by_issue[issue].shape[0]:
            continue
        field = raw_by_issue[issue][lead] * 31.0
        anomaly = (field - climatology[target.month - 1]) / (
            climatology[target.month - 1] + 1e-6
        )
        result[target] = zoom(
            anomaly, (2, 2), order=3, grid_mode=True, mode="nearest"
        ).astype(np.float32)
    return result


def standardize(field: np.ndarray, mask: np.ndarray) -> np.ndarray:
    result = np.zeros_like(field, dtype=np.float32)
    values = field[mask]
    result[mask] = (values - values.mean()) / (values.std() + 1e-8)
    return result


def acc(pred: np.ndarray, obs: np.ndarray, mask: np.ndarray) -> float:
    p = pred[mask] - pred[mask].mean()
    o = obs[mask] - obs[mask].mean()
    denom = np.sqrt(np.sum(p * p) * np.sum(o * o))
    return float(np.sum(p * o) / denom) if denom > 0 else 0.0


def main() -> None:
    all_dates = [
        date
        for date in pd.date_range(START, TEST_END, freq="MS")
        if date not in EXCLUDED
    ]
    observations = np.load(OBS_PATH)
    obs_by_date = dict(zip(all_dates, np.nan_to_num(observations, nan=0.0)))
    mask = ~np.isnan(observations[0])

    model_anom = {}
    model_raw = {}
    for model in ("ECMWF", "NCEP", "JMA"):
        fields, counts, failures, raw_by_issue = build_model_fields(model)
        model_anom[model] = fields
        model_raw[model] = raw_by_issue
        print(
            f"{model}: test fields={len(fields)}, "
            f"train climatology counts/lead min={counts.min(axis=1).tolist()}, "
            f"failures={len(failures)}"
        )
        if failures:
            print("  first failures:", failures[:3])

    test_dates = list(pd.date_range(TEST_START, TEST_END, freq="MS"))
    combinations = {
        "ECMWF": ("ECMWF",),
        "NCEP": ("NCEP",),
        "JMA": ("JMA",),
        "ECMWF+NCEP": ("ECMWF", "NCEP"),
        "ECMWF+NCEP+JMA": ("ECMWF", "NCEP", "JMA"),
    }

    print("\nMean test ACC (common available dates within each row)")
    print("model," + ",".join(f"lead{lead}" for lead in range(6)))
    for label, members in combinations.items():
        scores = []
        ns = []
        for lead in range(6):
            lead_scores = []
            for target in test_dates:
                keys = [(member, (lead, target)) for member in members]
                if not all(key in model_anom[member] for member, key in keys):
                    continue
                member_fields = [
                    standardize(model_anom[member][key], mask)
                    for member, key in keys
                ]
                prediction = np.mean(member_fields, axis=0)
                lead_scores.append(acc(prediction, obs_by_date[target], mask))
            scores.append(np.mean(lead_scores) if lead_scores else np.nan)
            ns.append(len(lead_scores))
        formatted = ",".join(
            f"{score:.6f}(n={n})" if np.isfinite(score) else "nan(n=0)"
            for score, n in zip(scores, ns)
        )
        print(f"{label},{formatted}")

    print("\nLeak-free rolling-CV selection for ECMWF/NCEP blend")
    print("lead,ncep_weight,cv_acc,ec_cv_acc,fold_gains,test_acc,ec_test_acc")
    fold_weights = np.arange(1.0, 6.0) ** 2
    fold_weights /= fold_weights.sum()
    candidate_weights = np.linspace(0.0, 1.0, 11)
    for lead in range(6):
        ec_dates = aligned_dates(model_raw["ECMWF"], lead)
        training_dates = ec_dates[:-21]
        fold_scores = {float(weight): [] for weight in candidate_weights}
        splitter = TimeSeriesSplit(n_splits=5, test_size=21)
        for train_idx, val_idx in splitter.split(training_dates):
            fold_train_dates = [training_dates[index] for index in train_idx]
            fold_val_dates = [training_dates[index] for index in val_idx]
            ec_fields = anomalies_for_dates(
                model_raw["ECMWF"], lead, fold_train_dates, fold_val_dates
            )
            ncep_fields = anomalies_for_dates(
                model_raw["NCEP"], lead, fold_train_dates, fold_val_dates
            )
            common_dates = [
                date for date in fold_val_dates
                if date in ec_fields and date in ncep_fields
            ]
            for weight in candidate_weights:
                sample_scores = []
                for target in common_dates:
                    ec_field = standardize(ec_fields[target], mask)
                    ncep_field = standardize(ncep_fields[target], mask)
                    prediction = (1.0 - weight) * ec_field + weight * ncep_field
                    sample_scores.append(acc(prediction, obs_by_date[target], mask))
                fold_scores[float(weight)].append(float(np.mean(sample_scores)))

        weighted_scores = {
            weight: float(np.average(scores, weights=fold_weights))
            for weight, scores in fold_scores.items()
        }
        best_weight = max(weighted_scores, key=weighted_scores.get)
        paired_gains = np.asarray(fold_scores[best_weight]) - np.asarray(
            fold_scores[0.0]
        )
        effective_folds = 1.0 / np.sum(fold_weights**2)
        weighted_gain = float(np.average(paired_gains, weights=fold_weights))
        weighted_variance = np.sum(
            fold_weights * (paired_gains - weighted_gain) ** 2
        ) / (1.0 - np.sum(fold_weights**2))
        gain_se = float(np.sqrt(weighted_variance / effective_folds))
        stable = np.count_nonzero(paired_gains > 0) >= 4
        if not stable or weighted_gain <= max(0.005, gain_se):
            best_weight = 0.0
        test_scores = []
        ec_test_scores = []
        for target in test_dates:
            ec_field = standardize(model_anom["ECMWF"][(lead, target)], mask)
            ncep_field = standardize(model_anom["NCEP"][(lead, target)], mask)
            prediction = (1.0 - best_weight) * ec_field + best_weight * ncep_field
            test_scores.append(acc(prediction, obs_by_date[target], mask))
            ec_test_scores.append(acc(ec_field, obs_by_date[target], mask))
        print(
            f"{lead},{best_weight:.1f},{weighted_scores[best_weight]:.6f},"
            f"{weighted_scores[0.0]:.6f},"
            f"{'/'.join(f'{gain:+.3f}' for gain in paired_gains)},"
            f"{np.mean(test_scores):.6f},"
            f"{np.mean(ec_test_scores):.6f}"
        )


if __name__ == "__main__":
    main()

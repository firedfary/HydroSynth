import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional


class PentadDataset(Dataset):
    """
    Dataset for Paradigm 1: Pentad (5-day) S2S precipitation forecast.
    """
    def __init__(
        self,
        pentad_precip: np.ndarray,      # (Total_Pentads, Num_Stations)
        macro_features: np.ndarray,     # (Total_Months, Num_Stations, C_macro)
        pentad_dates: pd.DatetimeIndex,
        monthly_dates: pd.DatetimeIndex,
        in_len: int = 6,                # 6 pentads in (30 days)
        out_len: int = 6                # 6 pentads out (30 days)
    ):
        self.precip = torch.tensor(pentad_precip, dtype=torch.float32)
        self.in_len = in_len
        self.out_len = out_len
        self.num_pentads, self.num_nodes = pentad_precip.shape
        
        # Build samples
        self.samples = []
        for t in range(in_len, self.num_pentads - out_len + 1):
            target_date = pentad_dates[t]
            # Find closest corresponding month index
            m_idx = np.argmin(np.abs((monthly_dates - target_date).days))
            self.samples.append((t, m_idx))
            
        self.macro_tensor = torch.tensor(macro_features, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        t, m_idx = self.samples[idx]
        
        # Historical pentads: (in_len, N) -> add log transformed feature -> (in_len, N, 2)
        hist_p = self.precip[t - self.in_len:t]  # (in_len, N)
        hist_log = torch.log1p(torch.clamp(hist_p, min=0.0))
        x = torch.stack([hist_p, hist_log], dim=-1)  # (in_len, N, 2)
        
        y = self.precip[t:t + self.out_len]  # (out_len, N)
        macro = self.macro_tensor[m_idx]      # (N, C_macro)
        
        return {
            "x": x,
            "y": y,
            "macro": macro
        }


class DailyHybridDataset(Dataset):
    """
    Dataset for Paradigm 2: Multi-scale Daily Hybrid ST-GNN with Trend + Residual.
    """
    def __init__(
        self,
        daily_features: np.ndarray,     # (Total_Days, Num_Stations, Num_Features)
        daily_trend: np.ndarray,        # (Total_Days, Num_Stations)
        daily_labels: np.ndarray,       # (Total_Days, Num_Stations)
        macro_features: np.ndarray,     # (Total_Months, Num_Stations, C_macro)
        daily_dates: pd.DatetimeIndex,
        monthly_dates: pd.DatetimeIndex,
        in_len: int = 30,               # 30 days history
        out_len: int = 30               # 30 days forecast
    ):
        self.x_feat = torch.tensor(daily_features, dtype=torch.float32)
        self.trend = torch.tensor(daily_trend, dtype=torch.float32)
        self.y_label = torch.tensor(daily_labels, dtype=torch.float32)
        self.macro_tensor = torch.tensor(macro_features, dtype=torch.float32)
        
        self.in_len = in_len
        self.out_len = out_len
        self.total_days = len(daily_dates)
        
        self.samples = []
        for t in range(in_len, self.total_days - out_len + 1):
            target_date = daily_dates[t]
            m_idx = np.argmin(np.abs((monthly_dates - target_date).days))
            self.samples.append((t, m_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        t, m_idx = self.samples[idx]
        
        x = self.x_feat[t - self.in_len:t]       # (in_len, N, D)
        trend = self.trend[t:t + self.out_len]   # (out_len, N)
        y = self.y_label[t:t + self.out_len]     # (out_len, N)
        macro = self.macro_tensor[m_idx]         # (N, C_macro)
        
        return {
            "x": x,
            "trend": trend,
            "y": y,
            "macro": macro
        }


class HurdleDataset(Dataset):
    """
    Dataset for Paradigm 3: Two-Stage Hurdle and Quantile Extreme ST-GNN.
    """
    def __init__(
        self,
        daily_features: np.ndarray,
        daily_labels: np.ndarray,
        macro_features: np.ndarray,
        daily_dates: pd.DatetimeIndex,
        monthly_dates: pd.DatetimeIndex,
        in_len: int = 30,
        out_len: int = 30,
        rain_threshold: float = 0.1  # mm
    ):
        self.x_feat = torch.tensor(daily_features, dtype=torch.float32)
        self.y_label = torch.tensor(daily_labels, dtype=torch.float32)
        self.macro_tensor = torch.tensor(macro_features, dtype=torch.float32)
        self.rain_threshold = rain_threshold
        
        self.in_len = in_len
        self.out_len = out_len
        self.total_days = len(daily_dates)
        
        self.samples = []
        for t in range(in_len, self.total_days - out_len + 1):
            target_date = daily_dates[t]
            m_idx = np.argmin(np.abs((monthly_dates - target_date).days))
            self.samples.append((t, m_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        t, m_idx = self.samples[idx]
        
        x = self.x_feat[t - self.in_len:t]       # (in_len, N, D)
        y_amt = self.y_label[t:t + self.out_len] # (out_len, N)
        y_occ = (y_amt >= self.rain_threshold).float()  # (out_len, N)
        macro = self.macro_tensor[m_idx]         # (N, C_macro)
        
        return {
            "x": x,
            "y_amt": y_amt,
            "y_occ": y_occ,
            "macro": macro
        }


def create_s2s_dataloaders(
    dataset_type: str,
    data_dict: dict,
    batch_size: int = 16,
    train_years: Tuple[int, int] = (1994, 2018),
    val_years: Tuple[int, int] = (2019, 2021),
    test_years: Tuple[int, int] = (2022, 2024),
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Split time-series data chronologically by years and create train/val/test DataLoaders.
    """
    if dataset_type == "pentad":
        p_precip = data_dict["pentad_precip"]
        p_dates = data_dict["pentad_dates"]
        macro = data_dict["macro_features"]
        m_dates = data_dict["monthly_dates"]
        
        # Split indices based on years
        train_mask = (p_dates.year >= train_years[0]) & (p_dates.year <= train_years[1])
        val_mask = (p_dates.year >= val_years[0]) & (p_dates.year <= val_years[1])
        test_mask = (p_dates.year >= test_years[0]) & (p_dates.year <= test_years[1])
        
        train_ds = PentadDataset(p_precip[train_mask], macro, p_dates[train_mask], m_dates)
        val_ds = PentadDataset(p_precip[val_mask], macro, p_dates[val_mask], m_dates)
        test_ds = PentadDataset(p_precip[test_mask], macro, p_dates[test_mask], m_dates)

    elif dataset_type == "daily_hybrid":
        d_feat = data_dict["daily_features"]
        d_trend = data_dict["daily_trend"]
        d_precip = data_dict["daily_precip"]
        d_dates = data_dict["daily_dates"]
        macro = data_dict["macro_features"]
        m_dates = data_dict["monthly_dates"]
        
        train_mask = (d_dates.year >= train_years[0]) & (d_dates.year <= train_years[1])
        val_mask = (d_dates.year >= val_years[0]) & (d_dates.year <= val_years[1])
        test_mask = (d_dates.year >= test_years[0]) & (d_dates.year <= test_years[1])
        
        train_ds = DailyHybridDataset(d_feat[train_mask], d_trend[train_mask], d_precip[train_mask], macro, d_dates[train_mask], m_dates)
        val_ds = DailyHybridDataset(d_feat[val_mask], d_trend[val_mask], d_precip[val_mask], macro, d_dates[val_mask], m_dates)
        test_ds = DailyHybridDataset(d_feat[test_mask], d_trend[test_mask], d_precip[test_mask], macro, d_dates[test_mask], m_dates)

    elif dataset_type == "hurdle":
        d_feat = data_dict["daily_features"]
        d_precip = data_dict["daily_precip"]
        d_dates = data_dict["daily_dates"]
        macro = data_dict["macro_features"]
        m_dates = data_dict["monthly_dates"]
        
        train_mask = (d_dates.year >= train_years[0]) & (d_dates.year <= train_years[1])
        val_mask = (d_dates.year >= val_years[0]) & (d_dates.year <= val_years[1])
        test_mask = (d_dates.year >= test_years[0]) & (d_dates.year <= test_years[1])
        
        train_ds = HurdleDataset(d_feat[train_mask], d_precip[train_mask], macro, d_dates[train_mask], m_dates)
        val_ds = HurdleDataset(d_feat[val_mask], d_precip[val_mask], macro, d_dates[val_mask], m_dates)
        test_ds = HurdleDataset(d_feat[test_mask], d_precip[test_mask], macro, d_dates[test_mask], m_dates)
        
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader

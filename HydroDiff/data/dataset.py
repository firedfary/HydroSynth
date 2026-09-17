"""Paired Meteorological Downscaling Dataset, Spatial Transforms, and Temporal Partitioning."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class FullDomainSpatialTransform:
    """
    Symmetric padding and unpadding transform for meteorological grids.

    Adapts non-power-of-two domain (e.g., 120 x 140) to divisibility-friendly
    dimensions (e.g., 128 x 144) for deep multi-stage UNet architectures.
    """

    def __init__(
        self,
        orig_shape: Tuple[int, int] = (120, 140),
        target_shape: Tuple[int, int] = (128, 144),
    ):
        self.orig_h, self.orig_w = orig_shape
        self.target_h, self.target_w = target_shape

        diff_h = self.target_h - self.orig_h
        diff_w = self.target_w - self.orig_w

        if diff_h < 0 or diff_w < 0:
            raise ValueError(f"Target shape {target_shape} must be >= original shape {orig_shape}")

        self.pad_top = diff_h // 2
        self.pad_bottom = diff_h - self.pad_top
        self.pad_left = diff_w // 2
        self.pad_right = diff_w - self.pad_left

    def pad_array(self, arr: np.ndarray, fill_value: float = 0.0) -> np.ndarray:
        """Pad a 2D, 3D, or 4D numpy array along last two dimensions (H, W)."""
        ndim = arr.ndim
        pad_width = [(0, 0)] * (ndim - 2) + [
            (self.pad_top, self.pad_bottom),
            (self.pad_left, self.pad_right),
        ]
        return np.pad(arr, pad_width, mode="constant", constant_values=fill_value)

    def unpad_array(self, arr: np.ndarray) -> np.ndarray:
        """Crop padded numpy array back to original (orig_h, orig_w)."""
        h_end = self.pad_top + self.orig_h
        w_end = self.pad_left + self.orig_w
        return arr[..., self.pad_top : h_end, self.pad_left : w_end]

    def pad_tensor(self, tensor: torch.Tensor, fill_value: float = 0.0) -> torch.Tensor:
        """Pad a PyTorch tensor along last two dimensions (F.pad takes left, right, top, bottom)."""
        return F.pad(
            tensor,
            (self.pad_left, self.pad_right, self.pad_top, self.pad_bottom),
            mode="constant",
            value=fill_value,
        )

    def unpad_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Crop padded PyTorch tensor back to original (orig_h, orig_w)."""
        h_end = self.pad_top + self.orig_h
        w_end = self.pad_left + self.orig_w
        return tensor[..., self.pad_top : h_end, self.pad_left : w_end]


def compute_normalizer_stats(hr_data: np.ndarray) -> Tuple[float, float]:
    """
    Compute mean and std of target high-resolution observations strictly over valid finite pixels.
    """
    valid_mask = np.isfinite(hr_data)
    valid_values = hr_data[valid_mask]
    if valid_values.size == 0:
        return 0.0, 1.0

    mean = float(np.mean(valid_values))
    std = float(np.std(valid_values))
    if std < 1e-6:
        std = 1.0
    return mean, std


class PairedDownscalingDataset(Dataset):
    """
    Paired dataset for conditional diffusion downscaling:
    - Target: High-resolution precipitation anomaly (1 channel)
    - Condition: Multi-channel low-resolution driving variables (C_cond channels)
    - Mask: Valid observation mask (1 for valid station/land grid, 0 for missing/ocean/padding)
    """

    def __init__(
        self,
        hr_data: np.ndarray | torch.Tensor,
        lr_data: np.ndarray | torch.Tensor,
        mean: Optional[float] = None,
        std: Optional[float] = None,
        spatial_crop: Optional[Tuple[int, int, int, int]] = None,  # (y1, y2, x1, x2)
        spatial_transform: Optional[FullDomainSpatialTransform] = None,
    ):
        super().__init__()

        if isinstance(hr_data, torch.Tensor):
            hr_data = hr_data.cpu().numpy()
        if isinstance(lr_data, torch.Tensor):
            lr_data = lr_data.cpu().numpy()

        if hr_data.ndim == 3:
            hr_data = np.expand_dims(hr_data, axis=1)  # [N, 1, H, W]

        # Spatial crop if specified
        if spatial_crop is not None:
            y1, y2, x1, x2 = spatial_crop
            hr_data = hr_data[:, :, y1:y2, x1:x2]
            lr_data = lr_data[:, :, y1:y2, x1:x2]

        self.mask = np.isfinite(hr_data)  # [N, 1, H, W]

        # Normalization parameters (must be computed on training set to prevent leakage)
        if mean is None or std is None:
            self.mean, self.std = compute_normalizer_stats(hr_data)
        else:
            self.mean, self.std = mean, std

        # Standardize HR target on valid mask
        hr_normalized = np.where(self.mask, (hr_data - self.mean) / self.std, 0.0).astype(np.float32)

        # LR condition: clean NaNs if any
        lr_clean = np.nan_to_num(lr_data, nan=0.0).astype(np.float32)

        mask_clean = self.mask.astype(np.float32)

        # Apply full-domain spatial padding if transform is supplied
        self.spatial_transform = spatial_transform
        if self.spatial_transform is not None:
            hr_normalized = self.spatial_transform.pad_array(hr_normalized, fill_value=0.0)
            lr_clean = self.spatial_transform.pad_array(lr_clean, fill_value=0.0)
            mask_clean = self.spatial_transform.pad_array(mask_clean, fill_value=0.0)

        self.hr_tensor = torch.from_numpy(hr_normalized)
        self.lr_tensor = torch.from_numpy(lr_clean)
        self.mask_tensor = torch.from_numpy(mask_clean)

    def __len__(self) -> int:
        return self.hr_tensor.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            target: [1, H, W]
            condition: [C_cond, H, W]
            mask: [1, H, W]
        """
        return self.hr_tensor[idx], self.lr_tensor[idx], self.mask_tensor[idx]

    def denormalize(self, target_norm: torch.Tensor | np.ndarray) -> np.ndarray:
        """
        Convert normalized model output back to physical precipitation anomaly units.
        Unpads automatically if spatial_transform was used.
        """
        if isinstance(target_norm, torch.Tensor):
            arr = target_norm.detach().cpu().numpy()
        else:
            arr = target_norm.copy()

        if self.spatial_transform is not None:
            arr = self.spatial_transform.unpad_array(arr)

        return arr * self.std + self.mean

    def get_stats(self) -> dict:
        return {
            "mean": self.mean,
            "std": self.std,
            "num_samples": len(self),
            "hr_shape": list(self.hr_tensor.shape),
            "lr_shape": list(self.lr_tensor.shape),
            "valid_ratio": float(self.mask_tensor.mean().item()),
            "padded": self.spatial_transform is not None,
        }


def create_spatiotemporal_splits(
    hr_file: Union[str, Path],
    lr_file: Union[str, Path],
    split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    pad_to: Optional[Tuple[int, int]] = (128, 144),
    spatial_crop: Optional[Tuple[int, int, int, int]] = None,
) -> Dict[str, Union[PairedDownscalingDataset, dict, FullDomainSpatialTransform]]:
    """
    Construct temporal Train / Val / Test partitions to strictly eliminate cross-year leakage.
    
    Normalization statistics (mean, std) are derived EXCLUSIVELY from the training split
    and then shared with validation and test partitions.
    """
    hr_path = Path(hr_file)
    lr_path = Path(lr_file)

    if not hr_path.exists():
        raise FileNotFoundError(f"High-resolution observation file not found: {hr_path}")
    if not lr_path.exists():
        raise FileNotFoundError(f"Low-resolution condition file not found: {lr_path}")

    # Load arrays via memory mapping
    hr_all = np.load(hr_path, mmap_mode="r")
    lr_all = np.load(lr_path, mmap_mode="r")

    n_total = len(hr_all)
    assert len(lr_all) == n_total, f"Length mismatch: hr={n_total}, lr={len(lr_all)}"

    train_ratio, val_ratio, test_ratio = split_ratios
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val

    train_slice = slice(0, n_train)
    val_slice = slice(n_train, n_train + n_val)
    test_slice = slice(n_train + n_val, n_total)

    # Compute training normalizer statistics STRICTLY on the training partition
    train_hr_subset = hr_all[train_slice]
    if spatial_crop is not None:
        y1, y2, x1, x2 = spatial_crop
        train_hr_subset = train_hr_subset[:, y1:y2, x1:x2]

    train_mean, train_std = compute_normalizer_stats(train_hr_subset)

    # Setup spatial transform if padding is requested
    orig_shape = (120, 140) if spatial_crop is None else (spatial_crop[1] - spatial_crop[0], spatial_crop[3] - spatial_crop[2])
    transform = FullDomainSpatialTransform(orig_shape=orig_shape, target_shape=pad_to) if pad_to is not None else None

    # Instantiate datasets sharing identical training-derived normalizer
    train_ds = PairedDownscalingDataset(
        hr_data=hr_all[train_slice],
        lr_data=lr_all[train_slice],
        mean=train_mean,
        std=train_std,
        spatial_crop=spatial_crop,
        spatial_transform=transform,
    )

    val_ds = PairedDownscalingDataset(
        hr_data=hr_all[val_slice],
        lr_data=lr_all[val_slice],
        mean=train_mean,
        std=train_std,
        spatial_crop=spatial_crop,
        spatial_transform=transform,
    )

    test_ds = PairedDownscalingDataset(
        hr_data=hr_all[test_slice],
        lr_data=lr_all[test_slice],
        mean=train_mean,
        std=train_std,
        spatial_crop=spatial_crop,
        spatial_transform=transform,
    )

    stats = {
        "train_mean": train_mean,
        "train_std": train_std,
        "n_total": n_total,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "train_indices": [0, n_train - 1],
        "val_indices": [n_train, n_train + n_val - 1],
        "test_indices": [n_train + n_val, n_total - 1],
        "split_ratios": list(split_ratios),
        "orig_shape": list(orig_shape),
        "target_shape": list(pad_to) if pad_to is not None else None,
    }

    return {
        "train": train_ds,
        "val": val_ds,
        "test": test_ds,
        "stats": stats,
        "transform": transform,
    }

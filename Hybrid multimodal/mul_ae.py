import os
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
import time
import threading
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
# from unetlite import UNetLite
import fucs  # 你原来的工具函数库（cal_acc 等）

# -----------------------
# ======= 模型部分 =======
# -----------------------
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.GroupNorm(num_groups=8, num_channels=out_channels))
    
    def forward(self, x):
        identity = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return F.relu(out)

class ClimateEncoder(nn.Module):
    def __init__(self, input_channels=10, dropout=0.5):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=dropout)
        )
        self.res_blocks = nn.Sequential(
            ResBlock(64, 64),
            ResBlock(64, 64)
        )
        self.upsample = nn.Sequential(
            nn.Upsample(size=(60, 70), mode='bilinear', align_corners=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=dropout),
            nn.Upsample(size=(120, 140), mode='bilinear', align_corners=True)
        )
    
    def forward(self, x):
        x = self.initial(x)
        x = self.down1(x)
        x = self.res_blocks(x)
        return self.upsample(x)

class SSTEncoder(nn.Module):
    def __init__(self, input_channels=3, dropout=0.5):
        super().__init__()
        self.upsample_init = nn.Upsample(size=(100, 180), mode='bilinear', align_corners=True)
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=dropout)
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=dropout)
        )
        self.res_block = ResBlock(64, 64)
        self.upsample = nn.Sequential(
            nn.Upsample(size=(60, 70), mode='bilinear', align_corners=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=dropout),
            nn.Upsample(size=(120, 140), mode='bilinear', align_corners=True))
    
    def forward(self, x):
        x = self.upsample_init(x)
        x = self.conv1(x)
        x = self.down1(x)
        x = self.res_block(x)
        return self.upsample(x)

class FeatureFusion(nn.Module):
    def __init__(self, output_channels=48):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.Conv2d(64, output_channels, kernel_size=1)
        )
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(128, 8, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(8, 128, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, climate_feats, sst_feats):
        combined = torch.cat([climate_feats, sst_feats], dim=1)
        attn_weights = self.attention(combined)
        attn_applied = attn_weights * combined
        return self.fusion(attn_applied)

class DownsampledAttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=4, reduction=4):
        """
        参数:
          channels: 输入通道数
          num_heads: 注意力头数
          reduction: 下采样缩放因子 (e.g., 4 表示缩小到 1/4 尺寸做注意力)
        """
        super().__init__()
        self.reduction = reduction
        self.norm = nn.GroupNorm(8, channels)

        # QKV 投影
        self.qkv = nn.Linear(channels, channels * 3)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.proj = nn.Linear(channels, channels)

    def forward(self, x):
        """
        输入: x [B, C, H, W]
        输出: out [B, C, H, W]
        """
        B, C, H, W = x.shape
        residual = x

        # ---- 下采样 ----
        x_ds = F.adaptive_avg_pool2d(x, (H // self.reduction, W // self.reduction))  # [B,C,H',W']
        Hs, Ws = x_ds.shape[2], x_ds.shape[3]

        # ---- 拉直为序列 ----
        x_seq = x_ds.flatten(2).transpose(1, 2)   # [B, H'*W', C]

        # ---- QKV & Attention ----
        qkv = self.qkv(x_seq)                     # [B, L, 3C]
        q, k, v = qkv.chunk(3, dim=-1)            # 三份 [B, L, C]
        out, _ = self.attn(q, k, v)               # [B, L, C]
        out = self.proj(out)                      # [B, L, C]

        # ---- reshape 回 feature map ----
        out_map = out.transpose(1, 2).reshape(B, C, Hs, Ws)

        # ---- 上采样回原始尺寸 ----
        out_up = F.interpolate(out_map, size=(H, W), mode="bilinear", align_corners=True)

        # ---- 残差连接 ----
        out_final = self.norm(residual + out_up)
        return out_final


# ---- GlobalContextBlock: 低成本引入全局信息到空间特征 ----
class GlobalContextBlock(nn.Module):
    def __init__(self, channels, hidden_ratio=4, spatial_bias=True):
        """
        channels: 输入通道数
        hidden_ratio: MLP 缩放倍数（通常 2-8）
        spatial_bias: 是否产生空间偏置 map
        """
        super().__init__()
        hidden = max(8, channels // hidden_ratio)
        # Global pooling -> MLP
        self.pool = nn.AdaptiveAvgPool2d(1)   # [B,C,1,1]
        self.mlp = nn.Sequential(
            nn.Flatten(),                     # [B, C]
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels * (1 + int(spatial_bias)))  # produce channel_scale and optionally spatial params
        )
        self.spatial_bias = spatial_bias
        # small conv to expand spatial bias if needed
        if spatial_bias:
            # 1x1 conv to map a small vector to spatial map if you prefer, but we'll broadcast
            # We'll use broadcasting: mlp outputs C (scale) + C (bias scalar that will be broadcast spatially)
            pass
        # normalization for stability
        self.gn = nn.GroupNorm(8, channels) if channels >= 8 else nn.Identity()

    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        pooled = self.pool(x)          # [B, C, 1, 1]
        mlp_out = self.mlp(pooled.view(B, C))  # [B, C*(1 + sb)]
        if self.spatial_bias:
            # split into scale and bias vectors, both length C
            scale_vec = mlp_out[:, :C].unsqueeze(-1).unsqueeze(-1)   # [B,C,1,1]
            bias_vec  = mlp_out[:, C:].unsqueeze(-1).unsqueeze(-1)   # [B,C,1,1]
            # apply
            out = x * (1.0 + torch.tanh(scale_vec)) + bias_vec  # tanh guards scale magnitude
        else:
            scale_vec = mlp_out.view(B, C, 1, 1)
            out = x * (1.0 + torch.tanh(scale_vec))
        out = self.gn(out)
        # residual connection
        return x + out



# ---- 修改 MultiModalEncoder，使用 GlobalContextBlock ----
class MultiModalEncoder(nn.Module):
    def __init__(self, climate_channels=10, sst_channels=3, latent_channels=48, use_global_ctx=True, dropout=0.5):
        super().__init__()
        self.climate_encoder = ClimateEncoder(climate_channels, dropout)
        self.sst_encoder = SSTEncoder(sst_channels, dropout)
        self.fusion = FeatureFusion(latent_channels)
        self.use_global_ctx = use_global_ctx
        if use_global_ctx:
            self.attn_block = GlobalContextBlock(latent_channels)


    def forward(self, climate_data, sst_data):
        climate_feats = self.climate_encoder(climate_data)   # [B, Cc, H, W]
        sst_feats = self.sst_encoder(sst_data)               # [B, Cs, H, W]
        fused = self.fusion(climate_feats, sst_feats)        # [B, latent_channels, H, W]
        # print("fused mean/std:", fused.mean().item(), fused.std().item())
        if self.use_global_ctx:
            fused = self.attn_block(fused)
            # print("after attn_block mean/std:", fused.mean().item(), fused.std().item())
        return fused



class PrecipitationUNet(nn.Module):
    def __init__(self, in_channels, out_channels=1, dropout=0.2):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=dropout)
        )
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=dropout)
        )
        self.pool2 = nn.MaxPool2d(2)
        self.mid = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=dropout)
        )
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=dropout)
        )
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=dropout)
        )
        # replace output sequential
        self.output = nn.Conv2d(32, out_channels, kernel_size=1)

        self.attn = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        enc1 = self.enc1(x)
        x = self.pool1(enc1)
        enc2 = self.enc2(x)
        x = self.pool2(enc2)
        x = self.mid(x)
        x = self.up2(x)
        attn_weights = self.attn(enc2)
        enc2 = enc2 * attn_weights
        x = torch.cat([x, enc2], dim=1)
        x = self.dec2(x)
        x = self.up1(x)
        x = torch.cat([x, enc1], dim=1)
        x = self.dec1(x)
        return self.output(x)

class ClimatePrecipModel(nn.Module):
    def __init__(self, climate_channels=10, sst_channels=3, latent_channels=48):
        super().__init__()
        self.encoder = MultiModalEncoder(climate_channels, sst_channels, latent_channels, dropout=0.6)
        self.unet = PrecipitationUNet(latent_channels, dropout=0.6)
        self.climate_encoder = ClimateEncoder(climate_channels, dropout=0.6)
        self.sst_encoder = SSTEncoder(sst_channels, dropout=0.6)
    def forward(self, climate_data, sst_data):
        if climate_data is None:
            latent = self.sst_encoder(sst_data)
        elif sst_data is None:
            latent = self.climate_encoder(climate_data)
        elif climate_data is not None and sst_data is not None:
            latent = self.encoder(climate_data, sst_data)
        else:
            raise ValueError("climate_data and sst_data cannot be both None")
        return self.unet(latent)

# -----------------------
# ===== Dataset & Transform
# -----------------------
class SaltAndPepperNoise:
    """
    改为添加椒盐噪声的数据增强。
    支持输入为 numpy.ndarray 或 torch.Tensor，返回同类型的数据。
    参数:
      prob: 每个像素被替换为椒盐噪声的概率（每次调用独立采样）
      salt_vs_pepper: 在被替换的像素中，变为“盐"(max) 的比例（其余为“椒"(min)）
    """
    def __init__(self, prob=0.02, salt_vs_pepper=0.5):
        # 使用传入的 prob 参数（之前误写为常量）
        self.prob = float(prob)
        self.salt_vs_pepper = float(salt_vs_pepper)

    def __call__(self, data):
        # numpy 输入处理
        if isinstance(data, np.ndarray):
            arr = data.copy()
            if arr.ndim < 2:
                return arr
            H, W = arr.shape[-2], arr.shape[-1]
            mask = np.random.rand(H, W) < self.prob
            salt_mask = mask & (np.random.rand(H, W) < self.salt_vs_pepper)
            pepper_mask = mask & (~salt_mask)
            # 计算替换值（对整个样本使用整体 max/min）
            maxv = np.nanmax(arr)
            minv = np.nanmin(arr)
            # 若通道在第一个轴, 使用广播
            if arr.ndim == 3:
                arr[:, salt_mask] = maxv
                arr[:, pepper_mask] = minv
            else:
                arr[salt_mask] = maxv
                arr[pepper_mask] = minv
            return arr

        # torch.Tensor 输入处理
        if isinstance(data, torch.Tensor):
            x = data.clone()
            if x.ndim < 2:
                return x
            device = x.device
            H, W = x.shape[-2], x.shape[-1]
            # 在 CPU 上生成随机 mask 更稳健（防止 GPU 内存占用），然后搬到 device
            rnd_mask = torch.from_numpy((np.random.rand(H, W) < self.prob).astype(np.bool_)).to(device)
            rnd_salt = torch.from_numpy((np.random.rand(H, W) < self.salt_vs_pepper).astype(np.bool_)).to(device)
            salt_mask = rnd_mask & rnd_salt
            pepper_mask = rnd_mask & (~rnd_salt)
            # 计算替换值（使用当前张量的 max/min）
            try:
                maxv = x.nanmean() if torch.isnan(x).all() else x[~torch.isnan(x)].max()
            except Exception:
                maxv = x.max()
            try:
                mask = ~torch.isnan(x)
                minv = x[mask].min()
            except Exception:
                minv = x.min()
            # expand masks to channel dim if needed
            if x.ndim == 3:
                salt_mask_e = salt_mask.unsqueeze(0).expand(x.shape[0], -1, -1)
                pepper_mask_e = pepper_mask.unsqueeze(0).expand(x.shape[0], -1, -1)
            else:
                salt_mask_e = salt_mask
                pepper_mask_e = pepper_mask
            x[salt_mask_e] = maxv
            x[pepper_mask_e] = minv
            return x

        # 其它类型直接返回
        return data

class GaussianNoise:
    """
    在样本上添加高斯噪声的 transform（支持 numpy.ndarray 与 torch.Tensor）。
    参数:
      std: 噪声标准差（相对于标准化数据，通常小于1）
      prob: 每次调用应用噪声的概率（0-1）
      per_channel: 若为 True，则每个通道独立采样噪声（保持 shape）
      clip: 若为 True，结果会被限制在原始非 NaN 值的 min/max 范围内
    注意: 不会在 NaN 位置注入噪声（保持 NaN 不变）。
    """
    def __init__(self, std=0.05, prob=1.0, per_channel=False, clip=False):
        self.std = float(std)
        self.prob = float(prob)
        self.per_channel = bool(per_channel)
        self.clip = bool(clip)

    def __call__(self, data):
        # numpy 版本
        if isinstance(data, np.ndarray):
            if np.random.rand() > self.prob:
                return data
            arr = data.copy()
            if arr.ndim < 2:
                return arr
            mask_nan = np.isnan(arr)
            # 生成噪声，若 per_channel 且有通道维则按通道生成
            if self.per_channel and arr.ndim >= 3:
                noise = np.random.normal(loc=0.0, scale=self.std, size=arr.shape)
            else:
                noise = np.random.normal(loc=0.0, scale=self.std, size=arr.shape)
            noise[mask_nan] = 0.0
            arr = arr + noise
            if self.clip:
                # 使用原始非 NaN 的 min/max 做裁剪
                valid = ~mask_nan
                if valid.any():
                    vmin = np.nanmin(data)
                    vmax = np.nanmax(data)
                    arr = np.clip(arr, vmin, vmax)
            return arr

        # torch 版本
        if isinstance(data, torch.Tensor):
            if np.random.rand() > self.prob:
                return data
            x = data.clone()
            if x.ndim < 2:
                return x
            device = x.device
            # 先在 CPU 上生成噪声再搬到 device 更稳健
            noise_np = np.random.normal(loc=0.0, scale=self.std, size=tuple(x.shape))
            noise = torch.from_numpy(noise_np).to(device).type_as(x)
            mask_nan = torch.isnan(x)
            noise = noise.masked_fill(mask_nan, 0.0)
            x = x + noise
            if self.clip:
                # 使用原始非 NaN 的 min/max 做裁剪（在 CPU 上计算）
                data_np = data.cpu().numpy()
                valid = ~np.isnan(data_np)
                if valid.any():
                    vmin = float(np.nanmin(data_np))
                    vmax = float(np.nanmax(data_np))
                    x = torch.clamp(x, vmin, vmax)
            return x
        # 其它类型直接返回
        return data


class ClimateDataset(Dataset):
    def __init__(self, climate_path, sst_path, precip_path, transform=None):
        # ---- Load with mmap to avoid full copy first ----
        climate_raw = np.load(climate_path, mmap_mode="r")    # [time, vars, H, W]
        sst_raw = np.load(sst_path, mmap_mode="r")            # [time, months, H, W]
        precip_raw = np.load(precip_path, mmap_mode="r")      # [time, H, W]

        # ---- Standardize climate (per-variable channel) ----
        climate_mean = np.nanmean(climate_raw, axis=(0,2,3), keepdims=True)
        climate_std  = np.nanstd(climate_raw, axis=(0,2,3), keepdims=True) + 1e-6
        climate_stdzd = (climate_raw - climate_mean) / climate_std
        # print("Standardized climate:", climate_stdzd.shape, climate_stdzd.dtype, climate_stdzd.nbytes/1e9, "GB")


        # ---- Standardize sst (per-month channel) ----
        sst_filled = np.nan_to_num(sst_raw, nan=0.0).astype(np.float64)
        T, M, H, W = sst_filled.shape

        sst_mean = np.zeros((1, M, 1, 1), dtype=np.float64)
        sst_std  = np.zeros((1, M, 1, 1), dtype=np.float64)

        for m in range(M):
            # print(f"Processing month {m}/{M} ...", flush=True)
            channel = sst_filled[:, m, :, :]  # [T,H,W]

            # --- 检查异常值 ---
            if not np.isfinite(channel).all():
                # print(f"Warning: channel {m} contains NaN/Inf")
                channel = np.nan_to_num(channel, nan=0.0, posinf=0.0, neginf=0.0)

            # --- 手写 mean/std 避免 MKL bug ---
            count = channel.size
            mean_val = channel.sum() / count
            var_val = ((channel - mean_val) ** 2).sum() / count
            std_val = np.sqrt(var_val) + 1e-6

            sst_mean[0, m, 0, 0] = mean_val
            sst_std[0, m, 0, 0]  = std_val

            # print(f"month {m}: mean={mean_val:.4f}, std={std_val:.4f}", flush=True)

        # 标准化
        sst_stdzd = (sst_filled - sst_mean) / sst_std
        # print("Standardized sst:", sst_stdzd.shape, sst_stdzd.dtype, sst_stdzd.nbytes/1e9, "GB")




        # ---- Standardize precip (global mean/std) ----
        precip_mean = np.nanmean(precip_raw)
        precip_std  = np.nanstd(precip_raw) + 1e-6
        precip_stdzd = (precip_raw - precip_mean) / precip_std
        precip_stdzd = np.clip(precip_stdzd, -3.0, 3.0)

        # ---- Convert to torch tensors (float32) ----
        self.climate_arr = torch.from_numpy(climate_stdzd.astype(np.float32))   # [T, vars, H, W]
        self.sst_arr     = torch.from_numpy(sst_stdzd.astype(np.float32))       # [T, months, H, W]
        self.precip_arr  = torch.from_numpy(precip_stdzd.astype(np.float32))    # [T, H, W]

        # ---- Mask from original precip NaN ----
        self.data_mask = torch.from_numpy(np.isnan(precip_raw)).unsqueeze(1)    # [T,1,H,W]
        # print("valid_pixels per sample:", (~self.data_mask).sum(dim=[1,2,3]).cpu().numpy())

        self.transform = transform

    def __len__(self):
        return self.precip_arr.shape[0]

    def __getitem__(self, idx):
        climate_t = self.climate_arr[idx]
        sst_t     = self.sst_arr[idx]
        precip_t  = self.precip_arr[idx].reshape(1, *self.precip_arr.shape[1:])
        mask_t    = self.data_mask[idx]
        if self.transform:
            climate_t = self.transform(climate_t)
            sst_t     = self.transform(sst_t)
        return climate_t, torch.zeros_like(sst_t), precip_t, mask_t

# 将 AugmentedSubset 移到模块顶层，避免 Windows multiprocess 无法 pickle 局部类的问题
class AugmentedSubset(Dataset):
    """
    对已有 Dataset 的 subset 进行包装，在 __getitem__ 时对 climate/sst 应用 transform（训练增强）。
    定义在模块顶层以便 DataLoader 的 worker 可序列化（Windows spawn 模式）。
    """
    def __init__(self, base_ds, indices, transform=None):
        super().__init__()
        self.base = base_ds
        self.indices = list(indices)
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        climate_t, sst_t, precip_t, mask_t = self.base[idx]
        if self.transform is not None:
            climate_t = self.transform(climate_t)
            sst_t = self.transform(sst_t)
        return climate_t, sst_t, precip_t, mask_t


loss_fn = torch.nn.MSELoss()
def masked_smooth_l1(pred, target):
    mask = ~torch.isnan(target)
    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    return F.smooth_l1_loss(pred[mask], target[mask], reduction='mean')

def variance_loss(pred, target):
    mask = ~torch.isnan(target)
    p = pred[mask]; t = target[mask]
    return ((p.std() - t.std()) ** 2)


def combined_loss(pred, target, alpha=1.0, beta=0.1, gamma=0.01):
    # alpha * SmoothL1 + beta * (1 - Pearson)
    mask = ~torch.isnan(target)
    l1 = masked_smooth_l1(pred, target) * alpha
    if mask.sum() < 2:
        return l1
    p = pred[mask]
    t = target[mask]
    p = p - p.mean()
    t = t - t.mean()
    denom = (p.std(unbiased=False) * t.std(unbiased=False) + 1e-8)
    rho = (p * t).mean() / denom
    return l1 + beta * (1.0 - rho) + gamma * variance_loss(pred, target)
# -----------------------
# ====== 训练/验证函数 ======
# -----------------------
def save_checkpoint_async(ckpt, path):
    def _save_fn(ckpt_copy, path_inner):
        try:
            torch.save(ckpt_copy, path_inner)
        except Exception as e:
            print("Async save failed:", e)

    ckpt_copy = {}
    for k,v in ckpt.items():
        if isinstance(v, dict):
            sub = {}
            for sk, sv in v.items():
                try:
                    sub[sk] = sv.cpu().clone()
                except Exception:
                    sub[sk] = v[sk]
            ckpt_copy[k] = sub
        else:
            try:
                ckpt_copy[k] = v.cpu().clone()
            except Exception:
                ckpt_copy[k] = v
    thread = threading.Thread(target=_save_fn, args=(ckpt_copy, path))
    thread.daemon = True
    thread.start()

def train_model(model, dataloader, optimizer, criterion, device, epoch, writer=None, acc_frequency=10):
    model.train()
    running_loss = []
    running_acc = []
    runnint_norm = []
    running_masked_outputs = []
    running_masked_precip = []

    # total_samples = 0
    progress_bar = tqdm(enumerate(dataloader), desc=f'Epoch {epoch+1}', total=len(dataloader))
    for batch_idx, (climate, sst, precip, mask) in progress_bar:
        climate = climate.to(device, non_blocking=True)
        sst = sst.to(device, non_blocking=True)
        precip = precip.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        # print(climate.shape, sst.shape, precip.shape, mask.shape)


        outputs = model(climate, sst)  # [B,1,H,W]
        # print(outputs.nanmean())
        masked_outputs = outputs[~mask]
        masked_precip = precip[~mask]
        # print(masked_outputs.mean(), masked_precip.mean(), masked_outputs.std(), masked_precip.std())

        # per-sample losses (可能包含 NaN 表示该样本无有效像素)
        loss = combined_loss(masked_outputs, masked_precip, alpha=1.0, beta=0.5, gamma=0.01)
        optimizer.zero_grad()
        loss.backward()
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        runnint_norm.append(total_norm)

        # before = model.encoder.fusion.fusion[0].weight.detach().cpu().clone() 
        optimizer.step()
        # after = model.encoder.fusion.fusion[0].weight.detach().cpu()
        # print("param change norm:", (after - before).norm().item())
        outputs[mask] = float('nan')
        Acc = fucs.cal_acc(outputs[:,0,:,:]*100, precip[:,0,:,:]*100).mean()
        running_loss.append(loss.item())
        running_acc.append(Acc.item())
        running_masked_outputs.append(masked_outputs)
        running_masked_precip.append(masked_precip)


        # running_loss += float(batch_loss.item()) * n_valid
        # if not math.isnan(batch_acc):
        #     running_acc += batch_acc * n_valid
        # total_samples += n_valid

        progress_bar.set_postfix(loss=loss.item(), acc=Acc.item())

    epoch_loss = np.nanmean(running_loss)
    epoch_acc = np.nanmean(running_acc)
    if writer is not None:
        writer.add_scalar('Loss/train', epoch_loss, epoch)
        writer.add_scalar('Acc/train', epoch_acc, epoch)
        writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar('Grad Norm/train', np.mean(runnint_norm), epoch)
        writer.add_scalar('Std/train_outputs', torch.cat(running_masked_outputs).std(), epoch)
        writer.add_scalar('Std/train_precip', torch.cat(running_masked_precip).std(), epoch)
    return epoch_loss, epoch_acc

def validate_model(model, dataloader, criterion, device, epoch, writer=None):
    model.eval()
    running_loss = []
    running_acc = []
    running_masked_outputs = []
    running_masked_precip = []

    # total_samples = 0
    with torch.no_grad():
        for climate, sst, precip, mask in dataloader:
            climate = climate.to(device, non_blocking=True)
            sst = sst.to(device, non_blocking=True)
            precip = precip.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            outputs = model(climate, sst)

            masked_outputs = outputs[~mask]
            masked_precip = precip[~mask]
            # print(masked_outputs.mean(), masked_precip.mean(), masked_outputs.std(), masked_precip.std())

            # per-sample losses (可能包含 NaN 表示该样本无有效像素)
            loss = combined_loss(masked_outputs, masked_precip, alpha=1.0, beta=0.5, gamma=0.01)

            Acc = fucs.cal_acc(outputs[:,0,:,:]*100, precip[:,0,:,:]*100).mean()
            running_loss.append(loss.item())
            running_acc.append(Acc.item())
            running_masked_outputs.append(masked_outputs)
            running_masked_precip.append(masked_precip)


    epoch_loss = np.nanmean(running_loss)
    epoch_acc = np.nanmean(running_acc)
    if writer is not None:
        writer.add_scalar('Loss/val', epoch_loss, epoch)
        writer.add_scalar('Acc/val', epoch_acc, epoch)
        writer.add_scalar('Std/val_outputs', torch.cat(running_masked_outputs).std(), epoch)
        writer.add_scalar('Std/val_precip', torch.cat(running_masked_precip).std(), epoch)

    return epoch_loss, epoch_acc

# -----------------------
# ======= main ==========
# -----------------------
def target_stats(dataset, idxs):
    arr = torch.nan_to_num(dataset.precip_arr[idxs].float())  # shape [N,H,W]
    mean = float(torch.nanmean(arr))
    std  = float(torch.std(arr))
    mn   = float(torch.min(arr))
    mx   = float(torch.max(arr))
    p90 = float(torch.nanquantile(arr, 0.9))
    p10 = float(torch.nanquantile(arr, 0.1))
    print("target mean,std,min,max,p10,p90:", mean, std, mn, mx, p10, p90)


def inspect_target_distribution(dataset, idxs, name="train"):
    arr = dataset.precip_arr[idxs].numpy().ravel()  # 标准化后的值
    arr = arr[~np.isnan(arr)]
    print(f"[{name}] count={arr.size}, mean={arr.mean():.6f}, std={arr.std():.6f}")
    for q in [0,1,5,10,25,50,75,90,95,99,100]:
        print(f"  p{q:02d} = {np.percentile(arr, q):.4f}")
    # baseline: predict zero (since data standardized -> mean≈0). correlation with zero is NaN,
    # but baseline MSE is simply arr.var()
    baseline_mse = np.mean((arr - 0.0)**2)
    print(f"[{name}] baseline (predict 0) mse = {baseline_mse:.6f}")


def main():
    print("Starting main() function...")
    torch.backends.cudnn.benchmark = True
    config = {
        'climate_channels': 10,
        'sst_channels': 6,
        'latent_channels': 8,
        'batch_size': 8,
        'epochs': 500,
        'lr': 1e-5,
        'weight_decay': 0.1,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'resume_checkpoint': None,
        'train_num_workers': 4,
        'val_num_workers': 2,
        'prefetch_factor': 2,
        'acc_frequency': 10,
        'ckpt_dir': './weights'
    }

    climate_path = r"E:\D1\diffusion\my_models\my_model_data\lr_unetall/1.npy"
    sst_path = r"E:/D1/diffusion/my_models/mulNet_data/sst_3chan.npy"
    precip_path = r"E:/D1/diffusion/my_models/mulNet_data/hr.npy"
    print("Loading data...")

    # 先创建一个不带增强的基础数据集（用于验证 & 统计）
    base_dataset = ClimateDataset(climate_path, sst_path, precip_path, transform=None)
    n_total = len(base_dataset)
    n_val = 70
    n_train = n_total - n_val

    # 使用模块顶层的 AugmentedSubset（避免在主函数内定义类导致 worker 无法序列化）
    # 训练集使用增强（GaussianNoise），验证集不使用增强
    train_dataset = AugmentedSubset(base_dataset, range(0, n_train), transform=GaussianNoise(std=0.5, prob=0.5, per_channel=True))
    val_dataset = torch.utils.data.Subset(base_dataset, range(n_train, n_total))

    # 统计仍然基于基础数据集（标准化后的目标）
    target_stats(base_dataset, list(range(0, n_train)))
    inspect_target_distribution(base_dataset, list(range(0, n_train)), "train")
    inspect_target_distribution(base_dataset, list(range(n_train, len(base_dataset))), "val")
    train_loader_kwargs = {
        'batch_size': config['batch_size'],
        'shuffle': True,
        'pin_memory': True,
        'num_workers': config['train_num_workers'],
        'prefetch_factor': config['prefetch_factor']
    }
    val_loader_kwargs = {
        'batch_size': config['batch_size'],
        'shuffle': False,
        'pin_memory': True,
        'num_workers': config['val_num_workers'],
        'prefetch_factor': config['prefetch_factor']
    }
    if config['train_num_workers'] > 0:
        train_loader_kwargs['persistent_workers'] = True
    if config['val_num_workers'] > 0:
        val_loader_kwargs['persistent_workers'] = True

    train_loader = DataLoader(train_dataset, **train_loader_kwargs)
    val_loader = DataLoader(val_dataset, **val_loader_kwargs)

    model = ClimatePrecipModel(
        climate_channels=config['climate_channels'],
        sst_channels=config['sst_channels'],
        latent_channels=config['latent_channels']
    ).to(config['device'])


    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'], eta_min=1e-6)

    writer = SummaryWriter(log_dir='./runs_mul_ae')

    start_epoch = 0
    best_val_loss = float('inf')
    resume_path = config.get('resume_checkpoint', None)
    if resume_path is not None and os.path.isfile(resume_path):
        ckpt = torch.load(resume_path, map_location=config['device'])
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
            if 'optimizer_state_dict' in ckpt:
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(config['device'])
            start_epoch = ckpt.get('epoch', -1) + 1
            best_val_loss = ckpt.get('best_val_loss', best_val_loss)
        else:
            model.load_state_dict(ckpt)
        print(f"Resumed from checkpoint: {resume_path}, start_epoch={start_epoch}, best_val_loss={best_val_loss}")

    os.makedirs(config['ckpt_dir'], exist_ok=True)

    for epoch in range(start_epoch, config['epochs']):
        t0 = time.time()
        train_loss, train_acc = train_model(
            model, train_loader, optimizer, loss_fn, config['device'], epoch,
            writer, acc_frequency=config['acc_frequency']
        )
        val_loss, val_acc = validate_model(model, val_loader, loss_fn, config['device'], epoch, writer)
        elapsed = time.time() - t0

        print(f'Epoch {epoch+1}/{config["epochs"]}  time: {elapsed:.1f}s')
        print(f'Train Loss: {train_loss}, Val Loss: {val_loss}')
        print(f'Train Acc: {train_acc}, Val Acc: {val_acc}')

        # 只有当 val_loss 有效 (非 NaN 且有限) 时才传给 scheduler，并参与最优模型判断
        if (not math.isnan(val_loss)) and math.isfinite(val_loss):
            scheduler.step()
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                ckpt_to_save = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'best_val_loss': best_val_loss
                }
                path_best = os.path.join(config['ckpt_dir'], 'best_model.pth')
                save_checkpoint_async(ckpt_to_save, path_best)
                print('Saved best model (async)')
        else:
            print("Warning: val_loss is NaN or Inf; skipping scheduler.step and best-model check.")

        # 周期性保存（异步）
        if epoch % 10 == 0:
            ckpt_to_save = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'best_val_loss': best_val_loss
            }
            path = os.path.join(config['ckpt_dir'], f"epoch_{epoch}.pth")
            save_checkpoint_async(ckpt_to_save, path)

    writer.close()

if __name__ == '__main__':
    main()

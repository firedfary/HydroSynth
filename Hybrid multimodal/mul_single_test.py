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
    def __init__(self, input_channels=10):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=0.5)
        )
        self.res_blocks = nn.Sequential(
            ResBlock(64, 64),
            ResBlock(64, 64)
        )
        self.upsample = nn.Sequential(
            nn.Upsample(size=(60, 70), mode='bilinear', align_corners=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=0.5),
            nn.Upsample(size=(120, 140), mode='bilinear', align_corners=True)
        )
    
    def forward(self, x):
        x = self.initial(x)
        x = self.down1(x)
        x = self.res_blocks(x)
        return self.upsample(x)

class SSTEncoder(nn.Module):
    def __init__(self, input_channels=3):
        super().__init__()
        self.upsample_init = nn.Upsample(size=(100, 180), mode='bilinear', align_corners=True)
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=0.5)
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=0.5)
        )
        self.res_block = ResBlock(64, 64)
        self.upsample = nn.Sequential(
            nn.Upsample(size=(60, 70), mode='bilinear', align_corners=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=0.5),
            nn.Upsample(size=(120, 140), mode='bilinear', align_corners=True))
    
    def forward(self, x):
        x = self.upsample_init(x)
        x = self.conv1(x)
        x = self.down1(x)
        x = self.res_block(x)
        return self.upsample(x)



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
        print("valid_pixels per sample:", (~self.data_mask).sum(dim=[1,2,3]).cpu().numpy())

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
        return climate_t, sst_t, precip_t, mask_t

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
    
class ClimateModel(nn.Module):
    """ 在 ClimateEncoder 后加一个 1x1 卷积作为预测头 """
    def __init__(self, input_channels=10):
        super().__init__()
        self.encoder = ClimateEncoder(input_channels)
        self.head = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        feat = self.encoder(x)  # [B,64,120,140]
        out = self.head(feat)   # [B,1,120,140]
        return out

class SSTModel(nn.Module):
    """ 在 SSTEncoder 后加一个 1x1 卷积作为预测头 """
    def __init__(self, input_channels=3):
        super().__init__()
        self.encoder = SSTEncoder(input_channels)
        self.head = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        feat = self.encoder(x)  # [B,64,120,140]
        out = self.head(feat)   # [B,1,120,140]
        return out

    
# （将所有与数据加载、DataLoader 创建和训练循环的顶层执行移动到 main()，并添加 Windows-friendly 的入口保护）
def main():
    climate_path = "E:/D1/diffusion/my_models/mulNet_data/lr.npy"
    sst_path = "E:/D1/diffusion/my_models/mulNet_data/sst_3chan.npy"
    precip_path = "E:/D1/diffusion/my_models/mulNet_data/hr.npy"
    print("Loading data...")

    # 先创建一个不带增强的基础数据集（用于验证 & 统计）
    base_dataset = ClimateDataset(climate_path, sst_path, precip_path, transform=None)
    n_total = len(base_dataset)
    n_val = 70
    n_train = n_total - n_val

    # 训练集使用增强（这里不传 transform，若需要在训练时加 augment，请传入GaussianNoise等实例）
    train_dataset = AugmentedSubset(base_dataset, range(0, n_train))
    val_dataset = torch.utils.data.Subset(base_dataset, range(n_train, n_total))

    config = {
        'climate_channels': 10,
        'sst_channels': 6,
        'latent_channels': 16,
        'batch_size': 32,
        'epochs': 500,
        'lr': 5e-4,
        'weight_decay': 1e-3,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'resume_checkpoint': None,
        'train_num_workers': 4,
        'val_num_workers': 2,
        'prefetch_factor': 2,
        'acc_frequency': 10,
        'ckpt_dir': './weights'
    }

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

    # model / optimizer / loss
    device = config['device']
    model = ClimateModel(input_channels=config['climate_channels']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    criterion = nn.MSELoss()

    # 训练循环（示例）
    for epoch in range(config['epochs']):
        model.train()
        total_loss = 0.0
        total_acc = 0.0
        for climate, sst, precip, mask in train_loader:
            climate, sst, precip, mask = climate.to(device), sst.to(device), precip.to(device), mask.to(device)
            pred = model(climate)
            # 只计算未被 mask 覆盖的区域
            masked_outputs = pred[~mask]
            masked_precip = precip[~mask]
            loss = criterion(masked_outputs, masked_precip)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pred[mask] = float('nan')
            Acc = fucs.cal_acc(pred[:,0,:,:]*100, precip[:,0,:,:]*100).mean()
            total_acc += Acc.item()


        print(f"[ClimateEncoder] Epoch {epoch+1}, Loss={total_loss/len(train_loader):.4f}, Acc={total_acc/len(train_loader):.4f}")
        if (epoch+1) % 10 == 0:
            model.eval()
            total_loss = 0.0
            total_acc = 0.0
            for climate, sst, precip, mask in val_loader:
                climate, sst, precip, mask = climate.to(device), sst.to(device), precip.to(device), mask.to(device)
                pred = model(climate)
                # 只计算未被 mask 覆盖的区域
                masked_outputs = pred[~mask]
                masked_precip = precip[~mask]
                loss = criterion(masked_outputs, masked_precip)
                total_loss += loss.item()
                pred[mask] = float('nan')
                Acc = fucs.cal_acc(pred[:,0,:,:]*100, precip[:,0,:,:]*100).mean()
                total_acc += Acc.item()
            print(f"[ClimateEncoder] Epoch {epoch+1}, Val Loss={total_loss/len(val_loader):.4f}, Val Acc={total_acc/len(val_loader):.4f}")

# Windows multiprocess spawn 要求：把执行入口放在 __main__ 下，并调用 freeze_support()
if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()

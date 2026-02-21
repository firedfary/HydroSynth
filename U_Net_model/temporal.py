import torch
import os
import tqdm
import numpy as np
import sys

# Ensure project root is on sys.path so absolute imports like 'HydroSynth' work
_proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_proj_root = os.path.normpath(_proj_root)
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

from HydroSynth.utils import utils
from HydroSynth import config
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from unetlitefilm_3D import UNetLiteFiLM
import random


def compute_pcs_from_sst(sst, n_pcs=3, window=1, step=1):
    """
    读取 SST，计算 EOF PCs，并支持时间窗口拼接。
    """

    print("Loaded SST shape:", sst.shape)
    B, T, H, W = sst.shape
    # sst_mean = np.nanmean(sst, axis=0, keepdims=True)(366, 6, 89, 180)
    # sst_anom = sst - sst_mean
    pcss = []
    eofs = []
    variance = []
    for t in range(T):
        X = sst[:,t].reshape(B, -1)
        X[~np.isfinite(X)] = 0.0

        pca = PCA(n_components=n_pcs)
        pcs = pca.fit_transform(X)  # [T, n_pcs]
        pcs = (pcs - pcs.mean(0, keepdims=True)) / (pcs.std(0, keepdims=True) + 1e-8)
        eof_patterns = pca.components_.reshape(n_pcs, H, W)
        pcss.append(pcs)
        eofs.append(eof_patterns)
        variance.append(pca.explained_variance_ratio_.sum())

    print(f"PCA done. Explained variance={np.mean(variance):.3f}")
    pcs = np.stack(pcss, axis=0) # [B, T, n_pcs]
    eof_patterns = np.stack(eofs, axis=0)  # [B, n_pcs, H, W]

    return pcs.astype(np.float32), eof_patterns.astype(np.float32)

def prepare_data():
    """Load target, condition, and indices (PCs). Return TensorDatasets."""
    # --- target (precip) ---
    target_file = config.modelconfig["hr_path"] + "/hr_data1.npy"
    target = np.load(target_file).astype(np.float32)  # [B,T,H,W]
    target = np.expand_dims(target, 1)  # [B,1,T,H,W]去掉expand_dims变为[B,T,H,W]
    target_t = torch.from_numpy(target)
    mask_t = torch.isnan(target_t)

    # --- condition (10-channel climate fields) ---
    cond_file = config.modelconfig["lr_path"] + "/lr_data1.npy" # [B,C,T,H,W]
    cond = np.load(cond_file).astype(np.float32) 
    # if cond.ndim == 5:
    #     Tdim = cond.shape[0]
    #     cond = cond.reshape(Tdim, -1, cond.shape[-2], cond.shape[-1])
    #     print("Reshaped 5D condition:", cond.shape)#361,60,120,140
    cond_t = torch.from_numpy(cond)

    # --- PCs from SST ---
    sst = np.load(config.modelconfig["sst_file"])#[B,M,H,W]
    n_pcs = config.modelconfig["n_pcs"]
    window = config.modelconfig["pc_window"]
    step = config.modelconfig["pc_step"]#366,5/5,89,180
    pcs, eof_patterns = compute_pcs_from_sst(sst, n_pcs=n_pcs, window=window, step=step)  # [T', n_pcs*window]366,10
    pcs_t = torch.from_numpy(pcs).permute(1, 0, 2)#366,6,5

    # --- 对齐时间长度 ---
    minT = min(target_t.shape[0], cond_t.shape[0], pcs_t.shape[0])
    target_t = target_t[:minT]#361,1,6,120,140
    mask_t = mask_t[:minT]#361,1,6,120,140
    cond_t = cond_t[:minT]#361,10,6,120,140
    pcs_t = pcs_t[:minT]#361,6,5

    # --- split train/test ---
    num_test_samples = 21
    total = len(target_t)
    train_end = total - num_test_samples

    train_set = TensorDataset(
        target_t[:train_end], cond_t[:train_end], mask_t[:train_end], pcs_t[:train_end]
    )
    test_set = TensorDataset(
        target_t[train_end:], cond_t[train_end:], mask_t[train_end:], pcs_t[train_end:]
    )
    return train_set, test_set


class IndexEncoder(nn.Module):
    """
    输入：SST EOF PCs + lead time
    输出：每一层的 FiLM γ, β（随时间变化）
    """
    def __init__(self, pc_dim=5, time_dim=6, hidden=128, n_layers=4):
        super().__init__()
        self.n_layers = n_layers

        self.time_embed = nn.Embedding(time_dim, hidden)

        self.mlp = nn.Sequential(
            nn.Linear(pc_dim + hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU()
        )

        self.gamma = nn.ModuleList([
            nn.Linear(hidden, 1) for _ in range(n_layers)
        ])
        self.beta = nn.ModuleList([
            nn.Linear(hidden, 1) for _ in range(n_layers)
        ])

    def forward(self, pcs, lead_time):
        """
        pcs:       [B, T, K]
        lead_time: [B, T]
        """
        B, T, _ = pcs.shape

        t_emb = self.time_embed(lead_time)          # [B,T,H]
        x = torch.cat([pcs, t_emb], dim=-1)         # [B,T,K+H]
        h = self.mlp(x)                             # [B,T,H]

        gammas, betas = [], []
        for i in range(self.n_layers):
            g = self.gamma[i](h).unsqueeze(-1).unsqueeze(-1)
            b = self.beta[i](h).unsqueeze(-1).unsqueeze(-1)
            gammas.append(g)
            betas.append(b)

        return gammas, betas
    
class FiLMConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self, x, gamma, beta):
        # x: [B*T, C, H, W]
        x = self.conv(x)
        x = gamma * x + beta
        return x
    
class Encoder(nn.Module):
    def __init__(self, in_ch=10):
        super().__init__()
        self.blocks = nn.ModuleList([
            FiLMConvBlock(in_ch, 32),
            FiLMConvBlock(32, 64),
            FiLMConvBlock(64, 128),
            FiLMConvBlock(128, 256)
        ])
        self.pool = nn.MaxPool2d(2)

    def forward(self, x, gammas, betas):
        skips = []
        for i, block in enumerate(self.blocks):
            x = block(x, gammas[i], betas[i])
            if i < len(self.blocks) - 1:
                skips.append(x)
                x = self.pool(x)
        return x, skips


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, 2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch + skip_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)
    
class TemporalFiLMUNet(nn.Module):
    def __init__(self, pc_dim=5):
        super().__init__()

        self.index_encoder = IndexEncoder(
            pc_dim=pc_dim,
            time_dim=6,
            n_layers=4
        )

        self.encoder = Encoder(in_ch=10)

        self.up1 = UpBlock(256, 128, 128)
        self.up2 = UpBlock(128, 64, 64)
        self.up3 = UpBlock(64, 32, 32)
        self.out_conv = nn.Conv2d(32, 1, 1)

    def forward(self, cond, pcs):
        """
        cond: [B, C=10, T=6, H, W]
        pcs:  [B, T, K]
        """
        B, C, T, H, W = cond.shape

        lead_time = torch.arange(T, device=cond.device).unsqueeze(0).repeat(B, 1)

        gammas, betas = self.index_encoder(pcs, lead_time)

        # reshape time → batch
            #16,10,6,120,140 -> 16*6,10,120,140
        x = cond.permute(0,2,1,3,4).reshape(B*T, C, H, W)

        # flatten FiLM params
        gammas = [g.reshape(B*T,1,1,1) for g in gammas]
        betas  = [b.reshape(B*T,1,1,1) for b in betas]

        x, skips = self.encoder(x, gammas, betas)
        x = self.up1(x, skips[-1])
        x = self.up2(x, skips[-2])
        x = self.up3(x, skips[-3])
        out = self.out_conv(x)

        if out.shape[-2:] != (H, W):
            out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)

        out = out.reshape(B, T, 1, H, W).permute(0,2,1,3,4)
        return out
    
device = config.modelconfig['device']
model = TemporalFiLMUNet(pc_dim=5).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()

train_set, test_set = prepare_data()

train_loader = DataLoader(
    train_set, batch_size=config.modelconfig["batch_size"],
    shuffle=True, num_workers=0, pin_memory=True, drop_last=True
)
test_loader = DataLoader(
    test_set, batch_size=config.modelconfig["batch_size"],
    shuffle=False, num_workers=0, pin_memory=True
)
writer = SummaryWriter(log_dir=config.modelconfig["log_path"])

for e in range(config.modelconfig["epoch"]):
    model.train()
    train_loss, train_acc = [], {i: [] for i in range(6)}
    for target, cond, mask, pcs in tqdm.tqdm(train_loader, desc=f"Epoch {e}"):
        target, cond, mask, pcs = target.to(device), cond.to(device), mask.to(device), pcs.to(device)

        pred = model(cond, pcs)
        loss = criterion(pred[~mask], target[~mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
        for i in range(6):
            acc = utils.cal_acc(pred[:,0,i], target[:,0,i]).mean()
            train_acc[i].append(acc.item())

    for i in range(6):
        avg_acc_i = np.mean(train_acc[i])
        writer.add_scalar(f"Acc/train_t{i}", avg_acc_i, e)
    writer.add_scalar("train_loss", np.mean(train_loss), e)
    model.eval()
    test_loss, test_acc = [], {i: [] for i in range(6)}
    with torch.no_grad():
        for target, cond, mask, pcs in test_loader:
            target, cond, mask, pcs = target.to(device), cond.to(device), mask.to(device), pcs.to(device)

            pred = model(cond, pcs)
            loss = criterion(pred[~mask], target[~mask])
            test_loss.append(loss.item())
            for i in range(6):
                acc = utils.cal_acc(pred[:,0,i], target[:,0,i]).mean()
                test_acc[i].append(acc.item())
        for i in range(6):
            avg_acc_i = np.mean(test_acc[i])
            writer.add_scalar(f"Acc/test_t{i}", avg_acc_i, e)
        writer.add_scalar("test_loss", np.mean(test_loss), e)

writer.close()

import model1
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
config.enable_auto_create_folders()
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
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
    # n_pcs = config.modelconfig["n_pcs"]
    # window = config.modelconfig["pc_window"]
    # step = config.modelconfig["pc_step"]#366,5/5,89,180
    # pcs, eof_patterns = compute_pcs_from_sst(sst, n_pcs=n_pcs, window=window, step=step)  # [T', n_pcs*window]366,10
    # pcs_t = torch.from_numpy(pcs).permute(1, 0, 2)#366,6,5

    # --- 对齐时间长度 ---
    sst_t = torch.from_numpy(sst)
    minT = min(target_t.shape[0], cond_t.shape[0], sst_t.shape[0])
    target_t = target_t[:minT]#361,1,6,120,140
    mask_t = mask_t[:minT]#361,1,6,120,140
    cond_t = cond_t[:minT]#361,10,6,120,140
    sst_t = sst_t[:minT]#361,6,89,180

    # --- split train/test ---
    num_test_samples = 21
    total = len(target_t)
    train_end = total - num_test_samples

    train_set = TensorDataset(
        target_t[:train_end], cond_t[:train_end], mask_t[:train_end], sst_t[:train_end]
    )
    test_set = TensorDataset(
        target_t[train_end:], cond_t[train_end:], mask_t[train_end:], sst_t[train_end:]
    )
    return train_set, test_set

device = config.modelconfig['device']
model = model1.SpatioTemporalCorrector(in_ch=10, cond_time=6, sst_m=6, base_channels=8).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

if config.modelconfig["train_load_weight"] is not None:
    model.load_state_dict(torch.load(
        os.path.join(config.modelconfig["save_weight_dir"], config.modelconfig["train_load_weight"]),
        map_location=device))

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
    for target, cond, mask, sst in tqdm.tqdm(train_loader, desc=f"Epoch {e}"):
        target, cond, mask, sst = target.to(device), cond.to(device), mask.to(device), sst.to(device)

        pred = model(cond, sst)
        nll = model1.zero_inflated_gamma_nll(pred, target)
        crps = model1.approx_crps_by_sampling(pred, target, n_samples=16)
        mse = F.mse_loss(pred['delta'][~mask], target[~mask])
        loss = nll + 0.1 * crps + 0.5 * mse
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
        # pred 是模型输出的 dict
        p0    = pred['p0']      # [B,1,T,H,W]
        alpha = pred['alpha']
        beta  = pred['beta']
        delta = pred['delta']

        # 1. 计算确定性预测（posterior mean）
        #    shape: [B,1,T,H,W]
        pred_mean = delta + (1.0 - p0) * alpha * beta

        # 2. 按 lead time 计算 ACC
        for i in range(pred_mean.shape[2]):  # T = 6
            acc = utils.cal_acc(pred_mean[:, 0, i], target[:, 0, i]).mean()
            train_acc[i].append(acc.item())

    for i in range(6):
        avg_acc_i = np.mean(train_acc[i])
        writer.add_scalar(f"Acc/train_t{i}", avg_acc_i, e)
    writer.add_scalar("train_loss", np.mean(train_loss), e)
    model.eval()
    test_loss, test_acc = [], {i: [] for i in range(6)}
    with torch.no_grad():
        for target, cond, mask, sst in test_loader:
            target, cond, mask, sst = target.to(device), cond.to(device), mask.to(device), sst.to(device)

            pred = model(cond, sst)
            nll = model1.zero_inflated_gamma_nll(pred, target)
            crps = model1.approx_crps_by_sampling(pred, target, n_samples=16)
            mse = F.mse_loss(pred['delta'][~mask], target[~mask])
            loss = nll + 0.1 * crps + 0.5 * mse
            test_loss.append(loss.item())
            # pred 是模型输出的 dict
            p0    = pred['p0']      # [B,1,T,H,W]
            alpha = pred['alpha']
            beta  = pred['beta']
            delta = pred['delta']

            # 1. 计算确定性预测（posterior mean）
            #    shape: [B,1,T,H,W]
            pred_mean = delta + (1.0 - p0) * alpha * beta

            # 2. 按 lead time 计算 ACC
            for i in range(pred_mean.shape[2]):  # T = 6
                acc = utils.cal_acc(pred_mean[:, 0, i], target[:, 0, i]).mean()
                test_acc[i].append(acc.item())
        for i in range(6):
            avg_acc_i = np.mean(test_acc[i])
            writer.add_scalar(f"Acc/test_t{i}", avg_acc_i, e)
        writer.add_scalar("test_loss", np.mean(test_loss), e)
    if e % 5 == 0:
        torch.save(model, os.path.join(config.modelconfig["save_weight_path"], f"epoch_{e}.pt"))

writer.close()

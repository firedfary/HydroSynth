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
from sklearn.decomposition import PCA
from unetlitefilm_3D import UNetLiteFiLM
import random

def add_physical_noise(cond, noise_level=0.15):
    """添加符合物理约束的噪声"""
    noise = torch.randn_like(cond) * noise_level
    
    # # 保持空间相关性（高斯平滑）
    # noise_smoothed = torch.nn.functional.conv2d(
    #     noise, 
    #     torch.ones(1, 1, 3, 3).to(cond.device) / 9,
    #     padding=1
    # )
    
    # # 对不同通道应用不同噪声水平
    # channel_weights = torch.tensor([1.0, 0.8, 0.6, ...]).to(cond.device)  # 根据变量重要性
    # noise_smoothed = noise_smoothed * channel_weights.view(1, -1, 1, 1)
    
    return (cond + noise)/cond.max()

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

    # # --- 窗口拼接 ---
    # pcs_window = []
    # for t in range(0, T - window + 1, step):
    #     pcs_window.append(pcs[t:t + window].reshape(-1))
    # pcs_window = np.stack(pcs_window, axis=0)  # [T-window+1, n_pcs*window]

    # print(f"After windowing: {pcs_window.shape}")
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
    sst = np.load(config.modelconfig["sst_file"])#[T,M,H,W]
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


def train():
    device = torch.device(config.modelconfig["device"])
    train_set, test_set = prepare_data()

    train_loader = DataLoader(
        train_set, batch_size=config.modelconfig["batch_size"],
        shuffle=True, num_workers=0, pin_memory=True, drop_last=True
    )
    test_loader = DataLoader(
        test_set, batch_size=config.modelconfig["batch_size"],
        shuffle=False, num_workers=0, pin_memory=True
    )

    input_channels = train_set[0][1].shape[0]  # condition channels
    index_dim = train_set[0][3].shape[0]       # PCs dimension (n_pcs * window)
    model = UNetLiteFiLM(
        n_channels=60,
        n_classes=6,
        index_dim=index_dim,
        base_filters=64,
        dropout=config.modelconfig["dropout"]
    ).to(device)

    if config.modelconfig["train_load_weight"] is not None:
        model.load_state_dict(torch.load(
            os.path.join(config.modelconfig["save_weight_dir"], config.modelconfig["train_load_weight"]),
            map_location=device))

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.modelconfig["lr"], weight_decay=1e-5)
    loss_fn = torch.nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.modelconfig["epoch"], eta_min=0
    )
    writer = SummaryWriter(config.modelconfig["log_path"])

    best_test_loss = float("inf")
    for e in range(config.modelconfig["epoch"]):
        # --- train ---
        model.train()
        train_losses = []
        train_accs_by_t = {i: [] for i in range(6)}  # 为每个时间步分别记录
        for x_0, cond, mask, pcs in tqdm.tqdm(train_loader, desc=f"Train {e}"):
            x_0, cond, mask, pcs = x_0.to(device), cond.to(device), mask.to(device), pcs.to(device)
            B, C, t, H, W = cond.shape

            cond = cond.view(B, t*C, H, W)
            x_0 = x_0.view(B, t, H, W)
            mask = mask.view(B, t, H, W)

            # pcs = pcs.unsqueeze(1).repeat(1, 6, 1)
            # pcs = pcs.view(B*6, -1)

            lead = torch.arange(6, device=device).repeat(B)

            optimizer.zero_grad()
            if random.random() < 0.5:
                pcs = add_physical_noise(pcs)
                cond = add_physical_noise(cond)
            out = model(cond, pcs)
            out[mask] = float("nan")
            # fucs.check_nan_status(out)
            # fucs.check_nan_status(x_0)
            loss = loss_fn(out[~mask], x_0[~mask])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.modelconfig["grad_clip"])
            optimizer.step()

            train_losses.append(loss.item())
            # 为每个时间步分别计算精度
            for i in range(t):
                acc_i = utils.cal_acc(out[:,i]*100, x_0[:,i]*100).mean()
                train_accs_by_t[i].append(acc_i.item())

        scheduler.step()
        avg_train_loss = np.mean(train_losses)
        writer.add_scalar("Loss/train", avg_train_loss, e)
        # 为每个时间步单独记录精度
        for i in range(6):
            avg_acc_i = np.mean(train_accs_by_t[i])
            writer.add_scalar(f"Acc/train_t{i}", avg_acc_i, e)

        # --- test ---
        model.eval()
        test_losses = []
        test_accs_by_t = {i: [] for i in range(6)}  # 为每个时间步分别记录
        with torch.no_grad():
            for x_0, cond, mask, pcs in tqdm.tqdm(test_loader, desc=f"Test {e}"):
                x_0, cond, mask, pcs = x_0.to(device), cond.to(device), mask.to(device), pcs.to(device)
                B, C, t, H, W = cond.shape

                cond = cond.view(B, t*C, H, W)
                x_0 = x_0.view(B, t, H, W)
                mask = mask.view(B, t, H, W)

                # pcs = pcs.unsqueeze(1).repeat(1, 6, 1)
                # pcs = pcs.view(B*6, -1)

                lead = torch.arange(6, device=device).repeat(B)
                
                out = model(cond, pcs)

                out[mask] = float("nan")
                loss = loss_fn(out[~mask], x_0[~mask])
                test_losses.append(loss.item())
                # 为每个时间步分别计算精度
                for i in range(t):
                    acc_i = utils.cal_acc(out[:,i]*100, x_0[:,i]*100).mean()
                    test_accs_by_t[i].append(acc_i.item())

        avg_test_loss = np.mean(test_losses)
        writer.add_scalar("Loss/test", avg_test_loss, e)
        # 为每个时间步单独记录精度
        for i in range(6):
            avg_acc_i = np.mean(test_accs_by_t[i])
            writer.add_scalar(f"Acc/test_t{i}", avg_acc_i, e)

        # 计算平均精度用于打印
        avg_train_acc = np.mean([np.mean(train_accs_by_t[i]) for i in range(6)])
        avg_test_acc = np.mean([np.mean(test_accs_by_t[i]) for i in range(6)])
        print(f"Epoch {e}: TrainLoss={avg_train_loss:.4f}, TestLoss={avg_test_loss:.4f}, "
              f"TrainAcc={avg_train_acc:.3f}, TestAcc={avg_test_acc:.3f}")

        if e % 5 == 0:
            torch.save(model, os.path.join(config.modelconfig["save_weight_path"], f"epoch_{e}.pt"))

    writer.close()


if __name__ == "__main__":
    train()

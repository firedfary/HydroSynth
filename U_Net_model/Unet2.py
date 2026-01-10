import torch
import os
import tqdm
import numpy as np
import fucs
import config
from torch.utils.tensorboard import SummaryWriter
from unetlite import UNetLite
from sklearn.decomposition import PCA
from torch.utils.data import TensorDataset, DataLoader


def compute_pcs_from_sst(sst_path, n_pcs=3):
    sst = np.load(sst_path)  # [T,H,W] or [T,M,H,W]
    print("Loaded SST shape:", sst.shape)

    if sst.ndim == 4:
        sst = np.mean(sst, axis=1)  # [T,H,W]
        print("Averaged over month-dim:", sst.shape)

    T, H, W = sst.shape
    sst_mean = np.nanmean(sst, axis=0, keepdims=True)
    sst_anom = sst - sst_mean
    X = sst_anom.reshape(T, -1)
    X[~np.isfinite(X)] = 0.0

    pca = PCA(n_components=n_pcs)
    pcs = pca.fit_transform(X)  # [T, n_pcs]
    pcs = (pcs - pcs.mean(0, keepdims=True)) / (pcs.std(0, keepdims=True) + 1e-8)

    print(f"PCA done. {n_pcs} PCs explain {pca.explained_variance_ratio_.sum():.3f} variance.")
    return pcs.astype(np.float32)


def prepare_data():
    device = config.modelconfig["device"]

    # --- target ---
    target_file = config.modelconfig["hr_path"] + "/hr_data1.npy"
    target = np.load(target_file).astype(np.float32)  # [T,H,W]
    target = np.expand_dims(target, 1)  # [T,1,H,W]
    target_t = torch.from_numpy(target)
    mask_t = torch.isnan(target_t)

    # --- condition (10-channel climate fields) ---
    cond_file = config.modelconfig["lr_path"] + "/lr_data1.npy"
    cond = np.load(cond_file).astype(np.float32)  # [T,C,H,W] (should be this shape)
    if cond.ndim == 5:
        Tdim = cond.shape[0]
        cond = cond.reshape(Tdim, -1, cond.shape[-2], cond.shape[-1])
        print("Reshaped 5D condition to:", cond.shape)

    # --- PCs from SST ---
    sst_path = config.modelconfig["sst_file"]
    n_pcs = config.modelconfig["n_pcs"]
    pcs = compute_pcs_from_sst(sst_path, n_pcs=n_pcs)  # [T,n_pcs]

    # trim time length to match
    minT = min(target_t.shape[0], cond.shape[0], pcs.shape[0])
    target_t = target_t[:minT]
    mask_t = mask_t[:minT]
    cond = cond[:minT]
    pcs = pcs[:minT]

    # broadcast PCs to spatial
    T, C, H, W = cond.shape
    pc_channels = np.repeat(pcs[:, :, None, None], H, axis=2)
    pc_channels = np.repeat(pc_channels, W, axis=3)  # [T,K,H,W]

    cond_aug = np.concatenate([cond, pc_channels], axis=1)  # [T,C+K,H,W]
    cond_t = torch.from_numpy(cond_aug)

    # --- split train/test ---
    num_test_samples = 21
    total = len(target_t)
    train_end = total - num_test_samples

    train_set = TensorDataset(
        target_t[:train_end], cond_t[:train_end], mask_t[:train_end]
    )
    test_set = TensorDataset(
        target_t[train_end:], cond_t[train_end:], mask_t[train_end:]
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
    model = UNetLite(n_channels=input_channels, n_classes=1).to(device)
    if config.modelconfig["train_load_weight"] is not None:
        model.load_state_dict(torch.load(
            os.path.join(config.modelconfig["save_weight_path"], config.modelconfig["train_load_weight"]),
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
        train_losses, train_accs = [], []
        for x_0, cond, mask in tqdm.tqdm(train_loader, desc=f"Train {e}"):
            x_0, cond, mask = x_0.to(device), cond.to(device), mask.to(device)
            optimizer.zero_grad()
            out = model(cond)
            out[mask] = float("nan")
            loss = loss_fn(out[~mask], x_0[~mask])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.modelconfig["grad_clip"])
            optimizer.step()

            acc = fucs.cal_acc(out[:,0]*100, x_0[:,0]*100).mean()
            train_losses.append(loss.item())
            train_accs.append(acc.item())

        scheduler.step()
        avg_train_loss, avg_train_acc = np.mean(train_losses), np.mean(train_accs)
        writer.add_scalar("Loss/train", avg_train_loss, e)
        writer.add_scalar("Acc/train", avg_train_acc, e)

        # --- test ---
        model.eval()
        test_losses, test_accs = [], []
        with torch.no_grad():
            for x_0, cond, mask in tqdm.tqdm(test_loader, desc=f"Test {e}"):
                x_0, cond, mask = x_0.to(device), cond.to(device), mask.to(device)
                out = model(cond)
                out[mask] = float("nan")
                loss = loss_fn(out[~mask], x_0[~mask])
                acc = fucs.cal_acc(out[:,0]*100, x_0[:,0]*100).mean()
                test_losses.append(loss.item())
                test_accs.append(acc.item())

        avg_test_loss, avg_test_acc = np.mean(test_losses), np.mean(test_accs)
        writer.add_scalar("Loss/test", avg_test_loss, e)
        writer.add_scalar("Acc/test", avg_test_acc, e)

        print(f"Epoch {e}: TrainLoss={avg_train_loss:.4f}, TestLoss={avg_test_loss:.4f}, "
              f"TrainAcc={avg_train_acc:.3f}, TestAcc={avg_test_acc:.3f}")

        if e % 10 == 0:
            torch.save(model.state_dict(),
                       os.path.join(config.modelconfig["save_weight_path"], f"unet_{e}.pt"))
            print("Saved best model.")

    writer.close()


if __name__ == "__main__":
    train()

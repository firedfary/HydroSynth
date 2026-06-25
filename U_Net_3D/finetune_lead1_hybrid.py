import os
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader
from sklearn.decomposition import PCA

# Add directories to path so Python can unpickle the model and import config
_curr_file = os.path.abspath(__file__)
_proj_root = os.path.dirname(os.path.dirname(_curr_file)) # /Users/huawei/HydroSynth
_proj_parent = os.path.dirname(_proj_root)                # /Users/huawei
if _proj_parent not in sys.path:
    sys.path.insert(0, _proj_parent)
if os.path.join(_proj_root, "U_Net_model") not in sys.path:
    sys.path.insert(0, os.path.join(_proj_root, "U_Net_model"))

from HydroSynth.utils import utils
from HydroSynth import config
from unetlitefilm import UNetLiteFiLM

# Set seed for reproducibility
torch.manual_seed(config.modelconfig["seed"])
np.random.seed(config.modelconfig["seed"])

def acc_hybrid_loss(pred, target, mask, mse_weight=0.05):
    """
    Hybrid loss: (1 - ACC) + mse_weight * MSE.
    Optimizes ACC directly while retaining scale information.
    """
    m = mask.float()
    count = m.sum(dim=(2, 3), keepdim=True).clamp_min(1.0)
    p_mean = (pred * m).sum(dim=(2, 3), keepdim=True) / count
    t_mean = (target * m).sum(dim=(2, 3), keepdim=True) / count
    pa = (pred - p_mean) * m
    ta = (target - t_mean) * m
    cov = (pa * ta).sum(dim=(2, 3))
    p_var = (pa ** 2).sum(dim=(2, 3))
    t_var = (ta ** 2).sum(dim=(2, 3))
    acc = cov / torch.sqrt(p_var * t_var + 1e-6)
    acc_loss = (1.0 - acc).mean()
    mse_loss = ((pred - target)**2 * m).sum() / m.sum().clamp_min(1.0)
    return acc_loss + mse_weight * mse_loss

def compute_pcs_from_sst(sst_path, n_pcs=5, window=1, step=1):
    sst = np.load(sst_path)
    if sst.ndim == 4:
        sst = np.mean(sst, axis=1)

    T, H, W = sst.shape
    sst_mean = np.nanmean(sst, axis=0, keepdims=True)
    sst_anom = sst - sst_mean
    X = sst_anom.reshape(T, -1)
    X[~np.isfinite(X)] = 0.0

    pca = PCA(n_components=n_pcs, svd_solver='full')
    pcs = pca.fit_transform(X)
    pcs = (pcs - pcs.mean(0, keepdims=True)) / (pcs.std(0, keepdims=True) + 1e-8)
    eof_patterns = pca.components_.reshape(n_pcs, H, W)

    pcs_window = []
    for t in range(0, T - window + 1, step):
        pcs_window.append(pcs[t:t + window].reshape(-1))
    pcs_window = np.stack(pcs_window, axis=0)
    return pcs_window.astype(np.float32), eof_patterns.astype(np.float32)

def prepare_data():
    data_dir = "/Users/huawei/workplace/unet3D"
    cond_file = os.path.join(data_dir, 'lr_data_lead1_aligned.npy')
    target_file = os.path.join(data_dir, 'hr_data_lead1_aligned.npy')
    
    if not os.path.exists(cond_file) or not os.path.exists(target_file):
        raise FileNotFoundError(f"Aligned datasets not found in {data_dir}. Run prepare_lead1_data.py first.")
        
    cond = np.load(cond_file).astype(np.float32)      # [N, 10, 120, 140]
    target = np.load(target_file).astype(np.float32)  # [N, 1, 120, 140]
    
    cond_t = torch.from_numpy(cond)
    target_t = torch.from_numpy(target)
    mask_t = torch.isnan(target_t)
    
    sst_path = config.modelconfig["sst_file"]
    n_pcs = config.modelconfig["n_pcs"]
    window = config.modelconfig["pc_window"]
    step = config.modelconfig["pc_step"]
    pcs, _ = compute_pcs_from_sst(sst_path, n_pcs=n_pcs, window=window, step=step)
    
    all_dates = pd.date_range(start='1994-01-01', end='2024-09-01', freq='MS')
    exclude_dates = [pd.to_datetime(d) for d in ['2017-01-01', '2011-09-01', '2011-10-01']]
    valid_issue_dates = [d for d in all_dates if d not in exclude_dates]
    date_to_idx = {d: i for i, d in enumerate(valid_issue_dates)}
    
    aligned_target_dates = []
    for target_date in all_dates:
        if target_date in exclude_dates:
            continue
        issue_date = target_date - pd.DateOffset(months=1)
        if issue_date in exclude_dates:
            continue
        aligned_target_dates.append(target_date)
        
    assert len(aligned_target_dates) == cond_t.shape[0], f"Mismatch in aligned dates length"
    
    pcs_aligned_list = []
    for target_date in aligned_target_dates:
        idx_target = date_to_idx[target_date]
        pcs_aligned_list.append(pcs[idx_target])
        
    pcs_aligned = np.stack(pcs_aligned_list)
    pcs_t = torch.from_numpy(pcs_aligned)
    
    print(f"Data ready. Total samples: {target_t.shape[0]}")
    print(f"Condition shape: {cond_t.shape}")
    print(f"Target shape: {target_t.shape}")
    print(f"PCs shape: {pcs_t.shape}")
    
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
    print(f"Training device: {device}")
    
    train_set, test_set = prepare_data()
    pin_memory = device.type == "cuda" or device.type == "mps"
    
    train_loader = DataLoader(
        train_set, batch_size=config.modelconfig["batch_size"],
        shuffle=True, num_workers=0, pin_memory=pin_memory, drop_last=True
    )
    test_loader = DataLoader(
        test_set, batch_size=config.modelconfig["batch_size"],
        shuffle=False, num_workers=0, pin_memory=pin_memory
    )
    
    # Load pre-trained lead-0 model
    pretrained_path = "/Users/huawei/workplace/weight_t0/run_20260624_202637/epoch_40.pt"
    print(f"Loading pre-trained lead-0 model from: {pretrained_path}")
    model = torch.load(pretrained_path, map_location=device, weights_only=False)
    
    # Fine-tuning learning rate (smaller)
    lr = 5e-5
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    epochs = 80
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=0
    )
    
    log_dir = os.path.join(config.modelconfig["log_path"] + "_finetune")
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard logs saved to: {log_dir}")
    
    best_test_acc = -1.0
    
    for e in range(epochs):
        model.train()
        train_losses, train_accs = [], []
        for x_0, cond, mask, pcs in train_loader:
            x_0, cond, mask, pcs = x_0.to(device), cond.to(device), mask.to(device), pcs.to(device)
            optimizer.zero_grad()
            out = model(cond, pcs)
            
            target_clean = x_0.clone()
            target_clean[mask] = 0.0
            
            loss = acc_hybrid_loss(out, target_clean, ~mask, mse_weight=0.05)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.modelconfig["grad_clip"])
            optimizer.step()
            
            out_masked = out.clone()
            out_masked[mask] = float("nan")
            acc = utils.cal_acc(out_masked[:,0]*100, x_0[:,0]*100).mean()
            
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
            for x_0, cond, mask, pcs in test_loader:
                x_0, cond, mask, pcs = x_0.to(device), cond.to(device), mask.to(device), pcs.to(device)
                out = model(cond, pcs)
                
                target_clean = x_0.clone()
                target_clean[mask] = 0.0
                
                loss = acc_hybrid_loss(out, target_clean, ~mask, mse_weight=0.05)
                
                out_masked = out.clone()
                out_masked[mask] = float("nan")
                acc = utils.cal_acc(out_masked[:,0]*100, x_0[:,0]*100).mean()
                
                test_losses.append(loss.item())
                test_accs.append(acc.item())
                
        avg_test_loss, avg_test_acc = np.mean(test_losses), np.mean(test_accs)
        writer.add_scalar("Loss/test", avg_test_loss, e)
        writer.add_scalar("Acc/test", avg_test_acc, e)
        
        if avg_test_acc > best_test_acc:
            best_test_acc = avg_test_acc
            save_path = os.path.join(config.modelconfig["save_weight_path"], "best_model_v2.pt")
            torch.save(model.state_dict(), save_path)
            
            if best_test_acc >= 0.2:
                print(f"--> Target reached! Saved best_model_v2.pt with Test ACC: {best_test_acc:.3f}")
            
        if e % 10 == 0 or e == epochs - 1:
            print(f"Epoch {e:2d}: TrainLoss={avg_train_loss:.4f}, TestLoss={avg_test_loss:.4f}, "
                  f"TrainAcc={avg_train_acc:.3f}, TestAcc={avg_test_acc:.3f} (Best={best_test_acc:.3f})")
            
    writer.close()
    print(f"\nFine-tuning finished! Best Test ACC: {best_test_acc:.3f}")

if __name__ == "__main__":
    train()

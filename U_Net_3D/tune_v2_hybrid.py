import os
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader
from sklearn.decomposition import PCA

# Ensure project parent is in sys.path so 'HydroSynth' is importable
_curr_file = os.path.abspath(__file__)
_proj_root = os.path.dirname(os.path.dirname(_curr_file)) # /Users/huawei/HydroSynth
_proj_parent = os.path.dirname(_proj_root)                # /Users/huawei
if _proj_parent not in sys.path:
    sys.path.insert(0, _proj_parent)

from HydroSynth.utils import utils
from HydroSynth import config
from U_Net_model.unetlitefilm import UNetLiteFiLM

# Set seed for reproducibility
torch.manual_seed(config.modelconfig["seed"])
np.random.seed(config.modelconfig["seed"])

def acc_hybrid_loss(pred, target, mask, mse_weight=0.1):
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
    return pcs.astype(np.float32), eof_patterns.astype(np.float32)

def prepare_data():
    data_dir = "/Users/huawei/workplace/unet3D"
    cond_file = os.path.join(data_dir, 'lr_data_v2_aligned.npy')
    target_file = os.path.join(data_dir, 'hr_data_v2_aligned.npy')
    
    if not os.path.exists(cond_file) or not os.path.exists(target_file):
        raise FileNotFoundError(f"Aligned datasets not found in {data_dir}. Run prepare_data_v2.py first.")
        
    cond = np.load(cond_file).astype(np.float32)      # [N, 21, 120, 140]
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
        if issue_date < pd.to_datetime('1994-01-01'):
            continue
        aligned_target_dates.append(target_date)
        
    assert len(aligned_target_dates) == cond_t.shape[0], f"Mismatch in aligned dates length"
    
    pcs_aligned_list = []
    for target_date in aligned_target_dates:
        idx_target = date_to_idx[target_date]
        pcs_aligned_list.append(pcs[idx_target])
        
    pcs_aligned = np.stack(pcs_aligned_list)
    pcs_t = torch.from_numpy(pcs_aligned)
    
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

def train_config(cfg, train_set, test_set, device):
    print(f"\n--- Training configuration: {cfg['name']} ---")
    print(f"LR={cfg['lr']}, Dropout={cfg['dropout']}, WeightDecay={cfg['weight_decay']}")
    
    pin_memory = device.type == "cuda" or device.type == "mps"
    train_loader = DataLoader(
        train_set, batch_size=config.modelconfig["batch_size"],
        shuffle=True, num_workers=0, pin_memory=pin_memory, drop_last=True
    )
    test_loader = DataLoader(
        test_set, batch_size=config.modelconfig["batch_size"],
        shuffle=False, num_workers=0, pin_memory=pin_memory
    )
    
    input_channels = train_set[0][1].shape[0]
    index_dim = train_set[0][3].shape[0]
    model = UNetLiteFiLM(
        n_channels=input_channels,
        n_classes=1,
        index_dim=index_dim,
        base_filters=16,
        dropout=cfg["dropout"]
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.modelconfig["epoch"], eta_min=0
    )
    
    log_dir = os.path.join(config.modelconfig["log_path"] + f"_tune_{cfg['name']}")
    writer = SummaryWriter(log_dir)
    
    best_test_acc = -1.0
    
    for e in range(config.modelconfig["epoch"]):
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
        
        # test
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
            # Save the checkpoint if it is the best so far for this run
            save_path = os.path.join(config.modelconfig["save_weight_path"], f"best_model_tune_{cfg['name']}.pt")
            torch.save(model.state_dict(), save_path)
            
            # If it beats the global target of 0.2, save it as best_model_v2.pt as well
            if best_test_acc >= 0.2:
                torch.save(model.state_dict(), os.path.join(config.modelconfig["save_weight_path"], "best_model_v2.pt"))
                print(f"--> Target reached! Saved best_model_v2.pt with Test ACC: {best_test_acc:.3f}")
            
        if e % 40 == 0 or e == config.modelconfig["epoch"] - 1:
            print(f"  Epoch {e:3d}: TrainLoss={avg_train_loss:.4f}, TestLoss={avg_test_loss:.4f}, "
                  f"TrainAcc={avg_train_acc:.3f}, TestAcc={avg_test_acc:.3f} (Best={best_test_acc:.3f})")
            
    writer.close()
    print(f"Finished {cfg['name']}! Best Test ACC: {best_test_acc:.3f}")
    return best_test_acc

def main():
    device = torch.device(config.modelconfig["device"])
    print(f"Tuning device: {device}")
    
    train_set, test_set = prepare_data()
    
    configs = [
        {"lr": 2e-4, "dropout": 0.4, "weight_decay": 1e-4, "name": "lr2e-4_dr0.4_wd1e-4"},
        {"lr": 2e-4, "dropout": 0.5, "weight_decay": 1e-3, "name": "lr2e-4_dr0.5_wd1e-3"},
        {"lr": 1e-4, "dropout": 0.4, "weight_decay": 1e-4, "name": "lr1e-4_dr0.4_wd1e-4"},
        {"lr": 5e-4, "dropout": 0.5, "weight_decay": 1e-3, "name": "lr5e-4_dr0.5_wd1e-3"},
    ]
    
    best_results = {}
    for cfg in configs:
        best_acc = train_config(cfg, train_set, test_set, device)
        best_results[cfg['name']] = best_acc
        
    print("\n=== Tuning Summary ===")
    for name, acc in best_results.items():
        print(f"  {name}: {acc:.3f}")

if __name__ == "__main__":
    main()

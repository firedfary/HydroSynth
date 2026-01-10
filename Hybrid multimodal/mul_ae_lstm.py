# mul_ae_lstm.py
import os
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import fucs  # 你现有的 cal_acc 等工具

# -------------------
# ConvLSTMCell / Encoder
# -------------------
class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(input_dim + hidden_dim, 4 * hidden_dim,
                              kernel_size=kernel_size, padding=padding)

    def forward(self, x, h, c):
        # x: [B, C, H, W], h,c: [B, hidden, H, W]
        combined = torch.cat([x, h], dim=1)
        conv_out = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(conv_out, conv_out.size(1) // 4, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c + i * g
        # clamp to avoid explosion on CPU/GPU
        c_next = torch.clamp(c_next, -1e2, 1e2)
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

class ConvLSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            self.layers.append(ConvLSTMCell(in_dim, hidden_dim))

    def forward(self, x_seq):
        # x_seq: [B, T, C, H, W]
        B, T, C, H, W = x_seq.shape
        device = x_seq.device
        h = [torch.zeros(B, layer.conv.out_channels // 4, H, W, device=device)
             for layer in self.layers]
        c = [torch.zeros_like(hh) for hh in h]

        for t in range(T):
            x = x_seq[:, t]
            for i, cell in enumerate(self.layers):
                h[i], c[i] = cell(x, h[i], c[i])
                x = h[i]
        return h[-1]  # [B, hidden_dim, H, W]

# -------------------
# UNet-like decoder (light)
# -------------------
class UNetDecoder(nn.Module):
    def __init__(self, in_channels, mid_channels=64):
        super().__init__()
        # small decoder: two transpose upsample steps then interpolate to final target
        self.up1 = nn.ConvTranspose2d(in_channels, mid_channels, kernel_size=4, stride=2, padding=1)
        self.gn1 = nn.GroupNorm(8, mid_channels)
        self.up2 = nn.ConvTranspose2d(mid_channels, mid_channels//2, kernel_size=4, stride=2, padding=1)
        self.gn2 = nn.GroupNorm(8, mid_channels//2)
        self.out_reg = nn.Conv2d(mid_channels//2, 1, kernel_size=1)
        self.out_cls = nn.Conv2d(mid_channels//2, 1, kernel_size=1)

    def forward(self, x, target_size=(120,140)):
        x = F.relu(self.gn1(self.up1(x)))
        x = F.relu(self.gn2(self.up2(x)))
        reg = self.out_reg(x)
        cls = self.out_cls(x)
        # final interp to match target resolution (safe)
        reg = F.interpolate(reg, size=target_size, mode='bilinear', align_corners=False)
        cls = F.interpolate(cls, size=target_size, mode='bilinear', align_corners=False)
        return reg, cls

# -------------------
# Full model
# -------------------
class ClimatePrecipModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, num_layers=1, decoder_mid=64):
        super().__init__()
        self.encoder = ConvLSTMEncoder(input_dim, hidden_dim, num_layers)
        self.decoder = UNetDecoder(hidden_dim, mid_channels=decoder_mid)

    def forward(self, x_seq, target_size=(120,140)):
        # x_seq: [B, T, C, H, W]
        latent = self.encoder(x_seq)           # [B, hidden, H, W]
        out_reg, out_cls = self.decoder(latent, target_size=target_size)
        return out_reg, out_cls

# -------------------
# Dataset (动态切片 + mmap + 下采样)
# -------------------
class ClimateSeqDataset(Dataset):
    """
    Dynamic sequence dataset that reads from numpy memmaps to avoid large memory.
    combined: path or np array with shape [time, C_climate, Hc, Wc]
    sst: path or np array with shape [time, C_sst, Hs, Ws]  (e.g. 6 channels)
    precip: path or np array with shape [time, Ht, Wt] (target)
    T: number of input timesteps
    downsample_to: tuple for spatial size for model input (H_small, W_small) to reduce memory
    """
    def __init__(self, climate_path, sst_path, precip_path, T=3, downsample_to=(90,180), dtype=np.float32):
        # use mmap_mode to avoid full memory load
        self.climate = np.load(climate_path, mmap_mode='r')
        self.sst = np.load(sst_path, mmap_mode='r')
        self.precip = np.load(precip_path, mmap_mode='r')
        self.T = T
        self.downsample_to = downsample_to
        self.dtype = dtype

        # Basic checks
        n_climate = self.climate.shape[0]
        n_sst = self.sst.shape[0]
        n_precip = self.precip.shape[0]
        assert n_climate == n_sst == n_precip, "time dimension mismatch among climate/sst/precip"

        self.length = n_climate - T

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # produce one sample: x_seq (T, C_total, Hs, Ws), y (1, Ht, Wt)
        # NOTE: we will convert to torch tensors and downsample with torch ops (on CPU)
        t0 = idx
        t1 = idx + self.T

        # slices from mmap arrays -> numpy arrays view, cast to float32
        climate_seq = np.asarray(self.climate[t0:t1]).astype(self.dtype)   # [T, Cc, Hc, Wc]
        sst_seq = np.asarray(self.sst[t0:t1]).astype(self.dtype)           # [T, Cs, Hs, Ws]
        precip = np.asarray(self.precip[t1]).astype(self.dtype)            # [Ht, Wt]

        # convert to torch tensors (CPU)
        climate_t = torch.from_numpy(climate_seq)   # [T, Cc, Hc, Wc]
        sst_t = torch.from_numpy(sst_seq)           # [T, Cs, Hs, Ws]

        # If sst spatial shape differs from climate, resize sst for concatenation
        # First, bring channel dimension contiguous: (T, C, H, W) -> (T*C, 1, H, W) -> reshape later
        # Simpler: for each timestep, resize to downsample_to and stack
        target_H, target_W = self.downsample_to
        Tn, Cs, Hs, Ws = sst_t.shape
        Tn2, Cc, Hc, Wc = climate_t.shape
        # We'll downsample both climate and sst to (target_H, target_W)
        climate_ds = F.interpolate(climate_t, size=(target_H, target_W), mode='bilinear', align_corners=False)
        sst_ds = F.interpolate(sst_t, size=(target_H, target_W), mode='bilinear', align_corners=False)

        # concatenate channels along channel dim for each timestep
        # climate_ds: [T, Cc, Hs, Ws], sst_ds: [T, Cs, Hs, Ws]
        x_seq = torch.cat([climate_ds, sst_ds], dim=1)  # [T, Cc+Cs, H, W]

        # reorder to [T, C, H, W] already; model expects batch shape [B, T, C, H, W]
        # target precipitation (keep NaNs if any). convert to torch
        y = torch.from_numpy(precip)  # [Ht, Wt] (target resolution, do not downsample here)
        # return x_seq (T,C,H,W) and y (H_target, W_target)
        return x_seq, y

# -------------------
# Losses (masked)
# -------------------
def masked_mse(pred, target, mask):
    valid = ~mask
    if valid.sum() == 0:
        return torch.zeros((), device=pred.device, requires_grad=True)
    diff = (pred - target)**2
    return diff[valid].mean()

def masked_bce_logits(logits, label, mask):
    valid = ~mask
    if valid.sum() == 0:
        return torch.zeros((), device=logits.device, requires_grad=True)
    bce = F.binary_cross_entropy_with_logits(logits, label, reduction='none')
    return bce[valid].mean()

def masked_pearson_loss(pred, target, mask, eps=1e-8):
    valid = ~mask
    if valid.sum() < 2:
        return torch.zeros((), device=pred.device, requires_grad=True)
    p = pred[valid]
    t = target[valid]
    p = p - p.mean()
    t = t - t.mean()
    denom = (p.std(unbiased=False) * t.std(unbiased=False) + eps)
    rho = (p * t).mean() / denom
    return 1.0 - rho

def combined_masked_loss(out_reg, out_logits, target, mask,
                         w_reg=1.0, w_cls=1.0, w_corr=0.1, threshold=0.5):
    target_label = (target > threshold).float()
    L_reg = masked_mse(out_reg, target, mask) * w_reg
    L_cls = masked_bce_logits(out_logits, target_label, mask) * w_cls
    L_corr = masked_pearson_loss(out_reg, target, mask) * w_corr
    return L_reg + L_cls + L_corr

# -------------------
# Trainer utils
# -------------------
def train_one_epoch(model, loader, optimizer, device, scaler=None, epoch=0, target_size=(120,140)):
    model.train()
    losses, accs = [], []
    pbar = tqdm(enumerate(loader), total=len(loader), desc=f"Train E{epoch+1}")
    for i, (x_seq, y) in pbar:
        # x_seq: [B, T, C, H, W] after collate?  NOTE: DataLoader will stack T-first dimension -> shape [B, T, C, H, W]
        # but our dataset returns (T,C,H,W), so collate makes [B, T, C, H, W] as desired
        x_seq = x_seq.to(device=device, dtype=torch.float32, non_blocking=True)
        # reorder to [B, T, C, H, W] is already correct
        y = y.to(device=device, dtype=torch.float32, non_blocking=True)
        y = y.unsqueeze(1)  # [B,1,Ht,Wt]
        mask = torch.isnan(y)

        # optionally replace NaN with zero in inputs (we already downsampled)
        x_seq = torch.nan_to_num(x_seq, nan=0.0)

        optimizer.zero_grad()
        if device.type == 'cuda' and scaler is not None:
            with torch.amp.autocast(device_type='cuda'):
                out_reg, out_cls = model(x_seq, target_size=target_size)
                loss = combined_masked_loss(out_reg, out_cls, y, mask)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            out_reg, out_cls = model(x_seq, target_size=target_size)
            loss = combined_masked_loss(out_reg, out_cls, y, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # compute acc (use your fucs.cal_acc that expects 2D rasters scaled by 100 as earlier)
        out_reg_masked = out_reg.clone()
        out_reg_masked[mask] = float('nan')
        try:
            Acc = fucs.cal_acc(out_reg_masked[:,0,:,:]*100, y[:,0,:,:]*100).mean()
        except Exception:
            Acc = float('nan')

        losses.append(loss.item())
        accs.append(float(Acc) if not torch.isnan(torch.tensor(Acc)) else float('nan'))
        pbar.set_postfix(loss=loss.item(), acc=float(Acc) if not math.isnan(float(Acc)) else -1.0)
    return np.nanmean(losses), np.nanmean(accs)

def validate_one_epoch(model, loader, device, epoch=0, target_size=(120,140)):
    model.eval()
    losses, accs = [], []
    with torch.no_grad():
        pbar = tqdm(enumerate(loader), total=len(loader), desc=f"Val   E{epoch+1}")
        for i, (x_seq, y) in pbar:
            x_seq = x_seq.to(device=device, dtype=torch.float32, non_blocking=True)
            y = y.to(device=device, dtype=torch.float32, non_blocking=True)
            y = y.unsqueeze(1)
            mask = torch.isnan(y)
            x_seq = torch.nan_to_num(x_seq, nan=0.0)

            out_reg, out_cls = model(x_seq, target_size=target_size)
            loss = combined_masked_loss(out_reg, out_cls, y, mask)
            out_reg_masked = out_reg.clone()
            out_reg_masked[mask] = float('nan')
            try:
                Acc = fucs.cal_acc(out_reg_masked[:,0,:,:]*100, y[:,0,:,:]*100).mean()
            except Exception:
                Acc = float('nan')

            losses.append(loss.item())
            accs.append(float(Acc) if not math.isnan(float(Acc)) else float('nan'))
            pbar.set_postfix(loss=loss.item(), acc=float(Acc) if not math.isnan(float(Acc)) else -1.0)
    return np.nanmean(losses), np.nanmean(accs)

# -------------------
# Main
# -------------------
def main():
    torch.backends.cudnn.benchmark = True

    # -------------- config --------------
    config = {
        'climate_path': "E:/D1/diffusion/my_models/mulNet_data/lr.npy",   # [time, 10, Hc, Wc]
        'sst_path': "E:/D1/diffusion/my_models/mulNet_data/sst_3chan.npy",# [time, Cs, Hs, Ws]
        'precip_path': "E:/D1/diffusion/my_models/mulNet_data/hr.npy",   # [time, Ht, Wt]
        'T': 3,
        # downsample input spatially to reduce memory (HxW). Choose tradeoff between resolution & speed.
        'downsample_to': (90, 180),
        'batch_size': 16,
        'epochs': 200,
        'lr': 1e-4,
        'weight_decay': 1e-4,
        'hidden_dim': 32,
        'num_layers': 1,
        'device': torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
        'num_workers': 2,
        'target_size': (120, 140),  # output resolution
    }

    device = config['device']
    print("Device:", device)

    # -------------- dataset & dataloader --------------
    ds = ClimateSeqDataset(config['climate_path'], config['sst_path'], config['precip_path'],
                           T=config['T'], downsample_to=config['downsample_to'])
    n_total = len(ds)
    n_val = 70
    n_train = n_total - n_val
    print("Total samples:", n_total, "Train:", n_train, "Val:", n_val)

    # split indices
    train_set = torch.utils.data.Subset(ds, list(range(0, n_train)))
    val_set = torch.utils.data.Subset(ds, list(range(n_train, n_total)))

    train_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True,
                              num_workers=config['num_workers'], pin_memory=(device.type=='cuda'))
    val_loader = DataLoader(val_set, batch_size=config['batch_size'], shuffle=False,
                            num_workers=max(0, config['num_workers']//2), pin_memory=(device.type=='cuda'))

    # -------------- model --------------
    in_channels = ds.climate.shape[1] + ds.sst.shape[1]  # climate channels + sst channels
    model = ClimatePrecipModel(input_dim=in_channels, hidden_dim=config['hidden_dim'],
                               num_layers=config['num_layers']).to(device)
    print(model)
    # def overfit_test(model, X, Y, device, n_steps=200, lr=1e-3):
    #     model.train()
    #     optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)
    #     Xs = X[:8].to(device)          # take 8 samples
    #     Ys = Y[:8].to(device).unsqueeze(1)
    #     for it in range(n_steps):
    #         optim.zero_grad()
    #         out_reg, _ = model(Xs)     # temporarily ignore classifier branch
    #         loss = F.mse_loss(out_reg[~torch.isnan(Ys)], Ys[~torch.isnan(Ys)]) if (~torch.isnan(Ys)).any() else torch.tensor(0., device=device)
    #         loss.backward()
    #         optim.step()
    #         if (it % 10) == 0:
    #             print(f"overfit step {it}: loss={loss.item():.6f}")
    #     return
    # overfit_test(model, X, Y, device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'], eta_min=1e-6)

    scaler = torch.amp.GradScaler() if device.type == 'cuda' else None

    writer = SummaryWriter(log_dir='./runs_mul_ae_lstm_opt')

    best_val = float('inf')
    for epoch in range(config['epochs']):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device, scaler=scaler,
                                                epoch=epoch, target_size=config['target_size'])
        if epoch % 10 == 0:
            val_loss, val_acc = validate_one_epoch(model, val_loader, device, epoch=epoch, target_size=config['target_size'])
            print(f"[Epoch {epoch+1}/{config['epochs']}]  ValLoss={val_loss:.4f} ValAcc={val_acc:.4f}")
        scheduler.step()
        elapsed = time.time() - t0
        print(f"[Epoch {epoch+1}/{config['epochs']}] time {elapsed:.1f}s TrainLoss={train_loss:.4f} TrainAcc={train_acc:.4f}")

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Acc/train', train_acc, epoch)
        writer.add_scalar('Acc/val', val_acc, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

        if val_loss < best_val:
            best_val = val_loss
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'val_loss': val_loss}, 'best_model.pth')
            print("Saved best model")

    writer.close()

if __name__ == '__main__':
    main()

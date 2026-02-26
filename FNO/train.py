import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

import model1

# Ensure project root is on sys.path so absolute imports like 'HydroSynth' work
_proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_proj_root = os.path.normpath(_proj_root)
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

from HydroSynth import config
from HydroSynth.utils import utils

config.enable_auto_create_folders()


def compute_pcs_from_sst(sst, n_pcs=3, window=1, step=1):
    """
    Compute EOF PCs from SST.
    Expected input shape: [B, T, H, W], output PCs: [T, B, n_pcs].
    """
    print("Loaded SST shape:", sst.shape)
    B, T, H, W = sst.shape
    pcss = []
    eofs = []
    variance = []
    for t in range(T):
        x = sst[:, t].reshape(B, -1)
        x[~np.isfinite(x)] = 0.0
        pca = PCA(n_components=n_pcs)
        pcs = pca.fit_transform(x)  # [B, n_pcs]
        pcs = (pcs - pcs.mean(0, keepdims=True)) / (pcs.std(0, keepdims=True) + 1e-8)
        eof_patterns = pca.components_.reshape(n_pcs, H, W)
        pcss.append(pcs)
        eofs.append(eof_patterns)
        variance.append(pca.explained_variance_ratio_.sum())

    print(f"PCA done. Explained variance={np.mean(variance):.3f}")
    pcs = np.stack(pcss, axis=0)  # [T, B, n_pcs]
    eof_patterns = np.stack(eofs, axis=0)  # [T, n_pcs, H, W]
    return pcs.astype(np.float32), eof_patterns.astype(np.float32)


def prepare_data():
    """Load target, condition and EOF PCs. Return TensorDatasets."""
    target_file = os.path.join(config.modelconfig["hr_path"], "hr_data1.npy")
    target = np.load(target_file).astype(np.float32)  # [B,T,H,W]
    target = np.expand_dims(target, 1)  # [B,1,T,H,W]
    target_t = torch.from_numpy(target)
    mask_t = torch.isnan(target_t)

    cond_file = os.path.join(config.modelconfig["lr_path"], "lr_data1.npy")
    cond = np.load(cond_file).astype(np.float32)  # [B,C,T,H,W]
    cond_t = torch.from_numpy(cond)

    sst = np.load(config.modelconfig["sst_file"])  # [B,T,H,W]
    n_pcs = int(config.modelconfig["n_pcs"])
    window = int(config.modelconfig["pc_window"])
    step = int(config.modelconfig["pc_step"])
    pcs, _ = compute_pcs_from_sst(sst, n_pcs=n_pcs, window=window, step=step)
    pcs_t = torch.from_numpy(pcs).permute(1, 0, 2)  # [B,T,n_pcs]

    min_t = min(target_t.shape[0], cond_t.shape[0], pcs_t.shape[0])
    target_t = target_t[:min_t]
    mask_t = mask_t[:min_t]
    cond_t = cond_t[:min_t]
    pcs_t = pcs_t[:min_t]

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


def build_model(train_set, device):
    cond_tensor = train_set.tensors[1]
    pcs_tensor = train_set.tensors[3]
    in_ch = int(cond_tensor.shape[1])
    cond_time = int(cond_tensor.shape[2])
    pc_dim = int(pcs_tensor.shape[-1])

    film_channels = config.modelconfig.get("film_channels")
    if film_channels is not None:
        film_channels = [int(v) for v in film_channels]

    fno_modes = config.modelconfig.get("fno_modes", (12, 12))
    if isinstance(fno_modes, list):
        fno_modes = tuple(fno_modes)

    model = model1.SpatioTemporalCorrector(
        in_ch=in_ch,
        cond_time=cond_time,
        sst_m=cond_time,
        pc_dim=pc_dim,
        base_channels=int(config.modelconfig.get("base_channels", 24)),
        film_channels=film_channels,
        use_fno=bool(config.modelconfig.get("use_fno", False)),
        temporal_mode=config.modelconfig.get("temporal_mode", "conv"),
        use_convlstm=bool(config.modelconfig.get("use_convlstm", False)),
        use_lead_adapter=bool(config.modelconfig.get("use_lead_adapter", False)),
        index_hidden=int(config.modelconfig.get("index_hidden", 128)),
        fno_modes=fno_modes,
        transformer_layers=int(config.modelconfig.get("transformer_layers", 1)),
        n_experts=int(config.modelconfig.get("n_experts", 2)),
    ).to(device)
    return model


def maybe_load_weights(model, device):
    weight_name = config.modelconfig.get("train_load_weight")
    if not weight_name:
        return

    load_dir = config.modelconfig.get("save_weight_dir", config.modelconfig["save_weight_path"])
    ckpt_path = os.path.join(load_dir, weight_name)
    ckpt = torch.load(ckpt_path, map_location=device)

    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        elif "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt.state_dict()

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(
        f"Loaded checkpoint: {ckpt_path}\n"
        f"Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}"
    )


def compute_loss(pred, target, mask, crps_weight, crps_samples, nll_weight, mse_weight):
    # Compute losses in FP32 for better numerical stability when AMP is enabled.
    pred32 = {k: v.float() for k, v in pred.items()}
    target32 = target.float()
    nll = model1.zero_inflated_gamma_nll(pred32, target32)
    valid = ~mask
    if valid.any():
        mse = F.mse_loss(pred32["delta"][valid], target32[valid])
    else:
        mse = target32.new_tensor(0.0)
    if crps_weight > 0:
        crps = model1.approx_crps_by_sampling(pred32, target32, n_samples=crps_samples)
    else:
        crps = target32.new_tensor(0.0)
    loss = nll_weight * nll + crps_weight * crps + mse_weight * mse
    return loss, nll, crps, mse


def train():
    device = config.modelconfig["device"]
    train_set, test_set = prepare_data()
    model = build_model(train_set, device)
    maybe_load_weights(model, device)

    batch_size = int(config.modelconfig["batch_size"])
    epochs = int(config.modelconfig["epoch"])
    lr = float(config.modelconfig.get("lr", 1e-4))
    weight_decay = float(config.modelconfig.get("weight_decay", 1e-5))
    grad_clip = float(config.modelconfig.get("grad_clip", 2.0))
    save_every = int(config.modelconfig.get("save_every", 5))
    early_stop_patience = int(config.modelconfig.get("early_stop_patience", 30))
    early_stop_min_delta = float(config.modelconfig.get("early_stop_min_delta", 1e-4))

    nll_weight = float(config.modelconfig.get("nll_weight", 1.0))
    mse_weight = float(config.modelconfig.get("mse_weight", 0.5))
    train_crps_weight = float(config.modelconfig.get("crps_weight", 0.0))
    eval_crps_weight = float(config.modelconfig.get("eval_crps_weight", 0.0))
    crps_samples = int(config.modelconfig.get("crps_samples", 4))

    # GTX 16xx often has poor AMP stability for this workload; keep AMP opt-in.
    use_amp = bool(config.modelconfig.get("use_amp", False)) and str(device).startswith("cuda")
    if use_amp and torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        if major < 8:
            print(
                f"AMP disabled automatically on this GPU (compute capability {major}.{minor}) "
                "for numerical stability. Set use_amp=False explicitly in config."
            )
            use_amp = False
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True
    )
    writer = SummaryWriter(log_dir=config.modelconfig["log_path"])

    print(
        "Train setup:",
        {
            "device": str(device),
            "batch_size": batch_size,
            "epochs": epochs,
            "use_amp": use_amp,
            "train_crps_weight": train_crps_weight,
            "eval_crps_weight": eval_crps_weight,
            "crps_samples": crps_samples,
            "early_stop_patience": early_stop_patience,
        },
    )

    best_test_loss = float("inf")
    early_stop_counter = 0

    for e in range(epochs):
        model.train()
        train_loss = []
        train_acc = {i: [] for i in range(6)}

        for target, cond, mask, pcs in tqdm.tqdm(train_loader, desc=f"Epoch {e}"):
            target = target.to(device, non_blocking=True)
            cond = cond.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            pcs = pcs.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                pred = model(cond, sst_pcs=pcs)
                loss, nll, crps, mse = compute_loss(
                    pred=pred,
                    target=target,
                    mask=mask,
                    crps_weight=train_crps_weight,
                    crps_samples=crps_samples,
                    nll_weight=nll_weight,
                    mse_weight=mse_weight,
                )
            if any((~torch.isfinite(v)).any().item() for v in pred.values()) or (not torch.isfinite(loss).item()):
                raise FloatingPointError(
                    "Non-finite values detected in training forward pass. "
                    "Please set config.modelconfig['use_amp'] = False."
                )

            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()

            train_loss.append(loss.item())

            pred_mean = pred["delta"] + (1.0 - pred["p0"]) * pred["alpha"] * pred["beta"]
            for i in range(pred_mean.shape[2]):
                acc = utils.cal_acc(pred_mean[:, 0, i], target[:, 0, i]).mean()
                train_acc[i].append(acc.item())

        for i in range(6):
            writer.add_scalar(f"Acc/train_t{i}", np.mean(train_acc[i]), e)
        writer.add_scalar("Loss/train", np.mean(train_loss), e)
        print(f"Epoch {e} train loss: {np.mean(train_loss):.4f}")

        model.eval()
        test_loss = []
        test_acc = {i: [] for i in range(6)}
        with torch.no_grad():
            for target, cond, mask, pcs in test_loader:
                target = target.to(device, non_blocking=True)
                cond = cond.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)
                pcs = pcs.to(device, non_blocking=True)

                with torch.cuda.amp.autocast(enabled=use_amp):
                    pred = model(cond, sst_pcs=pcs)
                    loss, nll, crps, mse = compute_loss(
                        pred=pred,
                        target=target,
                        mask=mask,
                        crps_weight=eval_crps_weight,
                        crps_samples=crps_samples,
                        nll_weight=nll_weight,
                        mse_weight=mse_weight,
                    )
                if any((~torch.isfinite(v)).any().item() for v in pred.values()) or (not torch.isfinite(loss).item()):
                    raise FloatingPointError(
                        "Non-finite values detected in validation forward pass. "
                        "Please set config.modelconfig['use_amp'] = False."
                    )

                test_loss.append(loss.item())
                pred_mean = pred["delta"] + (1.0 - pred["p0"]) * pred["alpha"] * pred["beta"]
                for i in range(pred_mean.shape[2]):
                    acc = utils.cal_acc(pred_mean[:, 0, i], target[:, 0, i]).mean()
                    test_acc[i].append(acc.item())

        for i in range(6):
            writer.add_scalar(f"Acc/test_t{i}", np.mean(test_acc[i]), e)
        epoch_test_loss = np.mean(test_loss)
        writer.add_scalar("Loss/test", epoch_test_loss, e)
        print(f"Epoch {e} test loss: {epoch_test_loss:.4f}")

        if e % save_every == 0:
            save_path = os.path.join(config.modelconfig["save_weight_path"], f"epoch_{e}.pt")
            torch.save(model.state_dict(), save_path)

        if epoch_test_loss + early_stop_min_delta < best_test_loss:
            best_test_loss = epoch_test_loss
            early_stop_counter = 0
            best_path = os.path.join(config.modelconfig["save_weight_path"], "best.pt")
            torch.save(model.state_dict(), best_path)
        else:
            early_stop_counter += 1
            if early_stop_counter >= early_stop_patience:
                print(f"Early stopping at epoch {e}: best test loss = {best_test_loss:.6f}")
                break

    writer.close()


if __name__ == "__main__":
    train()

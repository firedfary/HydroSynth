import os
import sys
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader
import tqdm

# Ensure repo parent is on sys.path so absolute imports like 'HydroSynth' work.
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_repo_root = os.path.normpath(_repo_root)
_repo_parent = os.path.dirname(_repo_root)
if _repo_parent not in sys.path:
    sys.path.insert(0, _repo_parent)

from HydroSynth import config
try:
    from HydroSynth.FNO import train
except Exception:
    import train  # fallback for direct script execution from HydroSynth/FNO


def _resolve_ckpt_path() -> str:
    cfg = config.modelconfig
    if cfg.get("infer_ckpt_path"):
        return cfg["infer_ckpt_path"]

    ckpt_name = cfg.get("eval_load_weight")
    if not ckpt_name:
        return ""

    if os.path.isabs(ckpt_name):
        return ckpt_name
    return os.path.join(cfg["save_weight_path"], ckpt_name)


def _load_weights(model: torch.nn.Module, device: torch.device) -> None:
    ckpt_path = r"D:\workplace\conv_data\weight_t0\run_20260321_172941\epoch_130.pt"
    if not ckpt_path:
        print("No checkpoint specified; running inference with current model weights.")
        return
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict):
        state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    else:
        state_dict = ckpt.state_dict()
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(
        f"Loaded checkpoint: {ckpt_path}\n"
        f"Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}"
    )


def build_full_loader(data: Dict[str, np.ndarray], device: torch.device) -> DataLoader:
    n = len(data["init_dates"])
    indices = np.arange(n, dtype=np.int64)
    ds = train.Hydro6LeadDataset(data, indices)

    batch_size = int(config.modelconfig.get("batch_size", 2))
    num_workers = int(config.modelconfig.get("num_workers", 0))
    pin_memory = str(device).startswith("cuda")

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )


def run_inference() -> str:
    device = config.modelconfig["device"]
    # Log raw input lengths to help verify alignment.
    lr_path = config.modelconfig["lr_path"]
    cond_path = os.path.join(lr_path, "cond.npy")
    anomaly_path = os.path.join(lr_path, "anomaly.npy")
    sst_path = config.modelconfig["sst_file"]
    try:
        cond_raw = np.load(cond_path, mmap_mode="r")
        ec_anomaly = np.load(anomaly_path, mmap_mode="r")
        sst_raw = np.load(sst_path, mmap_mode="r")
        print(
            "Raw lengths:",
            {
                "cond.npy": cond_raw.shape[0],
                "anomaly.npy": ec_anomaly.shape[0],
                "sst_file": sst_raw.shape[0],
                "cond_path": cond_path,
                "anomaly_path": anomaly_path,
                "sst_path": sst_path,
            },
        )
    except Exception as e:
        print(f"Warning: failed to read raw input lengths: {e}")

    data = train.prepare_data()
    loader = build_full_loader(data, device=device)
    print(f"Prepared sample count: {len(data['init_dates'])}")

    model = train.build_model(data, device=device)
    _load_weights(model, device=device)
    model.eval()
    autoregressive = bool(config.modelconfig.get("autoregressive", False))
    prev_pred_init = str(config.modelconfig.get("prev_pred_init", "ec_base"))

    preds: List[np.ndarray] = []
    with torch.no_grad():
        for cond, ec_base, _, _, sst_pcs in tqdm.tqdm(loader, desc="Inference"):
            cond = cond.to(device, non_blocking=True)
            ec_base = ec_base.to(device, non_blocking=True)
            sst_pcs = sst_pcs.to(device, non_blocking=True)

            if autoregressive:
                pred = train.autoregressive_rollout(
                    model=model,
                    cond=cond,
                    ec_base=ec_base,
                    sst_pcs=sst_pcs,
                    target=None,
                    teacher_forcing_ratio=0.0,
                    detach_rollout=True,
                    prev_pred_init=prev_pred_init,
                )
            else:
                pred = model(cond, ec_base=ec_base, sst_pcs=sst_pcs)
            preds.append(pred.detach().cpu().numpy())

    all_pred = np.concatenate(preds, axis=0)
    if all_pred.shape[0] != len(data["init_dates"]):
        raise RuntimeError(
            f"Inference length mismatch: preds={all_pred.shape[0]} vs init_dates={len(data['init_dates'])}"
        )

    save_dir = config.modelconfig["picture_save_path"]
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "inference_all.npy")
    np.save(save_path, all_pred)
    print(f"Saved inference results to: {save_path}")
    return save_path


if __name__ == "__main__":
    run_inference()

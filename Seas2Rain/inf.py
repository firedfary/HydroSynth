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
    from HydroSynth.Seas2Rain import train
except Exception:
    import train  # fallback for direct script execution from HydroSynth/Seas2Rain


def _candidate_ckpt_paths() -> list[str]:
    cfg = config.modelconfig
    candidates = []
    for key in ("infer_ckpt_path", "eval_load_weight", "train_load_weight"):
        val = cfg.get(key)
        if not val:
            continue
        if os.path.isabs(val):
            candidates.append(val)
        else:
            candidates.append(os.path.join(cfg["save_weight_path"], val))

    # Fallback: newest .pt/.pth in save_weight_path
    try:
        if os.path.isdir(cfg["save_weight_path"]):
            files = [
                os.path.join(cfg["save_weight_path"], f)
                for f in os.listdir(cfg["save_weight_path"])
                if f.endswith((".pt", ".pth"))
            ]
            files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            candidates.extend(files)
    except Exception:
        pass

    # de-dup, keep order
    seen = set()
    uniq = []
    for p in candidates:
        if p and p not in seen and os.path.exists(p):
            uniq.append(p)
            seen.add(p)
    return uniq


def _load_weights(model: torch.nn.Module, device: torch.device) -> None:
    candidates = ['D:/workplace/conv_data/weight_t0/run_20260402_145640/epoch_665.pt']
    if not candidates:
        print("No checkpoint found; running inference with current model weights.")
        return

    last_err = None
    for ckpt_path in candidates:
        try:
            ckpt = torch.load(ckpt_path, map_location=device)
            if isinstance(ckpt, dict):
                state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
            else:
                state_dict = ckpt.state_dict()
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            print(
                f"Loaded checkpoint: {ckpt_path}\nMissing keys: {len(missing)}, Unexpected keys: {len(unexpected)}"
            )
            return
        except Exception as e:
            last_err = e
            print(f"Failed to load checkpoint {ckpt_path}: {e}")

    raise RuntimeError(
        "All candidate checkpoints failed to load. "
        "Please set config.modelconfig['infer_ckpt_path'] to a valid .pt/.pth file."
    ) from last_err


def build_full_loader(data: Dict[str, np.ndarray], device: torch.device) -> DataLoader:
    n = len(data["init_dates"])
    indices = np.arange(n, dtype=np.int64)
    ds = train.Seas2RainDataset(data, indices)

    batch_size = int(config.modelconfig.get("batch_size", 4))
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

    data = train.prepare_data()
    loader = build_full_loader(data, device=device)
    print(f"Prepared sample count: {len(data['init_dates'])}")

    model = train.build_model(device=device)
    _load_weights(model, device=device)
    model.eval()

    autoregressive = bool(config.modelconfig.get("autoregressive", True))
    prev_pred_init = str(config.modelconfig.get("prev_pred_init", "ec_base"))

    preds: List[np.ndarray] = []
    with torch.no_grad():
        for cond, seas_anom, ec_base, _, _, sst_hist in tqdm.tqdm(loader, desc="Inference"):
            cond = cond.to(device, non_blocking=True)
            seas_anom = seas_anom.to(device, non_blocking=True)
            ec_base = ec_base.to(device, non_blocking=True)
            sst_hist = sst_hist.to(device, non_blocking=True)

            if autoregressive:
                pred = train.autoregressive_rollout(
                    net=model,
                    cond=cond,
                    seas_anom=seas_anom,
                    ec_base=ec_base,
                    sst_hist=sst_hist,
                    target=None,
                    teacher_forcing_ratio=0.0,
                    detach_rollout=True,
                    prev_pred_init=prev_pred_init,
                )
            else:
                pred = model(cond, seas_anom=seas_anom, ec_base=ec_base, sst_hist=sst_hist)

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

    meta_path = os.path.join(save_dir, "inference_meta.npz")
    np.savez_compressed(meta_path, preds=all_pred, init_dates=data["init_dates"])

    print(f"Saved inference results to: {save_path}")
    return save_path


if __name__ == "__main__":
    run_inference()

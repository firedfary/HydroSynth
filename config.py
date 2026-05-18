import json
import os
import re
from datetime import datetime

import torch


# Default behavior: do not create timestamped run folders on import.
AUTO_CREATE_FOLDERS_ON_IMPORT = False

lr_foldr = "lr_unet"
hr_foldr = "hr_unet"
save_weight_foldr = "weight_t0"
picture_foldr = "picture"
log_foldr = "log_ind"

local_data_path = "d:/workplace/unet_data/"
colab_data_path = "/content/drive/MyDrive/my_models/my_model_data/"


def _resolve_base_path() -> str:
    if os.name == "nt":
        return local_data_path
    if os.name == "posix":
        return colab_data_path
    raise ValueError(f"Unknown OS: {os.name}")


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


base_data_path = _resolve_base_path()

lr_path = os.path.join(base_data_path, lr_foldr)
hr_path = os.path.join(base_data_path, hr_foldr)
sst_file = os.path.join(base_data_path, "sst_6chan.npy")
save_weight_path = os.path.join(base_data_path, save_weight_foldr)
picture_save_path = os.path.join(base_data_path, picture_foldr)
log_path = os.path.join(base_data_path, log_foldr)

for _path in (save_weight_path, picture_save_path, log_path):
    _ensure_dir(_path)

run_id = "default"

RUN_ID_TIME_FORMAT = "%Y%m%d_%H%M%S"

modelconfig = {
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    "batch_size": 16,
    # "T": 200,
    "channels": [32, 64, 128, 256],
    # "channel_mult": [2, 4, 4, 2],
    # "atten": [0, 1, 2, 3],
    # "num_res_block": 6,
    "dropout": 0.6,
    "save_weight_path": save_weight_path,
    "train_load_weight": None,
    "eval_load_weight": "ckptunet_1.pt",
    "picture_save_path": picture_save_path,
    "log_path": log_path,
    "lr": 0.00001,
    "epoch": 1002,
    # "multiplier": 1.0,
    # "bata_1": 0.0001,
    # "bata_T": 0.02,
    "grad_clip": 2.0,
    "lr_path": lr_path,
    "hr_path": hr_path,
    "sst_file": sst_file,
    # "cond_dim": 10,
    # "test_ratio": 0.2,
    "seed": 42,
    "n_pcs": 5,
    # "pc_window": 1,
    # "pc_step": 1,
    # "horizon": 6,
    "num_workers": 0,
    "lead_embed_dim": 8,
    "global_dim": 128,
    "weight_decay": 0.0001,
    # "grad_accum": 1,
    "save_every": 5,
    "patience": 12,
    "early_stop_min_delta": 0.0001,
    "autoregressive": False,
    "ssr_start": 1.0,
    "ssr_end": 0.0,
    "ssr_decay_epochs": 300,
    "prev_pred_init": "ec_base",
    "detach_rollout": False,
    "spade_hidden": 256,
    "enc_spade1_hidden": 64,
    "enc_spade2_hidden": 1024,
    "dec_spade_hidden": 1024,
    "lead_gate_hidden": 256,
    "lead_gate_init_bias": 4.0,
}


def _update_runtime_paths(current_run_id: str) -> None:
    modelconfig["save_weight_path"] = os.path.join(save_weight_path, f"run_{current_run_id}")
    modelconfig["picture_save_path"] = os.path.join(picture_save_path, f"run_{current_run_id}")
    modelconfig["log_path"] = os.path.join(log_path, f"run_{current_run_id}")


def _default_run_id() -> str:
    return datetime.now().strftime(RUN_ID_TIME_FORMAT)


def _sanitize_run_id(user_run_id: str) -> str:
    sanitized = re.sub(r'[<>:"/\\|?*]+', "_", user_run_id.strip())
    sanitized = sanitized.rstrip(". ")
    return sanitized


def _get_run_id_from_user() -> str:
    try:
        user_run_id = input("Enter run_id (leave blank to use timestamp): ").strip()
    except EOFError:
        user_run_id = ""

    if not user_run_id:
        return _default_run_id()

    sanitized_run_id = _sanitize_run_id(user_run_id)
    if not sanitized_run_id:
        return _default_run_id()

    if sanitized_run_id != user_run_id:
        print(f"Invalid path characters were replaced. Using run_id: {sanitized_run_id}")

    return sanitized_run_id


def enable_auto_create_folders(enable: bool = True):
    """Enable/disable runtime folders and update modelconfig paths."""
    global AUTO_CREATE_FOLDERS_ON_IMPORT, run_id
    AUTO_CREATE_FOLDERS_ON_IMPORT = enable

    if enable:
        run_id = _get_run_id_from_user()
        _update_runtime_paths(run_id)
        for _path in (
            modelconfig["save_weight_path"],
            modelconfig["picture_save_path"],
            modelconfig["log_path"],
        ):
            _ensure_dir(_path)
        auto_save_config()
    else:
        run_id = "default"
        modelconfig["save_weight_path"] = save_weight_path
        modelconfig["picture_save_path"] = picture_save_path
        modelconfig["log_path"] = log_path


def load_config(json_file_path=None, merge_mode="update"):
    """Load JSON config and merge it into modelconfig."""
    if json_file_path is None:
        print("Warning: no JSON file path provided, return current modelconfig")
        return modelconfig

    if not os.path.exists(json_file_path):
        raise FileNotFoundError(f"Config file not found: {json_file_path}")

    try:
        with open(json_file_path, "r", encoding="utf-8") as f:
            json_config = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {e}")
    except Exception as e:
        raise RuntimeError(f"Failed to read config file: {e}")

    if merge_mode == "replace":
        modelconfig.clear()
        modelconfig.update(json_config)
    elif merge_mode == "deep_update":
        def deep_update(source, update):
            for key, value in update.items():
                if isinstance(value, dict) and key in source and isinstance(source[key], dict):
                    deep_update(source[key], value)
                else:
                    source[key] = value

        deep_update(modelconfig, json_config)
    else:
        modelconfig.update(json_config)

    print(f"Loaded config file: {json_file_path}")
    print(f"Merge mode: {merge_mode}")
    return modelconfig


def save_config(config_dict=None, file_path=None, indent=4):
    """Save config dict as JSON file."""
    if config_dict is None:
        config_dict = modelconfig

    serializable_config = {}
    for key, value in config_dict.items():
        if isinstance(value, torch.device):
            serializable_config[key] = str(value)
        elif isinstance(value, (list, dict, int, float, str, bool)) or value is None:
            serializable_config[key] = value
        else:
            serializable_config[key] = str(value)

    if file_path is None:
        file_path = os.path.join(modelconfig["save_weight_path"], f"config_{run_id}.json")

    _ensure_dir(os.path.dirname(file_path))

    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(serializable_config, f, indent=indent, ensure_ascii=False)
        print(f"Config saved to: {file_path}")
        return file_path
    except Exception as e:
        raise RuntimeError(f"Failed to save config file: {e}")


def auto_save_config():
    """Auto-save config to current save_weight_path."""
    try:
        save_config()
    except Exception as e:
        print(f"Auto-save config failed: {e}")


if AUTO_CREATE_FOLDERS_ON_IMPORT:
    enable_auto_create_folders(True)

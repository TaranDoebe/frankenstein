import os, re, yaml, json
from datetime import datetime
from torchinfo import summary  
import torch

def slugify(x: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", x)

def build_run_suffix(cfg: dict) -> str:
    parts = []

    if 'learning_rate' in cfg: parts.append(f"lr{cfg['learning_rate']:.0e}".replace("-0", ""))
    if 'batch_size' in cfg: parts.append(f"bs{cfg['batch_size']}")
    if 'model_type' in cfg:
        model_type_full = cfg['model_type']
        model_map = {
            "Simple3DCNN": "m1",
            "HighAccuracy3DCNN": "m2",
            "ResNet3DCNN": "m3"
        }
        parts.append(model_map.get(model_type_full, slugify(model_type_full)))

    if 'n_splits_cv' in cfg: parts.append(f"cv{cfg['n_splits_cv']}")
    if 'timepoints_segment' in cfg and isinstance(cfg['timepoints_segment'], int): parts.append(f"t{cfg['timepoints_segment']}")

    return "_".join(parts)

def create_experiment_root(base_dir: str, cfg: dict) -> str:
    date_tag = datetime.now().strftime("%m_%d")
    daily_exp_dir = os.path.join(base_dir, date_tag)
    os.makedirs(daily_exp_dir, exist_ok=True)

    experiment_number = 1
    candidate_exp_root = os.path.join(daily_exp_dir, str(experiment_number))

    while os.path.exists(candidate_exp_root):
        experiment_number += 1
        candidate_exp_root = os.path.join(daily_exp_dir, str(experiment_number))

    os.makedirs(candidate_exp_root)
    return candidate_exp_root

def dump_config_and_model(run_dir: str, cfg: dict, model: torch.nn.Module):
    config_save_path = os.path.join(run_dir, "config_snapshot_for_run.yaml")
    with open(config_save_path, "w") as fh: yaml.safe_dump(cfg, fh)

    hparams_to_save = {}
    keys_for_hparams = [
        "model_type", "learning_rate", "batch_size", "num_epochs",
        "n_splits_cv", "random_state", 
        "timepoints_segment_for_mean", 
        "data_loading_mode" 
    ]
    for k in keys_for_hparams:
        if k in cfg: hparams_to_save[k] = cfg[k]
            
    hparams_save_path = os.path.join(run_dir, "hparams_snapshot.json")
    with open(hparams_save_path, "w") as fh: json.dump(hparams_to_save, fh, indent=2)

    model_summary_path = os.path.join(run_dir, "model_summary.txt")
    model_stats = summary(model, depth=3, verbose=0)
    arch_txt = str(model_stats) 

    with open(model_summary_path, "w") as fh: fh.write(arch_txt)

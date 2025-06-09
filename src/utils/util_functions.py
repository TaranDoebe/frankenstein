import os
import shutil
import logging
import torch
import pandas as pd
import yaml

from utils.data_preparation import (
    get_sertraline_subject_pt_paths,
    load_and_clean_clinical_data,
)
from utils.experiment_io import create_experiment_root
from analysis.plotting import plot_experiment_summary_metrics

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def init_experiment(cfg, cfg_path):
    """
    Create directory if it's not there.
    """
    root = create_experiment_root(cfg["base_output_dir"], cfg)
    cfg["exp_root"] = root
    os.makedirs(root, exist_ok=True)
    shutil.copy(cfg_path, os.path.join(root, "config_loaded.yaml"))
    return root, cfg


def init_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    lg = logging.getLogger()
    lg.setLevel(logging.INFO)
    fh = logging.FileHandler(os.path.join(log_dir, "experiment.log"))
    fh.setFormatter(fmt)
    lg.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    lg.addHandler(sh)


def select_device(device_str):
    if "cuda" in device_str and torch.cuda.is_available(): return torch.device(device_str)
    return torch.device("cpu")


def prepare_data(cfg):
    """
    This function returns the dataframe with all the patients
    """

    clinical = load_and_clean_clinical_data(
        cfg["clinical_data_path"], cfg.get("subjects_to_remove", [])
    )
    fmri_df = get_sertraline_subject_pt_paths(clinical, cfg["raw_pt_dir"])
    return fmri_df


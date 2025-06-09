import argparse
import logging

import torch
from utils.seed import set_seed

from datasets.fmri_dataset import (
    preload_timepoints
)
from utils.util_functions import (init_experiment,
                                  select_device,
                                  prepare_data,
                                  load_config,
                                  init_logging)

from cross_validation.nested_cv import run_nested_cv

torch.set_float32_matmul_precision('high')

def parse_args():
    p = argparse.ArgumentParser(description="Run fMRI nested-CV experiment")
    p.add_argument("--config", default="configs/config.yaml")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    exp_root, config  = init_experiment(load_config(args.config), args.config)
    init_logging(exp_root)
    logging.info(f"Experiment root: {exp_root}")

    device = select_device(config.get("device_selection", "cpu"))

    SEED = set_seed(config.get("random_state", 42))  # <‑- choose default or cfg
    logging.info(f"Global seed set to {SEED}")

    # ---------------- data --------------------------------------------
    all_df = prepare_data(config)

    if config["data_loading_mode"] == "mean_timepoints":
        logging.info('mean image path')
        data_tv = preload_timepoints(all_df, config["mean_image_path"], mean=True)
    else: 
        logging.info('raw path')
        data_tv = preload_timepoints(all_df, config["raw_pt_dir"], mean=False)

    #  run nested cross-validation
    outer_df = run_nested_cv(all_df, data_tv, config, device)
    logging.info('Nested Cross Validation Executed Successfully.')

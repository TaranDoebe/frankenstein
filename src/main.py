import os
import argparse
import yaml
import time
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from datasets.fmri_dataset import fMRIDataset, preload_mean_images
from utils.data_preparation import (
    load_and_clean_clinical_data,
    get_sertraline_subject_pt_paths,
    precompute_and_save_mean_images,
    create_subject_part_df
)
from utils.experiment_io import create_experiment_root
from utils.log_results import run_cv

def main(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    exp_root = create_experiment_root(config['base_output_dir'], config)
    config['base_output_dir'] = exp_root      
    print(f"Experiment root: {exp_root}")

    if "cuda" in config['device_selection'] and not torch.cuda.is_available():
        print(f"CUDA device {config['device_selection']} requested but not available. Using CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(config['device_selection'])
    print(f"✅ Using device: {device}")

    print("Step 1: Loading and preparing clinical data...")
    base_clinical_data = load_and_clean_clinical_data(config['clinical_data_path'], config['subjects_to_remove'])
    
    # Ensure 'pt_files_dir' exists in your config.yaml
    if 'pt_files_dir' not in config:
        raise ValueError("Configuration 'pt_files_dir' is missing in config.yaml")
    processed_fmri_df = get_sertraline_subject_pt_paths(base_clinical_data, config['pt_files_dir'])
    
    print(f"Found {len(processed_fmri_df)} subjects with SERTRALINE and available .pt data.")

    if processed_fmri_df.empty:
        print("No subjects found after filtering for .pt files. Exiting.")
        return

    if config.get('run_precomputation', False): # Optional: Add a flag in config to run this
        print("Step 2: Precomputing mean images from .pt files...")
        precompute_and_save_mean_images(
            processed_fmri_df,
            config['mean_image_path'],
            config['min_timepoints_for_precompute'],
            config['timepoints_segment']
        )
    else:
        print(f"Step 2: Assuming mean images are precomputed in: {config['mean_image_path']}")

    print("Step 3: Loading precomputed mean images...")
    train_val_subjects_df, test_subjects_df = train_test_split(
        processed_fmri_df,
        test_size=config['test_size_split'],
        random_state=config['random_state'],
        stratify=processed_fmri_df['w8_responder']
    )
    
    print(f"Train/Validation subjects: {len(train_val_subjects_df)}, Test subjects: {len(test_subjects_df)}")

    data_dict_train_val = preload_mean_images(train_val_subjects_df, config['mean_image_path'])
    if not data_dict_train_val:
        print("No precomputed images loaded for training/validation. Check paths and precomputation. Exiting.")
        return

    test_loader = None
    if not test_subjects_df.empty:
        data_dict_test = preload_mean_images(test_subjects_df, config['mean_image_path'])
        if data_dict_test:
            mean_df_test = create_subject_part_df(test_subjects_df, data_dict_test)
            if not mean_df_test.empty:
                test_dataset = fMRIDataset(mean_df_test, data_dict_test)
                test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4, pin_memory=True)
                print(f"Test loader created with {len(test_dataset)} samples.")
            else:
                print("Warning: Test DataFrame is empty after loading images, no test evaluation will be performed.")
        else:
            print("Warning: No precomputed images for test subjects, no test evaluation will be performed.")
    else:
        print("No subjects in test set after split. No test evaluation will be performed.")

    print("\nStep 4: Starting training and evaluation runs...")
    all_runs_metrics = []
    for run_idx in range(1, config.get('n_runs', 1) + 1):
        start_time_run = time.time()
        # The returned values are now from the inlined test evaluation
        test_acc, test_prec, test_rec, test_f1 = run_cv(
            run_idx, config, train_val_subjects_df, data_dict_train_val, test_loader, device
        )
        all_runs_metrics.append({'run': run_idx, 'acc': test_acc, 'prec': test_prec, 'rec': test_rec, 'f1': test_f1})
        end_time_run = time.time()
        print(f"Run {run_idx} finished in {(end_time_run - start_time_run)/60:.2f} minutes.")

    if config.get('n_runs', 1) > 1 and all_runs_metrics:
        print("\n--- Summary of Test Set Performance Across All Runs ---")
        # Filter out runs where metrics might be zero due to no test data
        valid_metrics = [m for m in all_runs_metrics if m['acc'] > 0 or m['f1'] > 0] # Or some other check
        if valid_metrics:
            avg_acc = np.mean([m['acc'] for m in valid_metrics])
            avg_prec = np.mean([m['prec'] for m in valid_metrics])
            avg_rec = np.mean([m['rec'] for m in valid_metrics])
            avg_f1 = np.mean([m['f1'] for m in valid_metrics])
            print(f"Average Test Accuracy: {avg_acc:.4f}")
            print(f"Average Test Precision: {avg_prec:.4f}")
            print(f"Average Test Recall: {avg_rec:.4f}")
            print(f"Average Test F1-Score: {avg_f1:.4f}")
        else:
            print("No valid test metrics to average across runs.")


    print("\n--- Script Finished ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run fMRI classification training and evaluation.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to the configuration YAML file."
    )
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Error: Config file not found at {args.config}")
    else:
        main(args.config)
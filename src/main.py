# main.py

import os
import argparse
import yaml
import time
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from torch.utils.data import DataLoader
from tqdm import tqdm # Ensure tqdm is imported

# Import scikit-learn metric functions if they are not already imported at the top of main.py
from sklearn.metrics import precision_score, recall_score, f1_score

from datasets.fmri_dataset import fMRIDataset, preload_mean_images
from models.cnn_models import Simple3DCNN, HighAccuracy3DCNN
from utils.data_preparation import (
    load_and_clean_clinical_data,
    # get_sertraline_fmri_paths, # Assuming you've updated this to get_sertraline_subject_pt_paths
    get_sertraline_subject_pt_paths,
    precompute_and_save_mean_images,
    create_subject_part_df
)
# We will no longer call evaluate_model from training_evaluation for the final test step,
# but train_fold_routine will still use it for validation during CV.
from utils.training_evaluation import train_fold_routine, evaluate_model # evaluate_model is still used by train_fold_routine
from evaluation.plotting import save_training_plots, save_test_set_results_plots

def run_cv_and_save_results(run_id, config, mean_df_train_val, data_dict_train_val, test_loader, device):
    """
    Manages the cross-validation training, evaluation, and result saving for a single run.
    """
    outdir_run = os.path.join(config['base_output_dir'], f"run_{run_id:03d}")
    os.makedirs(outdir_run, exist_ok=True)

    epoch_log_path = os.path.join(outdir_run, "training_log_per_epoch.csv")
    with open(epoch_log_path, "w") as f:
        f.write("fold,epoch,train_loss,train_acc,train_f1,val_loss,val_acc,val_prec,val_rec,val_f1\n")

    skf = StratifiedKFold(n_splits=config['n_splits_cv'], shuffle=True, random_state=config['random_state'])
    all_fold_histories = []
    best_val_f1_overall = -1
    final_model_for_test = None

    # Define criterion here as it's needed for the inlined test evaluation too
    criterion = torch.nn.BCEWithLogitsLoss()

    print(f"\n--- Starting Run {run_id} with {config['n_splits_cv']}-Fold Cross-Validation ---")

    for fold, (train_idx, val_idx) in enumerate(skf.split(mean_df_train_val, mean_df_train_val["label"]), start=1):
        print(f"\n--- Fold {fold}/{config['n_splits_cv']} ---")
        train_df_fold = mean_df_train_val.iloc[train_idx].reset_index(drop=True)
        val_df_fold = mean_df_train_val.iloc[val_idx].reset_index(drop=True)

        train_loader_fold = DataLoader( # Renamed to avoid conflict with outer scope test_loader
            fMRIDataset(train_df_fold, data_dict_train_val),
            batch_size=config['batch_size'], shuffle=True, num_workers=4, pin_memory=True
        )
        val_loader_fold = DataLoader( # Renamed
            fMRIDataset(val_df_fold, data_dict_train_val),
            batch_size=config['batch_size'], shuffle=False, num_workers=4, pin_memory=True
        )

        if config['model_type'] == "Simple3DCNN":
            model = Simple3DCNN().to(device)
        elif config['model_type'] == "HighAccuracy3DCNN":
            model = HighAccuracy3DCNN().to(device)
        else:
            raise ValueError(f"Unknown model type: {config['model_type']}")

        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
        # criterion is already defined above

        fold_history = train_fold_routine(
            model, train_loader_fold, val_loader_fold, criterion, optimizer, device,
            config['num_epochs'], config.get('max_grad_norm', 1.0)
        )
        all_fold_histories.append(fold_history)

        current_best_val_f1_fold = max(fold_history['val_f1']) if fold_history['val_f1'] else -1
        if current_best_val_f1_fold > best_val_f1_overall:
             best_val_f1_overall = current_best_val_f1_fold
             final_model_for_test = model
        if final_model_for_test is None:
            final_model_for_test = model

        with open(epoch_log_path, "a") as f:
            for e_idx in range(len(fold_history["epoch"])):
                f.write(",".join(map(str, [
                    fold, fold_history["epoch"][e_idx],
                    fold_history["train_loss"][e_idx], fold_history["train_accuracy"][e_idx], fold_history["train_f1"][e_idx],
                    fold_history["val_loss"][e_idx], fold_history["val_accuracy"][e_idx],
                    fold_history["val_precision"][e_idx], fold_history["val_recall"][e_idx], fold_history["val_f1"][e_idx]
                ])) + "\n")

    save_training_plots(all_fold_histories, run_id, outdir_run)

    # --- Inlined Test Set Evaluation ---
    test_loss_val, test_acc_val, test_prec_val, test_rec_val, test_f1_val = 0.0, 0.0, 0.0, 0.0, 0.0 # Default values

    if final_model_for_test and test_loader:
        print("\n🔍 Evaluating on True Hold-out Test Set (manual loop in main.py)...")
        final_model_for_test.eval() # Set model to evaluation mode
        
        test_total_loss = 0
        all_test_labels = []
        all_test_predictions = []

        with torch.no_grad(): # Disable gradient calculations
            for images, labels in tqdm(test_loader, desc="Test Batches", leave=False):
                images, labels = images.to(device), labels.to(device)
                
                outputs = final_model_for_test(images).squeeze(1)
                loss = criterion(outputs, labels.float()) # Calculate loss
                test_total_loss += loss.item()

                predicted_probs = torch.sigmoid(outputs)
                predicted_classes = (predicted_probs > 0.5).int() # Threshold probabilities
                
                all_test_predictions.extend(predicted_classes.cpu().numpy())
                all_test_labels.extend(labels.cpu().numpy())

        if len(test_loader) > 0 and len(all_test_labels) > 0:
            test_loss_val = test_total_loss / len(test_loader)
            test_acc_val = (np.array(all_test_predictions) == np.array(all_test_labels)).mean()
            test_prec_val = precision_score(all_test_labels, all_test_predictions, zero_division=0)
            test_rec_val = recall_score(all_test_labels, all_test_predictions, zero_division=0)
            test_f1_val = f1_score(all_test_labels, all_test_predictions, zero_division=0)
        elif len(all_test_labels) == 0:
             print("Warning: No labels found in the test set during evaluation.")
        else: # len(test_loader) == 0
             print("Warning: Test loader was empty.")


        print(f"✅ Test Set Results (Run {run_id}): Loss: {test_loss_val:.4f} | Acc: {test_acc_val:.4f} | P: {test_prec_val:.4f} | R: {test_rec_val:.4f} | F1: {test_f1_val:.4f}")

        save_test_set_results_plots(final_model_for_test, test_loader, device, run_id, outdir_run)

        master_log_path = os.path.join(config['base_output_dir'], "master_test_results.csv")
        if not os.path.exists(master_log_path):
            with open(master_log_path, "w") as f:
                f.write("run_id,test_loss,test_accuracy,test_precision,test_recall,test_f1\n")
        with open(master_log_path, "a") as f:
            f.write(f"{run_id},{test_loss_val:.4f},{test_acc_val:.4f},{test_prec_val:.4f},{test_rec_val:.4f},{test_f1_val:.4f}\n")
    else:
        if not final_model_for_test:
            print("No model available for final test evaluation.")
        if not test_loader:
            print("No test loader available for final evaluation.")
    
    print(f"✅ Run {run_id} completed. Results and plots saved to {outdir_run}")
    return test_acc_val, test_prec_val, test_rec_val, test_f1_val # Return the computed test metrics


def main(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

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
    mean_df_train_val = create_subject_part_df(train_val_subjects_df, data_dict_train_val)
    if mean_df_train_val.empty:
        print("Failed to create training/validation DataFrame from preloaded images. Exiting.")
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
        test_acc, test_prec, test_rec, test_f1 = run_cv_and_save_results(
            run_idx, config, mean_df_train_val, data_dict_train_val, test_loader, device
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
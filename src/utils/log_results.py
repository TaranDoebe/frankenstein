
import os 
import torch
from tqdm import tqdm 
import numpy as np
from torch.utils.data import DataLoader
from utils.experiment_io import build_run_suffix, dump_config_and_model
from sklearn.model_selection import train_test_split, StratifiedKFold
from datasets.fmri_dataset import fMRIDataset
from models.cnn_models import Simple3DCNN, HighAccuracy3DCNN, ResNet3DCNN
from utils.training_evaluation import train_fold_routine
from sklearn.metrics import precision_score, recall_score, f1_score
from evaluation.plotting import save_training_plots, save_test_set_results_plots
from utils.data_preparation import create_subject_part_df

def run_cv(run_id, config, train_val_subjects_df, data_dict_train_val, test_loader, device): 
    """
    Manages the cross-validation training, evaluation, and result saving for a single run.
    """
    suffix     = build_run_suffix(config)
    outdir_run = os.path.join(config['base_output_dir'],
                            f"run_{run_id:03d}_{suffix}")
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

    for fold, (train_idx, val_idx) in enumerate(skf.split(train_val_subjects_df, train_val_subjects_df["w8_responder"]), start=1):
        train_df_fold = train_val_subjects_df.iloc[train_idx].reset_index(drop=True)
        val_df_fold = train_val_subjects_df.iloc[val_idx].reset_index(drop=True)

        train_df_fold = create_subject_part_df(train_df_fold, data_dict_train_val)
        val_df_fold = create_subject_part_df(val_df_fold, data_dict_train_val)

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
        elif config['model_type'] == "ResNet3DCNN":
            model = ResNet3DCNN().to(device)
        else:
            raise ValueError(f"Unknown model type: {config['model_type']}")

        # Only for the first fold, capture the run meta-data
        if fold == 1:
            dump_config_and_model(outdir_run, config, model)

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


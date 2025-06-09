import os
import logging
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from sklearn.metrics import (precision_score, 
                             recall_score, 
                             f1_score, 
                             roc_auc_score)
from cross_validation.run_cross_validation import run_cv
from analysis.plotting import (plot_model_selection_counts, 
                               plot_auc_per_model,
                               plot_inner_vs_outer_auc, 
                               plot_hp_heatmap, 
                               plot_auc_vs_loss,
                               plot_metrics)

logger = logging.getLogger(__name__)

def save_dfs(computed_metrics, records, experiment_root, name="inner_folds"):
    computed_metrics.to_csv(os.path.join(experiment_root, f"{name}_all_epoch_metrics.csv"), index=False)
    records.to_csv(os.path.join(experiment_root,f"{name}_cv_results.csv"), index=False)

def compute_metrics(df):

    df['prob_list'] = df['patient_probabilities']
    df['label_list'] = df['patient_labels']
    df['val_loss_list'] = df['val_loss']
    df['train_loss_list'] = df['train_loss']

    # Epoch count
    n_epochs = len(df['prob_list'].iloc[0])

    # Initialize metrics
    precisions, recalls, f1s, auc_list, mean_val_loss, mean_train_loss = ([] for _ in range(6))

    for e in range(n_epochs):
        all_probs = sum(df['prob_list'].str[e].tolist(), [])
        all_labels = sum(df['label_list'].str[e].tolist(), [])
        preds = [1 if p > 0.5 else 0 for p in all_probs]
        
        precisions.append(precision_score(all_labels, preds))
        recalls.append(recall_score(all_labels, preds))
        f1s.append(f1_score(all_labels, preds))
        roc_auc = roc_auc_score(all_labels, all_probs)
        auc_list.append(roc_auc)
        mean_val_loss.append(sum(v[e] for v in df['val_loss_list']) / len(df))
        mean_train_loss.append(sum(t[e] for t in df['train_loss_list']) / len(df))

    computed_metrics_df = pd.DataFrame({
        'epoch':           range(1, n_epochs + 1),
        'precision':       precisions,
        'recall':          recalls,
        'f1_score':        f1s,
        'auc':             auc_list,
        'mean_val_loss':   mean_val_loss,
        'mean_train_loss': mean_train_loss,
        })

    return computed_metrics_df

def inner_search(cfg: dict,
                 fold_id: int,
                 outer_train_df: pd.DataFrame,
                 data_tv: dict,
                 device: torch.device):
    """
    Run the inner folds, by splitting on the outer train-val fold.
    """
    nc = cfg["nested_cv"]
    k_in = nc["inner_splits"]
    grid = nc.get("param_grid", [{}])

    skf_inner = StratifiedKFold(
        n_splits=k_in, shuffle=True, random_state=fold_id
    )

    best_hp, best_auc = {}, -np.inf
    best_computed_metrics, best_records = None, None
    for hparams_i, hp in enumerate(ParameterGrid(grid)):
        records = []
        for fold_i, (tr_idx, val_idx) in enumerate(skf_inner.split(outer_train_df, outer_train_df["w8_responder"])):

            tr_df  = outer_train_df.iloc[tr_idx]
            val_df = outer_train_df.iloc[val_idx]

            # single-split mode (val_df passed)
            history = run_cv(
                run_id=fold_i,
                config={**cfg, **hp},
                train_df=tr_df,
                data_dict_train_val=data_tv,
                device=device,
                val_df=val_df,
                fold="val",
            )

            records.append({
            **history,
            "inner_fold": fold_i})

        inner_records_df = pd.DataFrame(records)
        computed_metrics_df = compute_metrics(inner_records_df)
        param_str = "_".join(f"{key}-{hp[key]}" for key in sorted(hp))
        computed_metrics_df.to_csv(os.path.join(cfg["exp_root"], f"inner_folds_all_epoch_metrics_{param_str}.csv"), index=False)
        
        max_auc_epochs = computed_metrics_df["auc"].tolist()[-1] # take the last epoch's auc value
        if max_auc_epochs > best_auc:
            best_auc = max_auc_epochs
            best_hp = hp
            best_computed_metrics = computed_metrics_df
            best_records = inner_records_df

    logger.info(
        f"  inner-CV best auc={best_auc:.2f} with HP {best_hp}"
    )

    save_dfs(best_computed_metrics, best_records, cfg["exp_root"], name="inner_folds_best_model")

    return best_hp, best_auc 

def run_nested_cv(train_val_df: pd.DataFrame,
                  data_tv: dict,
                  cfg: dict,
                  device: torch.device) -> pd.DataFrame:
    
    """
    First split for nested cross validation.
    Then run the outer training and testing folds using `run_cv`.
    
    """

    k_out = cfg["nested_cv"]["outer_splits"]
    
    outer = StratifiedKFold(
        n_splits=k_out, shuffle=True,
        random_state=cfg.get("random_state"),
    )

    records = []

    for fold_id, (tr_idx, hold_idx) in enumerate(
            outer.split(train_val_df, train_val_df["w8_responder"]), 1):
        outer_train_df = train_val_df.iloc[tr_idx].reset_index(drop=True)
        holdout_df = train_val_df.iloc[hold_idx].reset_index(drop=True)

        logger.info(f"\n\n=== Outer fold {fold_id}/{k_out} ===")

        best_hp, inner_auc = inner_search(cfg, fold_id,
                                  outer_train_df, data_tv, device) 

        history = run_cv(
            run_id=fold_id,
            config={**cfg, **best_hp},
            train_df=outer_train_df,
            data_dict_train_val=data_tv,
            val_df=holdout_df,
            device=device,
            fold="test",
        )

        records.append({
            **history,
            "inner_auc": inner_auc, 
            **{f"hp_{k}": v for k, v in best_hp.items()},
            "outer_fold": fold_id,
        })

    # output and plots
    exp_root = cfg["exp_root"]
    print(exp_root)
    outer_df = pd.DataFrame(records)
    computed_metrics_df = compute_metrics(outer_df)
    save_dfs(computed_metrics_df, outer_df, exp_root, name="outer_folds")

    plot_metrics(computed_metrics_df, exp_root)

    # plot_model_selection_counts(outer_df, exp_root)
    # plot_auc_per_model(outer_df, exp_root)
    # plot_inner_vs_outer_auc(outer_df, exp_root)
    # plot_hp_heatmap(outer_df, exp_root)
    # plot_auc_vs_loss(outer_df, exp_root)

    return outer_df



# src/evaluation/plotting.py
import os
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, auc,
    precision_recall_curve, average_precision_score,
    confusion_matrix, ConfusionMatrixDisplay,
)
from sklearn.calibration import calibration_curve  # ★ new

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------
#  (NEW) Inner-vs-Outer scatter
# -----------------------------------------------------------------
def plot_inner_vs_outer_auc(outer_df: pd.DataFrame, outdir: Path):
    if {"inner_mean_auc", "patient_auc"}.issubset(outer_df.columns):
        x = outer_df["inner_mean_auc"]
        y = outer_df["patient_auc"].apply(lambda z: z if np.isscalar(z) else z[-1])
        plt.figure(figsize=(6,6))
        for mdl in outer_df["hp_model_type"].unique():
            mask = outer_df["hp_model_type"] == mdl
            plt.scatter(x[mask], y[mask], label=mdl, alpha=.8)
        plt.plot([0,1],[0,1],"--",color="grey")
        plt.xlabel("Inner-CV mean AUC"); plt.ylabel("Outer-fold test AUC")
        plt.title("Optimism of inner CV"); plt.legend(); plt.grid(True)
        plt.tight_layout()
        plt.savefig(outdir / "inner_vs_outer_auc.png", dpi=300)
        plt.close()

# -----------------------------------------------------------------
#  (NEW) HP grid heat-map   (learning-rate × model → mean AUC)
# -----------------------------------------------------------------
def plot_hp_heatmap(outer_df: pd.DataFrame, outdir: Path):
    req = {"hp_learning_rate", "hp_model_type", "patient_auc"}
    if not req.issubset(outer_df.columns): return
    df = outer_df.copy()
    df["last_auc"] = df["patient_auc"].apply(lambda z: z if np.isscalar(z) else z[-1])
    pivot = df.pivot_table(index="hp_learning_rate",
                           columns="hp_model_type",
                           values="last_auc",
                           aggfunc="mean")
    plt.figure(figsize=(8,4))
    im = plt.imshow(pivot, aspect="auto", cmap="viridis",
                    vmin=pivot.min().min(), vmax=pivot.max().max())
    plt.colorbar(im, label="Mean outer-fold AUC")
    plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=45, ha="right")
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.title("Grid search – outer-fold mean AUC")
    plt.tight_layout()
    plt.savefig(outdir / "hp_heatmap_auc.png", dpi=300)
    plt.close()

    # -----------------------------------------------------------------
#  (NEW) Metric correlation scatter
# -----------------------------------------------------------------
def plot_auc_vs_loss(outer_df: pd.DataFrame, outdir: Path):
    if {"test_auc", "test_loss"}.issubset(outer_df.columns):
        plt.figure(figsize=(6,6))
        for mdl in outer_df["hp_model_type"].unique():
            m = outer_df["hp_model_type"] == mdl
            plt.scatter(outer_df.loc[m,"test_loss"],
                        outer_df.loc[m,"test_auc"],
                        label=mdl, alpha=.8)
        plt.xlabel("Test BCE loss"); plt.ylabel("Test AUC")
        plt.title("AUC vs loss by outer fold"); plt.legend(); plt.grid(True)
        plt.tight_layout()
        plt.savefig(outdir / "auc_vs_loss.png", dpi=300)
        plt.close()



# ---------------------------------------------------------------------
#  TRAIN / VAL CURVES  (callable with plot=False to skip)
# ---------------------------------------------------------------------
def save_training_plots(all_histories, run_id, outdir, *, plot: bool = True):
    """Aggregate histories from K folds and save diagnostic plots."""
    if not plot:
        return                                                   # ★ skip switch

    epochs = all_histories[0]["epoch"]
    if any(len(h["epoch"]) != len(epochs) for h in all_histories):
        min_ep = min(len(h["epoch"]) for h in all_histories)
        epochs = epochs[:min_ep]
        for h in all_histories:
            for k, v in h.items():
                if isinstance(v, list) and len(v) > min_ep:
                    h[k] = v[:min_ep]

    def safe_avg(key):
        vals = [h[key] for h in all_histories if key in h]
        return np.mean(vals, axis=0) if vals else np.zeros(len(epochs))

    def safe_std(key):
        vals = [h[key] for h in all_histories if key in h]
        return np.std(vals, axis=0) if len(vals) > 1 else np.zeros(len(epochs))

    # (a) Loss curve ---------------------------------------------------
    plt.figure(figsize=(8, 6))
    tl, vl = safe_avg("train_loss"), safe_avg("val_loss")
    tls, vls = safe_std("train_loss"), safe_std("val_loss")
    plt.plot(epochs, tl, label="Train")
    plt.fill_between(epochs, tl - tls, tl + tls, alpha=0.2)
    plt.plot(epochs, vl, label="Val")
    plt.fill_between(epochs, vl - vls, vl + vls, alpha=0.2)
    plt.xlabel("Epoch"); plt.ylabel("BCE Loss")
    plt.title(f"Run {run_id}: Loss"); plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(outdir, f"run_{run_id}_loss.png"),
                dpi=300, bbox_inches="tight")
    plt.close()

    # (b) Accuracy, Precision, Recall, F1 ------------------------------
    for key, name in [("accuracy", "Accuracy (%)"),
                      ("precision", "Precision"),
                      ("recall", "Recall"),
                      ("f1", "F1")]:
        plt.figure(figsize=(8, 6))
        t = safe_avg(f"train_{key}") * (100 if key == "accuracy" else 1)
        v = safe_avg(f"val_{key}")   * (100 if key == "accuracy" else 1)
        ts, vs = safe_std(f"train_{key}"), safe_std(f"val_{key}")
        if key == "accuracy":
            ts *= 100; vs *= 100
        plt.plot(epochs, t, label="Train"); plt.fill_between(epochs, t-ts, t+ts, alpha=0.2)
        plt.plot(epochs, v, label="Val");   plt.fill_between(epochs, v-vs, v+vs, alpha=0.2)
        plt.xlabel("Epoch"); plt.ylabel(name)
        plt.title(f"Run {run_id}: {name}"); plt.legend(); plt.grid(True)
        plt.savefig(os.path.join(outdir, f"run_{run_id}_{key}.png"),
                    dpi=300, bbox_inches="tight")
        plt.close()

    # (c) LR schedule (if logged)  ------------------------------------ ★
    if "lr" in all_histories[0]:
        plt.figure(figsize=(8, 4))
        plt.plot(epochs, all_histories[0]["lr"])
        plt.title(f"Run {run_id}: Learning-Rate schedule")
        plt.xlabel("Epoch"); plt.ylabel("LR"); plt.grid(True)
        plt.savefig(os.path.join(outdir, f"run_{run_id}_lr_schedule.png"),
                    dpi=300, bbox_inches="tight")
        plt.close()

# ---------------------------------------------------------------------
#  TEST-SET ROC / PR / Calibration etc.
# ---------------------------------------------------------------------
def save_test_set_results_plots(y_true, y_score, run_id, outdir, sids=None):
    y_true, y_score = np.asarray(y_true), np.asarray(y_score)
    if len(y_true) == 0 or len(y_true) != len(y_score):
        return

    # (1) ROC curve ----------------------------------------------------
    if len(np.unique(y_true)) > 1:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc_val = auc(fpr, tpr)
        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc_val:.3f}")
        plt.plot([0, 1], [0, 1], "--", color="grey")
        plt.title(f"Run {run_id}: ROC"); plt.xlabel("FPR"); plt.ylabel("TPR")
        plt.legend(); plt.grid(True)
        plt.savefig(os.path.join(outdir, f"run_{run_id}_roc.png"),
                    dpi=300, bbox_inches="tight")
        plt.close()

    # (2) Precision-Recall --------------------------------------------
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    plt.figure(figsize=(6, 6))
    plt.plot(rec, prec, label=f"AP = {ap:.3f}")
    base = sum(y_true) / len(y_true)
    plt.hlines(base, 0, 1, linestyles="--", color="grey", label="Baseline")
    plt.title(f"Run {run_id}: PR-Curve"); plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(outdir, f"run_{run_id}_pr.png"),
                dpi=300, bbox_inches="tight")
    plt.close()

    # (3) Reliability / Calibration ────────────────────────────────── ★
    prob_true, prob_pred = calibration_curve(y_true, y_score, n_bins=10, strategy="uniform")
    plt.figure(figsize=(6, 6))
    plt.plot(prob_pred, prob_true, marker="o", label="Model")
    plt.plot([0, 1], [0, 1], "--", color="grey")
    plt.title(f"Run {run_id}: Calibration"); plt.xlabel("Predicted prob"); plt.ylabel("Observed freq")
    plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(outdir, f"run_{run_id}_calibration.png"),
                dpi=300, bbox_inches="tight")
    plt.close()

    # (4) Confusion matrix --------------------------------------------
    y_pred = (y_score >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay(cm).plot(ax=ax, colorbar=False)
    ax.set_title(f"Run {run_id}: Confusion matrix")
    plt.savefig(os.path.join(outdir, f"run_{run_id}_cm.png"),
                dpi=300, bbox_inches="tight")
    plt.close()

    # (5) Patient-level ROC & PR (unchanged) --------------------------
    if sids is None:
        return

    df = pd.DataFrame({"sid": sids, "y_true": y_true, "y_score": y_score})
    pat = df.groupby("sid").agg(y_true=("y_true", "first"),
                                y_score=("y_score", "mean"))
    yt_p, ys_p = pat["y_true"].values, pat["y_score"].values

    if len(np.unique(yt_p)) > 1:
        fpr_p, tpr_p, _ = roc_curve(yt_p, ys_p)
        auc_p = auc(fpr_p, tpr_p)
        plt.figure(figsize=(6, 6))
        plt.plot(fpr_p, tpr_p, label=f"AUC = {auc_p:.3f}")
        plt.plot([0, 1], [0, 1], "--", color="grey")
        plt.title(f"Run {run_id}: Patient ROC"); plt.xlabel("FPR"); plt.ylabel("TPR")
        plt.legend(); plt.grid(True)
        plt.savefig(os.path.join(outdir, f"run_{run_id}_pat_roc.png"),
                    dpi=300, bbox_inches="tight")
        plt.close()

        prec_p, rec_p, _ = precision_recall_curve(yt_p, ys_p)
        ap_p = average_precision_score(yt_p, ys_p)
        plt.figure(figsize=(6, 6))
        plt.plot(rec_p, prec_p, label=f"AP = {ap_p:.3f}")
        base_p = sum(yt_p) / len(yt_p)
        plt.hlines(base_p, 0, 1, linestyles="--", color="grey")
        plt.title(f"Run {run_id}: Patient PR"); plt.xlabel("Recall"); plt.ylabel("Precision")
        plt.legend(); plt.grid(True)
        plt.savefig(os.path.join(outdir, f"run_{run_id}_pat_pr.png"),
                    dpi=300, bbox_inches="tight")
        plt.close()

        cm_p = confusion_matrix(yt_p, (ys_p >= 0.5).astype(int))
        fig, ax = plt.subplots(figsize=(5, 5))
        ConfusionMatrixDisplay(cm_p).plot(ax=ax, colorbar=False)
        ax.set_title(f"Run {run_id}: Patient CM")
        plt.savefig(os.path.join(outdir, f"run_{run_id}_pat_cm.png"),
                    dpi=300, bbox_inches="tight")
        plt.close()

# ---------------------------------------------------------------------
#  SUMMARY ACROSS OUTER FOLDS
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
#  convert list→scalar helper
# ---------------------------------------------------------------------
def _collapse(x, how: str = "last"):
    """Convert list → scalar (last / mean / median)."""
    if not isinstance(x, list):
        return x
    if len(x) == 0:
        return np.nan
    if how == "last":
        return x[-1]
    if how == "mean":
        return float(np.mean(x))
    if how == "median":
        return float(np.median(x))
    raise ValueError(f"Unknown collapse rule: {how}")

# ---------------------------------------------------------------------
#  SUMMARY ACROSS OUTER FOLDS
# ---------------------------------------------------------------------
def plot_experiment_summary_metrics(metrics_df: pd.DataFrame,
                                    exp_root: str,
                                    collapse_rule: str = "last"):
    """
    Box-plot distribution of key test-set metrics across outer folds.

    collapse_rule: how to turn a list column into a scalar
                   ("last", "mean", or "median")
    """
    keep = [c for c in metrics_df.columns
            if c in ("test_auc", "test_loss")
            or c.startswith("patient_")]

    if not keep:
        logger.warning("No recognised metric columns found for summary plot.")
        return

    df = metrics_df.copy()
    for col in keep:
        df[col] = df[col].apply(lambda x: _collapse(x, collapse_rule))

    melted = df.melt(value_vars=keep, var_name="Metric", value_name="Score")
    data   = [melted[melted["Metric"] == m]["Score"].dropna().values
              for m in keep]

    plt.figure(figsize=(max(8, 1.8 * len(keep)), 6))
    plt.boxplot(data, labels=keep, whis=1.5, showmeans=True)
    plt.title("Outer-fold metric distribution")
    plt.ylabel("Score")
    plt.xticks(rotation=15, ha="right")
    plt.grid(axis="y", ls="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(exp_root, "summary_outer_fold_metrics.png"),
                dpi=300, bbox_inches="tight")
    plt.close()

# ---------------------------------------------------------------------
#  MODEL-SELECTION VISUALS
# ---------------------------------------------------------------------
def plot_model_selection_counts(outer_df: pd.DataFrame, outdir: Path):
    counts = outer_df["hp_model_type"].value_counts()
    ax = counts.plot.bar()
    ax.set_ylabel("# outer folds selected")
    plt.tight_layout()
    plt.savefig(outdir / "model_selection_counts.png", dpi=300)
    plt.close()

def plot_auc_per_model(outer_df: pd.DataFrame,
                       outdir: Path,
                       collapse_rule: str = "last"):
    if "patient_auc" not in outer_df.columns:
        return

    df = outer_df.copy()
    df["scalar_auc"] = df["patient_auc"].apply(
        lambda x: _collapse(x, collapse_rule))

    plt.figure(figsize=(8, 6))
    df.boxplot(column="scalar_auc", by="hp_model_type")
    plt.suptitle("")
    plt.ylabel("Test Patient AUC")
    plt.tight_layout()
    plt.savefig(outdir / "auc_per_model.png", dpi=300)
    plt.close()


# plots
def plot_metrics(df_metrics, exp_root):
    # Plot Precision, Recall, F1
    plt.figure()
    plt.plot(df_metrics['epoch'], df_metrics['precision'], label='Precision')
    plt.plot(df_metrics['epoch'], df_metrics['recall'], label='Recall')
    plt.plot(df_metrics['epoch'], df_metrics['f1_score'], label='F1-score')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Precision, Recall & F1-score by Epoch (Threshold=0.5) Small Model GRU')
    plt.legend()
    plt.xticks(df_metrics['epoch'])
    plt.savefig(os.path.join(exp_root, "outer_folds_pr_rc_f1.png"))

    # Plot AUC by epoch
    plt.figure()
    plt.plot(df_metrics['epoch'], df_metrics['auc'])
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.title('AUC by Epoch Small Model GRU')
    plt.xticks(df_metrics['epoch'])
    plt.savefig(os.path.join(exp_root, "outer_folds_auc.png"))

    # Plot Mean Validation Loss by epoch
    plt.figure()
    plt.plot(df_metrics['epoch'], df_metrics['mean_val_loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Mean Validation Loss')
    plt.title('Mean Validation Loss by Epoch')
    plt.xticks(df_metrics['epoch'])
    plt.savefig(os.path.join(exp_root, "outer_folds_val_loss.png"))

    # Plot Mean Train Loss by epoch
    plt.figure()
    plt.plot(df_metrics['epoch'], df_metrics['mean_train_loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Mean Train Loss')
    plt.title('Mean Train Loss by Epoch')
    plt.xticks(df_metrics['epoch'])
    plt.savefig(os.path.join(exp_root, "outer_folds_train_loss.png"))


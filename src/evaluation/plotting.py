import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay

def save_training_plots(all_histories, run_id, outdir):
    if not all_histories:
        print("No history to plot.")
        return

    epochs = all_histories[0]["epoch"]
    avg = lambda key: np.mean([h[key] for h in all_histories], axis=0)
    std_dev = lambda key: np.std([h[key] for h in all_histories], axis=0)

    # (a) Train vs Val Loss
    plt.figure(figsize=(7, 5))
    plt.plot(epochs, avg("train_loss"), label="Avg Train Loss")
    plt.plot(epochs, avg("val_loss"),   label="Avg Val Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title(f"Run {run_id}: Avg Train/Val Loss across Folds")
    plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(outdir, f"run_{run_id}_avg_loss_curve.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # (b) Validation Accuracy
    plt.figure(figsize=(7, 5))
    val_acc_avg = avg("val_accuracy") * 100
    plt.plot(epochs, val_acc_avg, label="Avg Val Accuracy (%)")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy (%)")
    plt.title(f"Run {run_id}: Avg Val Accuracy across Folds")
    plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(outdir, f"run_{run_id}_avg_val_accuracy.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # (c) Precision / Recall / F1
    plt.figure(figsize=(7, 5))
    plt.plot(epochs, avg("val_precision"), label="Avg Val Precision")
    plt.plot(epochs, avg("val_recall"),    label="Avg Val Recall")
    plt.plot(epochs, avg("val_f1"),        label="Avg Val F1 Score")
    plt.xlabel("Epoch"); plt.ylabel("Score")
    plt.title(f"Run {run_id}: Avg Val Precision/Recall/F1 across Folds")
    plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(outdir, f"run_{run_id}_avg_prf1.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # (d) Per-Fold Validation Accuracy
    plt.figure(figsize=(7, 5))
    for i, h in enumerate(all_histories, start=1):
        plt.plot(h["epoch"], [v*100 for v in h["val_accuracy"]], label=f"Fold {i}")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy (%)")
    plt.title(f"Run {run_id}: Val Accuracy per Fold")
    plt.legend(ncol=min(2, len(all_histories)), loc='best'); plt.grid(True) #Adjust ncol
    plt.savefig(os.path.join(outdir, f"run_{run_id}_per_fold_val_accuracy.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # (e) Per-Fold F1 Score
    plt.figure(figsize=(7, 5))
    for i, h in enumerate(all_histories, start=1):
        plt.plot(h["epoch"], h["val_f1"], label=f"Fold {i}")
    plt.xlabel("Epoch"); plt.ylabel("F1 Score")
    plt.title(f"Run {run_id}: Val F1 per Fold")
    plt.legend(ncol=min(2, len(all_histories)), loc='best'); plt.grid(True) #Adjust ncol
    plt.savefig(os.path.join(outdir, f"run_{run_id}_per_fold_val_f1.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # (f) Metric Variance Across Folds (±1 std)
    plt.figure(figsize=(10, 7)) # Wider for multiple lines
    plot_idx = 1
    for key, ylabel, scale_by_100 in [
        ("val_accuracy", "Accuracy (%)", True),
        ("val_precision", "Precision", False),
        ("val_recall",    "Recall", False),
        ("val_f1",        "F1 Score", False)
    ]:
        ax = plt.subplot(2, 2, plot_idx)
        arr = np.array([h[key] for h in all_histories])
        mean_vals = arr.mean(axis=0)
        std_vals  = arr.std(axis=0)
        
        ys = mean_vals * (100 if scale_by_100 else 1)
        ss = std_vals  * (100 if scale_by_100 else 1)

        ax.plot(epochs, ys, label=f"{ylabel} Mean")
        ax.fill_between(epochs, ys-ss, ys+ss, alpha=0.2, label=f"±1 std")
        ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel)
        ax.set_title(f"Avg {ylabel}")
        ax.legend(loc='best'); ax.grid(True)
        plot_idx +=1
    plt.suptitle(f"Run {run_id}: Metric Variance Across Folds", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust for suptitle
    plt.savefig(os.path.join(outdir, f"run_{run_id}_metric_variance.png"), dpi=300, bbox_inches="tight")
    plt.close()


def save_test_set_results_plots(model, test_loader, device, run_id, outdir):
    model.eval()
    y_true, y_score = [], []
    with torch.no_grad():
        for imgs, labs in test_loader:
            imgs = imgs.to(device)
            logits = model(imgs).squeeze(1)
            probs  = torch.sigmoid(logits).cpu().numpy()
            y_score.extend(probs.tolist())
            y_true.extend(labs.numpy().tolist())

    if not y_true or not y_score:
        print("No data in test set for ROC/Confusion Matrix plotting.")
        return

    # (g) ROC Curve on Test Set
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc      = auc(fpr, tpr)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0,1], [0,1], color='navy', lw=2, linestyle="--")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(f"Run {run_id}: ROC Curve (Test Set)")
    plt.legend(loc="lower right"); plt.grid(True)
    plt.savefig(os.path.join(outdir, f"run_{run_id}_roc_curve.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # (h) Confusion Matrix on Test Set
    y_pred = [1 if p > 0.5 else 0 for p in y_score]
    cm     = confusion_matrix(y_true, y_pred)
    disp   = ConfusionMatrixDisplay(cm) #, display_labels=[0,1]) # if you know your classes
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, cmap=plt.cm.Blues)
    plt.title(f"Run {run_id}: Confusion Matrix (Test Set)")
    plt.savefig(os.path.join(outdir, f"run_{run_id}_confusion_matrix.png"), dpi=300, bbox_inches="tight")
    plt.close()
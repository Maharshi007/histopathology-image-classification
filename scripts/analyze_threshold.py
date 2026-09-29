"""
Threshold Sweep & ROC/PR Curve Analysis
========================================
Loads the best model checkpoint, runs inference on the test set (collecting raw
sigmoid probabilities), then sweeps decision thresholds from 0.05 -> 0.95 to
identify the optimal operating point for medical use (max F1, max Recall@Prec>=80%).

Outputs (saved to results/figures/):
  - roc_curve.png          : ROC curve with AUC
  - pr_curve.png           : Precision-Recall curve with AUC
  - threshold_sweep.png    : All metrics vs. threshold
  - threshold_analysis.json: Full numeric results table

Usage:
  .venv\\Scripts\\python.exe scripts\\analyze_threshold.py
  .venv\\Scripts\\python.exe scripts\\analyze_threshold.py --checkpoint results/checkpoints/best_model.pth --split test
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from src.config import (
    TEST_X_PATH, TEST_Y_PATH,
    VALID_X_PATH, VALID_Y_PATH,
    BATCH_SIZE, NUM_WORKERS, DEVICE,
    CHECKPOINTS_DIR, METRICS_DIR, FIGURES_DIR
)
from src.dataset import PCamDataset
from src.transforms import get_eval_transforms
from src.model import build_model
from src.utils import load_checkpoint


# ---- Styling -----------------------------------------------------------------
DARK_BG  = "#0f1117"
PANEL_BG = "#1a1d27"
ACCENT   = "#6c63ff"
TEAL     = "#00d4aa"
CORAL    = "#ff6b6b"
GOLD     = "#ffd166"
TEXT     = "#e8eaf6"
GRID     = "#2a2d3e"

plt.rcParams.update({
    "figure.facecolor": DARK_BG,
    "axes.facecolor":   PANEL_BG,
    "axes.edgecolor":   GRID,
    "axes.labelcolor":  TEXT,
    "axes.titlecolor":  TEXT,
    "xtick.color":      TEXT,
    "ytick.color":      TEXT,
    "grid.color":       GRID,
    "text.color":       TEXT,
    "legend.facecolor": PANEL_BG,
    "legend.edgecolor": GRID,
    "font.family":      "DejaVu Sans",
    "font.size":        11,
})


# ---- Helpers -----------------------------------------------------------------

def collect_probabilities(model, dataloader, device):
    """Run full inference; return (probs, targets) as numpy float32 arrays."""
    model.eval()
    all_probs, all_targets = [], []
    total = len(dataloader)
    print(f"  Running inference ({total} batches) ...")
    with torch.no_grad():
        for i, (images, targets) in enumerate(dataloader):
            images = images.to(device, non_blocking=True)
            logits = model(images).squeeze(1)
            probs  = torch.sigmoid(logits)
            all_probs.extend(probs.cpu().tolist())
            all_targets.extend(targets.cpu().tolist())
            if (i + 1) % 100 == 0 or (i + 1) == total:
                print(f"    Batch {i+1}/{total}", end="\r")
    print()
    return np.array(all_probs, dtype=np.float32), np.array(all_targets, dtype=int)


def sweep_thresholds(probs, targets, thresholds):
    """Compute medical metrics for every threshold in the given array."""
    eps = 1e-7
    results = []
    for t in thresholds:
        preds = (probs >= t).astype(int)
        tp = int(np.sum((targets == 1) & (preds == 1)))
        tn = int(np.sum((targets == 0) & (preds == 0)))
        fp = int(np.sum((targets == 0) & (preds == 1)))
        fn = int(np.sum((targets == 1) & (preds == 0)))
        total = tp + tn + fp + fn
        acc   = (tp + tn) / total
        prec  = tp / (tp + fp + eps)
        rec   = tp / (tp + fn + eps)
        spec  = tn / (tn + fp + eps)
        f1    = 2 * prec * rec / (prec + rec + eps)
        results.append({
            "threshold":   round(float(t), 3),
            "accuracy":    round(acc,  4),
            "precision":   round(prec, 4),
            "recall":      round(rec,  4),
            "specificity": round(spec, 4),
            "f1_score":    round(f1,   4),
            "tp": tp, "tn": tn, "fp": fp, "fn": fn
        })
    return results


def compute_roc(probs, targets, n_points=500):
    """Manually compute ROC curve without sklearn."""
    thresholds = np.linspace(0.0, 1.0, n_points)
    fprs, tprs = [], []
    eps = 1e-7
    for t in thresholds:
        preds = (probs >= t).astype(int)
        tp = np.sum((targets == 1) & (preds == 1))
        tn = np.sum((targets == 0) & (preds == 0))
        fp = np.sum((targets == 0) & (preds == 1))
        fn = np.sum((targets == 1) & (preds == 0))
        tprs.append(tp / (tp + fn + eps))
        fprs.append(fp / (fp + tn + eps))
    pairs  = sorted(zip(fprs, tprs))
    fprs_s = np.array([p[0] for p in pairs])
    tprs_s = np.array([p[1] for p in pairs])
    auc    = abs(float(np.trapz(tprs_s, fprs_s)))
    return fprs_s, tprs_s, auc


def compute_pr(probs, targets, n_points=500):
    """Manually compute Precision-Recall curve without sklearn."""
    thresholds = np.linspace(0.0, 1.0, n_points)
    precisions, recalls = [], []
    eps = 1e-7
    for t in thresholds:
        preds = (probs >= t).astype(int)
        tp = np.sum((targets == 1) & (preds == 1))
        fp = np.sum((targets == 0) & (preds == 1))
        fn = np.sum((targets == 1) & (preds == 0))
        precisions.append(tp / (tp + fp + eps))
        recalls.append(tp / (tp + fn + eps))
    pairs   = sorted(zip(recalls, precisions))
    rec_s   = np.array([p[0] for p in pairs])
    prec_s  = np.array([p[1] for p in pairs])
    auc     = abs(float(np.trapz(prec_s, rec_s)))
    return rec_s, prec_s, auc


# ---- Plots -------------------------------------------------------------------

def plot_roc(fprs, tprs, auc, op_fpr, op_tpr, save_path):
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(PANEL_BG)

    ax.plot(fprs, tprs, color=ACCENT, lw=2.5, label=f"ResNet50  (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], color=GRID, lw=1.2, linestyle="--", label="Random classifier")
    ax.fill_between(fprs, tprs, alpha=0.12, color=ACCENT)
    ax.scatter([op_fpr], [op_tpr], color=CORAL, s=110, zorder=6,
               label=f"Default threshold (0.50)\nFPR={op_fpr:.3f}  TPR={op_tpr:.3f}")

    ax.set_xlabel("False Positive Rate  (1 - Specificity)", fontsize=12)
    ax.set_ylabel("True Positive Rate  (Sensitivity / Recall)", fontsize=12)
    ax.set_title("ROC Curve  -  PCam Test Set", fontsize=14, fontweight="bold", pad=14)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  Saved: {save_path.name}")


def plot_pr(rec_curve, prec_curve, auc, op_recall, op_prec, save_path):
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(PANEL_BG)

    ax.plot(rec_curve, prec_curve, color=TEAL, lw=2.5, label=f"ResNet50  (AUC-PR = {auc:.4f})")
    ax.axhline(y=0.5, color=GRID, lw=1.2, linestyle="--", label="Random classifier baseline")
    ax.fill_between(rec_curve, prec_curve, alpha=0.12, color=TEAL)
    ax.scatter([op_recall], [op_prec], color=CORAL, s=110, zorder=6,
               label=f"Default threshold (0.50)\nRecall={op_recall:.3f}  Prec={op_prec:.3f}")

    ax.set_xlabel("Recall  (Sensitivity)", fontsize=12)
    ax.set_ylabel("Precision  (PPV)", fontsize=12)
    ax.set_title("Precision-Recall Curve  -  PCam Test Set", fontsize=14, fontweight="bold", pad=14)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  Saved: {save_path.name}")


def plot_threshold_sweep(sweep_results, best_f1_t, best_recall_t, save_path):
    thresholds = [r["threshold"]   for r in sweep_results]
    accs       = [r["accuracy"]    for r in sweep_results]
    precs      = [r["precision"]   for r in sweep_results]
    recs       = [r["recall"]      for r in sweep_results]
    f1s        = [r["f1_score"]    for r in sweep_results]
    tps        = [r["tp"]          for r in sweep_results]
    fns        = [r["fn"]          for r in sweep_results]
    fps        = [r["fp"]          for r in sweep_results]

    fig, axes = plt.subplots(2, 1, figsize=(10, 9), sharex=True)
    fig.patch.set_facecolor(DARK_BG)
    fig.suptitle("Metric vs. Decision Threshold  -  PCam Test Set",
                 fontsize=15, fontweight="bold", color=TEXT, y=0.99)

    # Top: metric curves
    ax = axes[0]
    ax.set_facecolor(PANEL_BG)
    ax.plot(thresholds, f1s,   color=ACCENT, lw=2.2, label="F1-Score")
    ax.plot(thresholds, recs,  color=TEAL,   lw=2.2, label="Recall (Sensitivity)")
    ax.plot(thresholds, precs, color=GOLD,   lw=2.2, label="Precision")
    ax.plot(thresholds, accs,  color=CORAL,  lw=1.5, linestyle="--", label="Accuracy")

    ax.axvline(x=0.50,          color="white", lw=1.2, ls=":",  alpha=0.7, label="Default (0.50)")
    ax.axvline(x=best_f1_t,     color=ACCENT,  lw=1.5, ls="--", alpha=0.9, label=f"Best F1 ({best_f1_t:.2f})")
    ax.axvline(x=best_recall_t, color=TEAL,    lw=1.5, ls="--", alpha=0.9, label=f"Best Recall@Prec>=80% ({best_recall_t:.2f})")

    ax.set_ylabel("Score", fontsize=12)
    ax.legend(loc="lower left", fontsize=9, ncol=2)
    ax.grid(True, alpha=0.25)
    ax.set_ylim(0, 1.05)

    # Bottom: confusion matrix counts
    ax2 = axes[1]
    ax2.set_facecolor(PANEL_BG)
    ax2.plot(thresholds, tps, color=TEAL,  lw=2.0, label="TP  (true positives)")
    ax2.plot(thresholds, fns, color=CORAL, lw=2.0, label="FN  (missed tumors)")
    ax2.plot(thresholds, fps, color=GOLD,  lw=2.0, label="FP  (false alarms)")

    ax2.axvline(x=0.50,          color="white", lw=1.2, ls=":", alpha=0.7)
    ax2.axvline(x=best_f1_t,     color=ACCENT,  lw=1.5, ls="--", alpha=0.9)
    ax2.axvline(x=best_recall_t, color=TEAL,    lw=1.5, ls="--", alpha=0.9)

    ax2.set_xlabel("Decision Threshold", fontsize=12)
    ax2.set_ylabel("Count", fontsize=12)
    ax2.legend(loc="center right", fontsize=9)
    ax2.grid(True, alpha=0.25)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  Saved: {save_path.name}")


def print_summary_table(sweep_results, best_f1_t, best_recall_t):
    hdr = f"{'Thresh':>8} {'Acc':>8} {'Prec':>8} {'Recall':>8} {'F1':>8} {'Spec':>8} {'TP':>7} {'FN':>7}"
    sep = "-" * len(hdr)
    print(f"\n{sep}\n{hdr}\n{sep}")
    show_at = {0.30, 0.35, 0.40, 0.45, 0.50, 0.55,
               round(best_f1_t, 2), round(best_recall_t, 2)}
    for r in sweep_results:
        t = round(r["threshold"], 2)
        if t not in show_at:
            continue
        tag = ""
        if abs(t - best_f1_t)     < 0.005: tag += "  <- best F1"
        if abs(t - best_recall_t) < 0.005: tag += "  <- best Recall@Prec>=80%"
        if abs(t - 0.50)          < 0.005: tag += "  <- default"
        print(
            f"{t:>8.2f} {r['accuracy']*100:>7.2f}% {r['precision']*100:>7.2f}% "
            f"{r['recall']*100:>7.2f}% {r['f1_score']*100:>7.2f}% "
            f"{r['specificity']*100:>7.2f}% {r['tp']:>7} {r['fn']:>7}{tag}"
        )
    print(sep)


# ---- Main --------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Threshold sweep + ROC/PR analysis")
    p.add_argument("--checkpoint", type=str,
                   default=str(CHECKPOINTS_DIR / "best_model.pth"))
    p.add_argument("--split", type=str, choices=["valid", "test"], default="test")
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    return p.parse_args()


def main():
    args      = parse_args()
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    x_path = TEST_X_PATH  if args.split == "test" else VALID_X_PATH
    y_path = TEST_Y_PATH  if args.split == "test" else VALID_Y_PATH

    print("=" * 62)
    print(f"  THRESHOLD SWEEP & ROC/PR ANALYSIS  [{args.split.upper()} SET]")
    print("=" * 62)
    print(f"  Checkpoint : {ckpt_path.name}")
    print(f"  Device     : {DEVICE}")

    # Load model
    model = build_model(pretrained=False).to(DEVICE)
    load_checkpoint(ckpt_path, model, device=DEVICE)

    # Load dataset
    dataset    = PCamDataset(image_path=x_path, label_path=y_path,
                             transform=get_eval_transforms())
    dataloader = DataLoader(dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=NUM_WORKERS,
                            pin_memory=torch.cuda.is_available())
    print(f"  Samples    : {len(dataset):,}\n")

    # Collect probabilities
    probs, targets = collect_probabilities(model, dataloader, DEVICE)

    # Threshold sweep
    print("[1/4] Sweeping thresholds ...")
    thresholds    = np.arange(0.05, 0.955, 0.01)
    sweep_results = sweep_thresholds(probs, targets, thresholds)

    f1s         = [r["f1_score"] for r in sweep_results]
    best_f1_idx = int(np.argmax(f1s))
    best_f1_t   = sweep_results[best_f1_idx]["threshold"]

    best_recall_t, best_recall = best_f1_t, 0.0
    for r in sweep_results:
        if r["precision"] >= 0.80 and r["recall"] > best_recall:
            best_recall   = r["recall"]
            best_recall_t = r["threshold"]

    # Default op point
    op = next(r for r in sweep_results if abs(r["threshold"] - 0.50) < 0.006)
    op_fpr = op["fp"] / (op["fp"] + op["tn"] + 1e-7)

    # ROC
    print("[2/4] Computing ROC curve ...")
    fprs, tprs, roc_auc = compute_roc(probs, targets)

    # PR
    print("[3/4] Computing Precision-Recall curve ...")
    rec_curve, prec_curve, pr_auc = compute_pr(probs, targets)

    # Plots
    print("[4/4] Generating plots ...")
    plot_roc(fprs, tprs, roc_auc, op_fpr, op["recall"],
             FIGURES_DIR / "roc_curve.png")
    plot_pr(rec_curve, prec_curve, pr_auc, op["recall"], op["precision"],
            FIGURES_DIR / "pr_curve.png")
    plot_threshold_sweep(sweep_results, best_f1_t, best_recall_t,
                         FIGURES_DIR / "threshold_sweep.png")

    # Save JSON
    output = {
        "split":       args.split,
        "checkpoint":  str(ckpt_path),
        "roc_auc":     round(roc_auc, 6),
        "pr_auc":      round(pr_auc,  6),
        "best_f1_threshold": best_f1_t,
        "best_recall_at_precision_80pct": {
            "threshold": best_recall_t,
            "recall":    round(best_recall, 4)
        },
        "default_threshold_0.50": op,
        "sweep": sweep_results,
    }
    json_path = METRICS_DIR / "threshold_analysis.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Saved JSON : {json_path.name}")

    # Console table
    print_summary_table(sweep_results, best_f1_t, best_recall_t)

    print(f"\n{'='*62}")
    print(f"  ROC-AUC  : {roc_auc:.4f}")
    print(f"  PR-AUC   : {pr_auc:.4f}")
    r_best = sweep_results[best_f1_idx]
    print(f"  Best F1 threshold  : {best_f1_t:.2f}  ->  F1={r_best['f1_score']*100:.2f}%  Recall={r_best['recall']*100:.2f}%")
    print(f"  Best Recall@Prec>=80%: threshold={best_recall_t:.2f}  Recall={best_recall*100:.2f}%")
    print("=" * 62)


if __name__ == "__main__":
    main()

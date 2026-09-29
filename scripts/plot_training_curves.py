"""
Training History Visualization
================================
Reads results/metrics/training_history.json and produces a rich,
dark-themed figure showing loss, accuracy, recall, precision, and F1
curves across training stages.

Outputs (saved to results/figures/):
  - training_curves.png  : 2x2 grid of training vs. validation curves

Usage:
  .venv\\Scripts\\python.exe scripts\\plot_training_curves.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from src.config import METRICS_DIR, FIGURES_DIR


# ---- Styling -----------------------------------------------------------------
DARK_BG  = "#0f1117"
PANEL_BG = "#1a1d27"
ACCENT   = "#6c63ff"
TEAL     = "#00d4aa"
CORAL    = "#ff6b6b"
GOLD     = "#ffd166"
TEXT     = "#e8eaf6"
GRID     = "#2a2d3e"
STAGE1   = "#9b59b6"
STAGE2   = "#3498db"

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


def shade_stages(ax, history):
    """Draw translucent background bands to highlight Stage 1 vs Stage 2."""
    epochs = history["epoch"]
    stages = history["stage"]
    if len(set(stages)) < 2:
        return  # only one stage, nothing to shade
    stage1_epochs = [e for e, s in zip(epochs, stages) if s == 1]
    stage2_epochs = [e for e, s in zip(epochs, stages) if s == 2]
    if stage1_epochs:
        ax.axvspan(min(stage1_epochs) - 0.5, max(stage1_epochs) + 0.5,
                   alpha=0.08, color=STAGE1, zorder=0)
    if stage2_epochs:
        ax.axvspan(min(stage2_epochs) - 0.5, max(stage2_epochs) + 0.5,
                   alpha=0.08, color=STAGE2, zorder=0)


def plot_metric_pair(ax, epochs, train_vals, val_vals,
                     title, ylabel, history, ylim=(0, 1)):
    """Plot a (train, val) pair on one axis with stage shading."""
    shade_stages(ax, history)
    ax.plot(epochs, train_vals, color=TEAL,  lw=2.2, marker="o", ms=6,
            label="Train", zorder=3)
    ax.plot(epochs, val_vals,   color=CORAL, lw=2.2, marker="s", ms=6,
            label="Val",   zorder=3)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.25)
    ax.set_xticks(epochs)
    if ylim:
        ax.set_ylim(*ylim)


def main():
    history_path = METRICS_DIR / "training_history.json"
    if not history_path.is_file():
        raise FileNotFoundError(f"History file not found: {history_path}")

    with open(history_path) as f:
        h = json.load(f)

    epochs = h["epoch"]
    stages = h.get("stage", [1] * len(epochs))

    # Percentage-scale accuracy, recall, precision, f1
    train_acc   = [v * 100 for v in h["train_acc"]]
    val_acc     = [v * 100 for v in h["val_acc"]]
    train_rec   = [v * 100 for v in h["train_recall"]]
    val_rec     = [v * 100 for v in h["val_recall"]]
    train_prec  = [v * 100 for v in h["train_precision"]]
    val_prec    = [v * 100 for v in h["val_precision"]]
    train_f1    = [v * 100 for v in h["train_f1"]]
    val_f1      = [v * 100 for v in h["val_f1"]]
    train_loss  = h["train_loss"]
    val_loss    = h["val_loss"]
    lrs         = h.get("lr", [])

    n_panels = 5 if lrs else 4
    fig, axes = plt.subplots(2, 3 if n_panels == 5 else 2,
                             figsize=(16 if n_panels == 5 else 12, 9))
    axes_flat = axes.flatten()
    fig.patch.set_facecolor(DARK_BG)
    fig.suptitle("Training History  -  ResNet50 PCam Classifier",
                 fontsize=16, fontweight="bold", color=TEXT, y=1.01)

    # Panel 0: Loss
    plot_metric_pair(axes_flat[0], epochs, train_loss, val_loss,
                     "Training & Validation Loss", "BCE Loss", h, ylim=None)
    axes_flat[0].set_ylim(0, max(max(train_loss), max(val_loss)) * 1.15)

    # Panel 1: Accuracy
    plot_metric_pair(axes_flat[1], epochs, train_acc, val_acc,
                     "Accuracy", "Accuracy (%)", h, ylim=(50, 100))

    # Panel 2: Recall (Sensitivity) - the critical medical metric
    plot_metric_pair(axes_flat[2], epochs, train_rec, val_rec,
                     "Recall / Sensitivity  (Critical: tumor detection)", "Recall (%)", h, ylim=(50, 100))

    # Panel 3: Precision
    plot_metric_pair(axes_flat[3], epochs, train_prec, val_prec,
                     "Precision  (PPV)", "Precision (%)", h, ylim=(50, 100))

    # Panel 4: F1
    plot_metric_pair(axes_flat[4], epochs, train_f1, val_f1,
                     "F1-Score", "F1 (%)", h, ylim=(50, 100))

    # Panel 5: Learning Rate (if available)
    if lrs and n_panels == 5:
        ax_lr = axes_flat[5]
        ax_lr.set_facecolor(PANEL_BG)
        shade_stages(ax_lr, h)
        ax_lr.semilogy(epochs, lrs, color=GOLD, lw=2.2, marker="^", ms=6)
        ax_lr.set_title("Learning Rate Schedule", fontsize=13, fontweight="bold", pad=10)
        ax_lr.set_ylabel("Learning Rate (log scale)", fontsize=11)
        ax_lr.set_xlabel("Epoch", fontsize=11)
        ax_lr.set_xticks(epochs)
        ax_lr.grid(True, alpha=0.25)
    elif len(axes_flat) > 5:
        axes_flat[5].set_visible(False)

    # Stage legend (shared)
    legend_patches = [
        mpatches.Patch(color=STAGE1, alpha=0.5, label="Stage 1: Head warmup (frozen backbone)"),
        mpatches.Patch(color=STAGE2, alpha=0.5, label="Stage 2: End-to-end fine-tuning"),
    ]
    if len(set(stages)) > 1:
        fig.legend(handles=legend_patches, loc="lower center",
                   ncol=2, fontsize=10, framealpha=0.3, bbox_to_anchor=(0.5, -0.03))

    plt.tight_layout()
    save_path = FIGURES_DIR / "training_curves.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)

    # ---- Print summary stats -------------------------------------------------
    print("=" * 55)
    print("  TRAINING SUMMARY")
    print("=" * 55)
    best_val_f1_idx = int(np.argmax(val_f1))
    print(f"  Epochs trained    : {len(epochs)}")
    print(f"  Best val F1       : {val_f1[best_val_f1_idx]:.2f}%  (epoch {epochs[best_val_f1_idx]})")
    print(f"  Final train loss  : {train_loss[-1]:.4f}")
    print(f"  Final val loss    : {val_loss[-1]:.4f}")
    print(f"  Final val recall  : {val_rec[-1]:.2f}%")
    print(f"  Final val prec    : {val_prec[-1]:.2f}%")
    overfitting = val_loss[-1] - train_loss[-1]
    flag = "  (likely overfitting)" if overfitting > 0.05 else ""
    print(f"  Loss gap (val-trn): {overfitting:.4f}{flag}")
    print("=" * 55)
    print(f"  Saved: {save_path}")


if __name__ == "__main__":
    main()

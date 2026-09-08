"""
plotting.py — Figure helpers shared by train.py and evaluate.py.

All functions use the non-interactive 'Agg' backend and SAVE figures (never
show), so they run headless on a server or in CI.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_training_curves(history: dict, out_path: Path) -> None:
    """Two panels: (left) train+val loss, (right) val accuracy + val macro-F1.

    `history` has parallel lists keyed:
        train_loss, val_loss, val_accuracy, val_macro_f1
    """
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax_loss, ax_metric) = plt.subplots(1, 2, figsize=(11, 4))

    ax_loss.plot(epochs, history["train_loss"], label="train loss")
    ax_loss.plot(epochs, history["val_loss"], label="val loss")
    ax_loss.set_xlabel("epoch"); ax_loss.set_ylabel("loss")
    ax_loss.set_title("Training / validation loss")
    ax_loss.legend(); ax_loss.grid(alpha=0.3)

    ax_metric.plot(epochs, history["val_accuracy"], label="val accuracy")
    ax_metric.plot(epochs, history["val_macro_f1"], label="val macro-F1")
    ax_metric.set_xlabel("epoch"); ax_metric.set_ylabel("score")
    ax_metric.set_title("Validation accuracy & macro-F1")
    ax_metric.legend(); ax_metric.grid(alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_confusion_matrix(cm, class_names, out_path: Path, normalize: bool = True) -> None:
    """Save a confusion-matrix heatmap (row-normalized by default).

    Row-normalization shows per-true-class RECALL patterns, which is the most
    readable view under imbalance (raw counts are swamped by Blip).
    """
    cm = np.asarray(cm, dtype=np.float64)
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0)

    n = len(class_names)
    fig, ax = plt.subplots(figsize=(max(8, n * 0.5), max(7, n * 0.5)))
    im = ax.imshow(cm, cmap="viridis", vmin=0, vmax=1 if normalize else None)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=90, fontsize=7)
    ax.set_yticklabels(class_names, fontsize=7)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title("Confusion matrix" + (" (row-normalized)" if normalize else ""))
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_prediction_examples(images, trues, preds, class_names, out_path: Path,
                             title: str = "") -> None:
    """Grid of example spectrograms titled 'true -> pred' (green=correct, red=wrong).

    `images` are display-ready (H, W) arrays in [0,1].
    """
    k = len(images)
    cols = min(5, k) if k else 1
    rows = int(np.ceil(k / cols)) if k else 1
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.4, rows * 2.6))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for i in range(k):
        ax = axes[i]
        ax.imshow(images[i], cmap="viridis", aspect="auto", origin="lower")
        correct = trues[i] == preds[i]
        ax.set_title(f"{class_names[trues[i]]}\n-> {class_names[preds[i]]}",
                     fontsize=7, color=("green" if correct else "red"))
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96] if title else None)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)

"""All figures. Every function saves a PNG and returns its path."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence

import matplotlib

# Agg = no display needed, so these scripts run over SSH or in a notebook.
# Must be set before pyplot is imported.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _save(fig, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)   # matplotlib keeps unclosed figures alive; evaluate.py makes many
    return path


def plot_confusion_matrix(cm, class_names: List[str], path: Path,
                          title: str = "") -> Path:
    """Row-normalised confusion matrix.

    Normalised by row because the raw counts are unreadable with this much
    imbalance -- the Blip row would swamp the colour scale. Each row becomes a
    recall profile: of the true Koi_Fish glitches, where did they go?
    """
    counts = np.asarray(cm, dtype=np.float64)
    row_sums = counts.sum(axis=1, keepdims=True)
    norm = np.divide(counts, row_sums, out=np.zeros_like(counts), where=row_sums > 0)

    n = len(class_names)
    fig, ax = plt.subplots(figsize=(max(8, n * 0.55), max(7, n * 0.5)))
    im = ax.imshow(norm, cmap="viridis", vmin=0.0, vmax=1.0)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Fraction of true class")

    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=90, fontsize=7)
    ax.set_yticklabels(class_names, fontsize=7)
    ax.set_xlabel("Predicted label"); ax.set_ylabel("True label")
    ax.set_title(title or "Confusion matrix (row-normalised)")

    # Annotate only non-empty cells -- 22x22 = 484 cells, printing "0.00" in
    # every empty one makes the figure unreadable.
    for i in range(n):
        for j in range(n):
            if counts[i, j] > 0:
                ax.text(j, i, "%.2f" % norm[i, j], ha="center", va="center",
                        fontsize=5.5,
                        color="white" if norm[i, j] < 0.5 else "black")
    return _save(fig, path)


def plot_training_curves(history: Dict[str, List[float]], path: Path,
                         title: str = "") -> Path:
    """Loss on the left; validation accuracy and macro-F1 on the right.

    The two right-hand lines are the point of this figure: accuracy climbs fast
    and plateaus high, macro-F1 climbs slowly and lands lower, and the gap
    between them is the class imbalance made visible.
    """
    epochs = range(1, len(history.get("train_loss", [])) + 1)
    fig, (ax_loss, ax_score) = plt.subplots(1, 2, figsize=(13, 4.5))

    ax_loss.plot(epochs, history["train_loss"], label="train loss", lw=1.8)
    ax_loss.plot(epochs, history["val_loss"], label="val loss", lw=1.8)
    ax_loss.set_xlabel("Epoch"); ax_loss.set_ylabel("Loss"); ax_loss.set_title("Loss")
    ax_loss.grid(alpha=0.3); ax_loss.legend()

    ax_score.plot(epochs, history["val_accuracy"], label="val accuracy", lw=1.8)
    ax_score.plot(epochs, history["val_macro_f1"], label="val macro-F1", lw=1.8)
    ax_score.plot(epochs, history["train_accuracy"], label="train accuracy",
                  lw=1.2, ls="--", alpha=0.7)

    best = history.get("best_epoch")
    if best:
        ax_score.axvline(best, color="grey", ls=":", lw=1.4)
        ax_score.annotate("best (epoch %d)" % best, xy=(best, 0.02),
                          xycoords=("data", "axes fraction"), fontsize=8,
                          rotation=90, va="bottom", ha="right", color="grey")

    ax_score.set_xlabel("Epoch"); ax_score.set_ylabel("Score")
    ax_score.set_ylim(0, 1.02)
    ax_score.set_title("Validation accuracy vs macro-F1")
    ax_score.grid(alpha=0.3); ax_score.legend()

    if title:
        fig.suptitle(title)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
    else:
        fig.tight_layout()
    return _save(fig, path)


def denormalise(img: np.ndarray, mean: Sequence[float], std: Sequence[float]) -> np.ndarray:
    """Undo normalisation so a tensor can be displayed. CHW in, HW out."""
    arr = np.asarray(img, dtype=np.float32)
    arr = arr * np.asarray(std, np.float32).reshape(-1, 1, 1) \
        + np.asarray(mean, np.float32).reshape(-1, 1, 1)
    # All 3 channels are copies of one greyscale image, so take the first.
    return np.clip(arr[0], 0.0, 1.0)


# The stored arrays are image-style: row 0 is the TOP of the spectrogram, i.e.
# the HIGHEST frequency. Draw them with origin="upper" so frequency increases
# upward. origin="lower" would flip every glitch vertically -- a display-only
# error, but it puts Low_Frequency_Burst at the top and Whistle at the bottom.


def plot_example_grid(images, true_labels, pred_labels, confidences, path: Path,
                      title: str, mean, std, ncols: int = 4) -> Path:
    """Example predictions; green title = correct, red = wrong."""
    n = len(images)
    if n == 0:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "No examples", ha="center", va="center")
        ax.axis("off"); ax.set_title(title)
        return _save(fig, path)

    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.1 * ncols, 3.0 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[n:]:
        ax.axis("off")

    for i in range(n):
        axes[i].imshow(denormalise(images[i], mean, std), cmap="viridis",
                       aspect="auto", origin="upper")
        axes[i].set_title("true: %s\npred: %s (%.2f)"
                          % (true_labels[i], pred_labels[i], confidences[i]),
                          fontsize=7.5,
                          color="darkgreen" if true_labels[i] == pred_labels[i]
                          else "firebrick")
        axes[i].set_xticks([]); axes[i].set_yticks([])

    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return _save(fig, path)


def plot_sample_grid(images, labels, path: Path, title: str, ncols: int = 6) -> Path:
    """Grid of raw spectrograms, used by inspect_data."""
    n = len(images)
    ncols = min(ncols, max(n, 1))
    nrows = int(np.ceil(n / ncols)) if n else 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.5 * ncols, 2.5 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[n:]:
        ax.axis("off")
    for i in range(n):
        axes[i].imshow(images[i], cmap="viridis", aspect="auto", origin="upper")
        axes[i].set_title(labels[i], fontsize=7)
        axes[i].set_xticks([]); axes[i].set_yticks([])
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return _save(fig, path)


def plot_class_distribution(counts_by_split: Dict[str, Dict[str, int]],
                            class_names: List[str], path: Path) -> Path:
    """Per-class counts per split.

    Log x-axis because the biggest class is ~67x the smallest -- on a linear
    axis the rare classes would be invisible.
    """
    order = sorted(class_names,
                   key=lambda c: -sum(counts_by_split.get(s, {}).get(c, 0)
                                      for s in counts_by_split))
    y = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(9, max(6, len(order) * 0.34)))
    left = np.zeros(len(order))
    for split in ("train", "validation", "test"):
        vals = np.array([counts_by_split.get(split, {}).get(c, 0) for c in order],
                        dtype=float)
        ax.barh(y, vals, left=left, label=split, height=0.72)
        left += vals

    ax.set_yticks(y); ax.set_yticklabels(order, fontsize=8); ax.invert_yaxis()
    ax.set_xlabel("Number of glitches (log scale)"); ax.set_xscale("log")
    ax.set_title("Gravity Spy class distribution by split")
    ax.legend(); ax.grid(axis="x", alpha=0.3)
    for i, total in enumerate(left):
        if total > 0:
            ax.text(total * 1.05, i, "%d" % int(total), va="center", fontsize=7)
    return _save(fig, path)


def plot_model_comparison(results: Dict[str, Dict[str, Any]], path: Path) -> Path:
    """Grouped bars comparing models on the shared test set."""
    metrics = ("macro_f1", "accuracy", "balanced_accuracy", "weighted_f1")
    models = list(results)
    x = np.arange(len(metrics))
    width = 0.8 / max(len(models), 1)

    fig, ax = plt.subplots(figsize=(1.9 * len(metrics) + 4, 5))
    for i, name in enumerate(models):
        vals = [float(results[name].get(m, 0.0)) for m in metrics]
        bars = ax.bar(x + (i - (len(models) - 1) / 2) * width, vals, width, label=name)
        ax.bar_label(bars, fmt="%.3f", fontsize=7, padding=2)

    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", " ") for m in metrics])
    ax.set_ylim(0, 1.08); ax.set_ylabel("Score")
    ax.set_title("Model comparison on the shared test set")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    return _save(fig, path)

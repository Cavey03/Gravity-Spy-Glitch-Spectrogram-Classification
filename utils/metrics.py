"""
metrics.py — Metric computations shared by train.py and evaluate.py.

We separate "headline" metrics (a few scalars, computed every epoch during
training) from the "full report" (per-class table + confusion matrix, computed
once at test time in evaluate.py). This keeps the training loop cheap.

Reminder on why macro-F1 is the headline: with heavy class imbalance, plain
accuracy is dominated by the big classes. Macro-F1 averages the per-class F1
scores with EQUAL weight per class, so a tiny class counts as much as Blip.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


def headline_metrics(y_true, y_pred) -> dict:
    """Cheap scalars computed every epoch during training/validation.

    macro_f1 is THE metric we monitor for early stopping / best checkpoint.
    zero_division=0 keeps things quiet when a class has no predictions yet
    (common early in training for rare classes).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(p),
        "macro_recall": float(r),
        "macro_f1": float(f1),
    }


def full_report(y_true, y_pred, class_names) -> dict:
    """Everything evaluate.py needs: overall + per-class + confusion matrix.

    Returns a JSON-serializable dict:
      overall     : accuracy, macro & weighted precision/recall/F1
      per_class   : {class_name: {precision, recall, f1, support}}
      confusion   : list-of-lists (rows=true, cols=pred), label order == class_names
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = list(range(len(class_names)))

    acc = float(accuracy_score(y_true, y_pred))
    mp, mr, mf1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    wp, wr, wf1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0
    )
    # Per-class (average=None -> one value per class).
    pc_p, pc_r, pc_f1, pc_sup = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    per_class = {
        class_names[i]: {
            "precision": float(pc_p[i]),
            "recall": float(pc_r[i]),
            "f1": float(pc_f1[i]),
            "support": int(pc_sup[i]),
        }
        for i in labels
    }
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    return {
        "overall": {
            "accuracy": acc,
            "macro_precision": float(mp),
            "macro_recall": float(mr),
            "macro_f1": float(mf1),
            "weighted_precision": float(wp),
            "weighted_recall": float(wr),
            "weighted_f1": float(wf1),
        },
        "per_class": per_class,
        "confusion": cm.tolist(),
    }

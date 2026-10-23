"""Metrics.

MACRO-F1 IS THE HEADLINE NUMBER, not accuracy. Blip is 23% of the data and the
common classes dominate, so accuracy sits around 97% almost regardless of how
the rare classes are handled. Macro-F1 averages the per-class F1 scores giving
every class equal weight, so ignoring Paired_Doves costs as much as ignoring
Blip. The gap between weighted-F1 and macro-F1 shows how much the tail is being
neglected.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


def compute_metrics(
    y_true: Sequence[int], y_pred: Sequence[int], class_names: List[str]
) -> Dict[str, Any]:
    """The full metric suite. Returns plain Python types so it JSON-serialises."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = list(range(len(class_names)))

    # zero_division=0: a weak model may predict some classes not at all, which
    # would otherwise produce NaN and break the JSON round-trip.
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0)
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    per_p, per_r, per_f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0)

    return {
        "macro_f1": float(macro_f1),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "weighted_precision": float(weighted_p),
        "weighted_recall": float(weighted_r),
        "weighted_f1": float(weighted_f1),
        "n_samples": int(y_true.size),
        "class_names": list(class_names),
        "per_class": {
            name: {
                "precision": float(per_p[i]),
                "recall": float(per_r[i]),
                "f1": float(per_f1[i]),
                "support": int(support[i]),
            }
            for i, name in enumerate(class_names)
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def macro_f1(y_true: Sequence[int], y_pred: Sequence[int], num_classes: int) -> float:
    """Used inside the training loop for early stopping."""
    return float(f1_score(y_true, y_pred, labels=list(range(num_classes)),
                          average="macro", zero_division=0))


def per_class_table(metrics: Dict[str, Any]) -> str:
    """Per-class metrics as a markdown table, most-common class first."""
    rows = sorted(
        ((n, v["precision"], v["recall"], v["f1"], v["support"])
         for n, v in metrics["per_class"].items()),
        key=lambda r: -r[4],
    )
    out = ["| Class | Precision | Recall | F1 | Support |", "|---|---:|---:|---:|---:|"]
    out += ["| %s | %.4f | %.4f | %.4f | %d |" % r for r in rows]
    out.append("| **macro avg** | **%.4f** | **%.4f** | **%.4f** | **%d** |"
               % (metrics["macro_precision"], metrics["macro_recall"],
                  metrics["macro_f1"], metrics["n_samples"]))
    out.append("| weighted avg | %.4f | %.4f | %.4f | %d |"
               % (metrics["weighted_precision"], metrics["weighted_recall"],
                  metrics["weighted_f1"], metrics["n_samples"]))
    return "\n".join(out)


def compute_class_weights(counts: Sequence[int]) -> np.ndarray:
    """Class weights for the loss, as 1/sqrt(n) normalised to mean 1.

    Not plain 1/n. The training split runs from Blip at 1,274 images down to
    Paired_Doves at 19, so 1/n would give a 67:1 weight ratio and gradients
    from nineteen images would dominate every batch, making training unstable.
    1/sqrt(n) compresses that to about 8:1 -- still a strong correction toward
    the rare classes, but stable.

    Normalising to mean 1 means turning weighting on or off does not also
    change the effective learning rate.
    """
    counts_arr = np.asarray(counts, dtype=np.float64)
    weights = 1.0 / np.sqrt(np.maximum(counts_arr, 1.0))
    weights[counts_arr == 0] = 0.0        # nothing to learn from an absent class
    nonzero = weights[weights > 0]
    if nonzero.size:
        weights = weights / nonzero.mean()
    return weights.astype(np.float32)

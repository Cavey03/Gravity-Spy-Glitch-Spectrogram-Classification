"""
evaluate.py — Load a trained checkpoint, run the TEST set, save all metrics.

Model-agnostic: it reads the model NAME from the checkpoint and rebuilds via the
registry, so the same command evaluates any of the three models.

    python evaluate.py --model custom_cnn
    python evaluate.py --model transformer --split test

Outputs (all into outputs/<model>/):
    metrics.json            accuracy, macro-F1, per-class P/R/F1, confusion, ...
    per_class_metrics.csv   the per-class table, spreadsheet-friendly
    confusion_matrix.png    row-normalized heatmap
    examples_correct.png    grid of correct predictions
    examples_incorrect.png  grid of mistakes

NOTE ON METRICS: overall ACCURACY will look high (~97-98%) because the big
classes (Blip) dominate. That is expected. MACRO-F1 is the HEADLINE metric --
it averages per-class F1 equally, so rare classes count fully.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import config
from data.dataset import GravitySpyDataset
from models.registry import build_model
from utils.metrics import full_report
from utils.plotting import plot_confusion_matrix, plot_prediction_examples
from utils.seed import set_seed


def load_checkpoint(path: Path):
    """Load a checkpoint saved by train.py (weights_only=False: it holds a dict)."""
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {path}. Train first: "
            f"python train.py --model <name>"
        )
    return torch.load(path, map_location="cpu", weights_only=False)


@torch.no_grad()
def run_inference(model, loader, device, max_batches=None):
    """Return parallel lists (index, y_true, y_pred) over the loader.

    Because the loader is unshuffled, the running counter equals the dataset
    index -- we keep it so we can fetch the original spectrogram for the grids.
    """
    model.eval()
    idxs, y_true, y_pred = [], [], []
    running = 0
    for b, (x, y) in enumerate(loader):
        if max_batches is not None and b >= max_batches:
            break
        x = x.to(device)
        logits = model(x)
        preds = logits.argmax(1).cpu().tolist()
        n = len(preds)
        idxs.extend(range(running, running + n))
        running += n
        y_true.extend(y.tolist())
        y_pred.extend(preds)
    return idxs, y_true, y_pred


def pick_examples(idxs, y_true, y_pred, n_each, seed):
    """Return (correct_idx, incorrect_idx) samples of up to n_each dataset indices."""
    rng = random.Random(seed)
    correct = [i for i, t, p in zip(idxs, y_true, y_pred) if t == p]
    wrong = [i for i, t, p in zip(idxs, y_true, y_pred) if t != p]
    rng.shuffle(correct)
    rng.shuffle(wrong)
    return correct[:n_each], wrong[:n_each]


def save_example_grid(ds, chosen_idx, y_true, y_pred, out_path, title):
    """Load raw spectrograms for chosen indices and save a labeled grid."""
    if not chosen_idx:
        print(f"[eval] no examples for {out_path.name} (none in this category)")
        return
    # y_true/y_pred are aligned with the full index list; build lookup by index.
    true_by_idx = dict(zip(range(len(y_true)), y_true))
    pred_by_idx = dict(zip(range(len(y_pred)), y_pred))
    images = [ds.raw_image(i) for i in chosen_idx]
    trues = [true_by_idx[i] for i in chosen_idx]
    preds = [pred_by_idx[i] for i in chosen_idx]
    plot_prediction_examples(images, trues, preds, config.CLASSES, out_path, title)


def main():
    ap = argparse.ArgumentParser(description="Evaluate a trained model on the test set.")
    ap.add_argument("--model", required=True, help="model name (must match training)")
    ap.add_argument("--checkpoint", default=None,
                    help="path to checkpoint (default outputs/<model>/checkpoint_best.pt)")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    ap.add_argument("--num-workers", type=int, default=config.NUM_WORKERS)
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--n-examples", type=int, default=10,
                    help="how many correct and incorrect examples to plot")
    ap.add_argument("--h5-path", default=None)
    ap.add_argument("--metadata-csv", default=None)
    ap.add_argument("--max-batches", type=int, default=None, help="smoke tests only")
    args = ap.parse_args()

    set_seed(args.seed)
    device = config.get_device()

    ckpt_path = Path(args.checkpoint) if args.checkpoint \
        else config.OUTPUTS_ROOT / args.model / "checkpoint_best.pt"
    ckpt = load_checkpoint(ckpt_path)
    model_name = ckpt.get("model_name", args.model)
    best_f1 = ckpt.get("best_val_macro_f1")
    best_f1_str = f"{best_f1:.4f}" if best_f1 is not None else "n/a"
    print(f"[eval] model={model_name}  checkpoint={ckpt_path.name}  "
          f"trained_best_val_macroF1={best_f1_str}")

    # Rebuild via registry (pretrained=False; we load the trained weights next).
    model = build_model(model_name, num_classes=ckpt["num_classes"], pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)

    # Test data: use the same duration/loader the model was trained with.
    ds = GravitySpyDataset(
        args.split,
        loader=ckpt.get("loader", config.LOADER),
        duration=ckpt.get("duration", config.DURATION),
        h5_path=Path(args.h5_path) if args.h5_path else None,
        metadata_csv=Path(args.metadata_csv) if args.metadata_csv else None,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers,
                        pin_memory=torch.cuda.is_available())
    print(f"[eval] {args.split} samples: {len(ds)}")

    idxs, y_true, y_pred = run_inference(model, loader, device, args.max_batches)

    # ---- metrics ----
    report = full_report(y_true, y_pred, config.CLASSES)
    out_dir = config.OUTPUTS_ROOT / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "model_name": model_name,
        "split": args.split,
        "n_samples": len(y_true),
        "checkpoint": {
            "best_epoch": ckpt.get("best_epoch"),
            "best_val_macro_f1": ckpt.get("best_val_macro_f1"),
            "duration": ckpt.get("duration"),
        },
        "overall": report["overall"],
        "per_class": report["per_class"],
        "confusion": report["confusion"],
        "_note": "Accuracy is inflated by class imbalance; macro_f1 is the headline metric.",
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Per-class table as CSV (nice for the report / spreadsheets).
    import csv
    with open(out_dir / "per_class_metrics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class", "precision", "recall", "f1", "support"])
        for cls in config.CLASSES:
            r = report["per_class"][cls]
            w.writerow([cls, f"{r['precision']:.4f}", f"{r['recall']:.4f}",
                        f"{r['f1']:.4f}", r["support"]])

    # ---- figures ----
    plot_confusion_matrix(report["confusion"], config.CLASSES,
                          out_dir / "confusion_matrix.png", normalize=True)
    correct_idx, wrong_idx = pick_examples(idxs, y_true, y_pred, args.n_examples, args.seed)
    save_example_grid(ds, correct_idx, y_true, y_pred,
                      out_dir / "examples_correct.png", "Correct predictions")
    save_example_grid(ds, wrong_idx, y_true, y_pred,
                      out_dir / "examples_incorrect.png", "Incorrect predictions (true -> pred)")
    ds.close()

    # ---- console summary ----
    o = report["overall"]
    print("\n==== TEST RESULTS ====")
    print(f"  accuracy    : {o['accuracy']:.4f}   (inflated by imbalance)")
    print(f"  MACRO-F1    : {o['macro_f1']:.4f}   <- HEADLINE METRIC")
    print(f"  weighted-F1 : {o['weighted_f1']:.4f}")
    print(f"  macro P/R   : {o['macro_precision']:.4f} / {o['macro_recall']:.4f}")
    print(f"\n[eval] artifacts -> {out_dir}")


if __name__ == "__main__":
    main()

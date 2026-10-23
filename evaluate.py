"""Evaluate a checkpoint on the test set.

    python evaluate.py --model custom_cnn

Reads the model name, input size and training config out of the checkpoint, so
it works for any registered model without changes.

Writes to outputs/<model>/: metrics.json, per_class_metrics.md/.csv,
predictions.csv, confusion_matrix.png, examples_correct.png,
examples_incorrect.png, training_curves.png

Expect accuracy around 97%. Report MACRO-F1: the class distribution is skewed
enough that accuracy stays high regardless of how the rare classes are handled.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
import torch.nn.functional as F

from config import CLASSES, NUM_CLASSES, Config
from data.dataset import build_dataloaders
from data.transforms import normalisation_stats
from models import registry
from utils.metrics import compute_metrics, per_class_table
from utils.plotting import plot_confusion_matrix, plot_example_grid, plot_training_curves
from utils.seeding import resolve_device, set_seed

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw):
        return it


@torch.no_grad()
def run_inference(model, loader, device) -> Dict[str, Any]:
    """Run the model over a split, keeping images so the example grids can be
    drawn without a second pass over the data."""
    model.eval()
    logits, targets, ids, images = [], [], [], []
    for batch_images, batch_targets, batch_ids in tqdm(loader, desc="inference"):
        # No autocast: evaluation is cheap and full fp32 keeps the saved
        # probabilities exactly reproducible.
        out = model(batch_images.to(device, non_blocking=True)).float().cpu()
        logits.append(out.numpy())
        targets.extend(batch_targets.tolist())
        ids.extend(list(batch_ids))
        images.append(batch_images.numpy())

    probs = F.softmax(torch.from_numpy(np.concatenate(logits)), dim=1).numpy()
    return {"probs": probs, "preds": probs.argmax(1),
            "targets": np.array(targets), "ids": ids,
            "images": np.concatenate(images)}


def select_examples(results, n: int, correct: bool):
    """Pick examples for the figures.

    Correct ones are spread across different classes -- taking the top n by
    confidence would show eight Blips and teach nothing. Incorrect ones are the
    MOST CONFIDENT mistakes, which are the informative ones: a confident error
    shows what the model genuinely misunderstood.
    """
    preds, targets = results["preds"], results["targets"]
    confidences = results["probs"].max(1)
    idx = np.flatnonzero((preds == targets) if correct else (preds != targets))
    if idx.size == 0:
        return [], [], [], []

    if correct:
        by_class: Dict[int, List[int]] = {}
        for i in idx:
            by_class.setdefault(int(targets[i]), []).append(int(i))
        chosen, rank = [], 0
        while len(chosen) < n:
            added = False
            for cls in sorted(by_class):
                if rank < len(by_class[cls]):
                    chosen.append(by_class[cls][rank])
                    added = True
                    if len(chosen) >= n:
                        break
            if not added:
                break
            rank += 1
        selected = chosen[:n]
    else:
        selected = list(idx[np.argsort(-confidences[idx])][:n])

    return ([results["images"][i] for i in selected],
            [CLASSES[int(targets[i])] for i in selected],
            [CLASSES[int(preds[i])] for i in selected],
            [float(confidences[i]) for i in selected])


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Evaluate a trained model.")
    p.add_argument("--model", required=True)
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--split", default="test", choices=("test", "validation", "train"))
    p.add_argument("--data-root", type=Path, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--device", choices=("auto", "cuda", "cpu"), default=None)
    p.add_argument("--n-examples", type=int, default=8)
    p.add_argument("--output-root", type=Path, default=None)
    args = p.parse_args(argv)

    out_dir = Path(args.output_root or Config().output_root) / args.model
    ckpt_path = args.checkpoint or (out_dir / "checkpoint_best.pt")
    if not ckpt_path.exists():
        print("error: no checkpoint at %s. Run:\n  python train.py --model %s"
              % (ckpt_path, args.model), file=sys.stderr)
        return 2

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    # The training config travels with the checkpoint, so evaluation
    # automatically uses the same duration, normalisation and blank-image setting.
    cfg = Config.from_dict(ckpt["config"]).apply_overrides(
        data_root=args.data_root, num_workers=args.num_workers,
        device=args.device, output_root=args.output_root)
    input_size = tuple(ckpt["input_size"])
    class_names = ckpt["class_names"]

    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)

    print("=" * 66)
    print("Evaluating '%s' on the %s split" % (args.model, args.split))
    print("=" * 66)
    print("checkpoint : %s (epoch %s)" % (ckpt_path, ckpt.get("epoch")))

    loaders, grouped, _ = build_dataloaders(cfg, input_size, splits=(args.split,),
                                            shuffle_train=False)
    print("samples    : %d" % len(grouped[args.split]))

    # pretrained=False: every weight is about to be overwritten from the
    # checkpoint, so downloading ImageNet weights first would be wasted.
    model = registry.build_model(args.model, NUM_CLASSES, pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)

    results = run_inference(model, loaders[args.split], device)
    metrics = compute_metrics(results["targets"], results["preds"], class_names)
    metrics.update({
        "model": args.model, "split": args.split,
        "checkpoint_epoch": ckpt.get("epoch"), "input_size": list(input_size),
        "duration": cfg.duration, "seed": cfg.seed,
        "pretrained": bool(cfg.pretrained),
        "class_weighted_loss": bool(cfg.class_weights),
        "drop_blank_images": bool(cfg.drop_blank_images),
        "n_parameters": int(sum(p.numel() for p in model.parameters())),
    })

    history_path = out_dir / "history.json"
    if history_path.exists():
        history = json.loads(history_path.read_text(encoding="utf-8"))
        metrics["epochs_trained"] = len(history["train_loss"])
        metrics["best_epoch"] = history.get("best_epoch")
        plot_training_curves(history, out_dir / "training_curves.png",
                             "%s -- training history" % args.model)

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2),
                                          encoding="utf-8")
    (out_dir / "per_class_metrics.md").write_text(
        "# Per-class metrics -- %s (%s split)\n\n%s\n"
        % (args.model, args.split, per_class_table(metrics)), encoding="utf-8")

    with (out_dir / "per_class_metrics.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["class", "precision", "recall", "f1", "support"])
        for name, v in metrics["per_class"].items():
            w.writerow([name, "%.6f" % v["precision"], "%.6f" % v["recall"],
                        "%.6f" % v["f1"], v["support"]])

    with (out_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["gravityspy_id", "true_label", "predicted_label",
                    "confidence", "correct"])
        for i, gid in enumerate(results["ids"]):
            true_name = class_names[int(results["targets"][i])]
            pred_name = class_names[int(results["preds"][i])]
            w.writerow([gid, true_name, pred_name,
                        "%.6f" % results["probs"][i].max(),
                        int(true_name == pred_name)])

    plot_confusion_matrix(metrics["confusion_matrix"], class_names,
                          out_dir / "confusion_matrix.png",
                          "%s -- confusion matrix" % args.model)

    mean, std = normalisation_stats(cfg)
    for correct, filename, title in (
            (True, "examples_correct.png", "correct predictions"),
            (False, "examples_incorrect.png", "most confident mistakes")):
        plot_example_grid(*select_examples(results, args.n_examples, correct),
                          out_dir / filename, "%s -- %s" % (args.model, title),
                          mean, std)

    print("-" * 66)
    print("macro-F1          : %.4f   <-- headline metric" % metrics["macro_f1"])
    print("accuracy          : %.4f   (inflated by class imbalance)"
          % metrics["accuracy"])
    print("balanced accuracy : %.4f" % metrics["balanced_accuracy"])
    print("weighted-F1       : %.4f" % metrics["weighted_f1"])
    print("\nweighted-F1 minus macro-F1 = %.4f  (larger gap = rare classes neglected)"
          % (metrics["weighted_f1"] - metrics["macro_f1"]))

    print("\nweakest classes by F1:")
    for name, v in sorted(metrics["per_class"].items(), key=lambda kv: kv[1]["f1"])[:5]:
        print("  %-22s F1=%.3f (support=%d)" % (name, v["f1"], v["support"]))

    print("-" * 66)
    print("written to %s" % out_dir)
    print("\nOnce every model is evaluated: python compare.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

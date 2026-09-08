"""
train.py — Model-agnostic training loop.

Works for ANY registered model with zero model-specific code beyond the single
registry call. Usage:

    python train.py --model custom_cnn
    python train.py --model transfer_cnn --epochs 30 --no-class-weights
    python train.py --model transformer --no-pretrained

What it does:
  * seeds everything (reproducible)
  * builds train/val datasets + loaders (Windows num_workers auto-retry)
  * builds the model via registry.build_model(name, num_classes, pretrained)
  * class-weighted CrossEntropy (toggleable), AdamW, ReduceLROnPlateau
  * early stopping on VALIDATION MACRO-F1 (the imbalance-aware metric)
  * saves best checkpoint + training curves + run snapshot to outputs/<model>/
"""

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config
from data.dataset import GravitySpyDataset, compute_class_weights
from models.registry import build_model, available_models
from utils import reproducibility as repro
from utils.metrics import headline_metrics
from utils.plotting import plot_training_curves
from utils.seed import make_generator, seed_worker, set_seed


# --------------------------------------------------------------------------- #
# DataLoader construction with Windows-safe num_workers fallback
# --------------------------------------------------------------------------- #
def build_loader(dataset, batch_size, shuffle, num_workers, seed):
    """Create a DataLoader; if multi-worker startup fails, retry with 0 workers.

    On Windows, worker processes are *spawned* (the script is re-imported per
    worker), which occasionally fails for environment reasons. We force the
    workers to start by pulling one batch inside a try/except; on failure we
    transparently rebuild single-process (num_workers=0).
    """
    def _make(nw):
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=nw,
            worker_init_fn=seed_worker if nw > 0 else None,
            generator=make_generator(seed),
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )

    loader = _make(num_workers)
    if num_workers > 0:
        try:
            next(iter(loader))          # force workers to spawn now
        except Exception as e:          # noqa: BLE001
            print(f"[train] num_workers={num_workers} failed ({type(e).__name__}); "
                  "falling back to num_workers=0.")
            loader = _make(0)
    return loader


# --------------------------------------------------------------------------- #
# One epoch
# --------------------------------------------------------------------------- #
def run_epoch(model, loader, criterion, optimizer, device, train: bool,
              max_batches=None):
    """Run one pass. If train=True, updates weights. Returns (avg_loss, y_true, y_pred)."""
    model.train(train)
    total_loss, n_seen = 0.0, 0
    all_true, all_pred = [], []

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for b, (x, y) in enumerate(loader):
            if max_batches is not None and b >= max_batches:
                break
            x, y = x.to(device), y.to(device)
            if train:
                optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * x.size(0)
            n_seen += x.size(0)
            all_true.extend(y.detach().cpu().tolist())
            all_pred.extend(logits.detach().argmax(1).cpu().tolist())

    avg_loss = total_loss / max(n_seen, 1)
    return avg_loss, all_true, all_pred


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Train a Gravity Spy classifier.")
    ap.add_argument("--model", required=True,
                    help=f"model name; registered: {available_models()}")
    ap.add_argument("--epochs", type=int, default=config.EPOCHS)
    ap.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    ap.add_argument("--weight-decay", type=float, default=config.WEIGHT_DECAY)
    ap.add_argument("--patience", type=int, default=config.EARLY_STOPPING_PATIENCE)
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--duration", type=float, default=config.DURATION)
    ap.add_argument("--loader", default=config.LOADER, choices=["h5", "png"])
    ap.add_argument("--num-workers", type=int, default=config.NUM_WORKERS,
                    help="DataLoader workers; use 0 on Windows if spawning hangs")
    # Toggles (store_false so the config default is "on").
    ap.add_argument("--no-class-weights", action="store_true",
                    help="disable class-weighted loss")
    ap.add_argument("--no-aug", action="store_true",
                    help="disable train augmentation")
    ap.add_argument("--no-pretrained", action="store_true",
                    help="build backbone without pretrained weights")
    # Overrides for quick tests / alternate data locations.
    ap.add_argument("--h5-path", default=None)
    ap.add_argument("--metadata-csv", default=None)
    ap.add_argument("--max-batches", type=int, default=None,
                    help="cap batches/epoch (smoke tests only)")
    args = ap.parse_args()

    # ---- reproducibility ----
    set_seed(args.seed)
    device = config.get_device()
    print(f"[train] model={args.model} device={device} seed={args.seed}")

    # ---- config toggles from CLI ----
    if args.no_aug:
        config.USE_AUGMENTATION = False   # affects build_transforms(train=True)
    use_class_weights = config.USE_CLASS_WEIGHTS and not args.no_class_weights

    ds_kwargs = dict(
        loader=args.loader,
        duration=args.duration,
        h5_path=Path(args.h5_path) if args.h5_path else None,
        metadata_csv=Path(args.metadata_csv) if args.metadata_csv else None,
    )
    train_ds = GravitySpyDataset("train", **ds_kwargs)
    val_ds = GravitySpyDataset("val", **ds_kwargs)
    print(f"[train] train={len(train_ds)}  val={len(val_ds)}")

    train_loader = build_loader(train_ds, args.batch_size, True,
                                config.NUM_WORKERS, args.seed)
    val_loader = build_loader(val_ds, args.batch_size, False,
                              config.NUM_WORKERS, args.seed)

    # ---- model ----  (the ONLY model-specific line in the whole file)
    model = build_model(args.model, num_classes=config.NUM_CLASSES,
                        pretrained=not args.no_pretrained).to(device)

    # ---- loss / optim / scheduler ----
    if use_class_weights:
        weights = compute_class_weights(train_ds).to(device)
        print("[train] class-weighted loss ON")
    else:
        weights = None
        print("[train] class-weighted loss OFF")
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3)

    # ---- output dir + run snapshot ----
    out_dir = config.OUTPUTS_ROOT / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    repro.save_snapshot(
        repro.snapshot_environment(config, extra=vars(args)),
        out_dir / "run_config.json",
    )

    # ---- training loop with early stopping on val macro-F1 ----
    history = {"train_loss": [], "val_loss": [], "val_accuracy": [], "val_macro_f1": []}
    best_metric = -1.0
    best_state = None
    best_epoch = -1
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, _, _ = run_epoch(model, train_loader, criterion, optimizer,
                                  device, train=True, max_batches=args.max_batches)
        va_loss, yt, yp = run_epoch(model, val_loader, criterion, optimizer,
                                    device, train=False, max_batches=args.max_batches)
        m = headline_metrics(yt, yp)
        scheduler.step(m["macro_f1"])

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["val_accuracy"].append(m["accuracy"])
        history["val_macro_f1"].append(m["macro_f1"])

        improved = m["macro_f1"] > best_metric
        flag = ""
        if improved:
            best_metric = m["macro_f1"]
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_no_improve = 0
            flag = "  <- best"
        else:
            epochs_no_improve += 1

        print(f"epoch {epoch:3d}/{args.epochs}  "
              f"train_loss={tr_loss:.4f}  val_loss={va_loss:.4f}  "
              f"val_acc={m['accuracy']:.4f}  val_macroF1={m['macro_f1']:.4f}  "
              f"({time.time()-t0:.1f}s){flag}")

        if epochs_no_improve >= args.patience:
            print(f"[train] early stopping: no macro-F1 improvement for "
                  f"{args.patience} epochs.")
            break

    # ---- persist best checkpoint + curves + history ----
    checkpoint = {
        "model_name": args.model,
        "state_dict": best_state if best_state is not None else model.state_dict(),
        "num_classes": config.NUM_CLASSES,
        "class_names": config.CLASSES,
        "pretrained": not args.no_pretrained,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_metric,
        "duration": args.duration,
        "loader": args.loader,
    }
    torch.save(checkpoint, out_dir / "checkpoint_best.pt")
    with open(out_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    plot_training_curves(history, out_dir / "training_curves.png")

    train_ds.close(); val_ds.close()
    print(f"[train] done. best val macro-F1={best_metric:.4f} @ epoch {best_epoch}")
    print(f"[train] artifacts -> {out_dir}")


if __name__ == "__main__":
    main()

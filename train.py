"""Train a model.

    python train.py --model custom_cnn
    python train.py --model transfer_cnn --lr 1e-4 --epochs 25
    python train.py --list-models

Model-agnostic: it asks models/registry.py for a build function and an input
size by name, and never imports a model module directly. That is what lets a
teammate add a model without touching this file.

Writes to outputs/<model>/: config.json, env.json, history.json,
checkpoint_best.pt, checkpoint_last.pt, training_curves.png
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

from config import CLASSES, NUM_CLASSES, Config
from data.dataset import build_dataloaders
from models import registry
from utils.env_info import format_env_summary, save_env_info
from utils.metrics import compute_class_weights, macro_f1
from utils.plotting import plot_training_curves
from utils.seeding import resolve_device, set_seed

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw):
        return it


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train a Gravity Spy classifier.")
    p.add_argument("--model", default=None, help="Registered model name.")
    p.add_argument("--list-models", action="store_true")

    p.add_argument("--data-root", type=Path, default=None)
    p.add_argument("--duration", default=None, help="0.5 / 1.0 / 2.0 / 4.0")

    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--warmup-epochs", type=int, default=None)
    p.add_argument("--patience", dest="early_stopping_patience", type=int, default=None)

    p.add_argument("--no-pretrained", dest="pretrained", action="store_false",
                   default=None)
    p.add_argument("--no-class-weights", dest="class_weights", action="store_false",
                   default=None)
    p.add_argument("--no-augment", dest="augment", action="store_false", default=None)
    p.add_argument("--horizontal-flip", action="store_true", default=None,
                   help="Enable time-reversal augmentation (see data/transforms.py).")
    p.add_argument("--drop-blank-images", action="store_true", default=None,
                   help="Exclude the 134 blank images. Off by default so the test "
                        "set stays canonical; all three models must match.")

    p.add_argument("--input-size", type=int, nargs=2, metavar=("H", "W"), default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", choices=("auto", "cuda", "cpu"), default=None)
    p.add_argument("--no-amp", dest="amp", action="store_false", default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--output-root", type=Path, default=None)
    return p


def build_scheduler(optimizer, cfg: Config) -> LambdaLR:
    """Linear warmup then cosine decay, as a per-epoch multiplier.

    The warmup exists because the randomly initialised head and the weighted
    loss both produce large gradients early on, which can knock a pretrained
    backbone off its weights before it has learned anything.
    """
    warmup = max(0, cfg.warmup_epochs)

    def lr_lambda(epoch: int) -> float:
        if warmup and epoch < warmup:
            return (epoch + 1) / warmup      # +1 so epoch 0 is not a dead epoch
        progress = (epoch - warmup) / max(1, cfg.epochs - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    return LambdaLR(optimizer, lr_lambda)


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, cfg,
                    epoch, total_epochs):
    model.train()
    running_loss = n_seen = n_correct = 0
    use_amp = cfg.amp and device.type == "cuda"
    bar = tqdm(loader, desc="epoch %d/%d" % (epoch, total_epochs), leave=False)

    for images, targets, _ in bar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, targets)

        if use_amp:
            scaler.scale(loss).backward()
            # Unscale before clipping, or the threshold would apply to
            # loss-scaled gradients and mean something different each step.
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

        running_loss += loss.item() * targets.size(0)
        n_seen += targets.size(0)
        n_correct += (outputs.argmax(1) == targets).sum().item()

    return running_loss / n_seen, n_correct / n_seen


@torch.no_grad()
def validate(model, loader, criterion, device, cfg):
    model.eval()
    running_loss = n_seen = 0
    preds, targets_all = [], []
    use_amp = cfg.amp and device.type == "cuda"

    for images, targets, _ in tqdm(loader, desc="  val", leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, targets)
        running_loss += loss.item() * targets.size(0)
        n_seen += targets.size(0)
        preds.extend(outputs.argmax(1).cpu().tolist())
        targets_all.extend(targets.cpu().tolist())

    preds = np.array(preds)
    targets_all = np.array(targets_all)
    return (running_loss / n_seen,
            float((preds == targets_all).mean()),
            macro_f1(targets_all, preds, NUM_CLASSES))


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list_models:
        print("Registered models:\n" + registry.describe_models())
        return 0
    if args.model is None:
        print("error: --model required. Registered models:\n"
              + registry.describe_models(), file=sys.stderr)
        return 2

    overrides = {k: v for k, v in vars(args).items()
                 if k not in ("list_models", "input_size") and v is not None}
    cfg = Config().apply_overrides(**overrides)
    if args.input_size:
        cfg = cfg.apply_overrides(input_size=tuple(args.input_size))

    spec = registry.get_spec(cfg.model)
    input_size = tuple(cfg.input_size) if cfg.input_size else spec.input_size
    out_dir = cfg.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Seed before anything that uses randomness: model init, shuffling, augmentation.
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)

    env = save_env_info(out_dir / "env.json")
    print("=" * 66)
    print("Training '%s'" % cfg.model)
    print("=" * 66)
    print(format_env_summary(env))
    print("device     : %s | input %dx%d | seed %d"
          % (device, input_size[0], input_size[1], cfg.seed))
    print("-" * 66)

    loaders, grouped, counts = build_dataloaders(
        cfg, input_size, splits=("train", "validation"))
    train_counts = [counts["train"].get(name, 0) for name in CLASSES]
    print("split      : train=%d validation=%d test=%d"
          % (len(grouped["train"]), len(grouped["validation"]), len(grouped["test"])))
    print("imbalance  : %d ... %d  (%.1fx)"
          % (min(train_counts), max(train_counts),
             max(train_counts) / max(min(train_counts), 1)))

    model = registry.build_model(cfg.model, NUM_CLASSES, cfg.pretrained).to(device)
    print("parameters : {:,}".format(sum(p.numel() for p in model.parameters())))

    if cfg.class_weights:
        weights = compute_class_weights(train_counts)
        weight_tensor = torch.tensor(weights, device=device)
        print("weighting  : 1/sqrt(n), range %.2f-%.2f"
              % (weights.min(), weights.max()))
    else:
        weight_tensor = None
        print("weighting  : disabled")

    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    # Validation loss is UNWEIGHTED, so it measures the validation set as it
    # really is. That is also why early stopping uses macro-F1 rather than val
    # loss -- the two would otherwise optimise different objectives.
    val_criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                  weight_decay=cfg.weight_decay)
    scheduler = build_scheduler(optimizer, cfg)
    use_amp = cfg.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None

    cfg.apply_overrides(input_size=input_size).save(out_dir / "config.json")

    history: Dict[str, object] = {k: [] for k in
                                  ("train_loss", "train_accuracy", "val_loss",
                                   "val_accuracy", "val_macro_f1", "lr")}
    best_f1, best_epoch, stale = -1.0, 0, 0

    print("-" * 66)
    for epoch in range(1, cfg.epochs + 1):
        started = time.time()
        current_lr = optimizer.param_groups[0]["lr"]

        train_loss, train_acc = train_one_epoch(
            model, loaders["train"], criterion, optimizer, scaler, device, cfg,
            epoch, cfg.epochs)
        val_loss, val_acc, val_f1 = validate(
            model, loaders["validation"], val_criterion, device, cfg)

        # After the optimizer -- stepping first would skip the initial LR.
        scheduler.step()

        for key, value in (("train_loss", train_loss), ("train_accuracy", train_acc),
                           ("val_loss", val_loss), ("val_accuracy", val_acc),
                           ("val_macro_f1", val_f1), ("lr", current_lr)):
            history[key].append(value)

        improved = val_f1 > best_f1
        print("epoch %3d/%d | lr %.2e | train %.4f/%.4f | val %.4f/%.4f "
              "macroF1 %.4f | %4.1fs%s"
              % (epoch, cfg.epochs, current_lr, train_loss, train_acc,
                 val_loss, val_acc, val_f1, time.time() - started,
                 "  *" if improved else ""))

        if improved:
            best_f1, best_epoch, stale = val_f1, epoch, 0
            torch.save({"model": cfg.model, "epoch": epoch,
                        "state_dict": model.state_dict(),
                        "config": cfg.apply_overrides(input_size=input_size).to_dict(),
                        "class_names": CLASSES, "input_size": list(input_size),
                        "val_macro_f1": val_f1},
                       out_dir / "checkpoint_best.pt")
        else:
            stale += 1

        if epoch >= cfg.min_epochs and stale >= cfg.early_stopping_patience:
            print("early stopping: no improvement for %d epochs" % stale)
            break

    torch.save({"model": cfg.model, "epoch": len(history["train_loss"]),
                "state_dict": model.state_dict(),
                "config": cfg.apply_overrides(input_size=input_size).to_dict(),
                "class_names": CLASSES, "input_size": list(input_size)},
               out_dir / "checkpoint_last.pt")

    history["best_epoch"] = best_epoch
    history["best_val_macro_f1"] = best_f1
    (out_dir / "history.json").write_text(json.dumps(history, indent=2),
                                          encoding="utf-8")
    plot_training_curves(history, out_dir / "training_curves.png",
                         "%s -- training history" % cfg.model)

    print("-" * 66)
    print("best epoch : %d (val macro-F1 %.4f)" % (best_epoch, best_f1))
    print("written to : %s" % out_dir)
    print("\nNext: python evaluate.py --model %s" % cfg.model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

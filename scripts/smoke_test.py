"""End-to-end check on synthetic data.

    python scripts/smoke_test.py
    python scripts/smoke_test.py --all-models

Builds a small fake dataset in the same HDF5 layout as the real file, then runs
inspect_data -> train -> evaluate -> compare against it. Runs in under a minute
and needs no download, so it works before the data arrives and is the quickest
way for a teammate to check their model satisfies the interface.

It proves the plumbing works. It proves nothing about accuracy -- the synthetic
images are crude per-class patterns.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import h5py
import numpy as np

from config import CLASSES, NATIVE_IMAGE_SIZE


def make_synthetic_h5(path: Path, seed: int = 0) -> int:
    """Write a fake trainingsetv1d1.h5 with the real group layout.

    All four duration views of a glitch go into the SAME split, which is the
    property the real dataset has and that data/dataset.py relies on.
    """
    rng = np.random.default_rng(seed)
    height, width = NATIVE_IMAGE_SIZE
    n = 0

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        for ci, label in enumerate(CLASSES):
            # Uneven on purpose, like the real data: a couple of big classes and
            # a long tail. Exercises class weighting and the metrics' guards.
            total = 40 if ci == 3 else (20 if ci < 6 else (12 if ci < 14 else 8))
            for split, count in (("train", int(total * 0.7)),
                                 ("validation", max(1, int(total * 0.15))),
                                 ("test", max(1, int(total * 0.15)))):
                for k in range(count):
                    # Include the class index in the id: truncated names collide
                    # (Low_Frequency_Burst / Low_Frequency_Lines) and would trip
                    # the leakage check for reasons unrelated to the pipeline.
                    gid = "c%02d_%s_%03d" % (ci, split[:2], k)
                    base = rng.normal(0.35, 0.06, (height, width)).astype(np.float32)
                    row = (ci * 37) % (height - 20) + 5
                    col = (ci * 53) % (width - 20) + 5
                    motif = ci % 3
                    if motif == 0:
                        base[row:row + 4, :] += 0.5          # horizontal line
                    elif motif == 1:
                        base[:, col:col + 5] += 0.5          # vertical burst
                    else:
                        base[row:row + 15, col:col + 15] += 0.6   # blob
                    for duration in ("0.5", "1.0", "2.0", "4.0"):
                        jitter = rng.normal(0, 0.02, base.shape).astype(np.float32)
                        f.create_dataset(
                            "%s/%s/%s/%s.png" % (label, split, gid, duration),
                            data=np.clip(base + jitter, 0, 1)[None, :, :])
                    n += 1
    return n


def run(cmd: List[str], label: str) -> bool:
    print("\n" + "=" * 66)
    print("RUN: %s" % label)
    print("=" * 66, flush=True)
    ok = subprocess.run(cmd, cwd=str(REPO_ROOT)).returncode == 0
    print("---> %s" % ("PASS" if ok else "FAIL"))
    return ok


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="End-to-end pipeline smoke test.")
    p.add_argument("--keep", action="store_true")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--all-models", "--with-teammate-stubs", dest="all_models",
                   action="store_true",
                   help="Also train transfer_cnn and transformer (downloads pretrained weights on first run).")
    args = p.parse_args(argv)

    work = Path(tempfile.mkdtemp(prefix="gravityspy_smoke_"))
    data_root, out_root = work / "dataset", work / "outputs"
    python = sys.executable

    n = make_synthetic_h5(data_root / "trainingsetv1d1.h5")
    print("synthetic dataset: %d glitches x 4 views in %s" % (n, work))

    common = ["--data-root", str(data_root)]
    results = {}
    for cmd, label in (
        ([python, "data/inspect_data.py", *common, "--show-augmented",
          "--out-dir", str(out_root / "data_inspection")], "inspect_data.py"),
        ([python, "train.py", "--model", "custom_cnn", *common,
          "--epochs", str(args.epochs), "--batch-size", "16", "--num-workers", "0",
          "--warmup-epochs", "1", "--output-root", str(out_root)], "train.py"),
        ([python, "evaluate.py", "--model", "custom_cnn", *common,
          "--num-workers", "0", "--n-examples", "6",
          "--output-root", str(out_root)], "evaluate.py"),
        ([python, "compare.py", "--output-root", str(out_root)], "compare.py"),
    ):
        results[label] = run(cmd, label)

    if args.all_models:
        for name in ("transfer_cnn", "transformer"):
            ok = run([python, "train.py", "--model", name, *common, "--epochs", "1",
                      "--batch-size", "8", "--num-workers", "0",
                      "--output-root", str(out_root)], "train.py --model %s" % name)
            results["train.py --model %s" % name] = ok

    required = [out_root / "custom_cnn" / f for f in (
        "metrics.json", "confusion_matrix.png", "training_curves.png",
        "examples_correct.png", "examples_incorrect.png", "per_class_metrics.md",
        "per_class_metrics.csv", "predictions.csv", "checkpoint_best.pt",
        "config.json", "env.json", "history.json")]
    required += [out_root / "comparison" / f for f in
                 ("comparison.md", "comparison.csv", "model_comparison.png")]
    required += [out_root / "data_inspection" / f for f in
                 ("class_distribution.png", "examples_per_class.png",
                  "class_counts.csv")]
    missing = [f for f in required if not f.exists()]

    print("\n" + "=" * 66)
    print("SUMMARY")
    print("=" * 66)
    for label, ok in results.items():
        print("  %-32s %s" % (label, "PASS" if ok else "FAIL"))
    print("  %-32s %s" % ("artefacts (%d)" % len(required),
                          "PASS" if not missing else "FAIL: %s" % missing))

    core_ok = all(results.values()) and not missing
    if args.keep:
        print("\nkept: %s" % work)
    else:
        shutil.rmtree(work, ignore_errors=True)

    print("\nSMOKE TEST: %s" % ("PASS" if core_ok else "FAIL"))
    return 0 if core_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

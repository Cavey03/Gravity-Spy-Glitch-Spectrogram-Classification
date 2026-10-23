"""Dataset sanity check. Run this before training.

    python data/inspect_data.py
    python data/inspect_data.py --show-augmented

Prints per-class counts per split, checks the split is well formed, reports the
blank images, and saves figures to outputs/data_inspection/.
"""

from __future__ import annotations

import argparse
import collections
import csv
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import CLASSES, SPLITS, Config
from data.dataset import (
    GravitySpyDataset, assert_no_id_leakage, build_index, counts_by_split,
    find_blank, split_records,
)
from data.transforms import Compose, Resize, build_transforms
from utils.plotting import plot_class_distribution, plot_sample_grid
from utils.seeding import set_seed


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Inspect the Gravity Spy dataset.")
    p.add_argument("--data-root", type=Path, default=None)
    p.add_argument("--duration", default=None)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--show-augmented", action="store_true",
                   help="Render one image under 8 independent augmentations.")
    args = p.parse_args(argv)

    cfg = Config().apply_overrides(data_root=args.data_root, duration=args.duration)
    set_seed(cfg.seed)
    out_dir = Path(args.out_dir or (cfg.output_root / "data_inspection"))
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 66)
    print("Gravity Spy dataset inspection")
    print("=" * 66)

    records, h5_path = build_index(cfg)
    print("source     : %s" % h5_path)
    print("duration   : %ss view" % cfg.duration)
    print("indexed    : %d glitches\n" % len(records))

    grouped = split_records(records)
    assert_no_id_leakage(grouped)
    print("[OK] no glitch id appears in more than one split\n")

    counts = counts_by_split(records)
    header = "%-24s %8s %12s %8s %8s" % ("Class", "train", "validation", "test", "total")
    print(header); print("-" * len(header))
    rows = []
    totals = {s: 0 for s in SPLITS}
    for name in CLASSES:
        vals = [counts[s].get(name, 0) for s in SPLITS]
        for s, v in zip(SPLITS, vals):
            totals[s] += v
        print("%-24s %8d %12d %8d %8d%s"
              % (name, vals[0], vals[1], vals[2], sum(vals),
                 "  <-- rare" if sum(vals) < 100 else ""))
        rows.append([name] + vals + [sum(vals)])
    print("-" * len(header))
    print("%-24s %8d %12d %8d %8d"
          % ("TOTAL", totals["train"], totals["validation"], totals["test"],
             sum(totals.values())))
    rows.append(["TOTAL"] + [totals[s] for s in SPLITS] + [sum(totals.values())])

    train_counts = [counts["train"].get(c, 0) for c in CLASSES]
    print("\nimbalance  : largest training class is %.1fx the smallest (%d vs %d)"
          % (max(train_counts) / max(min(train_counts), 1),
             max(train_counts), min(train_counts)))
    print("             -> class-weighted loss is on; report macro-F1.")

    # Blank images: a defect in the published dataset, reported up front so it
    # is not discovered halfway through writing up a confusion matrix.
    print("\nscanning for blank images...")
    blank = find_blank(records, h5_path)
    if blank:
        by_class = collections.Counter(r.label for r in records if r.locator in blank)
        totals_by_class = {c: sum(counts[s].get(c, 0) for s in SPLITS) for c in CLASSES}
        print("[WARN] %d/%d images (%.2f%%) are blank -- every pixel identical."
              % (len(blank), len(records), 100 * len(blank) / len(records)))
        print("       A dataset defect, not a loading error; most are blank at all")
        print("       four durations. Kept by default to preserve the canonical")
        print("       split (--drop-blank-images to omit).")
        for name, n in by_class.most_common():
            print("         %-22s %4d / %4d  (%.1f%%)"
                  % (name, n, totals_by_class[name], 100 * n / totals_by_class[name]))

    with (out_dir / "class_counts.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["class", "train", "validation", "test", "total"])
        w.writerows(rows)

    plot_class_distribution(counts, CLASSES, out_dir / "class_distribution.png")

    # One example per class, unnormalised -- this figure should show the data as
    # it is, not as the network sees it.
    seen, picks = set(), []
    for rec in grouped["train"]:
        if rec.label not in seen:
            seen.add(rec.label)
            picks.append(rec)
    ds = GravitySpyDataset(picks, h5_path, Compose([Resize((140, 170))]))
    plot_sample_grid([ds[i][0][0].numpy() for i in range(len(ds))],
                     [CLASSES[ds[i][1]] for i in range(len(ds))],
                     out_dir / "examples_per_class.png",
                     "Gravity Spy -- one example per class (%ss view)" % cfg.duration)

    if args.show_augmented:
        aug = build_transforms(cfg, (140, 170), train=True)
        aug.transforms = [t for t in aug.transforms if type(t).__name__ != "Normalise"]
        ds = GravitySpyDataset(grouped["train"][:1], h5_path, aug)
        plot_sample_grid([ds[0][0][0].numpy() for _ in range(8)],
                         ["augmented"] * 8, out_dir / "examples_augmented.png",
                         "One image under 8 augmentations (time shift + "
                         "SpecAugment + brightness/contrast)", ncols=4)

    print("\nfigures written to %s" % out_dir)
    print("Next: python train.py --model custom_cnn")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

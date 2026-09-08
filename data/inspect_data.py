"""
inspect_data.py — Look at the data BEFORE training.

Three jobs:
  1. Per-class counts per split (train/validation/test) -> see the imbalance.
  2. --h5-tree : print the real HDF5 layout (handy to confirm the file schema).
  3. A representative example grid: one real spectrogram per class, saved as a
     figure so you can eyeball that images and labels are sane.

Run:
    python -m data.inspect_data                 # counts + example grid
    python -m data.inspect_data --h5-tree       # also dump the h5 structure
    python -m data.inspect_data --no-grid       # counts only (fast, no h5 read)

Outputs go to  outputs/inspection/.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend: save files, never open a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config


# --------------------------------------------------------------------------- #
# 1. Per-class counts
# --------------------------------------------------------------------------- #
def class_counts(metadata_csv: Path) -> pd.DataFrame:
    """Return a table: rows = 22 classes, cols = train/validation/test/total.

    Built purely from the CSV (no h5 read), so it's instant. Sorted by total
    descending to make the imbalance obvious at a glance.
    """
    df = pd.read_csv(metadata_csv)
    df["sample_type"] = df["sample_type"].str.strip().str.lower()

    table = (
        df.groupby(["label", "sample_type"]).size().unstack(fill_value=0)
    )
    # Ensure all three split columns exist and are ordered.
    for col in ["train", "validation", "test"]:
        if col not in table.columns:
            table[col] = 0
    table = table[["train", "validation", "test"]]
    table["total"] = table.sum(axis=1)
    # Reindex to the canonical class order, then sort by total for display.
    table = table.reindex(config.CLASSES, fill_value=0)
    return table.sort_values("total", ascending=False)


def print_counts(table: pd.DataFrame) -> None:
    print("\nPer-class counts per split (sorted by total):")
    print("-" * 60)
    print(f"{'class':<22}{'train':>8}{'val':>8}{'test':>8}{'total':>8}")
    print("-" * 60)
    for cls, row in table.iterrows():
        print(f"{cls:<22}{row['train']:>8}{row['validation']:>8}"
              f"{row['test']:>8}{row['total']:>8}")
    print("-" * 60)
    print(f"{'TOTAL':<22}{table['train'].sum():>8}{table['validation'].sum():>8}"
          f"{table['test'].sum():>8}{table['total'].sum():>8}")
    # Imbalance headline: ratio of biggest to smallest class.
    biggest, smallest = table["total"].max(), table["total"].min()
    print(f"\nImbalance ratio (largest / smallest class): "
          f"{biggest} / {smallest} = {biggest / max(smallest, 1):.1f}x")


# --------------------------------------------------------------------------- #
# 2. HDF5 tree
# --------------------------------------------------------------------------- #
def print_h5_tree(h5_path: Path, max_examples: int = 2) -> None:
    import h5py

    print(f"\nHDF5 structure of {h5_path.name}:")
    with h5py.File(h5_path, "r") as f:
        top = list(f.keys())
        print(f"  top-level: {len(top)} groups (class labels), e.g. {top[:3]}")
        label = top[0]
        splits = list(f[label].keys())
        print(f"  f['{label}'] -> {splits}")
        split = splits[0]
        ids = list(f[label][split].keys())
        print(f"  f['{label}']['{split}'] -> {len(ids)} glitch ids, e.g. {ids[:2]}")
        gid = ids[0]
        views = list(f[label][split][gid].keys())
        print(f"  f['{label}']['{split}']['{gid}'] -> {views}")
        arr = f[label][split][gid][views[0]]
        print(f"    dataset '{views[0]}': shape={arr.shape} dtype={arr.dtype}")


# --------------------------------------------------------------------------- #
# 3. Example grid
# --------------------------------------------------------------------------- #
def example_grid(out_path: Path, split: str = "train") -> None:
    """Save a grid with one real spectrogram per class (from the given split)."""
    from data.dataset import GravitySpyDataset

    ds = GravitySpyDataset(split)  # uses config paths
    # Find the first index for each class so every class is represented.
    first_idx_per_class = {}
    for i in range(len(ds)):
        _, label = ds.get_id_label(i)
        if label not in first_idx_per_class:
            first_idx_per_class[label] = i
        if len(first_idx_per_class) == config.NUM_CLASSES:
            break

    n = config.NUM_CLASSES
    cols = 5
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.4, rows * 2.4))
    axes = axes.ravel()
    for ax in axes:
        ax.axis("off")

    for panel, cls in enumerate(config.CLASSES):
        ax = axes[panel]
        if cls in first_idx_per_class:
            img = ds.raw_image(first_idx_per_class[cls])   # (H, W) in [0,1]
            ax.imshow(img, cmap="viridis", aspect="auto", origin="lower")
        ax.set_title(cls, fontsize=7)
    ds.close()

    fig.suptitle(f"One example per class — split='{split}', {config.DURATION}s view",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"\nSaved example grid -> {out_path}")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Inspect the Gravity Spy dataset.")
    ap.add_argument("--h5-tree", action="store_true", help="print the HDF5 layout")
    ap.add_argument("--no-grid", action="store_true", help="skip the example grid")
    ap.add_argument("--grid-split", default="train", help="split to draw the grid from")
    args = ap.parse_args()

    out_dir = config.OUTPUTS_ROOT / "inspection"

    table = class_counts(config.METADATA_CSV)
    print_counts(table)
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "class_counts.csv")
    print(f"Saved counts table -> {out_dir / 'class_counts.csv'}")

    if args.h5_tree:
        print_h5_tree(config.H5_PATH)

    if not args.no_grid:
        example_grid(out_dir / "example_grid.png", split=args.grid_split)


if __name__ == "__main__":
    main()

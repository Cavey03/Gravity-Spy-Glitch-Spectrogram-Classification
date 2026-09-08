"""
make_synthetic.py — Generate a TINY fake dataset in the real file layout.

This is NOT scientific data. It exists so the pipeline can be run and tested
end-to-end without the 8.6 GB download, and so teammates can smoke-test their
models before getting the real data.

It writes (mirroring the CONFIRMED real Gravity Spy v1.1 layout):
  <out>/trainingsetv1d1.h5   datasets at "<label>/<split>/<gid>/<dur>.png" (1,140,170)
  <out>/trainingset_v1d1_metadata.csv   columns: gravityspy_id,label,sample_type

The class counts are deliberately IMBALANCED (like the real set) so class-weight
and macro-F1 code paths are exercised.

Usage:
    python -m data.make_synthetic --out dataset_synth
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

import config

# A few real class names (subset) with intentionally imbalanced counts, split
# across train/validation/test. Format: label -> (n_train, n_val, n_test).
_SYNTH_PLAN = {
    "Blip":                (40, 8, 8),   # dominant class
    "Whistle":             (16, 4, 4),
    "Koi_Fish":            (12, 3, 3),
    "Scattered_Light":     (10, 3, 3),
    "Low_Frequency_Burst": (6, 2, 2),
    "Chirp":               (3, 1, 1),    # tiny class
}


def _fake_spectrogram(rng: np.random.Generator, class_idx: int) -> np.ndarray:
    """A (1,140,170) grey 'spectrogram' with a faint class-dependent pattern.

    We bake in a weak, class-specific structure so a model can actually learn
    something above chance on synthetic data (useful for a real smoke test).
    """
    h, w = 140, 170
    base = rng.normal(0.3, 0.05, size=(h, w)).astype(np.float32)
    # A horizontal band whose frequency-row depends on the class -> learnable.
    row = 10 + (class_idx * 17) % (h - 20)
    base[row:row + 6, :] += 0.4
    return np.clip(base, 0, 1)[None, :, :]   # (1, H, W)


def make_synthetic(out_dir: Path, seed: int = 0) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    h5_path = out_dir / "trainingsetv1d1.h5"
    csv_path = out_dir / "trainingset_v1d1_metadata.csv"

    records = []
    with h5py.File(h5_path, "w") as hf:
        counter = 0
        for label, (n_tr, n_va, n_te) in _SYNTH_PLAN.items():
            class_idx = config.CLASS_TO_IDX[label]
            for split, n in (("train", n_tr), ("validation", n_va), ("test", n_te)):
                for _ in range(n):
                    gid = f"G{counter:06d}"
                    counter += 1
                    img = _fake_spectrogram(rng, class_idx)
                    # Real layout: <label>/<split>/<gid>/<dur>.png
                    grp = hf.require_group(f"{label}/{split}/{gid}")
                    for dur in config.VALID_DURATIONS:
                        grp.create_dataset(f"{dur:.1f}.png", data=img, compression="gzip")
                    records.append({"gravityspy_id": gid, "label": label,
                                    "sample_type": split})

    pd.DataFrame(records).to_csv(csv_path, index=False)
    print(f"Wrote {len(records)} glitches x {len(config.VALID_DURATIONS)} views")
    print(f"  h5:  {h5_path}")
    print(f"  csv: {csv_path}")
    return out_dir


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dataset_synth", help="output directory")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    make_synthetic(Path(args.out), seed=args.seed)

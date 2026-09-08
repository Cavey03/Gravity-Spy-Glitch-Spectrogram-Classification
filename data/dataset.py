"""
dataset.py — The PyTorch Dataset for Gravity Spy glitch spectrograms.

Core design (see the module-level insight in the project README):
  * The metadata CSV is the INDEX of samples: it provides, per glitch,
    (gravityspy_id, label, sample_type). We filter by sample_type to get a
    split. This makes "respect the canonical split" the default and the only
    behavior -- there is deliberately NO random-splitting code path here.
  * The .h5 (or PNG tree) is the PIXEL STORE, addressed by gravityspy_id and
    duration. All HDF5-hierarchy knowledge is confined to _read_h5_image().

Each __getitem__ returns:  (image_tensor [3,H,W] float, label_index int)
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

import config
from data.transforms import build_transforms

# Canonical CSV split values are train / validation / test. We accept a few
# friendly aliases so callers can say "val".
_SPLIT_ALIASES = {
    "train": "train",
    "val": "validation",
    "valid": "validation",
    "validation": "validation",
    "test": "test",
}


def _duration_strings(duration: float) -> list[str]:
    """Return the plausible string spellings of a duration used in keys/files.

    e.g. 2.0 -> ["2.0", "2"]. Different exports spell the duration differently;
    trying both keeps the loader robust across file variants.
    """
    return [f"{duration:.1f}", f"{duration:g}"]


def _to_grey_float(arr: np.ndarray) -> torch.Tensor:
    """Normalize any raw spectrogram array to a (1, H, W) float tensor in [0,1].

    Handles the shape variants we might encounter:
      (H, W)          grey, no channel dim
      (1, H, W)       grey, channel-first  (the pre-cropped .h5 case)
      (3/4, H, W)     channel-first colour -> average to grey
      (H, W, 3/4)     channel-last colour  -> average to grey
    and the value-range variants (uint8 0..255, or float already in [0,1]).
    """
    a = np.asarray(arr)

    # Collapse to a single (H, W) grey plane.
    if a.ndim == 2:
        grey = a
    elif a.ndim == 3:
        if a.shape[0] in (1, 3, 4):          # channel-first
            grey = a[0] if a.shape[0] == 1 else a[:3].mean(axis=0)
        elif a.shape[-1] in (1, 3, 4):       # channel-last
            grey = a[..., 0] if a.shape[-1] == 1 else a[..., :3].mean(axis=-1)
        else:
            raise ValueError(f"Unrecognized 3D spectrogram shape {a.shape}")
    else:
        raise ValueError(f"Unrecognized spectrogram ndim {a.ndim} (shape {a.shape})")

    grey = grey.astype(np.float32)
    # Scale to [0, 1]. uint8-like ranges get /255; already-normalized stays.
    if grey.max() > 1.0:
        grey = grey / 255.0

    return torch.from_numpy(grey).unsqueeze(0)   # (1, H, W)


class GravitySpyDataset(Dataset):
    def __init__(
        self,
        split: str,
        transform=None,
        loader: str = None,
        duration: float = None,
        h5_path: Path = None,
        png_root: Path = None,
        metadata_csv: Path = None,
    ):
        """
        split       : "train" | "val"/"validation" | "test"
        transform   : callable; if None, build_transforms(train=split=='train')
        loader      : "h5" or "png"  (defaults to config.LOADER)
        duration    : which duration-view (defaults to config.DURATION)
        *_path      : override the config paths (handy for synthetic tests)
        """
        if split not in _SPLIT_ALIASES:
            raise ValueError(f"split must be one of {list(_SPLIT_ALIASES)}, got {split!r}")
        self.split_key = _SPLIT_ALIASES[split]
        self.is_train = self.split_key == "train"

        self.loader = loader or config.LOADER
        self.duration = duration if duration is not None else config.DURATION
        if self.duration not in config.VALID_DURATIONS:
            raise ValueError(f"duration {self.duration} not in {config.VALID_DURATIONS}")

        self.h5_path = Path(h5_path) if h5_path else config.H5_PATH
        self.png_root = Path(png_root) if png_root else config.PNG_ROOT
        self.metadata_csv = Path(metadata_csv) if metadata_csv else config.METADATA_CSV

        self.transform = transform if transform is not None else build_transforms(self.is_train)

        # Build the sample index from the CSV, filtered to this split.
        self._index = self._build_index()          # list of (gid, label)
        self.labels = [config.CLASS_TO_IDX[lbl] for _, lbl in self._index]

        # HDF5 handle is opened lazily (per worker process) -- h5py handles do
        # not survive process forking/spawning, so we never open it in __init__.
        self._h5 = None
        self._h5_key_template = None               # cached once discovered

    # ------------------------------------------------------------------ index
    def _build_index(self) -> list[tuple[str, str]]:
        if not self.metadata_csv.exists():
            raise FileNotFoundError(
                f"Metadata CSV not found at {self.metadata_csv}. It is REQUIRED: "
                "it defines the canonical train/validation/test split "
                "(sample_type column). See README for download instructions."
            )
        df = pd.read_csv(self.metadata_csv)
        required = {"gravityspy_id", "label", "sample_type"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Metadata CSV missing columns: {missing}")

        # Normalize sample_type spellings (e.g. 'val' -> 'validation').
        df["sample_type"] = df["sample_type"].map(
            lambda s: _SPLIT_ALIASES.get(str(s).strip().lower(), str(s).strip().lower())
        )
        rows = df[df["sample_type"] == self.split_key]

        index = []
        unknown_labels = set()
        for gid, label in zip(rows["gravityspy_id"], rows["label"]):
            if label not in config.CLASS_TO_IDX:
                unknown_labels.add(label)
                continue
            index.append((str(gid), str(label)))

        if unknown_labels:
            # Surface, don't silently drop -- likely a CLASSES/spelling mismatch.
            raise ValueError(
                f"Labels in CSV not present in config.CLASSES: {sorted(unknown_labels)}. "
                "Fix config.CLASSES to match the dataset's exact label strings."
            )
        if not index:
            raise ValueError(f"No samples found for split '{self.split_key}' in {self.metadata_csv}")
        return index

    def __len__(self) -> int:
        return len(self._index)

    # ------------------------------------------------------------------ pixels
    def _ensure_h5(self):
        if self._h5 is None:
            import h5py
            if not self.h5_path.exists():
                raise FileNotFoundError(
                    f"HDF5 file not found at {self.h5_path}. See README for download."
                )
            self._h5 = h5py.File(self.h5_path, "r")

    def _candidate_templates(self) -> list[str]:
        """HDF5 key templates to try, most-likely first.

        Placeholders {label} {split} {gid} {d} are filled per sample, where
        {d} is a duration spelling (e.g. '2.0'). The FIRST template is the
        confirmed real Gravity Spy v1.1 layout:
            <label>/<sample_type>/<gravityspy_id>/<dur>.png
        The rest are fallbacks so the loader still adapts if a file variant
        differs. Whichever resolves first is cached and reused.
        """
        return [
            "{label}/{split}/{gid}/{d}.png",   # <- confirmed real layout
            "{label}/{split}/{gid}/{d}",
            "{label}/{gid}/{d}.png",
            "{label}/{gid}_{d}",               # <- layout our synthetic mirrors
            "{gid}/{d}.png",
            "{gid}_{d}",
        ]

    def _read_h5_image(self, gid: str, label: str) -> torch.Tensor:
        self._ensure_h5()
        fill = dict(label=label, split=self.split_key, gid=gid)

        # Fast path: reuse the template that worked before.
        if self._h5_key_template is not None:
            for d in _duration_strings(self.duration):
                key = self._h5_key_template.format(d=d, **fill)
                if key in self._h5:
                    return _to_grey_float(self._h5[key][()])

        # Discover: try templates x duration spellings; remember what resolves.
        tried = []
        for template in self._candidate_templates():
            for d in _duration_strings(self.duration):
                key = template.format(d=d, **fill)
                tried.append(key)
                if key in self._h5:
                    self._h5_key_template = template
                    return _to_grey_float(self._h5[key][()])

        raise KeyError(
            f"Could not locate glitch {gid} (label {label}, duration {self.duration}) "
            f"in {self.h5_path}. Tried e.g. {tried[:4]}. "
            "Run `python -m data.inspect_data --h5-tree` to see the real layout."
        )

    def _read_png_image(self, gid: str, label: str) -> torch.Tensor:
        from PIL import Image
        # Layout: <PNG_ROOT>/<label>/<sample_type>/<gid>_<dur>.png
        for d in _duration_strings(self.duration):
            path = self.png_root / label / self.split_key / f"{gid}_{d}.png"
            if path.exists():
                arr = np.array(Image.open(path))
                # Raw PNGs carry plot axes -> crop to the spectrogram region.
                arr = arr[66:532, 105:671, :3]
                return _to_grey_float(arr)
        raise FileNotFoundError(
            f"PNG for glitch {gid} (label {label}, dur {self.duration}) not found "
            f"under {self.png_root / label / self.split_key}."
        )

    # ------------------------------------------------------------------ item
    def __getitem__(self, i: int):
        gid, label = self._index[i]
        if self.loader == "h5":
            img = self._read_h5_image(gid, label)
        elif self.loader == "png":
            img = self._read_png_image(gid, label)
        else:
            raise ValueError(f"Unknown loader {self.loader!r} (use 'h5' or 'png')")

        if self.transform is not None:
            img = self.transform(img)
        return img, config.CLASS_TO_IDX[label]

    def raw_image(self, i: int) -> np.ndarray:
        """Return the un-transformed grey image (H, W) in [0,1] for display.

        Bypasses normalization/augmentation so inspect_data / evaluate can show
        human-viewable spectrograms. Use ONLY for visualization, never training.
        """
        gid, label = self._index[i]
        if self.loader == "h5":
            img = self._read_h5_image(gid, label)
        else:
            img = self._read_png_image(gid, label)
        return img.squeeze(0).numpy()          # (H, W)

    def get_id_label(self, i: int) -> tuple[str, str]:
        """Return (gravityspy_id, label_string) for index i."""
        return self._index[i]

    def close(self):
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None


def compute_class_weights(dataset: GravitySpyDataset) -> torch.Tensor:
    """Inverse-frequency class weights from a dataset's label distribution.

    Used for class-weighted CrossEntropyLoss to counter the strong imbalance
    (Blip dominates). Weight_c = N_total / (num_classes * count_c); classes not
    present get weight 0 (they contribute no loss). Normalized to mean 1.
    """
    counts = np.zeros(config.NUM_CLASSES, dtype=np.float64)
    for lbl in dataset.labels:
        counts[lbl] += 1
    total = counts.sum()
    weights = np.zeros(config.NUM_CLASSES, dtype=np.float64)
    nonzero = counts > 0
    weights[nonzero] = total / (config.NUM_CLASSES * counts[nonzero])
    # Normalize so the average weight is ~1 (keeps loss scale stable).
    if weights[nonzero].mean() > 0:
        weights[nonzero] /= weights[nonzero].mean()
    return torch.tensor(weights, dtype=torch.float32)

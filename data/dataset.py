"""Dataset for the Gravity Spy v1.1 spectrograms (trainingsetv1d1.h5).

The HDF5 layout is:  <label>/<sample_type>/<gravityspy_id>/<duration>.png
with each dataset a (1, 140, 170) float32 array already scaled to [0, 1].

THE SPLIT
---------
We take train/validation/test straight from the file's own `sample_type` groups
and never split randomly. Each glitch is stored as four images (0.5/1.0/2.0/4.0s)
which are four views of the SAME event. Splitting per image would put views of
one glitch on both sides of the train/test boundary, and the model would be
scored on events it had already memorised. Because the split is assigned per
glitch, honouring it makes that impossible. assert_no_id_leakage() checks it
rather than assuming it.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from config import CLASS_TO_IDX, CLASSES, H5_NAME, SPLITS, Config
from data.transforms import build_transforms
from utils.seeding import make_generator, seed_worker


@dataclass(frozen=True)
class Record:
    """One glitch, at one duration."""
    gravityspy_id: str
    label: str
    split: str
    locator: str          # path to the array inside the HDF5 file


def build_index(cfg: Config) -> Tuple[List[Record], Path]:
    """List every glitch at the configured duration."""
    h5_path = Path(cfg.data_root) / H5_NAME
    if not h5_path.exists():
        raise FileNotFoundError(
            "%s not found. See README.md 'Getting the data'." % h5_path)

    leaf = "%s.png" % cfg.duration
    records: List[Record] = []
    with h5py.File(h5_path, "r") as f:
        for label in f:
            if label not in CLASS_TO_IDX:
                continue
            for split in f[label]:
                if split not in SPLITS:
                    continue
                for gid in f[label][split]:
                    records.append(Record(
                        gid, label, split,
                        "%s/%s/%s/%s" % (label, split, gid, leaf)))

    if not records:
        raise RuntimeError(
            "Indexed 0 glitches at duration %ss from %s." % (cfg.duration, h5_path))

    # Sort so the order does not depend on HDF5 iteration order -- otherwise a
    # fixed seed would not actually reproduce a run.
    records.sort(key=lambda r: (r.label, r.split, r.gravityspy_id))
    return records, h5_path


def split_records(records: Sequence[Record]) -> Dict[str, List[Record]]:
    grouped: Dict[str, List[Record]] = {s: [] for s in SPLITS}
    for rec in records:
        grouped[rec.split].append(rec)
    return grouped


def class_counts(records: Sequence[Record]) -> Dict[str, int]:
    counts = Counter(r.label for r in records)
    return {name: counts.get(name, 0) for name in CLASSES}


def counts_by_split(records: Sequence[Record]) -> Dict[str, Dict[str, int]]:
    return {s: class_counts(r) for s, r in split_records(records).items()}


def assert_no_id_leakage(grouped: Dict[str, List[Record]]) -> None:
    """Fail loudly if one glitch appears in more than one split.

    This should be impossible given we read the split from the file, but it is
    cheap to check and it is exactly the kind of error that yields a
    suspiciously good score and an inexplicable write-up.
    """
    seen: Dict[str, str] = {}
    offenders = set()
    for split, recs in grouped.items():
        for rec in recs:
            prior = seen.setdefault(rec.gravityspy_id, split)
            if prior != split:
                offenders.add(rec.gravityspy_id)
    if offenders:
        raise RuntimeError(
            "Split leakage: %d glitch id(s) in multiple splits, e.g. %s"
            % (len(offenders), sorted(offenders)[:5]))


class GravitySpyDataset(Dataset):
    """Returns (image, label index, glitch id).

    The id is carried through so evaluate.py can label the example figures.
    """

    def __init__(self, records: Sequence[Record], h5_path: Path, transform=None):
        self.records = list(records)
        self.h5_path = Path(h5_path)
        self.transform = transform
        self._h5 = None       # opened lazily -- see _get_h5

    def __len__(self) -> int:
        return len(self.records)

    def _get_h5(self):
        """Open the file on first use, separately in each worker process.

        An h5py handle must not be created here in __init__ and then inherited
        by DataLoader workers: HDF5 is not fork-safe and several processes
        sharing one handle corrupt data or crash. Opening lazily gives each
        worker its own handle, and keeps the Dataset picklable, which Windows
        needs because it spawns workers rather than forking them.
        """
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_h5"] = None
        return state

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        arr = np.asarray(self._get_h5()[rec.locator], dtype=np.float32)[0]
        img = torch.from_numpy(np.ascontiguousarray(arr)).unsqueeze(0)
        # Greyscale -> 3 identical channels so ImageNet-pretrained backbones
        # accept the input without altering their first convolution.
        img = img.expand(3, -1, -1).contiguous()
        if self.transform is not None:
            img = self.transform(img)
        return img, CLASS_TO_IDX[rec.label], rec.gravityspy_id


# 134 images in this release are entirely constant -- no signal at all. They are
# 24% of Extremely_Loud and 30% of Paired_Doves, and 107 of them are blank at
# all four durations, so changing --duration does not recover them. See README.
BLANK_STD_THRESHOLD = 1e-6


def find_blank(records: Sequence[Record], h5_path: Path) -> set:
    """Locators whose image has zero variance (every pixel identical)."""
    blank = set()
    with h5py.File(h5_path, "r") as f:
        for rec in records:
            if float(np.asarray(f[rec.locator]).std()) < BLANK_STD_THRESHOLD:
                blank.add(rec.locator)
    return blank


def build_dataloaders(
    cfg: Config,
    input_size: Tuple[int, int],
    splits: Sequence[str] = SPLITS,
    shuffle_train: bool = True,
) -> Tuple[Dict[str, DataLoader], Dict[str, List[Record]], Dict[str, Dict[str, int]]]:
    """Build DataLoaders. Returns (loaders, records by split, counts by split)."""
    records, h5_path = build_index(cfg)
    grouped = split_records(records)
    assert_no_id_leakage(grouped)

    if cfg.drop_blank_images:
        blank = find_blank(records, h5_path)
        for split in grouped:
            before = len(grouped[split])
            grouped[split] = [r for r in grouped[split] if r.locator not in blank]
            if len(grouped[split]) != before:
                print("[data] %-11s dropped %d blank image(s)"
                      % (split, before - len(grouped[split])))

    counts = {s: class_counts(r) for s, r in grouped.items()}
    generator = make_generator(cfg.seed)

    loaders: Dict[str, DataLoader] = {}
    for split in splits:
        recs = grouped.get(split, [])
        if not recs:
            continue
        is_train = split == "train"
        loaders[split] = DataLoader(
            GravitySpyDataset(recs, h5_path,
                              build_transforms(cfg, input_size, train=is_train)),
            batch_size=cfg.batch_size,
            shuffle=is_train and shuffle_train,
            num_workers=cfg.num_workers,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=seed_worker,
            generator=generator if is_train else None,
            persistent_workers=cfg.num_workers > 0,
        )
    return loaders, grouped, counts

"""Central configuration.

Gravity Spy training set v1.1 -- Zenodo DOI 10.5281/zenodo.1476551 (CC BY 4.0).
7,966 glitches, 22 classes, pre-split 5,587 train / 1,200 validation / 1,179 test.
Each glitch is rendered at four durations (0.5/1.0/2.0/4.0s); we use 2.0s.

train.py saves the resolved Config to outputs/<model>/config.json, so every run
records exactly what produced it.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent

# Override with --data-root or the GRAVITYSPY_DATA environment variable.
DEFAULT_DATA_ROOT = Path(os.environ.get("GRAVITYSPY_DATA", PROJECT_ROOT / "dataset"))
H5_NAME = "trainingsetv1d1.h5"
METADATA_NAME = "trainingset_v1d1_metadata.csv"
OUTPUT_ROOT = PROJECT_ROOT / "outputs"

# Fixed alphabetical order. This defines the integer label for every class, so
# reordering it would invalidate saved checkpoints and break comparisons.
CLASSES: List[str] = [
    "1080Lines", "1400Ripples", "Air_Compressor", "Blip", "Chirp",
    "Extremely_Loud", "Helix", "Koi_Fish", "Light_Modulation",
    "Low_Frequency_Burst", "Low_Frequency_Lines", "No_Glitch",
    "None_of_the_Above", "Paired_Doves", "Power_Line", "Repeating_Blips",
    "Scattered_Light", "Scratchy", "Tomte", "Violin_Mode", "Wandering_Line",
    "Whistle",
]
NUM_CLASSES = len(CLASSES)
CLASS_TO_IDX: Dict[str, int] = {name: i for i, name in enumerate(CLASSES)}

# The dataset's own train/validation/test assignment. We use it verbatim and
# never split randomly -- see data/dataset.py for why that matters.
SPLITS: Tuple[str, str, str] = ("train", "validation", "test")

DEFAULT_DURATION = "2.0"
NATIVE_IMAGE_SIZE: Tuple[int, int] = (140, 170)   # (H, W) of the stored arrays

# Spectrograms are greyscale but get replicated to 3 channels so pretrained
# backbones accept them, so we use the statistics those backbones expect.
IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)


@dataclass
class Config:
    """One training/evaluation run, fully specified."""

    # --- model ---
    model: str = "custom_cnn"
    pretrained: bool = True                       # ignored by from-scratch models
    input_size: Optional[Tuple[int, int]] = None  # None => registry default

    # --- data ---
    data_root: Path = DEFAULT_DATA_ROOT
    duration: str = DEFAULT_DURATION

    # --- optimisation (defaults tuned for one RTX 5070) ---
    epochs: int = 60
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 2
    grad_clip: float = 1.0

    # Blip has 1,274 training images, Paired_Doves 19 -- a 67:1 imbalance.
    # Weighting the loss stops the model collapsing onto the common classes.
    class_weights: bool = True

    # Early stopping selects on macro-F1, not val loss: the training loss is
    # class-weighted and the val loss is not, so they measure different things.
    early_stopping_patience: int = 12
    min_epochs: int = 5

    # --- augmentation (train split only) ---
    augment: bool = True
    time_shift_frac: float = 0.08   # +/- fraction of width
    time_mask_frac: float = 0.12    # SpecAugment mask sizes
    freq_mask_frac: float = 0.12
    brightness: float = 0.15
    contrast: float = 0.15
    # Time reversal. Off by default -- see data/transforms.py.
    horizontal_flip: bool = False

    # 134 images in the release are entirely blank (24% of Extremely_Loud, 30%
    # of Paired_Doves). Kept by default so our test set stays the canonical one
    # and matches the other two models. See README.
    drop_blank_images: bool = False

    # --- runtime ---
    seed: int = 42
    device: str = "auto"            # "auto" | "cuda" | "cpu"
    amp: bool = True
    num_workers: int = 4

    output_root: Path = OUTPUT_ROOT

    @property
    def output_dir(self) -> Path:
        return Path(self.output_root) / self.model

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        for k, v in list(d.items()):
            if isinstance(v, Path):
                d[k] = str(v)
            elif isinstance(v, tuple):
                d[k] = list(v)
        return d

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "Config":
        known = {f.name for f in fields(cls)}
        kwargs: Dict[str, Any] = {}
        for key, value in raw.items():
            if key not in known:
                continue
            if key in ("data_root", "output_root"):
                value = Path(value)
            elif key == "input_size" and value is not None:
                value = tuple(value)
            kwargs[key] = value
        return cls(**kwargs)

    def apply_overrides(self, **overrides: Any) -> "Config":
        """Return a copy with the non-None overrides applied."""
        data = {f.name: getattr(self, f.name) for f in fields(self)}
        data.update({k: v for k, v in overrides.items() if v is not None})
        return Config(**data)

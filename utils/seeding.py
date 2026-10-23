"""Reproducibility: one seed controls everything."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed every random number generator this project uses.

    cuDNN is also forced into deterministic mode. By default it autotunes --
    it times several convolution algorithms and keeps the fastest, which
    depends on transient machine load. Different algorithms sum floating-point
    numbers in different orders, so two runs would differ in the last few bits
    and diverge from there. Determinism costs some speed and buys repeatability.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """Seed each DataLoader worker.

    torch gives each worker its own seed but does not pass it to NumPy or
    Python's `random`, which is what our augmentation uses. Without this the
    workers would draw identical augmentation parameters.
    """
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int) -> torch.Generator:
    """Generator for DataLoader shuffling, so batch order is reproducible."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def resolve_device(preference: str = "auto") -> torch.device:
    if preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if preference == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA requested but not available. On an RTX 50-series card this "
            "usually means torch came from the default PyPI index instead of "
            "the cu128 build -- see README.md."
        )
    return torch.device(preference)

"""
seed.py — Global reproducibility control.

Call set_seed(config.SEED) once, at the very start of any script that does
anything random (training, evaluation, even data inspection that samples
examples). It locks down every source of randomness we can control.

Why this matters: the same seed must give the same result, so a run is
reproducible from its logged config alone.
"""

import os
import random

import numpy as np


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed every RNG PyTorch training touches.

    There are THREE independent random number generators in a typical
    PyTorch pipeline, and seeding one does not seed the others:
      1. Python's built-in `random`  (some torchvision-style augmentations)
      2. NumPy's RNG                 (array ops, our synthetic data, shuffles)
      3. PyTorch's RNG               (weight init, dropout, sampling)

    On GPU, cuDNN also chooses convolution algorithms that can differ run to
    run. `deterministic=True` forces reproducible algorithms (a little slower).
    """
    # 1. Python
    random.seed(seed)
    # PYTHONHASHSEED affects hash-based ordering (e.g. set iteration). Set it
    # for completeness; note it only fully applies to child processes, but it
    # documents intent and helps DataLoader workers.
    os.environ["PYTHONHASHSEED"] = str(seed)

    # 2. NumPy
    np.random.seed(seed)

    # 3. PyTorch (imported lazily so non-training tools don't need torch)
    import torch

    torch.manual_seed(seed)              # CPU RNG
    torch.cuda.manual_seed_all(seed)     # all GPU RNGs (safe even with no GPU)

    if deterministic:
        # Force cuDNN to use deterministic algorithms. `benchmark=False`
        # disables the autotuner that would otherwise pick different (faster)
        # algorithms depending on input sizes / hardware state.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """Seed an individual DataLoader worker process.

    Each worker runs in its own process with its own RNG state. If we don't
    seed them, augmentation randomness is not reproducible across runs.

    PyTorch gives each worker a base seed via torch.initial_seed(); we derive
    NumPy's and Python's seeds from it so all three RNGs are aligned per worker.
    Pass this as `worker_init_fn=seed_worker` to the DataLoader.
    """
    import torch

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int):
    """Return a torch.Generator seeded for the DataLoader `generator=` argument.

    Together with seed_worker, this makes shuffling order reproducible.
    """
    import torch

    g = torch.Generator()
    g.manual_seed(seed)
    return g

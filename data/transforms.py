"""Preprocessing and augmentation.

WHY NOT THE STANDARD IMAGE AUGMENTATIONS
----------------------------------------
A spectrogram is not a photograph. The horizontal axis is TIME, the vertical
axis is FREQUENCY, and brightness is energy. The two axes are not
interchangeable, which rules out most of the usual toolkit:

  VERTICAL FLIP swaps low and high frequencies. Several classes are defined by
  frequency -- Violin_Mode is a specific resonance, Low_Frequency_Burst is named
  for its band, 1080Lines and 1400Ripples for their frequency in Hz. Flipping
  turns one class into a physically impossible signal.

  ROTATION / SHEAR mix the time and frequency axes together, producing images no
  detector could ever record.

  HORIZONTAL FLIP is time reversal. Harmless for symmetric glitches (Blip), but
  a Chirp sweeps upward in frequency as the binary inspirals; reversed it sweeps
  downward, which is not a chirp. Off by default, exposed as a flag so the
  effect can be measured rather than assumed.

What IS valid: small time shifts (the glitch is not always centred),
SpecAugment-style masking (Park et al. 2019, arXiv:1904.08779) and mild
brightness/contrast changes (glitch loudness varies and is not class-defining).

These operate on tensors rather than PIL images because the loader reads NumPy
arrays straight out of HDF5.
"""

from __future__ import annotations

import random
from typing import Callable, List, Sequence, Tuple

import torch
import torch.nn.functional as F

from config import IMAGENET_MEAN, IMAGENET_STD, Config

Tensor = torch.Tensor


class Resize:
    """Resize to the model's input size.

    antialias=True matters when shrinking: without it thin features -- which is
    exactly what 1080Lines and Violin_Mode are -- break into dotted lines or
    disappear entirely.
    """

    def __init__(self, size: Tuple[int, int]):
        self.size = tuple(size)

    def __call__(self, img: Tensor) -> Tensor:
        if tuple(img.shape[-2:]) == self.size:
            return img
        return F.interpolate(img.unsqueeze(0), size=self.size, mode="bilinear",
                             align_corners=False, antialias=True).squeeze(0)


class RandomTimeShift:
    """Shift along the time axis, padding with the edge column.

    Deliberately not a circular roll: that would move energy from the end of the
    window to the start and could invent a second glitch that never happened.
    """

    def __init__(self, max_frac: float = 0.08):
        self.max_frac = max_frac

    def __call__(self, img: Tensor) -> Tensor:
        width = img.shape[-1]
        max_shift = int(round(width * self.max_frac))
        if max_shift < 1:
            return img
        shift = random.randint(-max_shift, max_shift)
        if shift == 0:
            return img
        if shift > 0:
            pad = img[..., :1].expand(-1, -1, shift)
            return torch.cat([pad, img[..., :-shift]], dim=-1)
        shift = -shift
        pad = img[..., -1:].expand(-1, -1, shift)
        return torch.cat([img[..., shift:], pad], dim=-1)


class SpecAugment:
    """Hide a random band of time and of frequency.

    Forces the model to classify from partial evidence instead of latching onto
    one narrow band. Masks are filled with the image mean rather than zero:
    after normalisation zero is a real intensity the model could learn to read
    as "this is a mask", which defeats the purpose.
    """

    def __init__(self, time_frac: float = 0.12, freq_frac: float = 0.12):
        self.time_frac = time_frac
        self.freq_frac = freq_frac

    def __call__(self, img: Tensor) -> Tensor:
        height, width = img.shape[-2], img.shape[-1]
        fill = img.mean()

        max_w = int(round(width * self.time_frac))
        if max_w >= 1:
            w = random.randint(1, max_w)
            start = random.randint(0, width - w)
            img[..., :, start:start + w] = fill

        max_h = int(round(height * self.freq_frac))
        if max_h >= 1:
            h = random.randint(1, max_h)
            start = random.randint(0, height - h)
            img[..., start:start + h, :] = fill

        return img


class RandomBrightnessContrast:
    """Corresponds to glitch loudness and dynamic range, which are not
    class-defining. Applied in [0, 1] space and clamped, so the result stays a
    plausible intensity map."""

    def __init__(self, brightness: float = 0.15, contrast: float = 0.15):
        self.brightness = brightness
        self.contrast = contrast

    def __call__(self, img: Tensor) -> Tensor:
        img = img + random.uniform(-self.brightness, self.brightness)
        factor = 1.0 + random.uniform(-self.contrast, self.contrast)
        mean = img.mean()
        return ((img - mean) * factor + mean).clamp_(0.0, 1.0)


class RandomHorizontalFlip:
    """Time reversal. Off by default -- see the module docstring."""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, img: Tensor) -> Tensor:
        return torch.flip(img, dims=[-1]) if random.random() < self.p else img


class Normalise:
    def __init__(self, mean: Sequence[float], std: Sequence[float]):
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def __call__(self, img: Tensor) -> Tensor:
        return (img - self.mean) / self.std


class Compose:
    def __init__(self, transforms: List[Callable[[Tensor], Tensor]]):
        self.transforms = transforms

    def __call__(self, img: Tensor) -> Tensor:
        for t in self.transforms:
            img = t(img)
        return img


def normalisation_stats(cfg: Config):
    """The (mean, std) in use -- needed to undo normalisation for figures."""
    return IMAGENET_MEAN, IMAGENET_STD


def build_transforms(cfg: Config, input_size: Tuple[int, int], train: bool) -> Compose:
    """Build the pipeline for one split.

    Order matters: resize first so the mask/shift fractions mean the same thing
    at any input size; augment next, while values are still in [0, 1] where
    brightness and the mean-fill are meaningful; normalise last.

    Augmentation applies to the TRAIN split only, so validation and test numbers
    stay comparable across models and runs.
    """
    steps: List[Callable[[Tensor], Tensor]] = [Resize(input_size)]

    if train and cfg.augment:
        steps.append(RandomTimeShift(cfg.time_shift_frac))
        steps.append(SpecAugment(cfg.time_mask_frac, cfg.freq_mask_frac))
        steps.append(RandomBrightnessContrast(cfg.brightness, cfg.contrast))
        if cfg.horizontal_flip:
            steps.append(RandomHorizontalFlip())
        # No vertical flip and no rotation -- see the module docstring.

    steps.append(Normalise(IMAGENET_MEAN, IMAGENET_STD))
    return Compose(steps)

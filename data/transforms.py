"""
transforms.py — Preprocessing + domain-safe augmentation for spectrograms.

Coordinate convention for a spectrogram image tensor of shape (C, H, W):
    H (height, rows)    = FREQUENCY axis
    W (width, columns)  = TIME axis

This convention is why some ordinary image augmentations are FORBIDDEN here:
  * Vertical flip  -> flips frequency -> a rising chirp becomes a falling one.
                      That is a physically DIFFERENT glitch. Never allowed.
  * Rotation       -> mixes the time and frequency axes into nonsense. Never.
  * Horizontal flip-> reverses time. Off by default (a time-reversed glitch is
                      usually not a valid example), but exposed as a toggle.

ALLOWED, physically-plausible augmentations (train split only):
  * Small time shift          -> the glitch can sit slightly earlier/later.
  * SpecAugment time masking  -> zero out a few TIME columns (robustness).
  * SpecAugment freq masking  -> zero out a few FREQUENCY rows (robustness).
  * Mild brightness/contrast  -> detector/whitening intensity varies.

All transforms operate on a float tensor image in [0, 1] with shape (C, H, W).
"""

import random

import torch
import torchvision.transforms.v2 as T

import config

# ImageNet statistics. timm pretrained backbones were trained on ImageNet, so
# normalizing to these stats is what they expect. Our from-scratch CNN is fine
# with normalized input too, so we use one consistent normalization everywhere.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------
# Custom, domain-aware augmentations (written explicitly so the physics is
# visible in the code, not hidden inside a library call).
# ---------------------------------------------------------------------------
class RandomTimeShift:
    """Shift the spectrogram left/right along the TIME (width) axis.

    Uses torch.roll so content wraps around; for small shifts relative to the
    image width this is a good approximation of the glitch arriving slightly
    earlier or later. Frequency content is untouched.
    """

    def __init__(self, max_fraction: float = 0.08):
        self.max_fraction = max_fraction

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        w = img.shape[-1]
        max_shift = int(w * self.max_fraction)
        if max_shift < 1:
            return img
        shift = random.randint(-max_shift, max_shift)
        return torch.roll(img, shifts=shift, dims=-1)


class SpecAugment:
    """SpecAugment-style masking: zero out random time columns and freq rows.

    This is the image analogue of the well-known audio SpecAugment. Masking
    forces the model not to rely on any single time/frequency location, which
    improves generalization. We mask with 0.0 (black) after resizing.
    """

    def __init__(self, time_mask: int = 16, freq_mask: int = 16,
                 n_time: int = 1, n_freq: int = 1):
        self.time_mask = time_mask   # max width (in px) of each time mask
        self.freq_mask = freq_mask   # max height (in px) of each freq mask
        self.n_time = n_time         # how many time masks
        self.n_freq = n_freq         # how many freq masks

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        _, h, w = img.shape
        img = img.clone()
        # Time masks: contiguous columns.
        for _ in range(self.n_time):
            t = random.randint(0, self.time_mask)
            if t == 0 or t >= w:
                continue
            t0 = random.randint(0, w - t)
            img[:, :, t0:t0 + t] = 0.0
        # Frequency masks: contiguous rows.
        for _ in range(self.n_freq):
            f = random.randint(0, self.freq_mask)
            if f == 0 or f >= h:
                continue
            f0 = random.randint(0, h - f)
            img[:, f0:f0 + f, :] = 0.0
        return img


# ---------------------------------------------------------------------------
# Transform pipelines
# ---------------------------------------------------------------------------
def _to_float_chw(img: torch.Tensor) -> torch.Tensor:
    """Ensure input is a float32 (C, H, W) tensor scaled to [0, 1].

    The Dataset hands us a tensor that may be (H, W), (1, H, W) or (3, H, W),
    with values in [0, 1] already (the Dataset normalizes raw pixel ranges).
    Here we only guarantee the channel dimension exists and dtype is float.
    """
    if img.dim() == 2:                 # (H, W) -> (1, H, W)
        img = img.unsqueeze(0)
    return img.to(torch.float32)


class GreyTo3Channel:
    """Replicate a single grey channel to 3 channels.

    ImageNet-pretrained backbones expect 3-channel input. Doing this in the
    transform (not the model) keeps every model getting identical input, so
    the pipeline stays model-agnostic.
    """

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        return img


def build_transforms(train: bool):
    """Return the transform callable for the given split.

    train=True  -> preprocessing + domain-safe augmentation
    train=False -> preprocessing ONLY (deterministic; never augment val/test)

    Pipeline order (train):
        to float/CHW -> resize -> grey->3ch -> [augment] -> normalize
    Augmentation is inserted BEFORE normalization so brightness/contrast and
    masking operate in the natural [0, 1] pixel space.
    """
    resize = T.Resize((config.IMG_SIZE, config.IMG_SIZE), antialias=True)
    grey_to_3 = GreyTo3Channel()
    normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    if not train or not config.USE_AUGMENTATION:
        # Validation / test / augmentation-disabled: deterministic only.
        steps = [_to_float_chw, resize, grey_to_3, normalize]
        return T.Compose(steps)

    # Training augmentation. Note the deliberately ABSENT transforms:
    #   NO RandomVerticalFlip, NO RandomRotation, NO affine shear/rotate.
    augment = [
        RandomTimeShift(max_fraction=0.08),
        SpecAugment(time_mask=16, freq_mask=16, n_time=1, n_freq=1),
        # Mild brightness/contrast; small ranges so glitch morphology is kept.
        T.ColorJitter(brightness=0.15, contrast=0.15),
    ]
    # Horizontal flip is OFF by default. Enable via config if you decide a
    # time-reversed glitch is an acceptable example for your experiment.
    if getattr(config, "USE_HFLIP", False):
        augment.append(T.RandomHorizontalFlip(p=0.5))

    steps = [_to_float_chw, resize, grey_to_3, *augment, normalize]
    return T.Compose(steps)


def denormalize(img: torch.Tensor) -> torch.Tensor:
    """Invert ImageNet normalization for visualization (returns [0, 1] tensor).

    Used by inspect_data / evaluate to show human-viewable example grids.
    """
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (img * std + mean).clamp(0, 1)

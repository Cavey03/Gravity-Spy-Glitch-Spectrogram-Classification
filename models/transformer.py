"""Model 3 of 3: a pretrained Swin-Tiny vision transformer, fine-tuned on Gravity Spy.

Implemented by Morena-111 (originally commit 109b758).

timm downloads the ImageNet-pretrained weights and, because num_classes is
passed, replaces the classification head with a fresh 22-way one.

Input size is fixed at 224x224 (set in registry.py). Swin uses shifted-window
attention with a 7x7 window, and the feature map must stay divisible by that
window at every stage -- so unlike the CNNs it cannot take arbitrary sizes.

    python train.py --model transformer --lr 3e-5 --epochs 20 --weight-decay 0.05
    python evaluate.py --model transformer

TRAINING NOTES
  Learning rate matters most here. Transformers have no convolutional
  inductive bias, so on 5,587 images they lean almost entirely on their
  pretrained weights; too high an LR destroys those. Use 1e-5 to 5e-5.
  Weight decay: 0.05 is a better starting point than the CNN default.
  Warmup: train.py already warms up for 2 epochs. Keep it.
"""

from __future__ import annotations

import timm
import torch.nn as nn

BACKBONE = "swin_tiny_patch4_window7_224"


def build_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    """Swin-Tiny with a fresh num_classes-way head. Returns raw logits."""
    return timm.create_model(BACKBONE, pretrained=pretrained, num_classes=num_classes)

"""Model 2 of 3: a pretrained EfficientNetV2-S, fine-tuned on Gravity Spy.

Implemented by Morena-111 (originally commit 085c636).

timm downloads the ImageNet-pretrained weights and, because num_classes is
passed, replaces the 1000-way ImageNet head with a fresh 22-way one. The
shared pipeline feeds it 3-channel 224x224 input (registry.py), which is what
the backbone expects.

    python train.py --model transfer_cnn --lr 1e-4 --epochs 25
    python evaluate.py --model transfer_cnn

TRAINING NOTES
  Learning rate: the default 3e-4 is tuned for the from-scratch CNN and is too
  high for fine-tuning -- it washes out the pretrained features early. Use 1e-4.
  Batch size: on CUDA out-of-memory, drop to --batch-size 32.
  Epochs: fine-tuning converges fast; 15-25 is usually plenty.
"""

from __future__ import annotations

import timm
import torch.nn as nn

BACKBONE = "tf_efficientnetv2_s"


def build_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    """EfficientNetV2-S with a fresh num_classes-way head. Returns raw logits."""
    return timm.create_model(BACKBONE, pretrained=pretrained, num_classes=num_classes)

"""
transformer.py — Vision Transformer model.

Uses a pretrained Swin-Tiny backbone from timm.
The classifier head is automatically replaced to output the required
number of Gravity Spy glitch classes.
"""

import torch.nn as nn
import timm

from models.registry import register


@register("transformer")
def build_model(num_classes: int, pretrained: bool) -> nn.Module:
    """Build a Swin-Tiny vision transformer model."""
    backbone = "swin_tiny_patch4_window7_224"

    model = timm.create_model(
        backbone,
        pretrained=pretrained,
        num_classes=num_classes,
    )

    return model
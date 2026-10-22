"""
transfer_cnn.py — Transfer-learning CNN.

Uses a pretrained EfficientNetV2-S backbone from timm.
The classifier head is automatically replaced to output the required
number of Gravity Spy glitch classes.
"""

import torch.nn as nn
import timm

from models.registry import register


@register("transfer_cnn")
def build_model(num_classes: int, pretrained: bool) -> nn.Module:
    """Build an EfficientNetV2-S transfer-learning model."""
    backbone = "tf_efficientnetv2_s"

    model = timm.create_model(
        backbone,
        pretrained=pretrained,
        num_classes=num_classes,
    )

    return model

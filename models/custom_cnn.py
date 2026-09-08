"""
custom_cnn.py — MY reference model (fully implemented, from scratch).

A small, conventional convolutional network: 5 blocks of
    Conv -> BatchNorm -> ReLU -> MaxPool
followed by global average pooling and a linear classifier. No pretrained
weights — this is the "learn the features from the spectrograms directly"
baseline that the transfer-learning and transformer models are compared to.

This file is also the TEMPLATE teammates copy: note how it (1) defines an
nn.Module and (2) exposes build_model(num_classes, pretrained) registered
under a string name. That is the entire contract.
"""

import torch
import torch.nn as nn

from models.registry import register


def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    """One downsampling block: Conv3x3 -> BN -> ReLU -> MaxPool2x2.

    * padding=1 keeps spatial size through the conv, so only the MaxPool
      halves H and W -> easy to reason about the shrinking feature map.
    * BatchNorm before ReLU stabilizes/accelerates training (standard recipe).
    * bias=False in the conv because the following BatchNorm has its own shift,
      making the conv bias redundant.
    """
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
    )


class CustomCNN(nn.Module):
    """Compact 5-block CNN for 224x224x3 spectrogram inputs.

    Spatial size through the network (H=W shown):
        224 -> 112 -> 56 -> 28 -> 14 -> 7     (one halving per block)
    Channels grow 3 -> 32 -> 64 -> 128 -> 256 -> 256 as spatial size shrinks,
    the classic "trade resolution for semantic depth" pattern.

    Global average pooling collapses the final 256x7x7 map to a 256-vector,
    which a dropout + linear layer maps to the class logits. GAP (instead of
    flatten+big-FC) keeps the parameter count low and is naturally robust to
    small spatial shifts — a good fit for glitches that can sit at slightly
    different times.
    """

    def __init__(self, num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            _conv_block(3, 32),      # 224 -> 112
            _conv_block(32, 64),     # 112 -> 56
            _conv_block(64, 128),    # 56  -> 28
            _conv_block(128, 256),   # 28  -> 14
            _conv_block(256, 256),   # 14  -> 7
        )
        self.pool = nn.AdaptiveAvgPool2d(1)   # -> (N, 256, 1, 1); size-agnostic
        self.classifier = nn.Sequential(
            nn.Flatten(),                     # (N, 256)
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),      # (N, num_classes) logits
        )
        self._init_weights()

    def _init_weights(self):
        """Kaiming init for conv/linear layers (good default for ReLU nets)."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)             # raw logits (CrossEntropyLoss applies softmax)


@register("custom_cnn")
def build_model(num_classes: int, pretrained: bool) -> nn.Module:
    """Registry entry point. Signature matches the contract exactly.

    `pretrained` is accepted for interface compatibility but ignored: this
    model is trained from scratch (there are no ImageNet weights for it).
    """
    if pretrained:
        print("[custom_cnn] note: pretrained=True ignored (this model trains "
              "from scratch).")
    return CustomCNN(num_classes=num_classes)

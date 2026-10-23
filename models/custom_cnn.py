"""A small CNN trained from scratch. The project's baseline model.

Architecture: 5 blocks of (conv -> BatchNorm -> ReLU) x2 -> maxpool, then global
average pooling and one linear layer. 2.36M parameters.

    input       3 x 140 x 170
    block 1    32 x  70 x  85
    block 2    64 x  35 x  43
    block 3   128 x  18 x  22
    block 4   256 x   9 x  11
    block 5   256 x   5 x   6
    GAP       256
    linear     22

Design choices worth defending:

  Five blocks. Each halves the resolution, so a sixth would leave a 3x3 map with
  little left to convolve over. By block 5 one unit sees a ~156x156 pixel region
  of a 140x170 image -- i.e. essentially all of it, which is what lets the model
  use a glitch's overall shape. Blip and Koi_Fish differ by global morphology,
  not local texture.

  Small on purpose. 2.36M parameters against ~21M for EfficientNetV2-S. With
  5,587 training images a larger from-scratch network would memorise rather than
  generalise.

  Global average pooling, not flatten. Flattening the final 256x5x6 map would
  need a 7,680 -> 22 layer (169k parameters); averaging each channel needs
  256 -> 22 (5.6k). Thirty times fewer parameters where overfitting is the main
  risk, and it makes the model work at any input size.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """(conv -> BN -> ReLU) x2 -> maxpool.

    Two 3x3 convolutions rather than one 5x5: same receptive field, fewer
    parameters (2x9xC^2 vs 25xC^2), and an extra non-linearity in between.

    bias=False because BatchNorm subtracts its own learned mean immediately
    afterwards, which would cancel any bias the convolution learned.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)
        # ceil_mode so odd sizes (35 -> 18, not 17) keep the last row/column.
        self.pool = nn.MaxPool2d(2, 2, ceil_mode=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        return self.pool(x)


class CustomCNN(nn.Module):
    def __init__(self, num_classes: int = 22,
                 widths: Sequence[int] = (32, 64, 128, 256, 256),
                 dropout: float = 0.4):
        super().__init__()
        blocks = []
        prev = 3
        for width in widths:
            blocks.append(ConvBlock(prev, width))
            prev = width
        self.features = nn.Sequential(*blocks)

        # Channel count doubles as resolution halves, keeping roughly constant
        # capacity per layer. The last block does not double again: 512 would
        # add ~2.4M parameters, doubling the model for little gain at this size.
        self.pool = nn.AdaptiveAvgPool2d(1)      # global average pooling
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # Dropout only here. On convolutional feature maps it is far less
            # effective, because neighbouring pixels carry the same information
            # so dropping one changes little. BatchNorm regularises the blocks.
            nn.Dropout(dropout),
            nn.Linear(prev, num_classes),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        """Kaiming-normal init, which keeps activation variance roughly constant
        through the depth of the network so training is stable from epoch 1.
        PyTorch's default is a different (and largely historical) choice."""
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
        x = self.pool(self.features(x))
        # Raw logits. CrossEntropyLoss applies log-softmax itself, so adding a
        # softmax here would silently train a worse model.
        return self.classifier(x)


def build_model(num_classes: int, pretrained: bool = False) -> nn.Module:
    """The interface function registered in models/registry.py.

    `pretrained` is accepted for interface compatibility and ignored -- there
    are no pretrained weights for this architecture.
    """
    return CustomCNN(num_classes=num_classes)

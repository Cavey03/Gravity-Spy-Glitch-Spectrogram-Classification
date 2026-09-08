"""
transfer_cnn.py — TEAMMATE STUB (transfer-learning CNN).

>>> THIS IS A SKELETON FOR A TEAMMATE TO FILL IN. <<<
Assigned model: a pretrained CNN backbone via `timm` — EfficientNetV2-S
(timm name "tf_efficientnetv2_s") OR ConvNeXt-Tiny ("convnext_tiny").

You only need to implement build_model() below. Do NOT edit any shared code
(config, data, train, evaluate, registry). The @register line already wires
this model into the pipeline under the name "transfer_cnn".

HOW TO FILL THIS IN (about 3 lines of real code):
--------------------------------------------------------------------------
1. Pick a backbone name, e.g. BACKBONE = "tf_efficientnetv2_s".
2. Create it with the right number of classes and pretrained weights:

       import timm
       model = timm.create_model(
           BACKBONE,
           pretrained=pretrained,
           num_classes=num_classes,   # timm swaps the classifier head for you
       )
       return model

   timm handles the ImageNet head replacement and expects 3-channel,
   ImageNet-normalized input — which is exactly what our transforms produce.
3. Delete the `raise NotImplementedError` line.
4. Train it:   python train.py --model transfer_cnn
--------------------------------------------------------------------------
See the CONTRIBUTING section of the README for the full checklist.
"""

import torch.nn as nn

from models.registry import register


@register("transfer_cnn")
def build_model(num_classes: int, pretrained: bool) -> nn.Module:
    # TODO(teammate): replace the raise below with the ~3 lines described above.
    raise NotImplementedError(
        "transfer_cnn is not implemented yet. See the instructions at the top "
        "of models/transfer_cnn.py — build an EfficientNetV2-S or ConvNeXt-Tiny "
        "with timm.create_model(name, pretrained=pretrained, num_classes=num_classes)."
    )

"""
transformer.py — TEAMMATE STUB (vision transformer).

>>> THIS IS A SKELETON FOR A TEAMMATE TO FILL IN. <<<
Assigned model: a pretrained vision transformer via `timm` — Swin-Tiny
(timm name "swin_tiny_patch4_window7_224") OR ViT-B/16 ("vit_base_patch16_224").

You only need to implement build_model() below. Do NOT edit any shared code
(config, data, train, evaluate, registry). The @register line already wires
this model into the pipeline under the name "transformer".

HOW TO FILL THIS IN (about 3 lines of real code):
--------------------------------------------------------------------------
1. Pick a backbone name, e.g. BACKBONE = "swin_tiny_patch4_window7_224".
   NOTE: these transformers expect 224x224 input — config.IMG_SIZE is already
   224, so the shared transforms produce the right size. Good.
2. Create it:

       import timm
       model = timm.create_model(
           BACKBONE,
           pretrained=pretrained,
           num_classes=num_classes,
       )
       return model

3. Delete the `raise NotImplementedError` line.
4. Train it:   python train.py --model transformer
--------------------------------------------------------------------------
See the CONTRIBUTING section of the README for the full checklist.
"""

import torch.nn as nn

from models.registry import register


@register("transformer")
def build_model(num_classes: int, pretrained: bool) -> nn.Module:
    # TODO(teammate): replace the raise below with the ~3 lines described above.
    raise NotImplementedError(
        "transformer is not implemented yet. See the instructions at the top of "
        "models/transformer.py — build a Swin-Tiny or ViT-B/16 with "
        "timm.create_model(name, pretrained=pretrained, num_classes=num_classes)."
    )

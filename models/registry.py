"""Model registry -- the one place models are wired into the project.

THE CONTRACT
Every model file exposes:

    build_model(num_classes: int, pretrained: bool) -> torch.nn.Module

and gets one register(...) line at the bottom of this file. train.py and
evaluate.py look models up by name and never import a model module directly, so
adding a model means one new file plus one line, and no changes to shared code.

WHY input_size IS REGISTERED HERE rather than passed to build_model:
models disagree about what size they accept, and not cosmetically. A CNN with
global pooling takes anything. ViT-B/16 splits the image into fixed 16x16
patches with one learned positional embedding each, so a different size means
interpolating those embeddings. Swin-T needs the feature map divisible by its
7x7 window at every stage. If size were a build_model argument every model would
have to honour any size given -- so instead each model declares what it wants
and the data pipeline resizes to match.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import torch.nn as nn


@dataclass(frozen=True)
class ModelSpec:
    name: str
    build_fn: Callable[[int, bool], nn.Module]
    input_size: Tuple[int, int]              # (H, W) to feed this model
    description: str = ""
    supports_pretrained: bool = True         # False for from-scratch models


_REGISTRY: Dict[str, ModelSpec] = {}


def register(name: str, build_fn, input_size: Tuple[int, int],
             description: str = "", supports_pretrained: bool = True) -> None:
    _REGISTRY[name] = ModelSpec(name, build_fn, tuple(input_size),
                                description, supports_pretrained)


def available_models() -> List[str]:
    return sorted(_REGISTRY)


def get_spec(name: str) -> ModelSpec:
    if name not in _REGISTRY:
        raise KeyError(
            "Unknown model %r. Registered: %s.\nTo add one, write "
            "models/<name>.py with build_model(num_classes, pretrained) and add "
            "a register(...) line in models/registry.py."
            % (name, ", ".join(available_models())))
    return _REGISTRY[name]


def get_input_size(name: str) -> Tuple[int, int]:
    return get_spec(name).input_size


def build_model(name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    spec = get_spec(name)
    if pretrained and not spec.supports_pretrained:
        print("[registry] %r trains from scratch; ignoring pretrained=True." % name)
        pretrained = False
    return spec.build_fn(num_classes, pretrained)


def describe_models() -> str:
    return "\n".join(
        "  %-14s %3dx%-3d  %s" % (s.name, s.input_size[0], s.input_size[1],
                                  s.description)
        for s in (_REGISTRY[n] for n in available_models()))


# --- registrations ---------------------------------------------------------
# Imported here, after register() exists, so a model module can import from
# this file without a circular import.

from models import custom_cnn, transfer_cnn, transformer  # noqa: E402

register("custom_cnn", custom_cnn.build_model, input_size=(140, 170),
         description="5-block CNN trained from scratch (baseline).",
         supports_pretrained=False)

register("transfer_cnn", transfer_cnn.build_model, input_size=(224, 224),
         description="Pretrained CNN via timm (EfficientNetV2-S / ConvNeXt-Tiny).")

register("transformer", transformer.build_model, input_size=(224, 224),
         description="Pretrained transformer via timm (Swin-Tiny / ViT-B/16).")

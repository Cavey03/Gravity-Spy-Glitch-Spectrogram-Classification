"""
registry.py — THE INTERFACE CONTRACT for all models in this project.

=============================================================================
THE CONTRACT (read this if you are adding a model)
=============================================================================
Every model file exposes exactly one builder with this signature:

    build_model(num_classes: int, pretrained: bool) -> torch.nn.Module

and registers it here under a unique string name. train.py and evaluate.py
take `--model <name>` and work for ANY registered model with ZERO other
changes. To add a model you write ONE file and add ONE registration line —
nothing else in the codebase changes.

Two equivalent ways to register:
  1. Decorate the builder in your model file:      @register("my_name")
  2. Or add a line in _import_models() below to import your module so its
     decorator runs.

This file must not be imported BY the model files (one-way dependency), which
keeps imports acyclic.
=============================================================================
"""

from typing import Callable

import torch.nn as nn

# name -> builder function.  A builder is: (num_classes, pretrained) -> Module
_REGISTRY: dict[str, Callable[..., nn.Module]] = {}


def register(name: str):
    """Decorator that adds a builder to the registry under `name`.

    Usage in a model file:

        @register("custom_cnn")
        def build_model(num_classes: int, pretrained: bool) -> nn.Module:
            ...
    """
    def _wrap(build_fn: Callable[..., nn.Module]):
        if name in _REGISTRY:
            raise ValueError(f"Model name {name!r} is already registered.")
        _REGISTRY[name] = build_fn
        return build_fn
    return _wrap


def _import_models():
    """Import model modules so their @register decorators execute.

    Import errors for the teammate STUB files are swallowed on purpose: the
    stubs raise NotImplementedError only when BUILT, not when imported, so they
    should import fine — but if a teammate's WIP file has a syntax/import error,
    we don't want it to take down the whole registry for everyone else.
    """
    # Fully-implemented models: import eagerly (failure here is a real bug).
    from models import custom_cnn  # noqa: F401  -> registers "custom_cnn"

    # Teammate models: import defensively so one broken WIP file doesn't block
    # the others. Each still registers its name when importable.
    for module_name in ("transfer_cnn", "transformer"):
        try:
            __import__(f"models.{module_name}")
        except Exception as e:  # noqa: BLE001
            print(f"[registry] note: could not import models.{module_name}: {e}")


_IMPORTED = False


def _ensure_imported():
    global _IMPORTED
    if not _IMPORTED:
        _import_models()
        _IMPORTED = True


def build_model(name: str, num_classes: int, pretrained: bool) -> nn.Module:
    """Build a registered model by name. This is the ONLY entry point callers use.

    train.py / evaluate.py call exactly this and nothing model-specific.
    """
    _ensure_imported()
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown model {name!r}. Available: {sorted(_REGISTRY)}. "
            "Add a model by writing models/<name>.py with a build_model() "
            "decorated @register('<name>'), then import it in registry._import_models()."
        )
    return _REGISTRY[name](num_classes=num_classes, pretrained=pretrained)


def available_models() -> list[str]:
    """Return the sorted list of registered model names."""
    _ensure_imported()
    return sorted(_REGISTRY)

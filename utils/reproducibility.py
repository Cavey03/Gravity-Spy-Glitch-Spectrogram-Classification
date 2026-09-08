"""
reproducibility.py — Self-documenting runs.

snapshot_environment() captures WHAT produced a result: the exact library
versions, the hardware, and the full config used. train.py and evaluate.py
write this into each run's output folder so any result can be traced back to
the precise setup that generated it.
"""

import json
import platform
import sys
from datetime import datetime
from pathlib import Path


def _safe_version(module_name: str) -> str:
    """Return a module's __version__ or a marker if it isn't importable."""
    try:
        mod = __import__(module_name)
        return getattr(mod, "__version__", "unknown")
    except Exception:
        return "not-installed"


def collect_versions() -> dict:
    """Record versions of every library that can affect numerical results."""
    versions = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": _safe_version("numpy"),
        "torch": _safe_version("torch"),
        "torchvision": _safe_version("torchvision"),
        "timm": _safe_version("timm"),
        "sklearn": _safe_version("sklearn"),
        "matplotlib": _safe_version("matplotlib"),
        "h5py": _safe_version("h5py"),
        "pandas": _safe_version("pandas"),
    }
    # GPU details (only if torch + CUDA are actually present).
    try:
        import torch

        versions["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            versions["cuda_version"] = torch.version.cuda
            versions["gpu_name"] = torch.cuda.get_device_name(0)
    except Exception:
        versions["cuda_available"] = False
    return versions


def config_as_dict(config_module) -> dict:
    """Extract the UPPER_CASE constants from config.py into a plain dict.

    We only capture module-level names in ALL_CAPS (the convention for config
    constants), skipping functions, imports, and dunders. Path objects are
    stringified so the result is JSON-serializable.
    """
    snapshot = {}
    for name in dir(config_module):
        if not name.isupper():
            continue
        value = getattr(config_module, name)
        if isinstance(value, Path):
            value = str(value)
        # Keep only JSON-friendly types; represent anything else by its repr.
        if isinstance(value, (str, int, float, bool, list, tuple, dict)) or value is None:
            snapshot[name] = value
        else:
            snapshot[name] = repr(value)
    return snapshot


def snapshot_environment(config_module, extra: dict | None = None) -> dict:
    """Build the full run record: timestamp + versions + config (+ extras)."""
    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        "versions": collect_versions(),
        "config": config_as_dict(config_module),
    }
    if extra:
        record["run"] = extra
    return record


def save_snapshot(record: dict, out_path: Path) -> None:
    """Write the run record to disk as pretty JSON."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)

"""Record library versions and hardware for each run (saved to env.json)."""

from __future__ import annotations

import json
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import torch


def collect_env_info() -> Dict[str, Any]:
    import numpy
    import sklearn

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    # ITRI626 spec 7.3 requires the GPU model AND its available memory.
    gpu_memory_gb = (round(torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 2)
                     if torch.cuda.is_available() else None)
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": numpy.__version__,
        "sklearn": sklearn.__version__,
        "cuda": torch.version.cuda,
        "gpu": gpu,
        "gpu_memory_gb": gpu_memory_gb,
    }


def save_env_info(path: Path) -> Dict[str, Any]:
    info = collect_env_info()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info


def format_env_summary(info: Dict[str, Any]) -> str:
    return (
        "python %s | torch %s | numpy %s | sklearn %s\nCUDA %s | GPU %s"
        % (info["python"], info["torch"], info["numpy"], info["sklearn"],
           info["cuda"], info["gpu"])
    )

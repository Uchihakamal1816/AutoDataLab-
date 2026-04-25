"""Load trusted local PyTorch state-dict checkpoints (CoS policy weights)."""
from __future__ import annotations

from pathlib import Path
from typing import Any


def load_state_dict(ckpt_path: Path, map_location: str | Any = "cpu") -> Any:
    import torch

    p = Path(ckpt_path)
    try:
        return torch.load(p, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(p, map_location=map_location)

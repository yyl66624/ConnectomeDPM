"""Deterministic seeding across every RNG that can affect a result.

Every experiment records the seed it used; nothing in the pipeline is allowed to be
implicitly random (e.g. python hash randomisation is disabled via PYTHONHASHSEED).
"""

from __future__ import annotations

import os
import random


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed python, numpy and torch (CPU + all CUDA devices)."""
    os.environ["PYTHONHASHSEED"] = str(int(seed))
    random.seed(int(seed))

    try:
        import numpy as np
    except ImportError:  # pragma: no cover
        np = None
    if np is not None:
        np.random.seed(int(seed))

    try:
        import torch
    except ImportError:  # pragma: no cover
        return

    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def numpy_rng(seed: int):
    """Return an independent numpy Generator so callers never touch global state."""
    import numpy as np

    return np.random.default_rng(int(seed))


def torch_generator(seed: int, device: str = "cpu"):
    """Return an independent torch Generator."""
    import torch

    g = torch.Generator(device=device)
    g.manual_seed(int(seed))
    return g


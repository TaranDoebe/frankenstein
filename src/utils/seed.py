"""
Utility to fix all random seeds for reproducible runs in PyTorch + NumPy + scikit‑learn.
Drop this into utils/seed.py (or similar) and call `set_seed(<int>)` right after argument parsing.
"""
from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch

try:
    import torch.backends.cudnn as cudnn
except ImportError:  # CPU‑only installs
    cudnn = None  # type: ignore

try:
    from sklearn.utils import check_random_state  # noqa: F401  (import for side‑effect)
except ImportError:
    # scikit‑learn not installed – ignore.
    pass

__all__ = ["set_seed"]


def set_seed(seed: int | None = 42, *, deterministic_cudnn: bool = True) -> Optional[int]:
    """Set *all* relevant RNG seeds.

    Parameters
    ----------
    seed : int | None
        The seed value.  If ``None`` → do **not** change global state (returns ``None``).
    deterministic_cudnn : bool, default=True
        If *True* and CUDA is available, force CuDNN into deterministic mode.

    Returns
    -------
    int | None
        The seed actually used (useful when you want to log it).
    """
    if seed is None:
        return None

    # ---------------- Python / NumPy / Random -----------------------
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    # ---------------- PyTorch ---------------------------------------
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic_cudnn and torch.cuda.is_available():
        if cudnn is not None:
            cudnn.deterministic = True   # type: ignore[attr-defined]
            cudnn.benchmark = False      # type: ignore[attr-defined]

    # ---------------- scikit‑learn ----------------------------------
    # Most sklearn functions accept a `random_state` argument which you
    # should pass *explicitly* (with `seed`) rather than relying on a
    # global setting.  This util just documents that convention.

    return seed

# ----------------------------------------------------------------------
# Example:  -------------------------------------------------------------
# ----------------------------------------------------------------------
# from utils.seed import set_seed
# SEED = set_seed(123)
# print(f"Using seed: {SEED}")
# ----------------------------------------------------------------------

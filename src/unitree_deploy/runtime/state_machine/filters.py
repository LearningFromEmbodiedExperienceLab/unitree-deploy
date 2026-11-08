"""Joint-target smoothing shared by policy entry and arm transitions."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


class TargetEMA:
    """Filter the first N joint targets from a copy of the last sent target."""

    def __init__(self, config) -> None:
        if not isinstance(config, Mapping):
            raise TypeError("entry_ema must be a mapping with frames and alpha")
        self.frames = config.get("frames", 0)
        if type(self.frames) is not int or self.frames < 0:
            raise ValueError("entry_ema.frames must be a nonnegative integer")
        self.alpha = float(config.get("alpha", 0.3))
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError("entry_ema.alpha must be in (0, 1]")
        self.remaining = 0
        self.target: np.ndarray | None = None

    def reset(self, seed: np.ndarray) -> None:
        self.remaining = self.frames
        self.target = seed.copy() if self.frames else None

    def apply(self, target: np.ndarray) -> np.ndarray:
        if not self.remaining:
            return target
        self.target += self.alpha * (target - self.target)
        self.remaining -= 1
        return self.target

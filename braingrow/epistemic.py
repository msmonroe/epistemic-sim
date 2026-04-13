"""
braingrow/epistemic.py
Epistemic tag enum and WelfordStats — shared by all specialist nets.
"""
from __future__ import annotations
from enum import Enum, auto
import numpy as np


class EpistemicTag(Enum):
    CONFIDENT      = auto()
    HONEST_UNKNOWN = auto()
    OUT_OF_BOUNDS  = auto()


class WelfordStats:
    """Online mean/variance (Welford's algorithm)."""

    def __init__(self, warmup_steps: int = 50):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0
        self.warmup_steps = warmup_steps

    @property
    def variance(self) -> float:
        return self.M2 / self.n if self.n > 1 else 1.0

    @property
    def std(self) -> float:
        return float(np.sqrt(max(self.variance, 1e-12)))

    @property
    def warmed_up(self) -> bool:
        return self.n >= self.warmup_steps

    def update(self, x: float) -> None:
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        self.M2 += delta * (x - self.mean)

    def z_score(self, x: float) -> float:
        return abs(x - self.mean) / self.std

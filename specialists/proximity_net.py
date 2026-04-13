"""
specialists/proximity_net.py
-----------------------------
ProximityNet — MLP specialist for processing proximity/rangefinder arrays.

Architecture: simple 3-layer MLP (input is already low-dimensional)
OOB heuristics:
  - All rays return max range (0.0 after normalisation inversion)
  - All rays return min range (sensor buried / all blocked)
  - Input contains NaN or Inf
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from specialists.base import SpecialistNet


class ProximityNet(SpecialistNet):

    def __init__(
        self,
        n_rays: int = 8,
        device: str = "cpu",
    ):
        self.n_rays = n_rays
        super().__init__(modality="proximity", device=device)

    def _build_encoder(self) -> nn.Module:
        return nn.Sequential(
            nn.Linear(self.n_rays, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
        )

    def _encoder_out_dim(self) -> int:
        return 256

    def _preprocess(self, raw: np.ndarray) -> torch.Tensor:
        """
        raw: (n_rays,) float32 in [0, 1]  (normalised: 0=max range, 1=contact)
        → (n_rays,) float32
        """
        prox = raw.astype(np.float32)
        prox = np.clip(prox, 0.0, 1.0)
        return torch.from_numpy(prox)

    def _detect_oob(self, raw: np.ndarray) -> bool:
        """
        Heuristic OOB detection for proximity arrays.
        """
        prox = raw.astype(np.float32)

        # NaN or Inf
        if not np.isfinite(prox).all():
            return True

        # All rays at exactly 0 → max range (sensor dropout)
        if np.all(prox < 1e-6):
            return True

        # All rays at exactly 1 → all blocked (buried/stuck)
        if np.all(prox > 1.0 - 1e-6):
            return True

        return False

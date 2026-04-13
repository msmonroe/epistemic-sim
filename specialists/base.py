"""
specialists/base.py
-------------------
SpecialistNet ABC — enforces the epistemic output contract for all
modality-specific encoder networks.

Every specialist returns a SpecialistOutput containing:
  embedding      fixed-size latent vector  (torch.Tensor, EMBED_DIM)
  confidence     scalar [0, 1]             (Welford-normalised)
  epistemic_tag  CONFIDENT | HONEST_UNKNOWN | OUT_OF_BOUNDS
  modality       string identifier
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
import torch
import torch.nn as nn

from braingrow.epistemic import EpistemicTag, WelfordStats


# ── Output contract ───────────────────────────────────────────────────────────

@dataclass
class SpecialistOutput:
    embedding:     torch.Tensor
    confidence:    float
    epistemic_tag: EpistemicTag
    modality:      str
    latency_ms:    float = 0.0
    raw_logvar:    float = 0.0

    def is_usable(self) -> bool:
        return self.epistemic_tag != EpistemicTag.OUT_OF_BOUNDS

    def weight(self) -> float:
        if self.epistemic_tag == EpistemicTag.OUT_OF_BOUNDS:
            return 0.0
        if self.epistemic_tag == EpistemicTag.HONEST_UNKNOWN:
            return self.confidence * 0.5
        return self.confidence

    def to_dict(self) -> dict:
        return {
            f"{self.modality}/confidence":    self.confidence,
            f"{self.modality}/tag":           self.epistemic_tag.name,
            f"{self.modality}/weight":        self.weight(),
            f"{self.modality}/latency_ms":    self.latency_ms,
        }


# ── Abstract base ─────────────────────────────────────────────────────────────

class SpecialistNet(ABC, nn.Module):
    """
    All specialists share this forward pipeline:
      1. Hard OOB heuristic  (_detect_oob)
      2. Welford z-score OOB  (input norm)
      3. Encoder forward pass
      4. Confidence from log-variance head
      5. EpistemicTag assignment
      6. SpecialistOutput construction
    """

    EMBED_DIM: ClassVar[int] = 128

    def __init__(
        self,
        modality:            str,
        confident_threshold: float = 0.60,
        oob_z_threshold:     float = 3.5,
        device:              str   = "cpu",
    ):
        super().__init__()
        self.modality            = modality
        self.confident_threshold = confident_threshold
        self.oob_z_threshold     = oob_z_threshold
        self.device              = torch.device(device)

        self._conf_stats  = WelfordStats(warmup_steps=30)
        self._input_stats = WelfordStats(warmup_steps=30)

        # Built by subclass
        self.encoder = self._build_encoder()
        enc_dim = self._encoder_out_dim()
        self.embed_head  = nn.Linear(enc_dim, self.EMBED_DIM)
        self.logvar_head = nn.Linear(enc_dim, self.EMBED_DIM)

        self.to(self.device)

    # ── Subclass interface ────────────────────────────────────────────────────

    @abstractmethod
    def _build_encoder(self) -> nn.Module: ...

    @abstractmethod
    def _encoder_out_dim(self) -> int: ...

    @abstractmethod
    def _preprocess(self, raw: np.ndarray) -> torch.Tensor: ...

    @abstractmethod
    def _detect_oob(self, raw: np.ndarray) -> bool: ...

    # ── Forward ───────────────────────────────────────────────────────────────

    @torch.inference_mode()
    def forward(self, raw: np.ndarray) -> SpecialistOutput:
        t0 = time.perf_counter()

        # Hard heuristic OOB
        hard_oob = self._detect_oob(raw)

        # Welford z-score OOB
        input_norm = float(np.linalg.norm(raw.astype(np.float32).ravel()))
        self._input_stats.update(input_norm)
        soft_oob = (
            self._input_stats.warmed_up
            and self._input_stats.z_score(input_norm) > self.oob_z_threshold
        )
        is_oob = hard_oob or soft_oob

        # Encoder
        x = self._preprocess(raw).to(self.device)
        if x.dim() == x.ndim:          # ensure batch dim present
            x = x.unsqueeze(0)

        features = self.encoder(x).view(1, -1)
        embedding = self.embed_head(features)
        logvar    = self.logvar_head(features)

        # Confidence
        mean_logvar = float(logvar.mean().item())
        raw_conf    = float(torch.sigmoid(-logvar).mean().item())
        self._conf_stats.update(raw_conf)

        if self._conf_stats.warmed_up and self._conf_stats.std > 1e-6:
            norm_conf = float(np.clip(
                (raw_conf - self._conf_stats.mean) / (self._conf_stats.std * 2) + 0.5,
                0.0, 1.0
            ))
        else:
            norm_conf = raw_conf

        # Tag
        if is_oob:
            tag = EpistemicTag.OUT_OF_BOUNDS
        elif norm_conf >= self.confident_threshold:
            tag = EpistemicTag.CONFIDENT
        else:
            tag = EpistemicTag.HONEST_UNKNOWN

        return SpecialistOutput(
            embedding     = embedding.squeeze(0).cpu(),
            confidence    = norm_conf,
            epistemic_tag = tag,
            modality      = self.modality,
            latency_ms    = (time.perf_counter() - t0) * 1000,
            raw_logvar    = mean_logvar,
        )

    def reset_stats(self) -> None:
        self._conf_stats  = WelfordStats(warmup_steps=30)
        self._input_stats = WelfordStats(warmup_steps=30)

    def summary(self) -> dict:
        return {
            f"{self.modality}/conf_mean":  self._conf_stats.mean,
            f"{self.modality}/conf_std":   self._conf_stats.std,
            f"{self.modality}/n_samples":  self._conf_stats.n,
        }

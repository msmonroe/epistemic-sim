"""
specialists/vision_net.py
--------------------------
VisionNet — CNN-based specialist for processing RGB camera frames.

Architecture: lightweight 3-layer CNN (no pretrained weights needed for sim)
OOB heuristics:
  - Image is >95% a single value (occluded / all-black)
  - Mean pixel value is near 0 (camera off)
  - Pixel variance is near 0 (frozen frame)
"""
from __future__ import annotations

import time
import numpy as np
import torch
import torch.nn as nn

from specialists.base import SpecialistNet, SpecialistOutput
from braingrow.epistemic import EpistemicTag


class VisionNet(SpecialistNet):

    def __init__(
        self,
        image_h: int = 64,
        image_w: int = 64,
        device:  str = "cpu",
    ):
        self.image_h = image_h
        self.image_w = image_w
        # Synthetic frames have naturally high variance — loosen z-score threshold
        # Hard heuristics still catch true faults (black/white/frozen)
        super().__init__(
            modality="vision",
            confident_threshold=0.55,
            oob_z_threshold=6.0,    # wider — Welford OOB only for extreme outliers
            device=device,
        )

    def _build_encoder(self) -> nn.Module:
        return nn.Sequential(
            # Block 1: 3 → 16 channels, 64×64 → 32×32
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            # Block 2: 16 → 32 channels, 32×32 → 16×16
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # Block 3: 32 → 64 channels, 16×16 → 8×8
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # Flatten: 64 × 8 × 8 = 4096
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(inplace=True),
        )

    def _encoder_out_dim(self) -> int:
        return 256

    def _preprocess(self, raw: np.ndarray) -> torch.Tensor:
        """
        raw: (H, W, 3) uint8 or float32
        → (3, H, W) float32 normalised to [0, 1]
        """
        img = raw.astype(np.float32)
        if img.max() > 1.0:
            img = img / 255.0
        img = np.clip(img, 0.0, 1.0)
        # HWC → CHW
        tensor = torch.from_numpy(img.transpose(2, 0, 1))
        return tensor

    def _detect_oob(self, raw: np.ndarray) -> bool:
        """
        Heuristic OOB detection for camera images.
        Catches: black screen, frozen frame, fully saturated.
        Welford z-score disabled for vision — synthetic frames have
        too much natural variance to use input-norm OOB reliably.
        """
        img = raw.astype(np.float32)
        if img.max() > 1.0:
            img = img / 255.0

        pixel_var = float(np.var(img))
        mean_val  = float(np.mean(img))

        # Near-zero variance → blank or frozen frame
        if pixel_var < 5e-5:
            return True

        # All pixels near 0 → camera off / occluded
        if mean_val < 0.008:
            return True

        # All pixels near 1 → saturated
        if mean_val > 0.992:
            return True

        return False

    @torch.inference_mode()
    def forward(self, raw: np.ndarray):
        """Override to skip Welford input-norm OOB for vision modality.
        Vision relies solely on hard heuristics in _detect_oob."""
        import time
        from specialists.base import SpecialistOutput
        from braingrow.epistemic import EpistemicTag

        t0 = time.perf_counter()

        # Hard heuristic only — no Welford input-norm for vision
        is_oob = self._detect_oob(raw)

        x        = self._preprocess(raw).to(self.device).unsqueeze(0)
        features = self.encoder(x).view(1, -1)
        embedding = self.embed_head(features)
        logvar    = self.logvar_head(features)

        mean_logvar = float(logvar.mean().item())
        raw_conf    = float(torch.sigmoid(-logvar).mean().item())
        self._conf_stats.update(raw_conf)

        if self._conf_stats.warmed_up and self._conf_stats.std > 1e-6:
            norm_conf = float(np.clip(
                (raw_conf - self._conf_stats.mean) / (self._conf_stats.std * 2) + 0.5,
                0.0, 1.0
            ))
        else:
            norm_conf = 0.5   # default to mid-confidence during warmup

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

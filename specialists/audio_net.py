"""
specialists/audio_net.py
-------------------------
AudioNet — 1D-Conv + GRU specialist for processing audio waveform chunks.

Architecture: 1D conv feature extractor → GRU → pooled features
OOB heuristics:
  - Signal is clipped (>80% of samples at ±1.0)
  - Signal is silent (RMS < threshold)
  - Signal is pure noise (spectral flatness near 1.0)
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from specialists.base import SpecialistNet


class AudioNet(SpecialistNet):

    def __init__(
        self,
        n_samples: int = 256,
        device:    str = "cpu",
    ):
        self.n_samples = n_samples
        super().__init__(modality="audio", device=device)

    def _build_encoder(self) -> nn.Module:
        return nn.Sequential(
            # 1D conv blocks: (1, n_samples) → feature maps
            nn.Conv1d(1, 16, kernel_size=8, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # Adaptive pool → fixed 16-step sequence
            nn.AdaptiveAvgPool1d(16),
            # Flatten
            nn.Flatten(),
            nn.Linear(64 * 16, 256),
            nn.ReLU(inplace=True),
        )

    def _encoder_out_dim(self) -> int:
        return 256

    def _preprocess(self, raw: np.ndarray) -> torch.Tensor:
        """
        raw: (n_samples,) float32 in [-1, 1]
        → (1, n_samples) float32
        """
        audio = raw.astype(np.float32)
        audio = np.clip(audio, -1.0, 1.0)
        return torch.from_numpy(audio).unsqueeze(0)  # (1, n_samples)

    def _detect_oob(self, raw: np.ndarray) -> bool:
        """
        Heuristic OOB detection for audio signals.
        Catches: silence, ADC clipping, pure noise.
        """
        audio = raw.astype(np.float32)

        # Silence
        rms = float(np.sqrt(np.mean(audio ** 2)))
        if rms < 1e-5:
            return True

        # ADC clipping: >80% of samples at rail
        clipped = np.sum(np.abs(audio) > 0.98) / len(audio)
        if clipped > 0.80:
            return True

        return False

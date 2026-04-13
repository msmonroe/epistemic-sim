"""
gemma/fusion.py
---------------
Confidence-weighted embedding fusion — the input stage of Gemma.

Takes a list of SpecialistOutputs and produces a single fused tensor
weighted by each specialist's epistemic confidence. OOB specialists
are zero-weighted and excluded from the normalisation denominator.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from specialists.base import SpecialistOutput, EpistemicTag


class FusionLayer(nn.Module):
    """
    Confidence-weighted sum of specialist embeddings.

    If ALL specialists are OOB (total weight = 0), returns a zero vector
    and sets a flag so Gemma's policy head can issue a safe/stop action.
    """

    def __init__(self, embed_dim: int = 128, n_specialists: int = 3):
        super().__init__()
        self.embed_dim     = embed_dim
        self.n_specialists = n_specialists

        # Learned per-modality bias (small init — starts near equal weighting)
        self.modality_bias = nn.Parameter(
            torch.zeros(n_specialists)
        )

    def forward(
        self,
        outputs: list[SpecialistOutput],
    ) -> tuple[torch.Tensor, float, bool]:
        """
        Returns:
          fused:        (embed_dim,) weighted sum embedding
          total_weight: scalar — 0.0 means all sensors are OOB
          all_oob:      True if no usable signal exists
        """
        assert len(outputs) == self.n_specialists

        weighted_sum = torch.zeros(self.embed_dim)
        total_weight = 0.0

        for i, out in enumerate(outputs):
            w = out.weight() + float(self.modality_bias[i].item())
            w = max(w, 0.0)   # bias can't make OOB negative → still zero
            if out.epistemic_tag == EpistemicTag.OUT_OF_BOUNDS:
                w = 0.0
            weighted_sum = weighted_sum + out.embedding * w
            total_weight += w

        all_oob = total_weight < 1e-8

        if not all_oob:
            fused = weighted_sum / total_weight
        else:
            fused = torch.zeros(self.embed_dim)

        return fused, total_weight, all_oob

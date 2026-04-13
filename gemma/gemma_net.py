"""
gemma/gemma_net.py
------------------
Gemma — the central integrator network.

Receives fused specialist embeddings + proprioception,
outputs a 2D motor control signal [forward_vel, angular_vel].

Architecture: Fusion → MLP policy head
"""
from __future__ import annotations

import torch
import torch.nn as nn

from specialists.base import SpecialistOutput
from gemma.fusion import FusionLayer


class GemmaNet(nn.Module):

    def __init__(
        self,
        embed_dim:      int = 128,
        n_specialists:  int = 3,
        proprio_dim:    int = 5,    # [x, y, heading, vx, vy]
        action_dim:     int = 2,    # [forward_vel, angular_vel]
        hidden_dim:     int = 256,
    ):
        super().__init__()
        self.embed_dim   = embed_dim
        self.proprio_dim = proprio_dim

        self.fusion = FusionLayer(embed_dim=embed_dim, n_specialists=n_specialists)

        policy_in = embed_dim + proprio_dim + 1  # +1 for total_weight signal

        self.policy = nn.Sequential(
            nn.Linear(policy_in, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Tanh(),   # actions in [-1, 1]
        )

    def forward(
        self,
        specialist_outputs: list[SpecialistOutput],
        proprioception:     torch.Tensor,           # (proprio_dim,)
    ) -> tuple[torch.Tensor, dict]:
        """
        Returns:
          action:  (action_dim,)  float32
          info:    dict with fusion diagnostics
        """
        fused, total_weight, all_oob = self.fusion(specialist_outputs)

        # If all OOB → safe stop (zero action)
        if all_oob:
            action = torch.zeros(2)
            info = {"all_oob": True, "total_weight": 0.0, "fused_norm": 0.0}
            return action, info

        # Concatenate fused embedding + proprioception + weight signal
        weight_tensor = torch.tensor([total_weight], dtype=torch.float32)
        policy_in = torch.cat([fused, proprioception.float(), weight_tensor], dim=0)

        action = self.policy(policy_in)

        info = {
            "all_oob":       False,
            "total_weight":  total_weight,
            "fused_norm":    float(fused.norm().item()),
        }
        # Add per-specialist diagnostics
        for out in specialist_outputs:
            info.update(out.to_dict())

        return action, info

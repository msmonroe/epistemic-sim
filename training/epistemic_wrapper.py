"""
training/epistemic_wrapper.py
------------------------------
EpistemicWrapper — a gymnasium.Wrapper that:

  1. Receives raw dict observations from EpistemicEnv
  2. Runs them through the three specialist networks
  3. Concatenates [fused_embedding, proprio, weights, tag_flags]
     into a flat float32 vector for SB3's PPO

This keeps Gemma's fusion logic alive during training while SB3
handles the RL loop without needing to know about specialist nets.

Flat observation layout (total = 128 + 5 + 1 + 3 = 137):
  [0:128]    fused embedding (Gemma FusionLayer output)
  [128:133]  proprioception [x, y, yaw, vx, vy]
  [133]      total_weight   (epistemic health signal)
  [134:137]  per-modality OOB flags [vision_oob, audio_oob, prox_oob]
"""
from __future__ import annotations

import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces

from specialists.vision_net    import VisionNet
from specialists.audio_net     import AudioNet
from specialists.proximity_net import ProximityNet
from gemma.fusion              import FusionLayer
from braingrow.epistemic       import EpistemicTag


OBS_DIM = 128 + 5 + 1 + 3   # = 137


class EpistemicWrapper(gym.Wrapper):
    """
    Wraps EpistemicEnv with specialist networks + fusion.
    SB3 sees a flat Box(137,) observation.
    """

    def __init__(self, env: gym.Env, device: str = "cpu"):
        super().__init__(env)
        self.device = device

        # Specialist networks (untrained — will learn via gradient through Gemma)
        self.vision    = VisionNet(device=device)
        self.audio_net = AudioNet(device=device)
        self.prox_net  = ProximityNet(device=device)
        self.fusion    = FusionLayer(embed_dim=128, n_specialists=3)

        # Override observation space with flat vector
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
        )

        # Diagnostics — updated each step, logged by callback
        self.last_tags:    list[EpistemicTag] = []
        self.last_weights: list[float]        = []
        self.last_confs:   list[float]        = []

    def reset(self, **kwargs):
        obs_dict, info = self.env.reset(**kwargs)
        return self._encode(obs_dict), info

    def step(self, action):
        obs_dict, reward, terminated, truncated, info = self.env.step(action)
        flat_obs = self._encode(obs_dict)

        # Attach epistemic info to info dict for the callback
        info["epistemic"] = {
            "tags":         [t.name for t in self.last_tags],
            "weights":      self.last_weights,
            "confidences":  self.last_confs,
        }
        return flat_obs, reward, terminated, truncated, info

    # ── Encoding pipeline ─────────────────────────────────────────────────────

    def _encode(self, obs_dict: dict) -> np.ndarray:
        with torch.inference_mode():
            v_out = self.vision(obs_dict["image"])
            a_out = self.audio_net(obs_dict["audio"])
            p_out = self.prox_net(obs_dict["proximity"])

        specialist_outs = [v_out, a_out, p_out]

        # Record for logging
        self.last_tags    = [o.epistemic_tag for o in specialist_outs]
        self.last_weights = [o.weight()      for o in specialist_outs]
        self.last_confs   = [o.confidence    for o in specialist_outs]

        # Fusion (no grad needed here — specialists are frozen during PPO)
        with torch.no_grad():
            fused, total_weight, all_oob = self.fusion(specialist_outs)

        proprio = obs_dict["proprio"].astype(np.float32)   # (5,)

        # OOB flags as float [0/1]
        oob_flags = np.array([
            1.0 if v_out.epistemic_tag == EpistemicTag.OUT_OF_BOUNDS else 0.0,
            1.0 if a_out.epistemic_tag == EpistemicTag.OUT_OF_BOUNDS else 0.0,
            1.0 if p_out.epistemic_tag == EpistemicTag.OUT_OF_BOUNDS else 0.0,
        ], dtype=np.float32)

        flat = np.concatenate([
            fused.numpy(),                          # (128,)
            proprio,                                # (5,)
            np.array([total_weight], dtype=np.float32),  # (1,)
            oob_flags,                              # (3,)
        ])

        return flat.astype(np.float32)

"""
sim/sensors/microphone.py
Simulated directional microphone — louder when agent faces the target,
attenuated by distance and angle. Generates a tone + spatial envelope.
"""
from __future__ import annotations
import numpy as np


class MicrophoneSensor:
    def __init__(self, model, data, n_samples: int = 256,
                 base_freq: float = 440.0, sample_rate: float = 16000.0):
        self.model       = model
        self.data        = data
        self.n_samples   = n_samples
        self.base_freq   = base_freq
        self.sample_rate = sample_rate
        self._t          = 0.0          # running phase

    def sample(self, target_pos: np.ndarray) -> np.ndarray:
        """
        Returns (n_samples,) float32 audio signal in [-1, 1].
        Amplitude encodes spatial proximity to target.
        """
        agent_pos = self.data.qpos[:2].astype(np.float32)

        # Distance attenuation
        dist      = float(np.linalg.norm(agent_pos - target_pos)) + 0.01
        amplitude = float(np.clip(1.0 / (1.0 + dist * 0.4), 0.05, 1.0))

        # Directional gain: dot product of heading vector and target direction
        heading   = float(self.data.qpos[3] if self.data.qpos.size > 3 else 0.0)
        direction = target_pos - agent_pos
        if np.linalg.norm(direction) > 1e-3:
            direction = direction / np.linalg.norm(direction)
        facing    = np.array([np.cos(heading), np.sin(heading)])
        dot       = float(np.clip(np.dot(facing, direction), -1.0, 1.0))
        dir_gain  = 0.5 + 0.5 * dot      # [0, 1]

        amplitude *= dir_gain

        # Generate sine wave chunk
        t   = np.linspace(self._t, self._t + self.n_samples / self.sample_rate,
                          self.n_samples, endpoint=False, dtype=np.float32)
        sig = amplitude * np.sin(2.0 * np.pi * self.base_freq * t)

        # Add a small noise floor
        sig += np.random.normal(0.0, 0.02, size=self.n_samples).astype(np.float32)
        sig  = np.clip(sig, -1.0, 1.0)

        self._t += self.n_samples / self.sample_rate
        return sig

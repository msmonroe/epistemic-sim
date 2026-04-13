"""
sim/sensors/proximity.py
Simulated proximity sensor — 8 radial raycasts returning
normalised distances [0=far, 1=contact].
"""
from __future__ import annotations
import numpy as np
import mujoco


class ProximitySensor:
    def __init__(self, model, data, n_rays: int = 8, max_range: float = 3.0):
        self.model     = model
        self.data      = data
        self.n_rays    = n_rays
        self.max_range = max_range

        # Ray angles evenly distributed around the agent
        self.angles = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)

    def scan(self) -> np.ndarray:
        """Returns (n_rays,) float32 in [0,1]. 0=max range, 1=contact."""
        agent_pos = self.data.qpos[:2].astype(np.float64)
        heading   = float(self.data.qpos[3]) if self.data.qpos.size > 3 else 0.0

        distances = np.zeros(self.n_rays, dtype=np.float32)

        for i, angle in enumerate(self.angles):
            world_angle = heading + angle
            direction   = np.array([np.cos(world_angle), np.sin(world_angle), 0.0])
            origin      = np.array([agent_pos[0], agent_pos[1], 0.15])

            # MuJoCo raycast
            geom_id     = np.array([-1], dtype=np.int32)
            dist        = mujoco.mj_ray(
                self.model, self.data,
                origin, direction,
                None,           # geomgroup mask
                1,              # flg_static
                -1,             # bodyexclude (no exclusion)
                geom_id,
            )

            if dist < 0 or dist > self.max_range:
                distances[i] = 0.0      # max range → no obstacle
            else:
                distances[i] = float(1.0 - dist / self.max_range)

        return distances

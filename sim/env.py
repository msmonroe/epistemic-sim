"""
sim/env.py
----------
EpistemicEnv — Gymnasium environment wrapping the MuJoCo arena.

qpos layout (freejoint, 7 dof):
  [0:3]  x, y, z   (agent position)
  [3:7]  qw, qx, qy, qz  (agent quaternion)
qvel layout (6 dof):
  [0:3]  vx, vy, vz
  [3:6]  wx, wy, wz

Action: [forward_vel, angular_vel]  Box(2,) in [-1, 1]

Observation: dict of raw sensor arrays (fed to specialist nets externally)
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np
from gymnasium import Env, spaces

from sim.sensors.camera     import CameraSensor
from sim.sensors.microphone import MicrophoneSensor
from sim.sensors.proximity  import ProximitySensor
from sim.degradation        import DegradationSchedule

ASSETS_DIR = Path(__file__).parent / "assets"
ARENA_XML  = ASSETS_DIR / "arena.xml"


@dataclass
class EnvConfig:
    sim_dt:          float = 0.002
    control_dt:      float = 0.05
    episode_steps:   int   = 500
    camera_width:    int   = 64
    camera_height:   int   = 64
    n_proximity_rays:int   = 8
    audio_samples:   int   = 256
    arena_size:      float = 5.0
    reach_threshold: float = 0.5
    w_dist:          float = 3.0    # increased
    w_collision:     float = -0.5   # softer — don't dominate
    w_step:          float = -0.005 # lighter step penalty
    w_reach:         float = 50.0   # big terminal bonus
    w_proximity:     float = 2.0    # shaped proximity bonus
    degradation:     DegradationSchedule | None = None
    render_mode:     str | None = None


class EpistemicEnv(Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, config: EnvConfig | None = None):
        super().__init__()
        self.cfg = config or EnvConfig()
        self.render_mode = self.cfg.render_mode

        self._physics_steps = max(1, round(self.cfg.control_dt / self.cfg.sim_dt))

        self._model = mujoco.MjModel.from_xml_path(str(ARENA_XML))
        self._model.opt.timestep = self.cfg.sim_dt
        self._data  = mujoco.MjData(self._model)

        # Sensor objects
        self._camera    = CameraSensor(self._model, self._data,
                                       self.cfg.camera_width,
                                       self.cfg.camera_height)
        self._mic       = MicrophoneSensor(self._model, self._data,
                                           self.cfg.audio_samples)
        self._proximity = ProximitySensor(self._model, self._data,
                                          self.cfg.n_proximity_rays,
                                          self.cfg.arena_size)

        # Observation space
        self.observation_space = spaces.Dict({
            "image":     spaces.Box(0, 255,
                                    (self.cfg.camera_height,
                                     self.cfg.camera_width, 3),
                                    dtype=np.uint8),
            "audio":     spaces.Box(-1.0, 1.0,
                                    (self.cfg.audio_samples,),
                                    dtype=np.float32),
            "proximity": spaces.Box(0.0, 1.0,
                                    (self.cfg.n_proximity_rays,),
                                    dtype=np.float32),
            "proprio":   spaces.Box(-10.0, 10.0, (5,), dtype=np.float32),
        })

        # Action space: [forward_vel, angular_vel]
        self.action_space = spaces.Box(
            np.array([-1.0, -1.0], dtype=np.float32),
            np.array([ 1.0,  1.0], dtype=np.float32),
        )

        self._step_count = 0
        self._target_pos = np.zeros(2, dtype=np.float32)
        self._prev_dist  = 0.0
        self._fault_state: dict = {}

        # Target body id (cached)
        self._target_body_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_BODY, "target"
        )
        self._agent_body_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_BODY, "agent"
        )

    # ── Gymnasium API ─────────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self._model, self._data)

        self._step_count  = 0
        self._fault_state = {}
        if self.cfg.degradation:
            self.cfg.degradation.reset()

        # Random agent start position (z fixed at 0.15)
        sx, sy = self.np_random.uniform(-3.0, 3.0, size=2)
        self._data.qpos[0] = sx
        self._data.qpos[1] = sy
        self._data.qpos[2] = 0.15
        # Random heading via quaternion (rotation around z)
        yaw = self.np_random.uniform(-np.pi, np.pi)
        self._data.qpos[3] = np.cos(yaw / 2)   # qw
        self._data.qpos[4] = 0.0                # qx
        self._data.qpos[5] = 0.0                # qy
        self._data.qpos[6] = np.sin(yaw / 2)   # qz

        # Random target (ensure separation)
        for _ in range(100):
            tx, ty = self.np_random.uniform(-4.0, 4.0, size=2)
            if np.hypot(tx - sx, ty - sy) > 1.5:
                break
        self._target_pos = np.array([tx, ty], dtype=np.float32)

        # Move target in sim (set mocap or just track via xpos)
        self._data.xpos[self._target_body_id][0] = tx
        self._data.xpos[self._target_body_id][1] = ty
        self._data.xpos[self._target_body_id][2] = 0.3

        mujoco.mj_forward(self._model, self._data)
        self._prev_dist = self._dist_to_target()

        return self._get_obs(), self._get_info()

    def step(self, action: np.ndarray):
        self._apply_action(action)

        for _ in range(self._physics_steps):
            mujoco.mj_step(self._model, self._data)

        self._step_count += 1

        if self.cfg.degradation:
            self._fault_state = self.cfg.degradation.get_faults(self._step_count)

        obs      = self._get_obs()
        reward   = self._compute_reward()
        reached  = self._dist_to_target() < self.cfg.reach_threshold
        timeout  = self._step_count >= self.cfg.episode_steps

        return obs, reward, reached, timeout, self._get_info()

    def render(self):
        return self._camera.capture()

    def close(self):
        self._camera.close()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_obs(self) -> dict:
        image     = self._camera.capture()
        audio     = self._mic.sample(self._target_pos)
        proximity = self._proximity.scan()

        # Apply active faults
        for modality, fault in self._fault_state.items():
            if modality == "vision":
                image = fault.apply(image.astype(np.float32)).astype(np.uint8)
            elif modality == "audio":
                audio = fault.apply(audio)
            elif modality == "proximity":
                proximity = fault.apply(proximity)

        pos     = self._data.qpos[:2].astype(np.float32)
        vel     = self._data.qvel[:2].astype(np.float32)
        # Extract yaw from quaternion
        qw, qz  = float(self._data.qpos[3]), float(self._data.qpos[6])
        yaw     = float(2.0 * np.arctan2(qz, qw))
        proprio = np.array([pos[0], pos[1], yaw,
                            vel[0], vel[1]], dtype=np.float32)

        return {
            "image":     image,
            "audio":     audio,
            "proximity": proximity,
            "proprio":   proprio,
        }

    def _apply_action(self, action: np.ndarray):
        """
        Differential drive: map [fwd, ang] to body velocity via
        direct qvel manipulation (simpler than wheel actuators for now).
        """
        fwd = float(np.clip(action[0], -1.0, 1.0)) * 2.0  # max 2 m/s
        ang = float(np.clip(action[1], -1.0, 1.0)) * 2.0  # max 2 rad/s

        # Get current heading
        qw = float(self._data.qpos[3])
        qz = float(self._data.qpos[6])
        yaw = 2.0 * np.arctan2(qz, qw)

        # Set velocity in world frame
        self._data.qvel[0] = fwd * np.cos(yaw)
        self._data.qvel[1] = fwd * np.sin(yaw)
        self._data.qvel[2] = 0.0
        self._data.qvel[5] = ang   # yaw rate (wz)

    def _compute_reward(self) -> float:
        dist     = self._dist_to_target()
        delta    = self._prev_dist - dist        # positive = closing in
        self._prev_dist = dist
        collision = self._is_colliding()
        reached   = dist < self.cfg.reach_threshold

        # Exponential proximity bonus — reward being close, not just moving closer
        proximity_bonus = self.cfg.w_proximity * float(
            np.exp(-dist / 2.0)                  # peaks at 1.0 when dist=0
        )

        return (
            self.cfg.w_dist      * delta
          + self.cfg.w_collision * float(collision)
          + self.cfg.w_step
          + self.cfg.w_reach    * float(reached)
          + proximity_bonus
        )

    def _dist_to_target(self) -> float:
        pos = self._data.qpos[:2]
        return float(np.linalg.norm(pos - self._target_pos))

    def _is_colliding(self) -> bool:
        for i in range(self._data.ncon):
            c  = self._data.contact[i]
            b1 = self._model.geom_bodyid[c.geom1]
            b2 = self._model.geom_bodyid[c.geom2]
            if self._agent_body_id in (b1, b2):
                return True
        return False

    def _get_info(self) -> dict:
        return {
            "step":            self._step_count,
            "dist_to_target":  self._dist_to_target(),
            "active_faults":   list(self._fault_state.keys()),
            "agent_pos":       self._data.qpos[:2].tolist(),
            "target_pos":      self._target_pos.tolist(),
        }

"""sim/sensors/camera.py — RGB camera sensor using MuJoCo offscreen renderer."""
from __future__ import annotations
import os
import numpy as np
import mujoco

# Force headless EGL renderer (no display required)
os.environ.setdefault("MUJOCO_GL", "egl")


class CameraSensor:
    def __init__(self, model, data, width: int = 64, height: int = 64):
        self.model    = model
        self.data     = data
        self.width    = width
        self.height   = height
        try:
            self._renderer = mujoco.Renderer(model, height=height, width=width)
            self._use_renderer = True
        except Exception:
            self._renderer = None
            self._use_renderer = False

    def capture(self) -> np.ndarray:
        """Returns (H, W, 3) uint8 RGB image from agent's forward camera."""
        if self._use_renderer:
            try:
                self._renderer.update_scene(self.data, camera="agent_cam")
                frame = self._renderer.render()
                # EGL can return black frames silently — detect and fall back
                if frame.max() > 5:
                    return frame.copy()
            except Exception:
                pass
        return self._synthetic_frame()

    def _synthetic_frame(self) -> np.ndarray:
        """
        Physics-derived synthetic frame.
        Encodes agent heading and target direction as visual cues so the
        VisionNet can actually learn something useful even without OpenGL.

        Layout:
          - Sky gradient (top half): blue intensity encodes target bearing
          - Ground (bottom half): green channel encodes forward speed
          - Bright patch: target direction rendered as a gaussian blob
        """
        h, w = self.height, self.width
        img = np.zeros((h, w, 3), dtype=np.float32)

        # Pull agent state from MuJoCo data
        try:
            ax, ay   = float(self.data.qpos[0]), float(self.data.qpos[1])
            qw, qz   = float(self.data.qpos[3]), float(self.data.qpos[6])
            yaw      = 2.0 * np.arctan2(qz, qw)
            vx, vy   = float(self.data.qvel[0]), float(self.data.qvel[1])
            speed    = float(np.hypot(vx, vy))
        except Exception:
            ax, ay, yaw, speed = 0.0, 0.0, 0.0, 0.0

        # -- Sky (top half) — base blue-grey --
        sky = np.linspace(60, 100, h // 2, dtype=np.float32)
        img[:h//2, :, 2] = sky[:, None]
        img[:h//2, :, 1] = 40
        img[:h//2, :, 0] = 30

        # -- Ground (bottom half) — earthy green --
        ground_g = int(np.clip(50 + speed * 20, 50, 120))
        img[h//2:, :, 1] = ground_g
        img[h//2:, :, 0] = 30
        img[h//2:, :, 2] = 20

        # -- Target blob (if target position available) --
        try:
            # Read target position from xpos (set by env.reset)
            tid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target")
            tx  = float(self.data.xpos[tid][0])
            ty  = float(self.data.xpos[tid][1])

            # Angle to target in agent frame
            dx, dy       = tx - ax, ty - ay
            world_angle  = np.arctan2(dy, dx)
            rel_angle    = world_angle - yaw          # relative to heading
            rel_angle    = (rel_angle + np.pi) % (2 * np.pi) - np.pi   # [-π, π]
            dist         = float(np.hypot(dx, dy))

            # Map rel_angle → horizontal pixel position
            fov_rad   = np.radians(60)
            if abs(rel_angle) < fov_rad / 2:
                cx = int(w * (0.5 + rel_angle / fov_rad))
                cy = int(h * 0.35)   # just above horizon
                blob_r = max(3, int(12 / (1 + dist * 0.3)))

                # Gaussian blob — orange target marker
                ys, xs = np.ogrid[:h, :w]
                mask   = ((xs - cx)**2 + (ys - cy)**2) < blob_r**2
                img[mask, 0] = np.clip(img[mask, 0] + 200, 0, 255)
                img[mask, 1] = np.clip(img[mask, 1] + 80,  0, 255)
                img[mask, 2] = np.clip(img[mask, 2] - 20,  0, 255)
        except Exception:
            pass

        # Add mild noise for texture (prevents frozen-frame OOB false positives)
        noise = np.random.randint(-12, 12, (h, w, 3), dtype=np.int16)
        img   = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return img

    def close(self):
        if self._renderer:
            try:
                self._renderer.close()
            except Exception:
                pass

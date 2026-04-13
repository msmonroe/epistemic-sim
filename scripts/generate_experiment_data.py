"""
scripts/generate_experiment_data.py
------------------------------------
Generates experiment data using the specialist nets + Gemma directly,
bypassing MuJoCo physics (which is the CPU bottleneck in this container).

This is scientifically valid: we're measuring epistemic tag behaviour
and fusion weight dynamics, which are properties of the neural architecture
and degradation pipeline — not of the RL policy.

Each "episode" simulates:
  - A directed agent moving toward a target (distance decays over time)
  - Realistic sensor signals derived from the agent's position
  - Mid-episode fault injection per experiment schedule
  - Specialist forward passes producing real epistemic tags
  - Gemma fusion producing real weighted embeddings

Results saved to results/experiment_data.npz for visualise.py.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from collections import defaultdict
from dataclasses import dataclass

from specialists.vision_net    import VisionNet
from specialists.audio_net     import AudioNet
from specialists.proximity_net import ProximityNet
from gemma.fusion              import FusionLayer
from braingrow.epistemic        import EpistemicTag
from sim.degradation           import (
    DegradationSchedule, SensorFault, FaultType
)

# ── Config ────────────────────────────────────────────────────────────────────

N_EPISODES   = 15
EP_STEPS     = 300
FAULT_STEP   = 80
ARENA_SIZE   = 5.0
TARGET_DIST_INIT = 9.0   # starting distance to target (metres)
REACH_THRESH = 0.5

EXPERIMENTS = {
    "baseline":        None,
    "vision_degraded": DegradationSchedule.vision_occluded(start_step=FAULT_STEP),
    "audio_degraded":  DegradationSchedule.audio_noisy(start_step=FAULT_STEP, sigma=0.7),
    "multi_fault":     DegradationSchedule.multi_fault(start_step=FAULT_STEP),
}

# ── Sensor signal generators ──────────────────────────────────────────────────

def make_camera_frame(dist: float, heading_error: float) -> np.ndarray:
    """
    Realistic camera frame:
    - Target appears as orange blob, size and brightness scale with proximity
    - Background is textured floor + walls
    - heading_error shifts target horizontally
    """
    img = np.zeros((64, 64, 3), dtype=np.float32)

    # Floor gradient (grey-blue checker)
    for y in range(32, 64):
        t = (y - 32) / 32.0
        img[y, :, 0] = 0.15 + t * 0.08
        img[y, :, 1] = 0.18 + t * 0.10
        img[y, :, 2] = 0.22 + t * 0.12

    # Ceiling (darker blue)
    img[:32, :, 2] = 0.18
    img[:32, :, 0] = 0.08
    img[:32, :, 1] = 0.10

    # Add checker texture noise
    xx, yy = np.meshgrid(np.arange(64), np.arange(64))
    checker = ((xx // 8 + yy // 8) % 2).astype(np.float32) * 0.04
    img[:, :, 0] += checker * 0.6
    img[:, :, 1] += checker * 0.6
    img[:, :, 2] += checker * 0.7

    # Target blob — orange sphere
    if dist < ARENA_SIZE * 1.5:
        blob_size  = max(2, int(14 * (1.0 - dist / (ARENA_SIZE * 1.5))))
        cx = 32 + int(np.clip(heading_error * 20, -28, 28))
        cy = 30  # slightly above horizon
        yy, xx = np.ogrid[:64, :64]
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= blob_size ** 2
        bright = max(0.2, 1.0 - dist / ARENA_SIZE)
        img[mask, 0] = np.clip(bright * 1.0, 0, 1)
        img[mask, 1] = np.clip(bright * 0.45, 0, 1)
        img[mask, 2] = np.clip(bright * 0.05, 0, 1)

    # Obstacle (red box) if nearby obstacle in view
    if np.random.rand() < 0.15:
        ox = np.random.randint(5, 55)
        img[28:45, ox:ox+6, 0] = 0.55
        img[28:45, ox:ox+6, 1] = 0.15
        img[28:45, ox:ox+6, 2] = 0.15

    img = np.clip(img + np.random.normal(0, 0.015, img.shape), 0, 1)
    return (img * 255).astype(np.uint8)


def make_audio_signal(dist: float, heading_error: float) -> np.ndarray:
    """
    440 Hz tone + harmonics, attenuated by distance and heading error.
    """
    n = 256
    t = np.linspace(0, 1, n, dtype=np.float32)
    amplitude = float(np.clip(1.0 / (1.0 + dist * 0.35), 0.06, 1.0))
    dir_gain  = float(np.clip(0.5 + 0.5 * np.cos(heading_error), 0.1, 1.0))
    amplitude *= dir_gain

    sig  = amplitude * np.sin(2 * np.pi * 440 * t)
    sig += amplitude * 0.3 * np.sin(2 * np.pi * 880 * t)   # 2nd harmonic
    sig += amplitude * 0.1 * np.sin(2 * np.pi * 1320 * t)  # 3rd harmonic
    sig += np.random.normal(0, 0.018, n).astype(np.float32)
    return np.clip(sig, -1.0, 1.0).astype(np.float32)


def make_proximity_reading(dist: float, has_obstacle: bool) -> np.ndarray:
    """
    8-ray proximity: encodes nearby obstacles + distance cues.
    """
    prox = np.random.uniform(0.05, 0.2, 8).astype(np.float32)

    if has_obstacle:
        # One or two rays light up
        idx = np.random.choice(8, size=np.random.randint(1, 3), replace=False)
        for i in idx:
            prox[i] = np.random.uniform(0.6, 0.95)

    # Forward ray brightens as target gets close
    prox[0] = float(np.clip((ARENA_SIZE - dist) / ARENA_SIZE * 0.4 + prox[0], 0, 1))

    return np.clip(prox, 0, 1)


def simulate_reward(dist: float, prev_dist: float, collision: bool) -> float:
    delta = prev_dist - dist
    return (2.0 * delta
            - 0.01           # step penalty
            - 1.5 * collision
            + (20.0 if dist < REACH_THRESH else 0.0))


# ── Episode simulation ────────────────────────────────────────────────────────

def run_episode(
    vision:    VisionNet,
    audio_net: AudioNet,
    prox_net:  ProximityNet,
    fusion:    FusionLayer,
    degradation: DegradationSchedule | None,
    seed: int,
    ep_steps: int = EP_STEPS,
) -> dict:
    rng = np.random.default_rng(seed)

    # Agent starts far from target
    dist      = float(rng.uniform(4.0, TARGET_DIST_INIT))
    prev_dist = dist
    heading_err = float(rng.uniform(-np.pi / 4, np.pi / 4))
    ep_reward = 0.0

    step_data: dict[str, list] = defaultdict(list)

    if degradation:
        degradation.reset()

    for step in range(ep_steps):
        # Agent moves toward target (simple controller)
        speed = 0.03 + rng.uniform(-0.005, 0.01)
        dist  = max(0.0, dist - speed)

        # Heading error decays (agent steers toward target)
        heading_err *= 0.92
        heading_err += rng.uniform(-0.05, 0.05)
        has_obstacle = rng.random() < 0.12

        # Build raw sensor signals
        img  = make_camera_frame(dist, heading_err)
        aud  = make_audio_signal(dist, heading_err)
        prox = make_proximity_reading(dist, has_obstacle)

        # Apply faults
        faults = degradation.get_faults(step) if degradation else {}
        if "vision" in faults:
            img  = faults["vision"].apply(img.astype(np.float32)).astype(np.uint8)
        if "audio" in faults:
            aud  = faults["audio"].apply(aud)
        if "proximity" in faults:
            prox = faults["proximity"].apply(prox)

        # Specialist forward passes (real inference)
        v_out = vision(img)
        a_out = audio_net(aud)
        p_out = prox_net(prox)

        # Fusion
        import torch
        with torch.no_grad():
            fused, total_weight, all_oob = fusion([v_out, a_out, p_out])

        # Reward
        collision = has_obstacle and float(prox.max()) > 0.85
        reward    = simulate_reward(dist, prev_dist, collision)
        ep_reward += reward
        prev_dist  = dist

        # Record
        step_data["step"].append(step)
        step_data["dist"].append(dist)
        step_data["reward"].append(reward)
        step_data["weight"].append(float(total_weight))
        step_data["vision_conf"].append(v_out.confidence)
        step_data["audio_conf"].append(a_out.confidence)
        step_data["proximity_conf"].append(p_out.confidence)
        step_data["vision_oob"].append(1.0 if v_out.epistemic_tag == EpistemicTag.OUT_OF_BOUNDS else 0.0)
        step_data["audio_oob"].append(1.0 if a_out.epistemic_tag == EpistemicTag.OUT_OF_BOUNDS else 0.0)
        step_data["proximity_oob"].append(1.0 if p_out.epistemic_tag == EpistemicTag.OUT_OF_BOUNDS else 0.0)
        step_data["all_oob"].append(1.0 if all_oob else 0.0)

        if dist < REACH_THRESH:
            break

    step_data["ep_reward"] = [ep_reward]
    step_data["ep_len"]    = [len(step_data["step"])]
    return {k: np.array(v) for k, v in step_data.items()}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import time
    os.makedirs("results", exist_ok=True)

    print("\n  epistemic-sim — Fast Experiment Data Generator")
    print(f"  {N_EPISODES} episodes × {EP_STEPS} steps × 4 experiments")
    print(f"  Fault injection at step {FAULT_STEP}\n")

    # Shared network instances (weights are random but consistent across exps)
    vision    = VisionNet(device="cpu")
    audio_net = AudioNet(device="cpu")
    prox_net  = ProximityNet(device="cpu")
    fusion    = FusionLayer(embed_dim=128, n_specialists=3)

    all_flat: dict = {}
    t_global = time.time()

    for exp_name, degradation in EXPERIMENTS.items():
        t0 = time.time()
        print(f"  ── {exp_name}")

        exp_data: dict[str, list] = defaultdict(list)

        for ep in range(N_EPISODES):
            # Reset Welford stats between experiments for fairness
            if ep == 0:
                vision.reset_stats()
                audio_net.reset_stats()
                prox_net.reset_stats()

            ep_result = run_episode(
                vision, audio_net, prox_net, fusion,
                degradation=degradation,
                seed=ep * 31 + hash(exp_name) % 1000,
                ep_steps=EP_STEPS,
            )

            for key, arr in ep_result.items():
                exp_data[key].append(arr)

        # Concatenate episodes (except scalar ep-level metrics)
        ep_level = {"ep_reward", "ep_len"}
        merged: dict[str, np.ndarray] = {}
        for key, arrays in exp_data.items():
            if key in ep_level:
                merged[key] = np.concatenate(arrays)
            else:
                merged[key] = np.concatenate(arrays)

        # Print summary
        elapsed = time.time() - t0
        v_oob   = merged["vision_oob"].mean() * 100
        a_oob   = merged["audio_oob"].mean() * 100
        p_oob   = merged["proximity_oob"].mean() * 100
        w_mean  = merged["weight"].mean()
        ep_r    = merged["ep_reward"].mean()
        ep_l    = merged["ep_len"].mean()

        print(f"     ep_reward={ep_r:+.1f}  ep_len={ep_l:.0f}  weight={w_mean:.3f}")
        print(f"     OOB%  vision={v_oob:.1f}  audio={a_oob:.1f}  prox={p_oob:.1f}  ({elapsed:.1f}s)")

        # Store for npz
        for key, arr in merged.items():
            all_flat[f"{exp_name}__{key}"] = arr

    np.savez("results/experiment_data.npz", **all_flat)
    print(f"\n  Done in {time.time()-t_global:.1f}s")
    print("  Saved → results/experiment_data.npz\n")


if __name__ == "__main__":
    main()

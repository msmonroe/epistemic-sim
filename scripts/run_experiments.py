"""
scripts/run_experiments.py
---------------------------
Runs the full experiment matrix:
  1. baseline          — all sensors healthy
  2. vision_degraded   — camera occludes at step 150
  3. audio_degraded    — mic noise at step 150
  4. multi_fault       — vision + audio fail simultaneously
  5. ablation          — no epistemic weighting (control group)

For each experiment, collects:
  - Per-step reward, distance-to-target
  - Per-modality confidence and OOB rate
  - Fusion weight over time

Results saved to results/experiment_data.npz for the visualiser.
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
from collections import defaultdict
import multiprocessing as mp
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from sim.env              import EpistemicEnv, EnvConfig
from sim.degradation      import DegradationSchedule
from training.epistemic_wrapper import EpistemicWrapper, OBS_DIM
from training.callbacks         import EpistemicCallback


# ── Experiment definitions ────────────────────────────────────────────────────

EXPERIMENTS = {
    "baseline":        None,
    "vision_degraded": DegradationSchedule.vision_occluded(start_step=150),
    "audio_degraded":  DegradationSchedule.audio_noisy(start_step=150, sigma=0.7),
    "multi_fault":     DegradationSchedule.multi_fault(start_step=150),
}

TRAIN_STEPS   = 30_000
EVAL_EPISODES = 10
FAULT_STEP    = 150


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_env(degradation=None) -> EpistemicWrapper:
    cfg = EnvConfig(episode_steps=400, degradation=degradation)
    env = Monitor(EpistemicEnv(config=cfg))
    return EpistemicWrapper(env)


def run_experiment_worker(args: tuple) -> tuple[str, dict]:
    """Top-level worker for multiprocessing — trains + evaluates one experiment."""
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.environ.setdefault("MUJOCO_GL", "egl")
    name, degradation = args
    model = train_experiment(name, degradation, TRAIN_STEPS)
    data  = evaluate_experiment(name, model, degradation)
    return name, data


def train_experiment(name: str, degradation, steps: int) -> PPO:
    print(f"\n  {'─'*50}")
    print(f"  TRAINING: {name}  ({steps:,} steps)")
    print(f"  {'─'*50}")

    env   = make_env(degradation)
    model = PPO(
        "MlpPolicy", env,
        n_steps=512, batch_size=64, n_epochs=4,
        learning_rate=3e-4, gamma=0.99, gae_lambda=0.95,
        clip_range=0.2, ent_coef=0.01, verbose=0,
        device="cpu",
    )
    cb = EpistemicCallback(log_interval=5000, verbose=1)
    model.learn(total_timesteps=steps, callback=cb)

    os.makedirs("models", exist_ok=True)
    model.save(f"models/ppo_{name}")
    env.close()
    return model


def evaluate_experiment(
    name: str,
    model: PPO,
    degradation,
    n_episodes: int = EVAL_EPISODES,
) -> dict:
    """
    Run n_episodes with the trained policy.
    Returns per-step arrays of reward, dist, confidence, oob, weight.
    """
    print(f"  Evaluating {name} ({n_episodes} episodes)...")

    env = make_env(degradation)
    results = defaultdict(list)

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep * 7)
        ep_reward = 0.0
        step      = 0

        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_info = info.get("epistemic", {})

            results["reward"].append(reward)
            results["dist"].append(info.get("dist_to_target", 0.0))
            results["step"].append(step)
            results["episode"].append(ep)
            results["weight"].append(sum(ep_info.get("weights", [0,0,0])))

            for i, mod in enumerate(["vision", "audio", "proximity"]):
                confs = ep_info.get("confidences", [0,0,0])
                tags  = ep_info.get("tags", ["","",""])
                results[f"{mod}_conf"].append(confs[i] if i < len(confs) else 0.0)
                results[f"{mod}_oob"].append(
                    1.0 if (i < len(tags) and tags[i] == "OUT_OF_BOUNDS") else 0.0
                )

            ep_reward += reward
            step      += 1

            if terminated or truncated:
                results["ep_reward"].append(ep_reward)
                results["ep_len"].append(step)
                break

    env.close()

    # Convert to numpy
    return {k: np.array(v) for k, v in results.items()}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs("results", exist_ok=True)

    n_workers = min(len(EXPERIMENTS), mp.cpu_count())
    print(f"\n  Running {len(EXPERIMENTS)} experiments in parallel ({n_workers} workers)\n")

    args = list(EXPERIMENTS.items())   # [(name, degradation), ...]

    with mp.Pool(processes=n_workers, maxtasksperchild=1) as pool:
        results_list = pool.map(run_experiment_worker, args)

    all_results = dict(results_list)

    for name, data in all_results.items():
        mean_r = data["ep_reward"].mean()
        mean_l = data["ep_len"].mean()
        v_oob  = data["vision_oob"].mean() * 100
        a_oob  = data["audio_oob"].mean()  * 100
        p_oob  = data["proximity_oob"].mean() * 100
        print(f"  {name}:  reward={mean_r:.1f}  len={mean_l:.0f}  "
              f"OOB%  vision={v_oob:.1f}  audio={a_oob:.1f}  prox={p_oob:.1f}")

    # Save flat npz
    flat = {}
    for exp_name, data in all_results.items():
        for key, arr in data.items():
            flat[f"{exp_name}__{key}"] = arr

    np.savez("results/experiment_data.npz", **flat)
    print("\n  Results saved → results/experiment_data.npz")

    return all_results


if __name__ == "__main__":
    main()

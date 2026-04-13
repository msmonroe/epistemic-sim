"""
scripts/train.py
-----------------
Main training script. Runs PPO on EpistemicEnv via EpistemicWrapper.

Usage:
  python scripts/train.py                          # baseline, 50k steps
  python scripts/train.py --exp vision_degraded    # fault injection
  python scripts/train.py --steps 200000           # longer run
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from sim.env              import EpistemicEnv, EnvConfig
from sim.degradation      import DegradationSchedule
from training.epistemic_wrapper import EpistemicWrapper
from training.callbacks         import EpistemicCallback


EXPERIMENTS = {
    "baseline":        None,
    "vision_degraded": DegradationSchedule.vision_occluded(start_step=150),
    "audio_degraded":  DegradationSchedule.audio_noisy(start_step=150),
    "multi_fault":     DegradationSchedule.multi_fault(start_step=150),
}


def make_env(exp_name: str = "baseline") -> EpistemicWrapper:
    degradation = EXPERIMENTS.get(exp_name)
    cfg = EnvConfig(
        episode_steps=500,
        degradation=degradation,
    )
    env = EpistemicEnv(config=cfg)
    env = Monitor(env)
    env = EpistemicWrapper(env)
    return env


def train(exp_name: str = "baseline", total_steps: int = 50_000):
    print(f"\n  epistemic-sim PPO training")
    print(f"  experiment : {exp_name}")
    print(f"  steps      : {total_steps:,}")
    print(f"  obs dim    : 137  (128 embed + 5 proprio + 1 weight + 3 oob)\n")

    env = make_env(exp_name)

    model = PPO(
        "MlpPolicy",
        env,
        n_steps=512,
        batch_size=64,
        n_epochs=4,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=0,
    )

    callback = EpistemicCallback(log_interval=500, verbose=1)

    model.learn(total_timesteps=total_steps, callback=callback)

    save_path = f"models/ppo_{exp_name}"
    os.makedirs("models", exist_ok=True)
    model.save(save_path)
    print(f"  Model saved → {save_path}.zip\n")

    env.close()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp",   default="baseline", choices=list(EXPERIMENTS))
    parser.add_argument("--steps", default=50_000, type=int)
    args = parser.parse_args()
    train(args.exp, args.steps)

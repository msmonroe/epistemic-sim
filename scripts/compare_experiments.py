"""
scripts/compare_experiments.py
--------------------------------
Runs multiple experiments back-to-back and prints a comparison table.
This is the publishable result — does BrainGrow epistemic fusion
enable graceful degradation under sensor fault?

Usage:
    python scripts/compare_experiments.py --steps 25000
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time
import numpy as np
from collections import defaultdict, deque

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

from sim.env              import EpistemicEnv, EnvConfig
from sim.degradation      import DegradationSchedule
from training.epistemic_wrapper import EpistemicWrapper
from braingrow.epistemic  import EpistemicTag


# ── Metrics collector callback ────────────────────────────────────────────────

class MetricsCallback(BaseCallback):
    """Collects per-episode stats silently — no stdout."""

    def __init__(self):
        super().__init__(verbose=0)
        self.ep_rewards:    list[float] = []
        self.ep_lengths:    list[int]   = []
        self.ep_successes:  list[bool]  = []
        self.oob_counts:    dict        = defaultdict(int)
        self.total_steps:   int         = 0
        self._cur_reward    = 0.0
        self._cur_len       = 0

    def _on_step(self) -> bool:
        r    = self.locals["rewards"][0]
        info = self.locals["infos"][0]
        done = self.locals["dones"][0]

        self._cur_reward += r
        self._cur_len    += 1
        self.total_steps += 1

        ep_info = info.get("epistemic", {})
        for i, tag in enumerate(ep_info.get("tags", [])):
            if tag == "OUT_OF_BOUNDS":
                mods = ["vision", "audio", "proximity"]
                if i < len(mods):
                    self.oob_counts[mods[i]] += 1

        if done:
            reached = info.get("dist_to_target", 999) < 0.5
            self.ep_rewards.append(self._cur_reward)
            self.ep_lengths.append(self._cur_len)
            self.ep_successes.append(reached)
            self._cur_reward = 0.0
            self._cur_len    = 0

        return True

    def summary(self) -> dict:
        if not self.ep_rewards:
            return {}
        return {
            "n_episodes":    len(self.ep_rewards),
            "mean_reward":   np.mean(self.ep_rewards),
            "std_reward":    np.std(self.ep_rewards),
            "mean_ep_len":   np.mean(self.ep_lengths),
            "success_rate":  np.mean(self.ep_successes) * 100,
            "oob_vision":    self.oob_counts.get("vision",    0) / max(self.total_steps, 1) * 100,
            "oob_audio":     self.oob_counts.get("audio",     0) / max(self.total_steps, 1) * 100,
            "oob_proximity": self.oob_counts.get("proximity", 0) / max(self.total_steps, 1) * 100,
        }


# ── Experiment runner ─────────────────────────────────────────────────────────

EXPERIMENTS = {
    "baseline":        None,
    "vision_degraded": DegradationSchedule.vision_occluded(start_step=150),
    "audio_degraded":  DegradationSchedule.audio_noisy(start_step=150, sigma=0.7),
    "multi_fault":     DegradationSchedule.multi_fault(start_step=150),
}


def run_experiment(name: str, total_steps: int) -> dict:
    degradation = EXPERIMENTS[name]
    cfg = EnvConfig(episode_steps=500, degradation=degradation)
    env = EpistemicWrapper(Monitor(EpistemicEnv(config=cfg)))

    model = PPO(
        "MlpPolicy", env,
        n_steps=512, batch_size=64, n_epochs=4,
        learning_rate=3e-4, gamma=0.99, gae_lambda=0.95,
        clip_range=0.2, ent_coef=0.01, verbose=0,
    )

    cb = MetricsCallback()
    model.learn(total_timesteps=total_steps, callback=cb)
    env.close()

    result = cb.summary()
    result["name"] = name
    return result


# ── Pretty printer ────────────────────────────────────────────────────────────

def print_results(results: list[dict]):
    G  = "\033[92m"; Y = "\033[93m"; R = "\033[91m"
    C  = "\033[96m"; B = "\033[1m";  X = "\033[0m"

    print(f"\n{B}{'═'*75}{X}")
    print(f"{B}{C}  EPISTEMIC-SIM — EXPERIMENT COMPARISON{X}")
    print(f"{B}{'═'*75}{X}\n")

    # Header
    cols = ["Experiment", "Episodes", "Mean Reward", "Success%",
            "Ep Len", "OOB-V%", "OOB-A%", "OOB-P%"]
    widths = [18, 9, 13, 10, 8, 8, 8, 8]

    header = "  " + "  ".join(f"{c:<{w}}" for c, w in zip(cols, widths))
    print(f"{B}{header}{X}")
    print("  " + "─" * (sum(widths) + 2 * len(widths)))

    for r in results:
        name        = r.get("name", "?")
        n_ep        = r.get("n_episodes",  0)
        mean_r      = r.get("mean_reward", 0.0)
        succ        = r.get("success_rate",0.0)
        ep_len      = r.get("mean_ep_len", 0.0)
        oob_v       = r.get("oob_vision",  0.0)
        oob_a       = r.get("oob_audio",   0.0)
        oob_p       = r.get("oob_proximity",0.0)

        # Colour reward: green if best, red if worst
        best_r = max(x.get("mean_reward", -999) for x in results)
        r_col  = G if abs(mean_r - best_r) < 1.0 else (R if mean_r < best_r - 50 else Y)

        # Colour success
        best_s = max(x.get("success_rate", 0) for x in results)
        s_col  = G if abs(succ - best_s) < 1.0 else (R if succ < best_s - 5 else Y)

        # OOB coloring
        def oob_col(v): return R if v > 20 else (Y if v > 5 else G)

        row = (
            f"  {name:<{widths[0]}}"
            f"  {n_ep:<{widths[1]}}"
            f"  {r_col}{mean_r:>+{widths[2]-1}.1f}{X}"
            f"  {s_col}{succ:>{widths[3]-1}.1f}%{X}"
            f"  {ep_len:>{widths[4]}.1f}"
            f"  {oob_col(oob_v)}{oob_v:>{widths[5]-1}.1f}%{X}"
            f"  {oob_col(oob_a)}{oob_a:>{widths[6]-1}.1f}%{X}"
            f"  {oob_col(oob_p)}{oob_p:>{widths[7]-1}.1f}%{X}"
        )
        print(row)

    print("\n  " + "─" * (sum(widths) + 2 * len(widths)))
    print(f"\n  {B}Key:{X}")
    print(f"  {G}■{X} Best / healthy   "
          f"{Y}■{X} Moderate         "
          f"{R}■{X} Degraded / high OOB")
    print(f"\n  {B}Interpretation:{X}")

    results_by_reward = sorted(results, key=lambda x: x.get("mean_reward", -999), reverse=True)
    best  = results_by_reward[0]
    worst = results_by_reward[-1]
    gap   = best.get("mean_reward",0) - worst.get("mean_reward",0)

    print(f"  Best experiment:  {G}{best['name']}{X}  "
          f"(mean reward {best.get('mean_reward',0):+.1f})")
    print(f"  Worst experiment: {R}{worst['name']}{X}  "
          f"(mean reward {worst.get('mean_reward',0):+.1f})")
    print(f"  Reward gap:       {gap:.1f} points across conditions")

    # Check graceful degradation claim
    baseline = next((r for r in results if r["name"] == "baseline"), None)
    multi    = next((r for r in results if r["name"] == "multi_fault"), None)
    if baseline and multi:
        deg = baseline.get("mean_reward",0) - multi.get("mean_reward",0)
        if deg < 80:
            print(f"\n  {G}{B}✓ Graceful degradation observed:{X} "
                  f"multi-fault performance within {deg:.1f} pts of baseline")
        else:
            print(f"\n  {Y}⚠ Degradation gap {deg:.1f} pts — "
                  f"more training steps may improve robustness")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps",    default=20_000, type=int,
                        help="Training steps per experiment")
    parser.add_argument("--exp",      default="all",
                        help="Comma-separated experiment names, or 'all'")
    args = parser.parse_args()

    if args.exp == "all":
        exp_names = list(EXPERIMENTS.keys())
    else:
        exp_names = [e.strip() for e in args.exp.split(",")]

    print(f"\n  Running {len(exp_names)} experiments × {args.steps:,} steps each")
    print(f"  Experiments: {', '.join(exp_names)}\n")

    all_results = []
    for i, name in enumerate(exp_names):
        t0 = time.time()
        print(f"  [{i+1}/{len(exp_names)}] {name}...", end="", flush=True)
        result = run_experiment(name, args.steps)
        elapsed = time.time() - t0
        print(f" done ({elapsed:.0f}s)  "
              f"reward={result.get('mean_reward',0):+.1f}  "
              f"success={result.get('success_rate',0):.1f}%")
        all_results.append(result)

    print_results(all_results)

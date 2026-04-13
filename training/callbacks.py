"""
training/callbacks.py
----------------------
EpistemicCallback — SB3 BaseCallback that logs per-modality
epistemic stats to stdout (and optionally W&B) every N steps.
"""
from __future__ import annotations

import numpy as np
from collections import defaultdict, deque
from stable_baselines3.common.callbacks import BaseCallback


class EpistemicCallback(BaseCallback):
    """
    Logs:
      - Mean reward (rolling 20 episodes)
      - Per-modality confidence (vision / audio / proximity)
      - OOB rate per modality
      - Total fusion weight
      - Episode length
    """

    def __init__(self, log_interval: int = 500, verbose: int = 1):
        super().__init__(verbose)
        self.log_interval = log_interval
        self._ep_rewards:   deque = deque(maxlen=20)
        self._ep_lengths:   deque = deque(maxlen=20)
        self._conf_buf:     dict  = defaultdict(list)
        self._oob_buf:      dict  = defaultdict(list)
        self._weight_buf:   list  = []
        self._current_ep_reward = 0.0
        self._current_ep_len    = 0

    def _on_step(self) -> bool:
        reward = self.locals["rewards"][0]
        info   = self.locals["infos"][0]

        self._current_ep_reward += reward
        self._current_ep_len    += 1

        # Pull epistemic info from wrapper
        ep_info = info.get("epistemic", {})
        tags    = ep_info.get("tags",        [])
        confs   = ep_info.get("confidences", [])
        weights = ep_info.get("weights",     [])

        modalities = ["vision", "audio", "proximity"]
        for i, mod in enumerate(modalities):
            if i < len(confs):
                self._conf_buf[mod].append(confs[i])
            if i < len(tags):
                self._oob_buf[mod].append(1.0 if tags[i] == "OUT_OF_BOUNDS" else 0.0)

        if weights:
            self._weight_buf.append(sum(weights))

        # Episode done
        done = self.locals["dones"][0]
        if done:
            self._ep_rewards.append(self._current_ep_reward)
            self._ep_lengths.append(self._current_ep_len)
            self._current_ep_reward = 0.0
            self._current_ep_len    = 0

        # Log every N steps
        if self.n_calls % self.log_interval == 0 and self.n_calls > 0:
            self._log()

        return True

    def _log(self):
        step = self.num_timesteps

        mean_reward = np.mean(self._ep_rewards) if self._ep_rewards else 0.0
        mean_eplen  = np.mean(self._ep_lengths) if self._ep_lengths else 0.0
        mean_weight = np.mean(self._weight_buf) if self._weight_buf else 0.0

        G = "\033[92m"; Y = "\033[93m"; R = "\033[91m"
        C = "\033[96m"; B = "\033[1m";  X = "\033[0m"

        print(f"\n{B}{C}  ── Step {step:>7,} ──────────────────────────────{X}")
        print(f"  Reward (roll-20):  {B}{mean_reward:+.3f}{X}")
        print(f"  Episode length:    {mean_eplen:.1f}")
        print(f"  Fusion weight:     {mean_weight:.3f}")

        print(f"  {'Modality':<12} {'Conf':>6}  {'OOB%':>6}")
        print(f"  {'─'*12} {'─'*6}  {'─'*6}")

        for mod in ["vision", "audio", "proximity"]:
            conf_list = self._conf_buf[mod]
            oob_list  = self._oob_buf[mod]
            mean_conf = np.mean(conf_list) if conf_list else 0.0
            oob_pct   = np.mean(oob_list) * 100 if oob_list else 0.0

            conf_col = G if mean_conf > 0.65 else Y if mean_conf > 0.4 else R
            oob_col  = R if oob_pct > 20 else Y if oob_pct > 5 else G

            print(f"  {mod:<12} {conf_col}{mean_conf:.3f}{X}   {oob_col}{oob_pct:5.1f}%{X}")

        # Clear buffers
        for mod in self._conf_buf:
            self._conf_buf[mod].clear()
            self._oob_buf[mod].clear()
        self._weight_buf.clear()
        print()

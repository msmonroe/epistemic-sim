"""
scripts/visualise.py
---------------------
Loads results/experiment_data.npz and produces comparison plots:

  1. Reward curves per experiment
  2. Vision confidence over time (baseline vs vision_degraded)
  3. Per-modality OOB rate comparison (bar chart)
  4. Fusion weight collapse under multi_fault
  5. Distance-to-target over episode steps (all experiments)

Saves PNG files to results/.
Requires: matplotlib
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path


# ── Palette ───────────────────────────────────────────────────────────────────
COLORS = {
    "baseline":        "#00e5ff",
    "vision_degraded": "#ff6b35",
    "audio_degraded":  "#a8ff3e",
    "multi_fault":     "#d966ff",
    "ablation":        "#888888",
}
FAULT_STEP = 150
RESULTS_DIR = Path("results")


# ── Load data ─────────────────────────────────────────────────────────────────

def load(path: str = "results/experiment_data.npz") -> dict[str, dict]:
    raw = np.load(path, allow_pickle=False)
    out: dict[str, dict] = {}
    for key, arr in raw.items():
        exp, metric = key.split("__", 1)
        if exp not in out:
            out[exp] = {}
        out[exp][metric] = arr
    return out


# ── Shared style ──────────────────────────────────────────────────────────────

def apply_dark_style(ax, title: str, xlabel: str, ylabel: str):
    ax.set_facecolor("#0a1520")
    ax.set_title(title, color="#9cd", fontsize=11, pad=8,
                 fontfamily="monospace")
    ax.set_xlabel(xlabel, color="#4a8090", fontsize=9)
    ax.set_ylabel(ylabel, color="#4a8090", fontsize=9)
    ax.tick_params(colors="#4a8090", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#1a3a4a")
    ax.grid(True, color="#1a3a4a", linewidth=0.5, linestyle="--", alpha=0.7)


def fault_vline(ax, label: bool = True):
    ax.axvline(FAULT_STEP, color="#ff4488", linewidth=1.2,
               linestyle="--", alpha=0.7,
               label="fault injection" if label else None)


# ── Plot 1: Episode reward curves ─────────────────────────────────────────────

def plot_rewards(data: dict, out_dir: Path):
    fig, ax = plt.subplots(figsize=(9, 4))
    fig.patch.set_facecolor("#050a0f")
    apply_dark_style(ax, "Episode Reward — All Experiments",
                     "Episode", "Total Reward")

    for exp, d in data.items():
        if "ep_reward" not in d:
            continue
        rewards = d["ep_reward"]
        eps     = np.arange(len(rewards))
        # Smooth
        if len(rewards) >= 5:
            kernel  = np.ones(3) / 3
            smoothed = np.convolve(rewards, kernel, mode="same")
        else:
            smoothed = rewards
        ax.plot(eps, smoothed, color=COLORS.get(exp, "#aaa"),
                linewidth=2, label=exp, alpha=0.9)
        ax.scatter(eps, rewards, color=COLORS.get(exp, "#aaa"),
                   s=12, alpha=0.3)

    ax.legend(facecolor="#0a1520", edgecolor="#1a3a4a",
              labelcolor="#8ab", fontsize=8, loc="lower right")
    fig.tight_layout()
    path = out_dir / "01_reward_curves.png"
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved {path}")


# ── Plot 2: Vision confidence — baseline vs vision_degraded ──────────────────

def plot_vision_confidence(data: dict, out_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    fig.patch.set_facecolor("#050a0f")
    fig.suptitle("Vision Confidence — Healthy vs Degraded",
                 color="#9cd", fontsize=12, fontfamily="monospace")

    for ax, exp in zip(axes, ["baseline", "vision_degraded"]):
        apply_dark_style(ax, exp, "Step (within episode)", "Vision Confidence")
        if exp not in data or "vision_conf" not in data[exp]:
            continue
        conf  = data[exp]["vision_conf"]
        steps = data[exp]["step"]

        # Average across episodes for each step index
        max_step = int(steps.max()) + 1
        mean_conf = np.zeros(max_step)
        count     = np.zeros(max_step)
        for s, c in zip(steps.astype(int), conf):
            if s < max_step:
                mean_conf[s] += c
                count[s]     += 1
        mask = count > 0
        xs   = np.where(mask)[0]
        ys   = mean_conf[mask] / count[mask]

        ax.plot(xs, ys, color=COLORS[exp], linewidth=1.5, alpha=0.9)
        ax.fill_between(xs, ys, alpha=0.15, color=COLORS[exp])
        ax.axhline(0.60, color="#00e5ff", linewidth=0.8,
                   linestyle=":", alpha=0.5, label="CONFIDENT threshold")

        if exp == "vision_degraded":
            fault_vline(ax, label=True)
            ax.legend(facecolor="#0a1520", edgecolor="#1a3a4a",
                      labelcolor="#8ab", fontsize=7)

        ax.set_ylim(0, 1.05)

    fig.tight_layout()
    path = out_dir / "02_vision_confidence.png"
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved {path}")


# ── Plot 3: OOB rate comparison bar chart ─────────────────────────────────────

def plot_oob_rates(data: dict, out_dir: Path):
    experiments = list(data.keys())
    modalities  = ["vision", "audio", "proximity"]
    x           = np.arange(len(experiments))
    width       = 0.25
    mod_colors  = {"vision": "#00e5ff", "audio": "#a8ff3e", "proximity": "#ff6b35"}

    fig, ax = plt.subplots(figsize=(9, 4))
    fig.patch.set_facecolor("#050a0f")
    apply_dark_style(ax, "OOB Rate per Modality (full eval period)",
                     "Experiment", "OOB Rate (%)")

    for i, mod in enumerate(modalities):
        rates = []
        for exp in experiments:
            key = f"{mod}_oob"
            if exp in data and key in data[exp]:
                rates.append(data[exp][key].mean() * 100)
            else:
                rates.append(0.0)
        bars = ax.bar(x + i * width, rates, width,
                      label=mod, color=mod_colors[mod], alpha=0.85,
                      edgecolor="#050a0f", linewidth=0.5)
        for bar, rate in zip(bars, rates):
            if rate > 2:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.5,
                        f"{rate:.0f}%",
                        ha="center", va="bottom",
                        fontsize=7, color=mod_colors[mod],
                        fontfamily="monospace")

    ax.set_xticks(x + width)
    ax.set_xticklabels(experiments, fontsize=8, fontfamily="monospace")
    ax.legend(facecolor="#0a1520", edgecolor="#1a3a4a",
              labelcolor="#8ab", fontsize=8)

    fig.tight_layout()
    path = out_dir / "03_oob_rates.png"
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved {path}")


# ── Plot 4: Fusion weight collapse under multi_fault ─────────────────────────

def plot_fusion_weight(data: dict, out_dir: Path):
    fig, ax = plt.subplots(figsize=(9, 4))
    fig.patch.set_facecolor("#050a0f")
    apply_dark_style(ax, "Fusion Weight Over Episode Steps",
                     "Step (within episode)", "Total Fusion Weight")

    for exp in ["baseline", "multi_fault", "vision_degraded"]:
        if exp not in data or "weight" not in data[exp]:
            continue
        weight = data[exp]["weight"]
        steps  = data[exp]["step"]
        max_s  = int(steps.max()) + 1
        mw     = np.zeros(max_s)
        cnt    = np.zeros(max_s)
        for s, w in zip(steps.astype(int), weight):
            if s < max_s:
                mw[s] += w; cnt[s] += 1
        mask = cnt > 0
        xs   = np.where(mask)[0]
        ys   = mw[mask] / cnt[mask]
        ax.plot(xs, ys, color=COLORS[exp], linewidth=2,
                label=exp, alpha=0.9)

    fault_vline(ax, label=True)
    ax.axhline(0.0, color="#ff4488", linewidth=0.5, linestyle=":", alpha=0.4)
    ax.legend(facecolor="#0a1520", edgecolor="#1a3a4a",
              labelcolor="#8ab", fontsize=8)
    ax.set_ylim(-0.05, None)

    fig.tight_layout()
    path = out_dir / "04_fusion_weight.png"
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved {path}")


# ── Plot 5: Distance to target ────────────────────────────────────────────────

def plot_distance(data: dict, out_dir: Path):
    fig, ax = plt.subplots(figsize=(9, 4))
    fig.patch.set_facecolor("#050a0f")
    apply_dark_style(ax, "Distance-to-Target Over Episode",
                     "Step (within episode)", "Distance (m)")

    for exp, d in data.items():
        if "dist" not in d:
            continue
        dist  = d["dist"]
        steps = d["step"]
        max_s = int(steps.max()) + 1
        md    = np.zeros(max_s)
        cnt   = np.zeros(max_s)
        for s, dv in zip(steps.astype(int), dist):
            if s < max_s:
                md[s] += dv; cnt[s] += 1
        mask = cnt > 0
        xs   = np.where(mask)[0]
        ys   = md[mask] / cnt[mask]
        ax.plot(xs, ys, color=COLORS.get(exp, "#aaa"),
                linewidth=2, label=exp, alpha=0.9)

    fault_vline(ax)
    ax.legend(facecolor="#0a1520", edgecolor="#1a3a4a",
              labelcolor="#8ab", fontsize=8)
    fig.tight_layout()
    path = out_dir / "05_distance_to_target.png"
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(npz_path: str = "results/experiment_data.npz"):
    if not Path(npz_path).exists():
        print(f"  No results found at {npz_path}")
        print("  Run: python scripts/run_experiments.py first")
        return

    print(f"\n  Loading {npz_path}...")
    data = load(npz_path)
    print(f"  Experiments: {list(data.keys())}")

    RESULTS_DIR.mkdir(exist_ok=True)

    plot_rewards(data, RESULTS_DIR)
    plot_vision_confidence(data, RESULTS_DIR)
    plot_oob_rates(data, RESULTS_DIR)
    plot_fusion_weight(data, RESULTS_DIR)
    plot_distance(data, RESULTS_DIR)

    print(f"\n  All charts saved to results/\n")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="results/experiment_data.npz")
    args = p.parse_args()
    main(args.data)

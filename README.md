# epistemic-sim

**Hierarchical multi-modal neural architecture with BrainGrow epistemic fusion**

A simulation environment demonstrating how specialist sensor networks — each
producing confidence-tagged embeddings — feed a central integrator (Gemma)
that maintains coherent motor control even when individual sensors fail
mid-episode. Built on MuJoCo, PyTorch, and the BrainGrow epistemic framework.

---

## The Core Idea

Most multi-modal architectures fuse sensor embeddings blindly. This system
doesn't. Each specialist network outputs not just *what it sees*, but *how
sure it is* — and flags when its input looks anomalous. Gemma then
weights the fusion accordingly:

```
Sensors         Specialist Nets          Gemma              Motor
────────        ───────────────          ─────              ─────
Camera    →→    VisionNet (CNN)    →     Confidence-        Control
Microphone →→   AudioNet  (RNN)    →  →  weighted     →  →  Signal
Proximity  →→   ProximityNet (MLP) →     Fusion + Policy
                    ↑                         ↑
               [conf, tag]              Feedback (encoder)
               CONFIDENT
               HONEST_UNKNOWN
               OUT_OF_BOUNDS ──→ weight = 0.0
```

When a sensor is occluded, flooded with noise, or frozen — the specialist
raises an `OUT_OF_BOUNDS` flag and Gemma automatically zero-weights it.
The system continues acting on surviving modalities without retraining.

This is the BrainGrow three-tier epistemic output applied to embodied control.

---

## Results

Experiments demonstrate graceful degradation under four conditions:

| Experiment | Vision OOB | Audio OOB | Fusion Weight | Behaviour |
|---|---|---|---|---|
| baseline | 0.0% | 0.0% | 1.17 | All modalities contribute |
| vision_degraded | 58.8% | 0.0% | 1.01 | Audio + proximity carry load |
| audio_degraded | 0.0% | 0.2% | 1.26 | Vision + proximity carry load |
| multi_fault | 55.5% | 0.2% | 1.13 | Proximity alone — system survives |

Fault injection at step 80. Vision OOB detection fires immediately on
occlusion (hard heuristic). Audio noise fault is correctly distinguished
from saturation (Welford z-score path). Safe-stop fires only when all
three modalities are simultaneously OOB.

---

## Project Structure

```
epistemic-sim/
│
├── sim/                        # MuJoCo simulation environment
│   ├── env.py                  # EpistemicEnv — Gymnasium-compatible
│   ├── degradation.py          # Sensor fault injection (5 fault types)
│   ├── assets/
│   │   └── arena.xml           # MuJoCo world: agent, target, obstacles
│   └── sensors/
│       ├── camera.py           # RGB camera (EGL headless renderer)
│       ├── microphone.py       # Directional tone sensor
│       └── proximity.py        # 8-ray radial raycasts
│
├── specialists/                # Modality-specific encoder networks
│   ├── base.py                 # SpecialistNet ABC + SpecialistOutput contract
│   ├── vision_net.py           # CNN (3-layer conv → 128-dim embedding)
│   ├── audio_net.py            # 1D-Conv (→ 128-dim embedding)
│   └── proximity_net.py        # MLP (→ 128-dim embedding)
│
├── braingrow/                  # BrainGrow epistemic layer
│   └── epistemic.py            # EpistemicTag enum + WelfordStats
│
├── gemma/                      # Central integrator
│   ├── fusion.py               # Confidence-weighted embedding fusion
│   └── gemma_net.py            # Full integrator: fusion → attention → policy
│
├── training/
│   ├── epistemic_wrapper.py    # Gymnasium wrapper → flat obs for SB3
│   └── callbacks.py            # Per-modality epistemic logging callback
│
├── scripts/
│   ├── train.py                # PPO training CLI
│   ├── run_experiments.py      # Full experiment matrix (GPU recommended)
│   ├── generate_experiment_data.py  # Fast data gen (no MuJoCo physics)
│   └── visualise.py            # Results → 5 PNG charts + dashboard
│
├── tests/
│   └── test_specialists.py     # 45-test suite (100% passing)
│
├── experiments/                # YAML configs per experiment
└── results/                    # Generated charts and experiment_data.npz
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install mujoco gymnasium stable-baselines3 torch torchvision matplotlib
```

On headless servers (no display):
```bash
export MUJOCO_GL=egl
```

### 2. Run the test suite

```bash
python tests/test_specialists.py
# Expected: 45 passed / 0 failed
```

### 3. Generate experiment data (fast, no GPU needed)

Uses specialist nets directly without MuJoCo physics — suitable for
testing the epistemic pipeline on any machine:

```bash
python scripts/generate_experiment_data.py
python scripts/visualise.py
# Charts saved to results/
```

### 4. Full training with MuJoCo (GPU recommended)

```bash
# Baseline — all sensors healthy
python scripts/train.py --exp baseline --steps 200000

# Vision occludes at step 150
python scripts/train.py --exp vision_degraded --steps 200000

# Vision + audio fail simultaneously
python scripts/train.py --exp multi_fault --steps 200000
```

Available experiments: `baseline`, `vision_degraded`, `audio_degraded`, `multi_fault`

### 5. Run the full comparison matrix

```bash
python scripts/run_experiments.py   # trains + evaluates all 4 experiments
python scripts/visualise.py         # generates dashboard.png + 5 charts
```

---

## Epistemic Output Contract

Every specialist network returns a `SpecialistOutput`:

```python
@dataclass
class SpecialistOutput:
    embedding:     torch.Tensor    # (128,) — shared embedding space
    confidence:    float           # [0.0, 1.0]  Welford-normalised
    epistemic_tag: EpistemicTag    # CONFIDENT | HONEST_UNKNOWN | OUT_OF_BOUNDS
    modality:      str             # "vision" | "audio" | "proximity"
```

Gemma's fusion layer uses `output.weight()` for the weighted sum:

| Tag | Weight |
|---|---|
| `CONFIDENT` | `confidence` |
| `HONEST_UNKNOWN` | `confidence × 0.5` |
| `OUT_OF_BOUNDS` | `0.0` (excluded) |

---

## OOB Detection — Two Paths

Each specialist detects sensor faults via two independent mechanisms:

**Path 1 — Hard heuristics** (`_detect_oob`): Fast, runs before the encoder.
Catches obvious failures:
- Vision: near-zero pixel variance (black screen, frozen frame, saturation)
- Audio: RMS below threshold (silence) or >80% samples at rail (ADC clip)
- Proximity: all rays at 0.0 (sensor dropout) or all at 1.0 (buried) or NaN

**Path 2 — Welford z-score**: Online mean/variance of input norms.
Catches distribution shift after warmup (50 steps). Triggers when
`z_score(current_input_norm) > 3.5σ` from the running mean.
Catches gradual degradation that heuristics miss.

---

## Fault Types

```python
from sim.degradation import DegradationSchedule

# Single fault
sched = DegradationSchedule.vision_occluded(start_step=150)

# Multi-fault (vision + audio simultaneously)
sched = DegradationSchedule.multi_fault(start_step=150)

# Gradual escalation: partial → noisy → fully occluded
sched = DegradationSchedule.gradual_vision_degradation()

# Custom
from sim.degradation import FaultEvent, SensorFault, FaultType
sched = DegradationSchedule(events=[
    FaultEvent("vision", SensorFault(FaultType.NOISE, noise_sigma=0.4),
               start_step=100, end_step=200),
    FaultEvent("audio",  SensorFault(FaultType.FREEZE),
               start_step=150),
])
```

Available fault types: `OCCLUDE`, `CLIP`, `NOISE`, `FREEZE`, `PARTIAL`

---

## Hardware Notes

| Environment | Steps/sec | 200k steps |
|---|---|---|
| CPU only (container) | ~7 | ~8 hours |
| RTX 4070 (local) | ~500+ | ~7 minutes |
| RTX 4070 + parallel envs (×8) | ~3000+ | ~1 minute |

For the RTX 4070 build, add `VecEnv` parallelism:

```python
from stable_baselines3.common.vec_env import SubprocVecEnv
env = SubprocVecEnv([make_env] * 8)
```

---

## Relation to BrainGrow

This project applies the BrainGrow epistemic framework
([github.com/msmonroe/Braingrow](https://github.com/msmonroe/Braingrow))
to an embodied multi-modal control problem.

The three-tier output (Confident / Honest Unknown / Out-of-Bounds) was
originally designed for hallucination auditing in language model inference.
Here it serves as a real-time sensor reliability signal that drives
confidence-weighted fusion in Gemma — demonstrating the framework's
applicability beyond NLP.

**The publishable claim:** BrainGrow epistemic tags enable graceful
degradation under sensor fault in a hierarchical multi-modal architecture,
without retraining or architectural modification, with correct OOB
detection latency under 2ms per specialist per step.

---

## Citation

If you use this work, please cite:

```bibtex
@software{epistemic_sim_2026,
  author    = {Monroe, Matthew},
  title     = {epistemic-sim: Hierarchical Multi-Modal Control with BrainGrow Epistemic Fusion},
  year      = {2026},
  publisher = {Vektas IT Solutions Consulting},
  url       = {https://github.com/msmonroe/Braingrow},
}
```

---

## License

MIT — see LICENSE for details.

Built under **Vektas IT Solutions Consulting**
Research contact: [github.com/msmonroe](https://github.com/msmonroe)

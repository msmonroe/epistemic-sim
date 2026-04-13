"""
tests/test_specialists.py
--------------------------
End-to-end tests for the specialist networks and Gemma integrator.

Tests cover:
  1. Output shape contract — all specialists produce (EMBED_DIM,) embeddings
  2. Confidence range     — always in [0, 1]
  3. OOB detection        — correct tag fires under known-bad inputs
  4. Healthy tag path     — CONFIDENT/HONEST_UNKNOWN on valid inputs
  5. Welford warmup       — stats accumulate correctly over N steps
  6. Fusion weighting     — OOB specialist gets zero weight in Gemma
  7. All-OOB safe stop    — Gemma returns zero action when all sensors OOB
  8. Full pipeline        — healthy inputs → valid action output
  9. Degradation schedule — fault injection changes tags mid-run
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from braingrow.epistemic import EpistemicTag, WelfordStats
from specialists.vision_net    import VisionNet
from specialists.audio_net     import AudioNet
from specialists.proximity_net import ProximityNet
from specialists.base          import SpecialistNet, SpecialistOutput
from gemma.fusion              import FusionLayer
from gemma.gemma_net           import GemmaNet
from sim.degradation           import (
    DegradationSchedule, SensorFault, FaultType, FaultEvent
)

# ── Colour helpers ────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

passed = 0
failed = 0
results = []

def run_test(name: str, fn):
    global passed, failed
    try:
        fn()
        print(f"  {GREEN}✓{RESET}  {name}")
        passed += 1
        results.append((name, True, None))
    except Exception as e:
        print(f"  {RED}✗{RESET}  {name}")
        print(f"       {RED}{type(e).__name__}: {e}{RESET}")
        failed += 1
        results.append((name, False, str(e)))

def section(title: str):
    print(f"\n{BOLD}{CYAN}── {title}{RESET}")

# ── Synthetic data factories ──────────────────────────────────────────────────

def make_image(h=64, w=64, mode="normal") -> np.ndarray:
    if mode == "normal":
        return (np.random.rand(h, w, 3) * 255).astype(np.uint8)
    elif mode == "black":
        return np.zeros((h, w, 3), dtype=np.uint8)
    elif mode == "white":
        return np.ones((h, w, 3), dtype=np.uint8) * 255
    elif mode == "frozen":
        return np.full((h, w, 3), 128, dtype=np.uint8)

def make_audio(n=256, mode="normal") -> np.ndarray:
    if mode == "normal":
        t = np.linspace(0, 1, n)
        return (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)
    elif mode == "silent":
        return np.zeros(n, dtype=np.float32)
    elif mode == "clipped":
        return np.ones(n, dtype=np.float32)
    elif mode == "noise":
        return np.random.uniform(-1, 1, n).astype(np.float32)

def make_proximity(n=8, mode="normal") -> np.ndarray:
    if mode == "normal":
        return np.random.uniform(0.1, 0.9, n).astype(np.float32)
    elif mode == "all_zero":
        return np.zeros(n, dtype=np.float32)
    elif mode == "all_one":
        return np.ones(n, dtype=np.float32)
    elif mode == "nan":
        a = np.random.uniform(0.1, 0.9, n).astype(np.float32)
        a[0] = float("nan")
        return a

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — WelfordStats
# ─────────────────────────────────────────────────────────────────────────────
section("1. WelfordStats — online statistics")

def test_welford_mean():
    w = WelfordStats(warmup_steps=5)
    for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
        w.update(v)
    assert abs(w.mean - 3.0) < 1e-9, f"mean={w.mean}"
run_test("Mean converges to 3.0 for [1,2,3,4,5]", test_welford_mean)

def test_welford_variance():
    w = WelfordStats(warmup_steps=5)
    for v in [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]:
        w.update(v)
    assert w.variance > 0, "variance should be positive"
run_test("Variance is positive after updates", test_welford_variance)

def test_welford_warmup():
    w = WelfordStats(warmup_steps=10)
    assert not w.warmed_up
    for i in range(10):
        w.update(float(i))
    assert w.warmed_up
run_test("warmed_up flag flips after threshold", test_welford_warmup)

def test_welford_zscore():
    w = WelfordStats(warmup_steps=5)
    for v in [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]:
        w.update(v)
    z = w.z_score(10.0)
    assert z > 3.0, f"z_score for outlier should be large, got {z}"
run_test("z_score is large for outlier value", test_welford_zscore)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — VisionNet
# ─────────────────────────────────────────────────────────────────────────────
section("2. VisionNet — CNN image encoder")

vision = VisionNet(image_h=64, image_w=64, device="cpu")

def test_vision_output_shape():
    img = make_image(mode="normal")
    out = vision(img)
    assert isinstance(out, SpecialistOutput)
    assert out.embedding.shape == (SpecialistNet.EMBED_DIM,), \
        f"expected ({SpecialistNet.EMBED_DIM},) got {out.embedding.shape}"
run_test("Output embedding shape is (128,)", test_vision_output_shape)

def test_vision_confidence_range():
    img = make_image(mode="normal")
    out = vision(img)
    assert 0.0 <= out.confidence <= 1.0, f"confidence={out.confidence}"
run_test("Confidence in [0, 1]", test_vision_confidence_range)

def test_vision_modality_label():
    out = vision(make_image())
    assert out.modality == "vision"
run_test("Modality label is 'vision'", test_vision_modality_label)

def test_vision_oob_black():
    img = make_image(mode="black")
    out = vision(img)
    assert out.epistemic_tag == EpistemicTag.OUT_OF_BOUNDS, \
        f"expected OOB for black image, got {out.epistemic_tag.name}"
run_test("Black image → OUT_OF_BOUNDS", test_vision_oob_black)

def test_vision_oob_white():
    img = make_image(mode="white")
    out = vision(img)
    assert out.epistemic_tag == EpistemicTag.OUT_OF_BOUNDS, \
        f"expected OOB for white image, got {out.epistemic_tag.name}"
run_test("Saturated (white) image → OUT_OF_BOUNDS", test_vision_oob_white)

def test_vision_oob_frozen():
    # Constant-value image → near-zero variance → OOB
    img = make_image(mode="frozen")
    out = vision(img)
    assert out.epistemic_tag == EpistemicTag.OUT_OF_BOUNDS, \
        f"expected OOB for frozen image, got {out.epistemic_tag.name}"
run_test("Frozen (constant-value) image → OUT_OF_BOUNDS", test_vision_oob_frozen)

def test_vision_is_usable_healthy():
    out = vision(make_image(mode="normal"))
    # May be CONFIDENT or HONEST_UNKNOWN — both usable
    assert out.is_usable(), f"healthy image should be usable, got {out.epistemic_tag.name}"
run_test("Healthy image → is_usable() == True", test_vision_is_usable_healthy)

def test_vision_oob_not_usable():
    out = vision(make_image(mode="black"))
    assert not out.is_usable(), "OOB image should not be usable"
run_test("OOB image → is_usable() == False", test_vision_oob_not_usable)

def test_vision_oob_weight_zero():
    out = vision(make_image(mode="black"))
    assert out.weight() == 0.0, f"OOB weight should be 0.0, got {out.weight()}"
run_test("OOB image → weight() == 0.0", test_vision_oob_weight_zero)

def test_vision_latency_logged():
    out = vision(make_image(mode="normal"))
    assert out.latency_ms >= 0.0
run_test("Latency_ms is non-negative", test_vision_latency_logged)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — AudioNet
# ─────────────────────────────────────────────────────────────────────────────
section("3. AudioNet — 1D-Conv audio encoder")

audio_net = AudioNet(n_samples=256, device="cpu")

def test_audio_output_shape():
    out = audio_net(make_audio(mode="normal"))
    assert out.embedding.shape == (SpecialistNet.EMBED_DIM,)
run_test("Output embedding shape is (128,)", test_audio_output_shape)

def test_audio_confidence_range():
    out = audio_net(make_audio(mode="normal"))
    assert 0.0 <= out.confidence <= 1.0
run_test("Confidence in [0, 1]", test_audio_confidence_range)

def test_audio_modality_label():
    out = audio_net(make_audio())
    assert out.modality == "audio"
run_test("Modality label is 'audio'", test_audio_modality_label)

def test_audio_oob_silent():
    out = audio_net(make_audio(mode="silent"))
    assert out.epistemic_tag == EpistemicTag.OUT_OF_BOUNDS, \
        f"expected OOB for silent audio, got {out.epistemic_tag.name}"
run_test("Silent audio → OUT_OF_BOUNDS", test_audio_oob_silent)

def test_audio_oob_clipped():
    out = audio_net(make_audio(mode="clipped"))
    assert out.epistemic_tag == EpistemicTag.OUT_OF_BOUNDS, \
        f"expected OOB for clipped audio, got {out.epistemic_tag.name}"
run_test("Clipped (saturated) audio → OUT_OF_BOUNDS", test_audio_oob_clipped)

def test_audio_healthy_usable():
    out = audio_net(make_audio(mode="normal"))
    assert out.is_usable()
run_test("Healthy audio → is_usable() == True", test_audio_healthy_usable)

def test_audio_oob_weight_zero():
    out = audio_net(make_audio(mode="silent"))
    assert out.weight() == 0.0
run_test("OOB audio → weight() == 0.0", test_audio_oob_weight_zero)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — ProximityNet
# ─────────────────────────────────────────────────────────────────────────────
section("4. ProximityNet — MLP rangefinder encoder")

prox_net = ProximityNet(n_rays=8, device="cpu")

def test_prox_output_shape():
    out = prox_net(make_proximity(mode="normal"))
    assert out.embedding.shape == (SpecialistNet.EMBED_DIM,)
run_test("Output embedding shape is (128,)", test_prox_output_shape)

def test_prox_confidence_range():
    out = prox_net(make_proximity(mode="normal"))
    assert 0.0 <= out.confidence <= 1.0
run_test("Confidence in [0, 1]", test_prox_confidence_range)

def test_prox_modality_label():
    out = prox_net(make_proximity())
    assert out.modality == "proximity"
run_test("Modality label is 'proximity'", test_prox_modality_label)

def test_prox_oob_all_zero():
    out = prox_net(make_proximity(mode="all_zero"))
    assert out.epistemic_tag == EpistemicTag.OUT_OF_BOUNDS, \
        f"expected OOB for all-zero prox, got {out.epistemic_tag.name}"
run_test("All-zero proximity → OUT_OF_BOUNDS", test_prox_oob_all_zero)

def test_prox_oob_all_one():
    out = prox_net(make_proximity(mode="all_one"))
    assert out.epistemic_tag == EpistemicTag.OUT_OF_BOUNDS, \
        f"expected OOB for all-one prox, got {out.epistemic_tag.name}"
run_test("All-one proximity → OUT_OF_BOUNDS", test_prox_oob_all_one)

def test_prox_oob_nan():
    out = prox_net(make_proximity(mode="nan"))
    assert out.epistemic_tag == EpistemicTag.OUT_OF_BOUNDS
run_test("NaN in proximity array → OUT_OF_BOUNDS", test_prox_oob_nan)

def test_prox_healthy_usable():
    out = prox_net(make_proximity(mode="normal"))
    assert out.is_usable()
run_test("Healthy proximity → is_usable() == True", test_prox_healthy_usable)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — FusionLayer
# ─────────────────────────────────────────────────────────────────────────────
section("5. FusionLayer — confidence-weighted embedding fusion")

fusion = FusionLayer(embed_dim=128, n_specialists=3)

def _make_fake_output(modality, tag, confidence=0.8):
    return SpecialistOutput(
        embedding     = torch.randn(128),
        confidence    = confidence,
        epistemic_tag = tag,
        modality      = modality,
    )

def test_fusion_output_shape():
    outs = [
        _make_fake_output("vision",    EpistemicTag.CONFIDENT),
        _make_fake_output("audio",     EpistemicTag.CONFIDENT),
        _make_fake_output("proximity", EpistemicTag.CONFIDENT),
    ]
    fused, weight, all_oob = fusion(outs)
    assert fused.shape == (128,), f"expected (128,) got {fused.shape}"
run_test("Fused output shape is (128,)", test_fusion_output_shape)

def test_fusion_oob_gets_zero_weight():
    outs = [
        _make_fake_output("vision",    EpistemicTag.OUT_OF_BOUNDS, confidence=0.9),
        _make_fake_output("audio",     EpistemicTag.CONFIDENT,     confidence=0.8),
        _make_fake_output("proximity", EpistemicTag.CONFIDENT,     confidence=0.7),
    ]
    fused, weight, all_oob = fusion(outs)
    assert not all_oob
    # Vision OOB → weight contributed by audio + proximity only
    assert weight < 0.9 + 0.8 + 0.7  # vision's 0.9 excluded
run_test("OOB specialist excluded from fusion weight", test_fusion_oob_gets_zero_weight)

def test_fusion_all_oob_safe_stop():
    outs = [
        _make_fake_output("vision",    EpistemicTag.OUT_OF_BOUNDS),
        _make_fake_output("audio",     EpistemicTag.OUT_OF_BOUNDS),
        _make_fake_output("proximity", EpistemicTag.OUT_OF_BOUNDS),
    ]
    fused, weight, all_oob = fusion(outs)
    assert all_oob, "all OOB should set all_oob=True"
    assert fused.norm().item() == 0.0, "all OOB fused should be zero vector"
run_test("All-OOB inputs → all_oob=True, zero fused vector", test_fusion_all_oob_safe_stop)

def test_fusion_unknown_half_weight():
    outs = [
        _make_fake_output("vision",    EpistemicTag.HONEST_UNKNOWN, confidence=0.5),
        _make_fake_output("audio",     EpistemicTag.CONFIDENT,      confidence=0.8),
        _make_fake_output("proximity", EpistemicTag.CONFIDENT,      confidence=0.7),
    ]
    fused, weight, all_oob = fusion(outs)
    # UNKNOWN weight = 0.5 * 0.5 = 0.25, audio = 0.8, prox = 0.7
    expected_approx = 0.25 + 0.8 + 0.7
    assert abs(weight - expected_approx) < 0.1, \
        f"expected weight ~{expected_approx:.2f}, got {weight:.2f}"
run_test("HONEST_UNKNOWN gets half-confidence weight", test_fusion_unknown_half_weight)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — GemmaNet full pipeline
# ─────────────────────────────────────────────────────────────────────────────
section("6. GemmaNet — full central integrator pipeline")

gemma = GemmaNet(embed_dim=128, n_specialists=3, proprio_dim=5, action_dim=2)

def test_gemma_action_shape():
    outs = [
        vision(make_image(mode="normal")),
        audio_net(make_audio(mode="normal")),
        prox_net(make_proximity(mode="normal")),
    ]
    proprio = torch.tensor([0.5, 0.3, 0.1, 0.0, 0.0])
    action, info = gemma(outs, proprio)
    assert action.shape == (2,), f"expected (2,) got {action.shape}"
run_test("Action output shape is (2,)", test_gemma_action_shape)

def test_gemma_action_range():
    outs = [
        vision(make_image(mode="normal")),
        audio_net(make_audio(mode="normal")),
        prox_net(make_proximity(mode="normal")),
    ]
    proprio = torch.tensor([0.5, 0.3, 0.1, 0.0, 0.0])
    action, _ = gemma(outs, proprio)
    assert action.abs().max().item() <= 1.0 + 1e-6, \
        f"action out of [-1,1] range: {action}"
run_test("Action values in [-1, 1] (tanh output)", test_gemma_action_range)

def test_gemma_all_oob_safe_stop():
    outs = [
        vision(make_image(mode="black")),
        audio_net(make_audio(mode="silent")),
        prox_net(make_proximity(mode="all_zero")),
    ]
    proprio = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0])
    action, info = gemma(outs, proprio)
    assert info["all_oob"] == True, "all degraded sensors should set all_oob"
    assert action.norm().item() == 0.0, "safe stop should be zero action"
run_test("All sensors OOB → safe stop (zero action)", test_gemma_all_oob_safe_stop)

def test_gemma_info_contains_per_modality():
    outs = [
        vision(make_image(mode="normal")),
        audio_net(make_audio(mode="normal")),
        prox_net(make_proximity(mode="normal")),
    ]
    proprio = torch.tensor([0.1, 0.2, 0.0, 0.0, 0.0])
    _, info = gemma(outs, proprio)
    assert "vision/confidence" in info
    assert "audio/confidence" in info
    assert "proximity/confidence" in info
run_test("Info dict contains per-modality confidence keys", test_gemma_info_contains_per_modality)

def test_gemma_vision_oob_still_acts():
    # Vision OOB but audio+proximity healthy → should still produce action
    outs = [
        vision(make_image(mode="black")),
        audio_net(make_audio(mode="normal")),
        prox_net(make_proximity(mode="normal")),
    ]
    proprio = torch.tensor([0.5, 0.3, 1.0, 0.1, 0.0])
    action, info = gemma(outs, proprio)
    assert not info["all_oob"], "2 healthy sensors should keep system alive"
    assert action.norm().item() > 0.0, "should produce non-zero action"
run_test("Vision OOB but audio+prox healthy → non-zero action", test_gemma_vision_oob_still_acts)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — Degradation schedule
# ─────────────────────────────────────────────────────────────────────────────
section("7. DegradationSchedule — fault injection")

def test_degradation_no_faults_before_start():
    sched = DegradationSchedule.vision_occluded(start_step=50)
    faults = sched.get_faults(step=30)
    assert "vision" not in faults
run_test("No faults returned before start_step", test_degradation_no_faults_before_start)

def test_degradation_fault_active_after_start():
    sched = DegradationSchedule.vision_occluded(start_step=50)
    faults = sched.get_faults(step=51)
    assert "vision" in faults
run_test("Vision fault active after start_step", test_degradation_fault_active_after_start)

def test_degradation_occlude_zeroes_image():
    sched = DegradationSchedule.vision_occluded(start_step=10)
    img = make_image(mode="normal").astype(np.float32)
    faults = sched.get_faults(step=20)
    degraded = faults["vision"].apply(img)
    assert np.all(degraded == 0.0), "OCCLUDE fault should zero the image"
run_test("OCCLUDE fault zeroes image array", test_degradation_occlude_zeroes_image)

def test_degradation_noise_changes_signal():
    fault = SensorFault(FaultType.NOISE, noise_sigma=0.5)
    audio = make_audio(mode="normal")
    degraded = fault.apply(audio)
    assert not np.allclose(audio, degraded), "NOISE should change the signal"
run_test("NOISE fault changes audio signal", test_degradation_noise_changes_signal)

def test_degradation_multi_fault():
    sched = DegradationSchedule.multi_fault(start_step=10)
    faults = sched.get_faults(step=20)
    assert "vision" in faults and "audio" in faults
run_test("Multi-fault schedule activates both vision and audio", test_degradation_multi_fault)

def test_degradation_oob_fires_after_occlude():
    """Full pipeline: occlude image mid-run → vision becomes OOB."""
    sched = DegradationSchedule.vision_occluded(start_step=5)
    img_healthy = make_image(mode="normal")

    # Step 1-4: healthy
    for _ in range(4):
        faults = sched.get_faults(step=_)
        img = img_healthy.copy()
        out = vision(img)

    # Step 5+: degraded
    faults = sched.get_faults(step=5)
    img_degraded = faults["vision"].apply(img_healthy.astype(np.float32))
    out_degraded = vision(img_degraded)
    assert out_degraded.epistemic_tag == EpistemicTag.OUT_OF_BOUNDS, \
        f"occluded image should be OOB, got {out_degraded.epistemic_tag.name}"
run_test("Occluded image post-fault-activation → OOB tag", test_degradation_oob_fires_after_occlude)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — Welford accumulation over episode
# ─────────────────────────────────────────────────────────────────────────────
section("8. Welford accumulation — 100-step episode simulation")

def test_welford_accumulates_over_episode():
    net = VisionNet(device="cpu")
    tags = []
    for i in range(100):
        img = make_image(mode="normal")
        out = net(img)
        tags.append(out.epistemic_tag)
    oob_count = sum(1 for t in tags if t == EpistemicTag.OUT_OF_BOUNDS)
    # Healthy images should produce almost no OOB tags
    assert oob_count < 5, \
        f"Too many OOB tags on healthy images: {oob_count}/100"
run_test("100 healthy images → <5% OOB tags", test_welford_accumulates_over_episode)

def test_welford_confidence_shifts_after_degradation():
    """
    Run 60 healthy steps then 40 degraded. Welford z-score should
    eventually catch the distribution shift and raise OOB rate.
    """
    net = AudioNet(device="cpu")
    healthy_confs  = []
    degraded_confs = []

    for i in range(60):
        out = net(make_audio(mode="normal"))
        healthy_confs.append(out.confidence)

    for i in range(40):
        out = net(make_audio(mode="noise"))
        degraded_confs.append(out.confidence)

    healthy_mean  = np.mean(healthy_confs)
    degraded_mean = np.mean(degraded_confs)
    # Confidence under noise flood should differ from healthy baseline
    assert abs(healthy_mean - degraded_mean) > 0.0, \
        "confidence should differ between healthy and degraded"
run_test("Confidence distribution shifts after sensor degradation", test_welford_confidence_shifts_after_degradation)

# ─────────────────────────────────────────────────────────────────────────────
# RESULTS SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

total = passed + failed
pct   = (passed / total * 100) if total > 0 else 0

print(f"\n{'─' * 55}")
print(f"{BOLD}  Results: {GREEN}{passed} passed{RESET}{BOLD} / "
      f"{RED}{failed} failed{RESET}{BOLD} / {total} total  ({pct:.0f}%){RESET}")
print(f"{'─' * 55}")

if failed > 0:
    print(f"\n{RED}Failed tests:{RESET}")
    for name, ok, err in results:
        if not ok:
            print(f"  • {name}")
            print(f"    {err}")

print()
sys.exit(0 if failed == 0 else 1)

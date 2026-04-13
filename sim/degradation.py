"""
sim/degradation.py
------------------
Sensor fault injection for mid-episode degradation experiments.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
import numpy as np


class FaultType(Enum):
    OCCLUDE  = auto()
    CLIP     = auto()
    NOISE    = auto()
    FREEZE   = auto()
    PARTIAL  = auto()


@dataclass
class SensorFault:
    fault_type:   FaultType
    severity:     float = 1.0
    clip_val:     float = 0.05
    noise_sigma:  float = 0.5
    partial_frac: float = 0.5
    _frozen:      np.ndarray | None = field(default=None, repr=False)

    def apply(self, raw: np.ndarray) -> np.ndarray:
        out = raw.astype(np.float32, copy=True)
        if self.fault_type == FaultType.OCCLUDE:
            out[:] = 0.0
        elif self.fault_type == FaultType.CLIP:
            out = np.clip(out, -self.clip_val, self.clip_val)
        elif self.fault_type == FaultType.NOISE:
            noise = np.random.normal(0.0, self.noise_sigma * self.severity,
                                     size=out.shape).astype(np.float32)
            out = np.clip(out + noise, -1.0, 1.0)
        elif self.fault_type == FaultType.FREEZE:
            if self._frozen is None:
                object.__setattr__(self, "_frozen", out.copy())
            out = self._frozen.copy()
        elif self.fault_type == FaultType.PARTIAL:
            n = max(1, int(out.size * self.partial_frac * self.severity))
            idx = np.random.choice(out.size, size=n, replace=False)
            flat = out.ravel()
            flat[idx] = 0.0
            out = flat.reshape(raw.shape)
        return out.reshape(raw.shape).astype(np.float32)

    def reset_freeze(self):
        object.__setattr__(self, "_frozen", None)


@dataclass
class FaultEvent:
    modality:   str
    fault:      SensorFault
    start_step: int
    end_step:   int | None = None


@dataclass
class DegradationSchedule:
    events: list[FaultEvent] = field(default_factory=list)

    def get_faults(self, step: int) -> dict[str, SensorFault]:
        active: dict[str, SensorFault] = {}
        for event in self.events:
            if step >= event.start_step:
                if event.end_step is None or step < event.end_step:
                    active[event.modality] = event.fault
        return active

    def reset(self):
        for event in self.events:
            event.fault.reset_freeze()

    @classmethod
    def vision_occluded(cls, start_step: int = 50) -> "DegradationSchedule":
        return cls(events=[FaultEvent("vision", SensorFault(FaultType.OCCLUDE), start_step)])

    @classmethod
    def audio_noisy(cls, start_step: int = 50, sigma: float = 0.6) -> "DegradationSchedule":
        return cls(events=[FaultEvent("audio", SensorFault(FaultType.NOISE, noise_sigma=sigma), start_step)])

    @classmethod
    def multi_fault(cls, start_step: int = 50) -> "DegradationSchedule":
        return cls(events=[
            FaultEvent("vision", SensorFault(FaultType.OCCLUDE), start_step),
            FaultEvent("audio",  SensorFault(FaultType.NOISE, noise_sigma=0.7), start_step),
        ])

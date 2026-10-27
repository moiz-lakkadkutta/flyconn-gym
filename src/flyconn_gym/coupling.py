"""Level A coupling: senses -> input neurons -> brain -> DN readouts -> FlyGym descending drive.

At Level A the leg movement comes from FlyGym's hand-written hybrid CPG controller; the
connectome only sets its two-number descending drive (left, right). Every output says so.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import numpy as np
from flyconn.graph import ConnectivityMatrix

from flyconn_gym.brain import BrainVariant, RateBrain, variant_matrix
from flyconn_gym.iomap import IOMap

COUPLING_LEVEL = "A: brain -> DN command -> FlyGym hybrid turning controller"

# Readouts that share one calibration scale (spread-v2, docs/metrics.md). The two turn sets
# share a scale so that their left/right asymmetry survives normalisation.
SCALE_GROUPS: dict[str, tuple[str, ...]] = {
    "turn": ("turn_left", "turn_right"),
    "forward": ("forward",),
    "backward": ("backward",),
    "escape": ("escape",),
}
UNREACHABLE = 1e-9


def looming_features(
    fly_xy: np.ndarray,
    heading_xy: np.ndarray,
    obj_xy: np.ndarray,
    radius: float,
    approach_speed: float,
) -> np.ndarray:
    """Angular expansion rate (rad/s) of an approaching sphere, split into (left, right) eyes.

    A geometric stand-in for looming detectors: no rendering, no retina. Eye weights are
    ``0.5 +/- 0.5 sin(bearing)``, so an object straight ahead or behind drives both eyes.
    """
    rel = obj_xy - fly_xy
    d = float(np.hypot(*rel))
    expansion = 2 * radius * approach_speed / (d * d + radius * radius)
    h = heading_xy / np.linalg.norm(heading_xy)
    sin_bearing = (h[0] * rel[1] - h[1] * rel[0]) / max(d, 1e-9)  # >0: object on the left
    left = expansion * (0.5 + 0.5 * sin_bearing)
    right = expansion * (0.5 - 0.5 * sin_bearing)
    return np.array([left, right])


@dataclass(frozen=True)
class DNDecoder:
    """Map DN readout rates to FlyGym's (left, right) descending drive.

    ``tonic`` is a constant drive that does not come from the connectome (the rate model is
    silent without input); ``no_brain`` runs show what the tonic drive alone produces.
    FlyGym convention: left drive > right drive turns the fly right. Optional readouts
    ``backward`` and ``escape`` lower both drives; a walking body cannot take off, so
    escape -> slowing is a hand-written Level A choice. Absent readouts count as zero.
    """

    tonic: float = 1.0
    forward_gain: float = 0.5
    turn_gain: float = 1.0
    backward_gain: float = 1.0
    escape_gain: float = 1.0
    low: float = -1.0
    high: float = 1.5

    def __call__(self, readouts: Mapping[str, float]) -> np.ndarray:
        base = (
            self.tonic
            + self.forward_gain * readouts["forward"]
            - self.backward_gain * readouts.get("backward", 0.0)
            - self.escape_gain * readouts.get("escape", 0.0)
        )
        turn = self.turn_gain * (readouts["turn_right"] - readouts["turn_left"])
        return np.clip(np.array([base + turn, base - turn]), self.low, self.high)


def _versions() -> dict[str, str]:
    out: dict[str, str] = {}
    for pkg in ("flyconn", "flygym", "flyconn-gym", "mujoco", "numpy"):
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:  # pragma: no cover
            out[pkg] = "missing"
    return out


@dataclass
class LevelAPolicy:
    """The brain as a policy for :class:`flyconn_gym.env.LoomingWalkEnv`."""

    iomap: IOMap
    variant: BrainVariant
    seed: int
    brain: RateBrain | None
    input_names: tuple[str, ...] = ("loom_left", "loom_right")
    input_gain: float = 0.5
    decoder: DNDecoder = field(default_factory=DNDecoder)
    readout_scale: dict[str, float] | None = None
    scale_groups: dict[str, float] = field(default_factory=dict[str, float])
    last_readouts: dict[str, float] = field(default_factory=dict[str, float])
    _in_idx: dict[str, np.ndarray] = field(default_factory=dict[str, np.ndarray])
    _out_idx: dict[str, np.ndarray] = field(default_factory=dict[str, np.ndarray])

    @classmethod
    def build(
        cls,
        m: ConnectivityMatrix,
        iomap: IOMap,
        variant: BrainVariant | str,
        seed: int,
        **brain_kwargs: Any,
    ) -> LevelAPolicy:
        variant = BrainVariant(variant)
        iomap.validate(m)
        wired = variant_matrix(m, variant, seed)
        brain = RateBrain(wired, **brain_kwargs) if wired is not None else None
        policy = cls(iomap=iomap, variant=variant, seed=seed, brain=brain)
        policy._in_idx = iomap.input_index(m)
        policy._out_idx = iomap.readout_index(m)
        return policy

    def reset(self) -> None:
        if self.brain is not None:
            self.brain.reset()
        self.last_readouts = {k: 0.0 for k in self.iomap.readouts}

    def _raw_readouts(self, features: np.ndarray) -> dict[str, float]:
        if self.brain is None:
            return {k: 0.0 for k in self.iomap.readouts}
        external = np.zeros(self.brain.n)
        for name, value in zip(self.input_names, features, strict=True):
            external[self._in_idx[name]] += self.input_gain * float(value)
        rates = self.brain.step(external)
        return {k: float(rates[i].mean()) for k, i in self._out_idx.items()}

    def calibrate(self, reference: float = 1.0, steps: int = 300) -> dict[str, float]:
        """Set :attr:`readout_scale` from open-loop responses to a reference stimulus.

        Each input set in turn gets ``reference`` alone for ``steps`` brain steps (from rest).
        A group's scale is the largest steady readout of any of its sets over those stimuli;
        afterwards each readout is divided by its group's scale and clipped to [0, 1]. A scale
        below :data:`UNREACHABLE` marks the group unreachable (its readouts read 0). Returns
        the group scales and leaves the brain reset.
        """
        peak = {k: 0.0 for k in self.iomap.readouts}
        for i in range(len(self.input_names)):
            features = np.zeros(len(self.input_names))
            features[i] = reference
            self.reset()
            raw: dict[str, float] = {}
            for _ in range(steps):
                raw = self._raw_readouts(features)
            peak = {k: max(peak[k], raw.get(k, 0.0)) for k in peak}
        self.reset()
        self.scale_groups = {}
        scale: dict[str, float] = {}
        for g, names in SCALE_GROUPS.items():
            present = [k for k in names if k in peak]
            if not present:
                continue
            s = max(peak[k] for k in present)
            self.scale_groups[g] = s if s >= UNREACHABLE else 0.0
            scale.update(dict.fromkeys(present, self.scale_groups[g]))
        self.readout_scale = scale
        return dict(self.scale_groups)

    def act(self, features: np.ndarray) -> np.ndarray:
        """One brain step on sensory ``features`` (ordered as ``input_names``) -> drive (2,)."""
        raw = self._raw_readouts(features)
        if self.readout_scale is None:
            self.last_readouts = raw
        else:
            sc = self.readout_scale
            self.last_readouts = {
                k: min(max(v / sc[k], 0.0), 1.0) if sc.get(k, 0.0) > 0 else 0.0
                for k, v in raw.items()
            }
        return self.decoder(self.last_readouts)

    def provenance(self) -> dict[str, Any]:
        brain = self.brain
        return {
            "coupling_level": COUPLING_LEVEL,
            "brain_variant": self.variant.value,
            "seed": self.seed,
            "dataset": self.iomap.dataset,
            "iomap": self.iomap.label(),
            "brain_model": None
            if brain is None
            else {
                "kind": "rate",
                "tau_ms": brain.tau_ms,
                "dt_ms": brain.dt_ms,
                "gain": brain.gain,
                "matrix_provenance": brain.matrix.provenance,
            },
            "decoder": vars(self.decoder),
            "readout_scale": None
            if self.readout_scale is None
            else {"groups": self.scale_groups, "unreachable_below": UNREACHABLE},
            "input_gain": self.input_gain,
            "packages": _versions(),
        }

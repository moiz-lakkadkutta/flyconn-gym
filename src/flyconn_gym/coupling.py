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
    FlyGym convention: left drive > right drive turns the fly right.
    """

    tonic: float = 1.0
    forward_gain: float = 0.5
    turn_gain: float = 1.0
    low: float = -1.0
    high: float = 1.5

    def __call__(self, readouts: Mapping[str, float]) -> np.ndarray:
        base = self.tonic + self.forward_gain * readouts["forward"]
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

    def act(self, features: np.ndarray) -> np.ndarray:
        """One brain step on sensory ``features`` (ordered as ``input_names``) -> drive (2,)."""
        if self.brain is not None:
            external = np.zeros(self.brain.n)
            for name, value in zip(self.input_names, features, strict=True):
                external[self._in_idx[name]] += self.input_gain * float(value)
            rates = self.brain.step(external)
            self.last_readouts = {k: float(rates[i].mean()) for k, i in self._out_idx.items()}
        else:
            self.last_readouts = {k: 0.0 for k in self.iomap.readouts}
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
            "input_gain": self.input_gain,
            "packages": _versions(),
        }

"""Rate-model brain over a flyconn connectome, and the null variants it runs beside.

v0 uses a rate model because flyconn 0.4's LIF engine cannot be advanced in chunks
with persistent state (see ``docs/upstream-notes.md``). Dynamics, per neuron ``i``::

    tau * dr_i/dt = -r_i + f(gain * sum_j s_j * w_ji / in_i * r_j + I_i),  f(x) = tanh(max(x, 0))

with ``s_j`` the presynaptic sign, ``w_ji`` the synapse count and ``in_i`` the total input
synapses of ``i``. Rates are dimensionless in [0, 1]; they are not firing rates in Hz.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
import scipy.sparse as sp
from flyconn.graph import ConnectivityMatrix
from flyconn.uncertainty import degree_preserving_rewire, shuffle_signs


class BrainVariant(StrEnum):
    """Which wiring drives the body. Every run names one; nulls are on by default."""

    REAL = "real"
    REWIRED = "rewired"  # flyconn degree_preserving_rewire
    SIGN_SHUFFLED = "sign_shuffled"  # flyconn shuffle_signs
    NO_BRAIN = "no_brain"  # constant (tonic) descending drive, no connectome
    LAT_REWIRED = "lat_rewired"  # flyconn_gym.nulls.lateral_rewire (spread-v3)


# The variant sets of the registered designs (docs/metrics.md); older designs keep theirs.
V2_VARIANTS = (
    BrainVariant.REAL,
    BrainVariant.REWIRED,
    BrainVariant.SIGN_SHUFFLED,
    BrainVariant.NO_BRAIN,
)
V3_VARIANTS = (*V2_VARIANTS[:3], BrainVariant.LAT_REWIRED, BrainVariant.NO_BRAIN)


def variant_matrix(
    m: ConnectivityMatrix, variant: BrainVariant, seed: int
) -> ConnectivityMatrix | None:
    """Return the connectome for ``variant`` (``None`` for ``NO_BRAIN``)."""
    match variant:
        case BrainVariant.REAL:
            return m
        case BrainVariant.REWIRED:
            return degree_preserving_rewire(m, seed=seed)
        case BrainVariant.SIGN_SHUFFLED:
            return shuffle_signs(m, seed=seed)
        case BrainVariant.LAT_REWIRED:
            from flyconn_gym.nulls import lateral_rewire

            return lateral_rewire(m, seed=seed)
        case BrainVariant.NO_BRAIN:
            return None


@dataclass
class RateBrain:
    """Steppable rate network; state persists across :meth:`step` calls until :meth:`reset`."""

    matrix: ConnectivityMatrix
    tau_ms: float = 20.0
    dt_ms: float = 1.0
    gain: float = 2.0
    rates: np.ndarray = field(init=False)
    _w_in: sp.csr_matrix = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not 0 < self.dt_ms <= self.tau_ms:
            msg = f"need 0 < dt_ms <= tau_ms for a stable Euler step (dt_ms={self.dt_ms})"
            raise ValueError(msg)
        signed = sp.diags(self.matrix.signs.astype(float)) @ self.matrix.input_normalized()
        # Rows are presynaptic; transpose so that w_in @ r is each neuron's input.
        self._w_in = sp.csr_matrix(self.gain * signed.T)
        self.rates = np.zeros(self.matrix.n)

    @property
    def n(self) -> int:
        return self.matrix.n

    def reset(self) -> None:
        self.rates = np.zeros(self.n)

    def step(self, external: np.ndarray) -> np.ndarray:
        """Advance one ``dt_ms`` with external input ``external`` (length ``n``)."""
        drive = self._w_in @ self.rates + external
        target = np.tanh(np.maximum(drive, 0.0))
        self.rates = self.rates + (self.dt_ms / self.tau_ms) * (target - self.rates)
        return self.rates

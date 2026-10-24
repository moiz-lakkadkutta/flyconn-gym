"""Named, versioned maps from simulator senses to input neurons and from DN sets to drive."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from flyconn.graph import ConnectivityMatrix


@dataclass(frozen=True)
class IOMap:
    """Input neuron sets (keyed by sensory channel) and DN readout sets for one dataset.

    Every map is a modelling assumption: ``citation`` names its source, or says why none
    exists (toy maps).
    """

    name: str
    version: str
    dataset: str
    inputs: Mapping[str, tuple[int, ...]]
    readouts: Mapping[str, tuple[int, ...]]
    citation: str = ""

    def validate(self, m: ConnectivityMatrix) -> None:
        """Raise ``KeyError`` if any id is missing from ``m``."""
        for ids in (*self.inputs.values(), *self.readouts.values()):
            m.index_of(np.asarray(ids, dtype=np.int64))

    def input_index(self, m: ConnectivityMatrix) -> dict[str, np.ndarray]:
        return {k: m.index_of(np.asarray(v, dtype=np.int64)) for k, v in self.inputs.items()}

    def readout_index(self, m: ConnectivityMatrix) -> dict[str, np.ndarray]:
        return {k: m.index_of(np.asarray(v, dtype=np.int64)) for k, v in self.readouts.items()}

    def label(self) -> dict[str, str]:
        return {"name": self.name, "version": self.version}

"""Named, versioned maps from simulator senses to input neurons and from DN sets to drive."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
    set_citations: Mapping[str, str] = field(default_factory=dict[str, str])

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

    def save(self, path: Path, extra: Mapping[str, Any] | None = None) -> None:
        """Write the map as JSON (ids as lists), with optional ``extra`` provenance."""
        doc = {
            "name": self.name,
            "version": self.version,
            "dataset": self.dataset,
            "citation": self.citation,
            "inputs": {k: list(v) for k, v in self.inputs.items()},
            "readouts": {k: list(v) for k, v in self.readouts.items()},
            "set_citations": dict(self.set_citations),
            "extra": dict(extra or {}),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(doc, indent=1) + "\n")

    @classmethod
    def load(cls, path: Path) -> IOMap:
        doc = json.loads(Path(path).read_text())
        return cls(
            name=doc["name"],
            version=doc["version"],
            dataset=doc["dataset"],
            inputs={k: tuple(v) for k, v in doc["inputs"].items()},
            readouts={k: tuple(v) for k, v in doc["readouts"].items()},
            citation=doc.get("citation", ""),
            set_citations=doc.get("set_citations", {}),
        )

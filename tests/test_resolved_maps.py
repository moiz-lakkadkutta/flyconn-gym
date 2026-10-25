"""The committed, resolved maps in data/iomaps stay loadable and consistent (offline)."""

from __future__ import annotations

from pathlib import Path

import pytest

from flyconn_gym.iomap import IOMap
from flyconn_gym.maps import SPECS

ROOT = Path(__file__).resolve().parents[1] / "data" / "iomaps"
FILES = sorted(ROOT.glob("*/*.json"))


def test_some_maps_are_committed() -> None:
    assert FILES


@pytest.mark.parametrize("path", FILES, ids=lambda p: f"{p.parent.name}/{p.stem}")
def test_resolved_map_matches_its_spec(path: Path) -> None:
    io = IOMap.load(path)
    spec = SPECS[path.parent.name]
    assert (io.name, io.version) == (spec.name, spec.version)
    assert io.dataset == path.stem
    for name, s in spec.sets.items():
        present = name in io.inputs or name in io.readouts
        assert present or not s.required, name
    assert not set(io.inputs["loom_left"]) & set(io.inputs["loom_right"])
    assert not set(io.readouts["turn_left"]) & set(io.readouts["turn_right"])

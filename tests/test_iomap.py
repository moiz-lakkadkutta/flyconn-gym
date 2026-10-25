from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from flyconn.graph import ConnectivityMatrix

from flyconn_gym.iomap import IOMap
from flyconn_gym.toy import toy_iomap


def test_toy_iomap_sets_are_valid(toy_matrix: ConnectivityMatrix) -> None:
    io = toy_iomap(toy_matrix)
    io.validate(toy_matrix)
    assert set(io.inputs) == {"loom_left", "loom_right"}
    assert set(io.readouts) == {"forward", "turn_left", "turn_right"}
    all_ids = [i for s in (*io.inputs.values(), *io.readouts.values()) for i in s]
    assert len(all_ids) == len(set(all_ids)), "sets must be disjoint"
    meta = toy_matrix.meta.set_index("neuron_id")
    for name, ids in io.inputs.items():
        assert len(ids) > 0
        assert set(meta.loc[list(ids), "super_class"]) == {"sensory"}, name
    for ids in io.readouts.values():
        assert set(meta.loc[list(ids), "super_class"]) == {"descending"}


def test_validate_rejects_unknown_ids(toy_matrix: ConnectivityMatrix) -> None:
    io = IOMap("bad", "0", "synthetic@0", {"x": (1,)}, {"forward": (toy_matrix.neuron_ids[0],)})
    with pytest.raises(KeyError):
        io.validate(toy_matrix)


def test_index_arrays_match_matrix(toy_matrix: ConnectivityMatrix) -> None:
    io = toy_iomap(toy_matrix)
    idx = io.input_index(toy_matrix)
    for name, ids in io.inputs.items():
        assert np.array_equal(toy_matrix.neuron_ids[idx[name]], np.array(ids))


def test_json_round_trip(tmp_path: Path, toy_matrix: ConnectivityMatrix) -> None:
    io = toy_iomap(toy_matrix)
    path = tmp_path / "m.json"
    io.save(path, extra={"store": {"sha": "abc"}})
    back = IOMap.load(path)
    assert back == io
    assert json.loads(path.read_text())["extra"] == {"store": {"sha": "abc"}}

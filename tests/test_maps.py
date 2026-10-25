from __future__ import annotations

import pandas as pd
import pytest

from flyconn_gym.maps import (
    LOOMING_V1,
    MapSpec,
    NeuronSet,
    UnresolvedSetError,
    aliases_from_column,
    resolve,
)


def _neurons() -> pd.DataFrame:
    rows = [
        (1, "LPLC2", "left"),
        (2, "LPLC2", "left"),
        (3, "LPLC2", "right"),
        (4, "LC4", "left"),
        (5, "LC4", "right"),
        (6, "DNa02", "left"),
        (7, "DNa02", "right"),
        (8, "DNp09", "left"),
        (9, "DNp09", "right"),
        (10, "DNp01", "left"),
        (11, "DNp01", "right"),
        (12, "MDN", "left"),
        (13, "MDN", "right"),
        (14, "DNa02", None),
    ]
    return pd.DataFrame(rows, columns=["neuron_id", "cell_type", "side"])


SPEC = MapSpec(
    name="t",
    version="1",
    inputs={"loom_left": NeuronSet(("LPLC2", "LC4"), "left", "c1")},
    readouts={
        "turn_left": NeuronSet(("DNa02",), "left", "c2"),
        "forward": NeuronSet(("DNp09",), None, "c3"),
    },
)


def test_resolve_selects_by_type_and_side() -> None:
    io, report = resolve(SPEC, _neurons(), "toy@0")
    assert io.inputs["loom_left"] == (1, 2, 4)
    assert io.readouts["turn_left"] == (6,)
    assert io.readouts["forward"] == (8, 9)
    assert io.name == "t" and io.version == "1" and io.dataset == "toy@0"
    assert io.set_citations["turn_left"] == "c2"
    assert report.loc["loom_left", "n"] == 3
    assert report.loc["loom_left", "by_type"] == "LPLC2:2 LC4:1"


def test_neurons_without_side_never_enter_sided_sets() -> None:
    io, _ = resolve(SPEC, _neurons(), "toy@0")
    assert 14 not in io.readouts["turn_left"]


def test_missing_set_raises_with_names() -> None:
    neurons = _neurons()
    neurons = neurons[neurons["cell_type"] != "DNp09"]
    with pytest.raises(UnresolvedSetError, match="forward"):
        resolve(SPEC, neurons, "toy@0")


def test_partial_types_are_reported_not_hidden() -> None:
    neurons = _neurons()
    neurons = neurons[neurons["cell_type"] != "LC4"]
    io, report = resolve(SPEC, neurons, "toy@0")
    assert io.inputs["loom_left"] == (1, 2)
    assert report.loc["loom_left", "missing_types"] == "LC4"


def test_aliases_rename_types_for_other_vocabularies() -> None:
    neurons = _neurons().replace({"cell_type": {"DNp09": "DNp09_alt"}})
    io, report = resolve(SPEC, neurons, "toy@0", aliases={"DNp09": "DNp09_alt"})
    assert io.readouts["forward"] == (8, 9)
    assert report.loc["forward", "by_type"] == "DNp09->DNp09_alt:2"


def test_aliases_from_column_takes_the_modal_counterpart() -> None:
    ref = pd.DataFrame(
        {
            "cell_type": ["MDN", "MDN", "MDN", "DNp09"],
            "fafb_783_cell_type": ["DNx_mdn", "DNx_mdn", "other", "DNp09"],
        }
    )
    assert aliases_from_column(ref, "fafb_783_cell_type", ["MDN", "DNp09", "LC4"]) == {
        "MDN": "DNx_mdn"
    }


def test_looming_v1_spec_is_complete_and_cited() -> None:
    assert set(LOOMING_V1.inputs) == {"loom_left", "loom_right"}
    assert set(LOOMING_V1.readouts) >= {"forward", "turn_left", "turn_right", "escape"}
    for s in (*LOOMING_V1.inputs.values(), *LOOMING_V1.readouts.values()):
        assert s.citation, s


def test_optional_sets_may_be_missing_and_are_reported() -> None:
    spec = MapSpec(
        name="t",
        version="1",
        inputs=SPEC.inputs,
        readouts={**SPEC.readouts, "backward": NeuronSet(("MDN",), None, "c4", required=False)},
    )
    neurons = _neurons()
    io, report = resolve(spec, neurons[neurons["cell_type"] != "MDN"], "toy@0")
    assert "backward" not in io.readouts
    assert report.loc["backward", "n"] == 0
    assert report.loc["backward", "missing_types"] == "MDN"

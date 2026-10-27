from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from flyconn_gym.iomap import IOMap
from flyconn_gym.subgraph import khop_ids, load_subgraph

# s -> a -> b -> t is the only source-to-target path; x is a dead end off a, y only feeds t,
# z is a second input to b from outside any path.
EDGES = pd.DataFrame(
    [
        (1, 10, 5),  # s -> a
        (10, 11, 5),  # a -> b
        (11, 2, 5),  # b -> t
        (10, 20, 5),  # a -> x (dead end)
        (30, 2, 5),  # y -> t (not reachable from s)
        (40, 11, 5),  # z -> b
        (1, 2, 1),  # s -> t, below the weight threshold
    ],
    columns=["pre", "post", "weight"],
)


class FakeStore:
    ref = "fake@0"
    provenance: dict[str, Any] = {"counts": {}}  # noqa: RUF012

    def __init__(self, edges: pd.DataFrame, totals: bool) -> None:
        self._edges = edges
        ids = sorted(set(edges["pre"]) | set(edges["post"]))
        self._neurons = pd.DataFrame(
            {"neuron_id": ids, "nt_pred": "acetylcholine", "side": "left", "cell_type": "x"}
        )
        if totals:
            self._neurons["input_synapses_total"] = 100

    def neurons(self) -> pd.DataFrame:
        return self._neurons.copy()

    def edges(self, *, min_weight: int = 1, pre: Any = None, post: Any = None) -> pd.DataFrame:
        e = self._edges[self._edges["weight"] >= min_weight]
        if pre is not None:
            e = e[e["pre"].isin(list(pre))]
        if post is not None:
            e = e[e["post"].isin(list(post))]
        return e.reset_index(drop=True)


def _fetch(min_weight: int = 5) -> Any:
    return lambda **kw: FakeStore(EDGES, totals=False).edges(min_weight=min_weight, **kw)


def test_khop_keeps_only_short_source_to_target_paths() -> None:
    assert khop_ids(_fetch(), [1], [2], k=3).tolist() == [1, 2, 10, 11]


def test_khop_too_short_keeps_only_endpoints() -> None:
    # The s -> t shortcut is below threshold, so no path has <= 2 hops.
    assert khop_ids(_fetch(), [1], [2], k=2).tolist() == [1, 2]
    assert khop_ids(_fetch(min_weight=1), [1], [2], k=2).tolist() == [1, 2]


def test_khop_rejects_bad_k() -> None:
    with pytest.raises(ValueError, match="k"):
        khop_ids(_fetch(), [1], [2], k=0)


def _io() -> IOMap:
    return IOMap(
        name="t",
        version="0",
        dataset="fake@0",
        inputs={"loom_left": (1,)},
        readouts={"turn_left": (2,)},
        citation="test",
    )


@pytest.mark.parametrize("totals", [True, False])
def test_load_subgraph_normalises_by_whole_dataset_input(totals: bool) -> None:
    m = load_subgraph(FakeStore(EDGES, totals=totals), _io(), k=3, min_weight=5)  # type: ignore[arg-type]
    assert m.neuron_ids.tolist() == [1, 2, 10, 11]
    norm = m.input_normalized().toarray()
    b = int(m.index_of(np.array([11]))[0])
    a = int(m.index_of(np.array([10]))[0])
    # b gets 5 synapses from a inside the subgraph and 5 from z outside it: its in-subgraph
    # input must stay a fraction of its real total, not be renormalised to 1.
    expected = 5 / 100 if totals else 5 / 10
    assert norm[a, b] == pytest.approx(expected)
    sub = m.provenance["subgraph"]
    assert sub["k"] == 3 and sub["min_weight"] == 5
    assert sub["totals"] == ("input_synapses_total" if totals else "stored edges")


def test_readout_coverage_is_inside_input_over_total() -> None:
    from flyconn_gym.subgraph import readout_coverage

    m = load_subgraph(FakeStore(EDGES, totals=True), _io(), k=3, min_weight=5)  # type: ignore[arg-type]
    # Readout t (id 2) gets 5 synapses from b inside the subgraph; its total is 100.
    assert readout_coverage(m, _io(), sets=("turn_left",)) == pytest.approx(5 / 100)

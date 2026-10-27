"""Small real-connectome brains: the k-hop subgraph between an I/O map's inputs and readouts.

The rate model never sees the whole brain. :func:`khop_ids` keeps every neuron on some
path of at most ``k`` hops from an input neuron to a readout neuron (edges of at least
``min_weight`` synapses), plus all input and readout neurons. Edges are fetched from the
store a frontier at a time, so only the subgraph and its BFS frontiers are ever loaded.

Weights are divided by each neuron's *whole-dataset* input (``input_synapses_total`` when
the store has it, else the sum over all stored edges), so input from outside the subgraph
still counts in the denominator: the subgraph is not renormalised into a stronger network.
Everything outside the subgraph is silent: an assumption, recorded in provenance.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, Protocol, cast

import numpy as np
import pandas as pd
from flyconn.graph import ConnectivityMatrix

from flyconn_gym.iomap import IOMap

TOTAL_COLUMN = "input_synapses_total"

EdgeFetch = Callable[..., pd.DataFrame]


class EdgeStore(Protocol):
    """The parts of :class:`flyconn.data.store.Store` this module uses."""

    @property
    def ref(self) -> str: ...

    @property
    def provenance(self) -> dict[str, Any]: ...

    def neurons(self) -> pd.DataFrame: ...

    def edges(self, *, min_weight: int = 1, pre: Any = None, post: Any = None) -> pd.DataFrame: ...


def _bfs(fetch: EdgeFetch, start: Iterable[int], hops: int, forward: bool) -> dict[int, int]:
    dist = {int(i): 0 for i in start}
    frontier = set(dist)
    src, dst = ("pre", "post") if forward else ("post", "pre")
    for d in range(1, hops + 1):
        if not frontier:
            break
        e = fetch(**{src: sorted(frontier)})
        frontier = {int(i) for i in cast("pd.Series", e[dst])} - dist.keys()
        dist.update(dict.fromkeys(frontier, d))
    return dist


def khop_ids(
    fetch: EdgeFetch, sources: Iterable[int], targets: Iterable[int], k: int
) -> np.ndarray:
    """Sorted ids on a source-to-target path of at most ``k`` hops, plus all endpoints.

    ``fetch(pre=ids)`` / ``fetch(post=ids)`` must return the (already thresholded) edges
    with those pre / post neurons. An intermediate neuron is at least one hop from both
    ends, so each search goes ``k - 1`` hops.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    sources, targets = list(sources), list(targets)
    fwd = _bfs(fetch, sources, k - 1, forward=True)
    bwd = _bfs(fetch, targets, k - 1, forward=False)
    keep = {i for i, d in fwd.items() if i in bwd and d + bwd[i] <= k}
    return np.array(sorted(keep | set(sources) | set(targets)), dtype=np.int64)


def readout_coverage(
    m: ConnectivityMatrix,
    io: IOMap,
    sets: tuple[str, ...] = ("turn_left", "turn_right", "escape"),
) -> float:
    """Mean over the neurons in ``sets`` of simulated input synapses / whole-dataset input.

    Simulated = from subgraph neurons on the matrix's edges (already ``>= min_weight``);
    the denominator is the ``input_synapses_total`` used for normalisation (spread-v3).
    """
    ids = sorted({i for k in sets if k in io.readouts for i in io.readouts[k]})
    idx = m.index_of(np.array(ids))
    inside = np.asarray(m.weights[:, idx].sum(axis=0)).ravel()
    total = cast("pd.Series", m.meta[TOTAL_COLUMN]).to_numpy(dtype=float)[idx]
    return float(np.mean(inside / total))


def _io_ids(io: IOMap) -> tuple[list[int], list[int]]:
    src = sorted({i for ids in io.inputs.values() for i in ids})
    tgt = sorted({i for ids in io.readouts.values() for i in ids})
    return src, tgt


def load_subgraph(
    store: EdgeStore, io: IOMap, *, k: int = 3, min_weight: int = 5
) -> ConnectivityMatrix:
    """The k-hop input-to-readout subgraph of ``store`` as a :class:`ConnectivityMatrix`."""
    if store.ref != io.dataset:
        raise ValueError(f"I/O map is for {io.dataset}, store is {store.ref}")
    src, tgt = _io_ids(io)

    def fetch(**kw: Any) -> pd.DataFrame:
        return store.edges(min_weight=min_weight, **kw)

    ids = khop_ids(fetch, src, tgt, k)
    all_neurons = store.neurons()
    neurons = cast("pd.DataFrame", all_neurons[all_neurons["neuron_id"].isin(ids.tolist())])
    neurons = neurons.sort_values("neuron_id").reset_index(drop=True)
    if TOTAL_COLUMN in neurons.columns and bool(neurons[TOTAL_COLUMN].notna().all()):
        totals = TOTAL_COLUMN
    else:
        inbound = store.edges(post=ids.tolist())
        per_post = cast("pd.Series", inbound.groupby("post")["weight"].sum())
        nid = cast("pd.Series", neurons["neuron_id"])
        neurons[TOTAL_COLUMN] = nid.map(per_post).fillna(0).astype(float)
        totals = "stored edges"
    edges = store.edges(min_weight=min_weight, pre=ids.tolist(), post=ids.tolist())
    prov = {
        "dataset": store.ref,
        "store_provenance": store.provenance.get("counts", {}),
        "subgraph": {
            "method": "k-hop input-to-readout paths",
            "k": k,
            "min_weight": min_weight,
            "iomap": io.label(),
            "n_inputs": len(src),
            "n_readouts": len(tgt),
            "totals": totals,
            "outside": "silent (not simulated)",
        },
    }
    return ConnectivityMatrix.from_frames(neurons, edges, min_weight=min_weight, provenance=prov)

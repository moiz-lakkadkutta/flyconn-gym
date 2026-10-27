"""Null wirings not (yet) in flyconn: a laterality-preserving degree-preserving rewire.

:func:`lateral_rewire` is flyconn's ``degree_preserving_rewire`` algorithm (permute edge
destinations, then resolve duplicate edges and self-loops by swaps) run separately inside
each (pre side, post side) block. Each edge keeps its source, its weight and its side
relation, so the rewired graph keeps the real graph's lateralisation (spread-v3, check 3).
Proposed upstream in ``docs/upstream-notes.md``.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd
import scipy.sparse as sp
from flyconn.graph import ConnectivityMatrix

SIDES = ("left", "right")


def side_labels(m: ConnectivityMatrix) -> np.ndarray:
    """``left`` / ``right`` / ``other`` per neuron, from ``meta['side']``."""
    side = cast("pd.Series", m.meta["side"]).astype("string").fillna("other").to_numpy()
    return np.where(np.isin(side, SIDES), side, "other")


def _shuffle_block(
    rows: np.ndarray, cols: np.ndarray, n: int, rng: np.random.Generator, max_rounds: int
) -> np.ndarray | None:
    k = len(rows)
    cols = cols[rng.permutation(k)]
    for _ in range(max_rounds):
        key = rows.astype(np.int64) * n + cols
        order = np.argsort(key, kind="stable")
        dup = np.zeros(k, dtype=bool)
        dup[order[1:]] = key[order[1:]] == key[order[:-1]]
        bad = np.flatnonzero(dup | (rows == cols))
        if len(bad) == 0:
            return cols
        partners = rng.integers(0, k, size=len(bad))
        for i, j in zip(bad.tolist(), partners.tolist(), strict=True):
            cols[i], cols[j] = cols[j], cols[i]
    return None


def lateral_rewire(m: ConnectivityMatrix, seed: int, max_rounds: int = 1000) -> ConnectivityMatrix:
    """Degree-preserving rewire within each (pre side, post side) block of edges.

    A block that cannot be resolved in ``max_rounds`` stays as it was; the number of such
    edges is recorded as ``provenance['unshuffled_edges']``.
    """
    rng = np.random.default_rng(seed)
    side = side_labels(m)
    coo = sp.coo_matrix(m.weights)
    rows, cols, data = coo.row.copy(), coo.col.copy(), coo.data.copy()
    block = np.char.add(np.char.add(side[rows].astype(str), "|"), side[cols].astype(str))
    unshuffled = 0
    for b in np.unique(block):
        idx = np.flatnonzero(block == b)
        new = _shuffle_block(rows[idx], cols[idx], m.n, rng, max_rounds)
        if new is None:
            unshuffled += len(idx)
        else:
            cols[idx] = new
    w = sp.csr_matrix((data, (rows, cols)), shape=m.weights.shape)
    w.sum_duplicates()
    prov = dict(m.provenance, null_model="lateral_rewire", seed=seed, unshuffled_edges=unshuffled)
    return ConnectivityMatrix(m.neuron_ids, w, m.signs.copy(), m.meta, prov)

from __future__ import annotations

from collections import Counter

import numpy as np
import scipy.sparse as sp
from flyconn.graph import ConnectivityMatrix

from flyconn_gym.nulls import lateral_rewire, side_labels


def _edges(m: ConnectivityMatrix) -> list[tuple[int, int, int]]:
    coo = sp.coo_matrix(m.weights)
    return list(zip(coo.row.tolist(), coo.col.tolist(), coo.data.tolist(), strict=True))


def test_lateral_rewire_preserves_side_blocks_degrees_and_weights(
    toy_matrix: ConnectivityMatrix,
) -> None:
    m = toy_matrix
    r = lateral_rewire(m, seed=0)
    side = side_labels(m)
    before, after = _edges(m), _edges(r)
    assert after != before, "something must move"
    assert len(after) == len(before)
    assert all(i != j for i, j, _ in after), "no self-loops"

    def block_stats(edges: list[tuple[int, int, int]]) -> tuple[Counter, Counter, Counter]:
        out = Counter((i, side[j]) for i, j, _ in edges)  # out-degree into each side
        inn = Counter((j, side[i]) for i, j, _ in edges)  # in-degree from each side
        wts = Counter((i, side[j], w) for i, j, w in edges)  # weights stay with their source
        return out, inn, wts

    assert block_stats(after) == block_stats(before)
    assert np.array_equal(r.signs, m.signs)
    assert r.provenance["null_model"] == "lateral_rewire"
    assert r.provenance["unshuffled_edges"] == 0


def test_lateral_rewire_is_seeded(toy_matrix: ConnectivityMatrix) -> None:
    a = lateral_rewire(toy_matrix, seed=1)
    b = lateral_rewire(toy_matrix, seed=1)
    assert (a.weights != b.weights).nnz == 0

"""Load a connectome and its looming I/O map by name: the toy, or a registered real subgraph."""

from __future__ import annotations

from pathlib import Path

from flyconn.graph import ConnectivityMatrix

from flyconn_gym.iomap import IOMap
from flyconn_gym.toy import toy_connectome, toy_iomap

# Registered in docs/metrics.md (spread-v1 on real subgraphs; reused by spread-v2).
SUBGRAPH_K = 3
SUBGRAPH_MIN_WEIGHT = 5
REAL_DATASETS = ("malecns@1.0", "banc@888", "flywire@783")
IOMAP_DIR = Path(__file__).resolve().parents[2] / "data" / "iomaps" / "looming@1"


def load_looming(dataset: str) -> tuple[ConnectivityMatrix, IOMap]:
    """``"toy"`` or a real dataset ref (reads the cached flyconn store; needs FLYCONN_CACHE)."""
    if dataset == "toy":
        m = toy_connectome()
        return m, toy_iomap(m)
    from flyconn.data.store import Store

    from flyconn_gym.subgraph import load_subgraph

    io = IOMap.load(IOMAP_DIR / f"{dataset}.json")
    m = load_subgraph(Store.open(dataset), io, k=SUBGRAPH_K, min_weight=SUBGRAPH_MIN_WEIGHT)
    return m, io

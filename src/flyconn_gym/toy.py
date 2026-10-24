"""Toy connectome and I/O map on flyconn's synthetic generator: offline, fast, not biology.

The neuron choices are arbitrary (by ``super_class`` and ``side`` only). The toy exists to
exercise the loop and the controls; its behaviour means nothing about flies.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd
from flyconn.graph import ConnectivityMatrix
from flyconn.testing import synthetic_connectome

from flyconn_gym.iomap import IOMap

TOY_DATASET = "synthetic@0"


def toy_connectome(n_neurons: int = 120, n_edges: int = 1200, seed: int = 0) -> ConnectivityMatrix:
    neurons, edges = synthetic_connectome(n_neurons, n_edges, seed=seed)
    return ConnectivityMatrix.from_frames(
        neurons, edges, provenance={"dataset": TOY_DATASET, "generator_seed": seed}
    )


def _ids(meta: pd.DataFrame, super_class: str, side: str | None = None) -> list[int]:
    keep = meta["super_class"] == super_class
    if side is not None:
        keep &= meta["side"] == side
    return sorted(int(i) for i in cast("pd.Series", meta.loc[keep, "neuron_id"]))


def toy_iomap(m: ConnectivityMatrix) -> IOMap:
    """Sensory neurons by side receive looming; descending neurons are split into three sets."""
    meta = m.meta
    dn = np.array(_ids(meta, "descending"))
    forward = dn[::3].tolist()
    rest = set(dn.tolist()) - set(forward)
    return IOMap(
        name="toy-looming",
        version="0",
        dataset=TOY_DATASET,
        inputs={
            "loom_left": tuple(_ids(meta, "sensory", "left")),
            "loom_right": tuple(_ids(meta, "sensory", "right")),
        },
        readouts={
            "forward": tuple(forward),
            "turn_left": tuple(i for i in _ids(meta, "descending", "left") if i in rest),
            "turn_right": tuple(i for i in _ids(meta, "descending", "right") if i in rest),
        },
        citation="none: toy map on flyconn.testing.synthetic_connectome, arbitrary by design",
    )

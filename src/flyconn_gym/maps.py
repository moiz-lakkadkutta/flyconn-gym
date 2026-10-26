"""Named, versioned, cited I/O map specs, resolved per dataset into :class:`IOMap` id lists.

A spec names cell types and a side; :func:`resolve` turns it into neuron ids for one
dataset and a report of what matched. ``side`` is the dataset's own ``side`` column (soma or
hemisphere side as each dataset defines it). For optic-lobe projection neurons it is the
optic lobe, so it is the ipsilateral visual field. Neurons without a side never enter a
sided set.

Every set is a modelling assumption. Citations were checked against the papers (by DOI) on
2026-10-26; where a paper supports a set only in part, the citation says so.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Literal, cast

import pandas as pd

from flyconn_gym.iomap import IOMap

Side = Literal["left", "right"]


class UnresolvedSetError(KeyError):
    """A spec set matched no neurons in the dataset."""


@dataclass(frozen=True)
class NeuronSet:
    cell_types: tuple[str, ...]
    side: Side | None
    citation: str
    required: bool = True


@dataclass(frozen=True)
class MapSpec:
    name: str
    version: str
    inputs: Mapping[str, NeuronSet]
    readouts: Mapping[str, NeuronSet]
    notes: str = ""
    sets: dict[str, NeuronSet] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "sets", {**self.inputs, **self.readouts})


_LOOM_CITE = (
    "LPLC2 is looming-selective: Klapoetke et al. 2017 Nature, doi:10.1038/nature24626. "
    "LC4 carries looming expansion velocity to the giant fibre: von Reyn et al. 2017 Neuron, "
    "doi:10.1016/j.neuron.2017.05.036. Both synapse directly onto the giant fibre: Ache et al. "
    "2019 Curr Biol, doi:10.1016/j.cub.2019.01.079."
)
_TURN_CITE = (
    "DNa02 is recruited on the side of the upcoming turn and unilateral activation biases "
    "steering ipsilaterally: Rayshubskiy et al. 2025 eLife, doi:10.7554/eLife.102230; "
    "unilateral DNa02 stimulation gives an ipsilateral turn: Yang et al. 2024 Cell, "
    "doi:10.1016/j.cell.2024.08.033."
)

LOOMING_V1 = MapSpec(
    name="looming",
    version="1",
    inputs={
        "loom_left": NeuronSet(("LPLC2", "LC4"), "left", _LOOM_CITE),
        "loom_right": NeuronSet(("LPLC2", "LC4"), "right", _LOOM_CITE),
    },
    readouts={
        "forward": NeuronSet(
            ("DNp09",),
            None,
            "DNp09 (P9) initiates forward walking, with an ipsilateral turning component (partial "
            "support: BPN is the straight-walking pathway): Bidaye et al. 2020 Neuron, "
            "doi:10.1016/j.neuron.2020.07.032.",
        ),
        "backward": NeuronSet(
            ("MDN",),
            None,
            "MDN (moonwalker) activation is sufficient for backward walking: Bidaye et al. 2014 "
            "Science, doi:10.1126/science.1249964.",
            required=False,
        ),
        "turn_left": NeuronSet(("DNa02",), "left", _TURN_CITE),
        "turn_right": NeuronSet(("DNa02",), "right", _TURN_CITE),
        "escape": NeuronSet(
            ("DNp01",),
            None,
            "DNp01 is the giant fibre: Namiki et al. 2018 eLife, doi:10.7554/eLife.34272; it "
            "drives the looming escape take-off: von Reyn et al. 2017 Neuron, "
            "doi:10.1016/j.neuron.2017.05.036. FlyGym's walking body cannot take off; at Level A "
            "the decoder maps it to slowing (a modelling choice, not from the papers).",
        ),
    },
    notes="v1: one stimulus family (looming), walking-body readouts. Sides are dataset `side`.",
)

SPECS: dict[str, MapSpec] = {f"{LOOMING_V1.name}@{LOOMING_V1.version}": LOOMING_V1}


def aliases_from_column(
    ref_neurons: pd.DataFrame, column: str, types: Iterable[str]
) -> dict[str, str]:
    """Map each type in ``types`` to its most common counterpart in ``column`` of a reference.

    For example ``column="fafb_783_cell_type"`` in MaleCNS gives the FlyWire v783 names.
    Types that map to themselves or have no counterpart are left out.
    """
    out: dict[str, str] = {}
    for t in types:
        vals = cast("pd.Series", ref_neurons.loc[ref_neurons["cell_type"] == t, column]).dropna()
        if len(vals) == 0:
            continue
        modal = str(vals.value_counts().index[0])
        if modal != t:
            out[t] = modal
    return out


def resolve(
    spec: MapSpec,
    neurons: pd.DataFrame,
    dataset: str,
    *,
    aliases: Mapping[str, str] | None = None,
) -> tuple[IOMap, pd.DataFrame]:
    """Resolve ``spec`` against a harmonized ``neurons`` table.

    Returns the :class:`IOMap` and a report indexed by set name, with columns ``n``,
    ``by_type`` and ``missing_types``. Raises :class:`UnresolvedSetError` naming every set
    that matched nothing, unless the set is optional (then it is reported and left out).
    """
    aliases = dict(aliases or {})
    ct = neurons["cell_type"].astype("string")
    side = neurons["side"].astype("string")
    ids: dict[str, tuple[int, ...]] = {}
    rows: list[dict[str, object]] = []
    for name, s in spec.sets.items():
        found: list[int] = []
        parts: list[str] = []
        missing: list[str] = []
        for t in s.cell_types:
            local = aliases.get(t, t)
            keep = (ct == local).fillna(False)
            if s.side is not None:
                keep &= (side == s.side).fillna(False)
            sel = sorted(int(i) for i in cast("pd.Series", neurons.loc[keep, "neuron_id"]))
            label = t if local == t else f"{t}->{local}"
            if sel:
                parts.append(f"{label}:{len(sel)}")
                found.extend(sel)
            else:
                missing.append(label)
        ids[name] = tuple(sorted(found))
        rows.append(
            {
                "set": name,
                "n": len(found),
                "by_type": " ".join(parts),
                "missing_types": " ".join(missing),
            }
        )
    report = pd.DataFrame(rows).set_index("set")
    empty = [k for k, v in ids.items() if not v and spec.sets[k].required]
    if empty:
        msg = f"{spec.name}@{spec.version} on {dataset}: no neurons for sets {empty}"
        raise UnresolvedSetError(msg)
    io = IOMap(
        name=spec.name,
        version=spec.version,
        dataset=dataset,
        inputs={k: ids[k] for k in spec.inputs if ids[k]},
        readouts={k: ids[k] for k in spec.readouts if ids[k]},
        citation=spec.notes,
        set_citations={k: s.citation for k, s in spec.sets.items()},
    )
    return io, report

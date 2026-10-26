"""Sanity checks on the Level A coupling that need no brain.

:func:`dn_direction` clamps DN readouts (bypassing the brain), drives the body through the
same :class:`~flyconn_gym.coupling.DNDecoder` the policies use, and measures the heading
change against the plain tonic drive on the same seed. Positive heading change = a turn to
the fly's left (counter-clockwise seen from above). Papers report that unilateral DNa02
activity turns the fly toward the active side (Yang et al. 2024 Cell; Rayshubskiy et al.
2025 eLife), so ``turn_left`` clamped high should give a positive change.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from flyconn_gym.coupling import DNDecoder

QUIET: dict[str, float] = {"forward": 0.0, "turn_left": 0.0, "turn_right": 0.0}


def _heading_change_deg(drive: np.ndarray, seed: int, steps: int) -> float:
    from flyconn_gym.env import LoomingWalkEnv  # MuJoCo is only needed here

    env = LoomingWalkEnv(max_steps=steps, loom_side="none")
    _, info = env.reset(seed=seed)
    h0 = float(np.arctan2(info["heading"][1], info["heading"][0]))
    terminated = truncated = False
    while not (terminated or truncated):
        _, _, terminated, truncated, info = env.step(drive)
    env.close()
    h1 = float(np.arctan2(info["heading"][1], info["heading"][0]))
    return float(np.degrees(np.angle(np.exp(1j * (h1 - h0)))))


def dn_direction(
    conditions: Mapping[str, Mapping[str, float]],
    seeds: range | list[int],
    steps: int = 300,
    decoder: DNDecoder | None = None,
) -> pd.DataFrame:
    """Heading change per condition and seed, minus the quiet (tonic-only) run on that seed.

    ``conditions`` maps a name to clamped readouts (missing readouts count as zero).
    """
    dec = decoder or DNDecoder()
    rows: list[dict[str, object]] = []
    for seed in seeds:
        base = _heading_change_deg(dec(QUIET), seed, steps)
        for name, clamp in conditions.items():
            drive = dec({**QUIET, **clamp})
            dh = _heading_change_deg(drive, seed, steps)
            rows.append(
                {
                    "condition": name,
                    "seed": seed,
                    "drive_left": float(drive[0]),
                    "drive_right": float(drive[1]),
                    "dheading_deg": dh,
                    "vs_quiet_deg": dh - base,
                }
            )
    return pd.DataFrame(rows)

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from flyconn_gym.metrics import bootstrap_ci, spread_v1, turn_away_deg


def test_turn_away_sign_convention() -> None:
    # Loom on the left: turning right (negative heading change) is away.
    assert turn_away_deg(-20.0, "left") == 20.0
    assert turn_away_deg(-20.0, "right") == -20.0


def test_bootstrap_ci_brackets_the_mean_and_is_seeded() -> None:
    x = np.arange(10, dtype=float)
    lo, hi = bootstrap_ci(x)
    assert lo < x.mean() < hi
    assert (lo, hi) == bootstrap_ci(x)


def _episodes(real_extra: float) -> pd.DataFrame:
    rows = []
    for seed in range(10):
        drift = 10.0 + seed  # same controller drift for every variant
        for side, sigma in (("left", 1), ("right", -1)):
            for variant, turn in (
                ("no_brain", 0.0),
                ("rewired", 2.0),
                ("sign_shuffled", 2.0),
                ("real", 2.0 + real_extra),
            ):
                # heading = drift + turning away (away = -sigma direction)
                rows.append((variant, seed, side, drift - sigma * turn))
    return pd.DataFrame(rows, columns=["variant", "seed", "side", "dheading_deg"])


def test_spread_v1_removes_drift_and_finds_planted_effect() -> None:
    res = spread_v1(_episodes(real_extra=5.0))
    brain = res["brain_effect"].set_index("variant")
    assert brain.loc["rewired", "mean"] == pytest.approx(2.0)
    assert brain.loc["real", "mean"] == pytest.approx(7.0)
    wiring = res["wiring_effect"].set_index("null")
    assert wiring.loc["rewired", "mean"] == pytest.approx(5.0)
    assert wiring.loc["rewired", "verdict"] == "real > null"


def test_spread_v1_reports_no_evidence_without_effect() -> None:
    res = spread_v1(_episodes(real_extra=0.0))
    wiring = res["wiring_effect"].set_index("null")
    assert set(wiring["verdict"]) == {"no evidence"}

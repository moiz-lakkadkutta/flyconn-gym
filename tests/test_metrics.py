from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from flyconn_gym.metrics import bootstrap_ci, spread_v1, spread_v2, turn_away_deg


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


def _episodes_v2(real_extra: float, noise: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    rows = []
    for seed in range(10):
        for side, sigma in (("left", 1), ("right", -1)):
            # Replay heading carries drift scaled by a side-dependent slowing: v1's confound.
            replay = (10.0 + seed) * (0.5 if side == "left" else 1.0)
            for variant, turn in (
                ("no_brain", 0.0),
                ("rewired", 1.0),
                ("sign_shuffled", 1.0),
                ("real", 1.0 + real_extra),
            ):
                t = 0.0 if variant == "no_brain" else turn + noise * rng.standard_normal()
                rows.append((variant, seed, side, replay - sigma * t, replay))
    cols = ["variant", "seed", "side", "dheading_deg", "dheading_replay_deg"]
    return pd.DataFrame(rows, columns=cols)


def test_spread_v2_subtracts_the_replay_and_uses_bonferroni() -> None:
    res = spread_v2(_episodes_v2(real_extra=3.0, noise=0.5), n_comparisons=6)
    brain = res["brain_effect"].set_index("variant")
    assert brain.loc["rewired", "mean"] == pytest.approx(1.0, abs=0.3)
    wiring = res["wiring_effect"].set_index("null")
    assert wiring.loc["rewired", "verdict"] == "real > null"
    # The Bonferroni interval is wider than the 95 % one.
    w = wiring.loc["rewired"]
    assert w["bonf_lo"] < w["ci_lo"] < w["mean"] < w["ci_hi"] < w["bonf_hi"]
    assert res["replay_check"]["max_abs_no_brain_deg"] == 0.0


def test_spread_v2_verdict_uses_the_bonferroni_interval() -> None:
    df = _episodes_v2(real_extra=0.0)
    res = spread_v2(df, n_comparisons=6)
    assert set(res["wiring_effect"]["verdict"]) == {"no evidence"}


def test_spread_v2_flags_an_inexact_replay() -> None:
    df = _episodes_v2(real_extra=0.0)
    df.loc[df["variant"] == "no_brain", "dheading_deg"] += 0.01
    res = spread_v2(df, n_comparisons=6)
    assert not res["replay_check"]["valid"]

from __future__ import annotations

import pytest

from flyconn_gym.checks import dn_direction

pytestmark = pytest.mark.physics


def test_dna02_turns_the_body_toward_its_own_side() -> None:
    df = dn_direction(
        {"turn_left": {"turn_left": 0.5}, "turn_right": {"turn_right": 0.5}},
        seeds=[0, 1],
        steps=300,
    )
    by = df.groupby("condition")["vs_quiet_deg"]
    assert (by.min()["turn_left"]) > 5.0, df
    assert (by.max()["turn_right"]) < -5.0, df

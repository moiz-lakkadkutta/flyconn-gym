from __future__ import annotations

import importlib.util
import os

import pytest
from flyconn.graph import ConnectivityMatrix

from flyconn_gym.toy import toy_connectome


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip ``physics`` tests when MuJoCo/FlyGym is unusable or FLYCONN_GYM_SKIP_PHYSICS=1."""
    reason = None
    if os.environ.get("FLYCONN_GYM_SKIP_PHYSICS") == "1":
        reason = "FLYCONN_GYM_SKIP_PHYSICS=1"
    elif importlib.util.find_spec("mujoco") is None or importlib.util.find_spec("flygym") is None:
        reason = "mujoco/flygym not importable"
    if reason is None:
        return
    for item in items:
        if "physics" in item.keywords:
            item.add_marker(pytest.mark.skip(reason=reason))


@pytest.fixture(scope="session")
def toy_matrix() -> ConnectivityMatrix:
    return toy_connectome()

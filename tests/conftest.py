from __future__ import annotations

import pytest
from flyconn.graph import ConnectivityMatrix

from flyconn_gym.toy import toy_connectome


@pytest.fixture(scope="session")
def toy_matrix() -> ConnectivityMatrix:
    return toy_connectome()

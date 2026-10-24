from __future__ import annotations

import numpy as np
import pytest
from flyconn.graph import ConnectivityMatrix
from hypothesis import given, settings
from hypothesis import strategies as st

from flyconn_gym.brain import BrainVariant, RateBrain, variant_matrix


def test_brain_at_rest_stays_silent(toy_matrix: ConnectivityMatrix) -> None:
    brain = RateBrain(toy_matrix)
    for _ in range(50):
        r = brain.step(np.zeros(toy_matrix.n))
    assert np.all(r == 0)


def test_driven_neuron_rises_towards_saturation(toy_matrix: ConnectivityMatrix) -> None:
    brain = RateBrain(toy_matrix, tau_ms=10.0, dt_ms=1.0)
    drive = np.zeros(toy_matrix.n)
    drive[0] = 5.0
    rates = [brain.step(drive)[0] for _ in range(100)]
    assert rates[0] < rates[10] < rates[-1] <= 1.0
    assert rates[-1] > 0.9


@settings(max_examples=25, deadline=None)
@given(scale=st.floats(0.0, 50.0), seed=st.integers(0, 1000))
def test_rates_stay_bounded(toy_matrix: ConnectivityMatrix, scale: float, seed: int) -> None:
    brain = RateBrain(toy_matrix)
    drive = np.random.default_rng(seed).random(toy_matrix.n) * scale
    for _ in range(30):
        r = brain.step(drive)
    assert np.all((r >= 0) & (r <= 1))


def test_reset_clears_state(toy_matrix: ConnectivityMatrix) -> None:
    brain = RateBrain(toy_matrix)
    brain.step(np.ones(toy_matrix.n))
    brain.reset()
    assert np.all(brain.rates == 0)


def test_dt_must_be_below_tau(toy_matrix: ConnectivityMatrix) -> None:
    with pytest.raises(ValueError, match="dt_ms"):
        RateBrain(toy_matrix, tau_ms=1.0, dt_ms=2.0)


def test_variant_matrices_follow_flyconn_nulls(toy_matrix: ConnectivityMatrix) -> None:
    real = variant_matrix(toy_matrix, BrainVariant.REAL, seed=0)
    rewired = variant_matrix(toy_matrix, BrainVariant.REWIRED, seed=0)
    shuffled = variant_matrix(toy_matrix, BrainVariant.SIGN_SHUFFLED, seed=0)
    assert real is toy_matrix
    assert rewired is not None and shuffled is not None
    out_deg = np.diff(toy_matrix.weights.indptr)
    assert np.array_equal(np.diff(rewired.weights.indptr), out_deg)
    assert (rewired.weights != toy_matrix.weights).nnz > 0
    assert np.array_equal(np.sort(shuffled.signs), np.sort(toy_matrix.signs))
    assert not np.array_equal(shuffled.signs, toy_matrix.signs)
    assert rewired.provenance["null_model"] == "degree_preserving_rewire"
    assert variant_matrix(toy_matrix, BrainVariant.NO_BRAIN, seed=0) is None

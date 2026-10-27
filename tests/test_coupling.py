from __future__ import annotations

import numpy as np
from flyconn.graph import ConnectivityMatrix

from flyconn_gym.brain import BrainVariant
from flyconn_gym.coupling import COUPLING_LEVEL, DNDecoder, LevelAPolicy, looming_features
from flyconn_gym.toy import toy_iomap


def test_looming_ahead_is_symmetric_and_grows() -> None:
    heading = np.array([1.0, 0.0])
    far = looming_features(np.zeros(2), heading, np.array([10.0, 0.0]), 1.0, approach_speed=20.0)
    near = looming_features(np.zeros(2), heading, np.array([3.0, 0.0]), 1.0, approach_speed=20.0)
    assert np.isclose(far[0], far[1])
    assert near[0] > far[0] > 0


def test_looming_from_left_drives_left_eye() -> None:
    heading = np.array([1.0, 0.0])
    left, right = looming_features(
        np.zeros(2), heading, np.array([0.0, 5.0]), 1.0, approach_speed=20.0
    )
    assert left > 0 and right == 0


def test_decoder_tonic_and_turn_convention() -> None:
    dec = DNDecoder(tonic=1.0)
    assert np.allclose(dec({"forward": 0.0, "turn_left": 0.0, "turn_right": 0.0}), [1.0, 1.0])
    # FlyGym hybrid turning: left drive > right drive turns the fly right.
    left, right = dec({"forward": 0.0, "turn_left": 0.0, "turn_right": 0.8})
    assert left > right


def _dn_trace(m: ConnectivityMatrix, variant: BrainVariant) -> np.ndarray:
    policy = LevelAPolicy.build(m, toy_iomap(m), variant, seed=0)
    policy.reset()
    trace = []
    for t in range(200):
        loom = np.array([min(t / 50, 4.0), 0.0])  # a growing stimulus on the left only
        policy.act(loom)
        trace.append([policy.last_readouts[k] for k in ("forward", "turn_left", "turn_right")])
    return np.array(trace)


def test_variants_produce_different_dn_outputs(toy_matrix: ConnectivityMatrix) -> None:
    traces = {v: _dn_trace(toy_matrix, v) for v in BrainVariant}
    real = traces[BrainVariant.REAL]
    assert real.max() > 0, "the stimulus must reach the DN readouts in the toy graph"
    for v in (BrainVariant.REWIRED, BrainVariant.SIGN_SHUFFLED):
        assert not np.allclose(real, traces[v]), v
    assert np.all(traces[BrainVariant.NO_BRAIN] == 0)


def test_no_brain_gives_constant_tonic_drive(toy_matrix: ConnectivityMatrix) -> None:
    policy = LevelAPolicy.build(toy_matrix, toy_iomap(toy_matrix), BrainVariant.NO_BRAIN, seed=0)
    policy.reset()
    drives = {tuple(policy.act(np.array([x, 0.0]))) for x in (0.0, 1.0, 5.0)}
    assert len(drives) == 1


def test_provenance_labels_level_a(toy_matrix: ConnectivityMatrix) -> None:
    policy = LevelAPolicy.build(toy_matrix, toy_iomap(toy_matrix), BrainVariant.REWIRED, seed=3)
    prov = policy.provenance()
    assert COUPLING_LEVEL.startswith("A")
    assert prov["coupling_level"] == COUPLING_LEVEL
    assert prov["brain_variant"] == "rewired"
    assert prov["seed"] == 3
    assert prov["dataset"] == "synthetic@0"
    assert prov["iomap"] == {"name": "toy-looming", "version": "0"}
    assert {"flyconn", "flygym", "flyconn-gym"} <= set(prov["packages"])


def test_decoder_backward_and_escape_slow_the_fly() -> None:
    dec = DNDecoder(tonic=1.0)
    still = {"forward": 0.0, "turn_left": 0.0, "turn_right": 0.0}
    base = dec(still)
    assert np.all(dec({**still, "backward": 0.5}) < base)
    assert np.all(dec({**still, "escape": 0.5}) < base)
    # Sets a map leaves out (e.g. no MDN in flywire@783) count as zero.
    assert np.allclose(dec({**still, "backward": 0.0, "escape": 0.0}), base)


def test_calibrated_readouts_are_normalised_with_a_shared_turn_scale(
    toy_matrix: ConnectivityMatrix,
) -> None:
    policy = LevelAPolicy.build(toy_matrix, toy_iomap(toy_matrix), BrainVariant.REAL, seed=0)
    groups = policy.calibrate(reference=1.0, steps=300)
    assert policy.brain is not None and not policy.brain.rates.any(), "calibration resets"
    scale = policy.readout_scale
    assert scale is not None and scale["turn_left"] == scale["turn_right"] == groups["turn"]
    policy.reset()
    for _ in range(300):
        policy.act(np.array([1.0, 0.0]))
    vals = np.array(list(policy.last_readouts.values()))
    assert np.all((vals >= 0) & (vals <= 1))
    # At the reference stimulus, the strongest set in each reachable group reads ~1.
    assert max(policy.last_readouts["turn_left"], policy.last_readouts["turn_right"]) <= 1.0


def test_unreachable_group_reads_zero(toy_matrix: ConnectivityMatrix) -> None:
    policy = LevelAPolicy.build(toy_matrix, toy_iomap(toy_matrix), BrainVariant.REAL, seed=0)
    policy.calibrate()
    assert policy.readout_scale is not None
    policy.readout_scale = {**policy.readout_scale, "forward": 0.0}
    policy.reset()
    for _ in range(50):
        policy.act(np.array([2.0, 2.0]))
    assert policy.last_readouts["forward"] == 0.0


def test_no_brain_calibrates_to_unreachable(toy_matrix: ConnectivityMatrix) -> None:
    policy = LevelAPolicy.build(toy_matrix, toy_iomap(toy_matrix), BrainVariant.NO_BRAIN, seed=0)
    assert set(policy.calibrate().values()) == {0.0}
    policy.reset()
    assert np.allclose(policy.act(np.array([1.0, 1.0])), [1.0, 1.0])

from __future__ import annotations

import numpy as np
import pytest

from flyconn_gym.coupling import COUPLING_LEVEL

pytestmark = pytest.mark.physics


def test_env_reset_step_contract() -> None:
    from flyconn_gym.env import LoomingWalkEnv

    env = LoomingWalkEnv(max_steps=20)
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    assert info["coupling_level"] == COUPLING_LEVEL
    for _ in range(20):
        obs, reward, terminated, truncated, info = env.step(np.array([1.0, 1.0]))
        assert env.observation_space.contains(obs)
        assert reward == 0.0
        if terminated or truncated:
            break
    assert truncated
    env.close()


def test_env_walks_forward_under_symmetric_drive() -> None:
    from flyconn_gym.env import LoomingWalkEnv

    env = LoomingWalkEnv(max_steps=200, loom_side="none")
    env.reset(seed=0)
    for _ in range(200):
        _, _, terminated, truncated, info = env.step(np.array([1.0, 1.0]))
        if terminated or truncated:
            break
    assert info["fly_xy"][0] > 1.0  # mm forward in 0.2 s
    env.close()


def test_replay_of_an_episode_drive_is_exact() -> None:
    from flyconn_gym.brain import BrainVariant
    from flyconn_gym.coupling import LevelAPolicy
    from flyconn_gym.env import LoomingWalkEnv
    from flyconn_gym.rollout import replay_drive, run_episode
    from flyconn_gym.toy import toy_connectome, toy_iomap

    m = toy_connectome()
    env = LoomingWalkEnv(max_steps=150)
    ep = run_episode(env, LevelAPolicy.build(m, toy_iomap(m), BrainVariant.REAL, seed=0), seed=3)
    again = replay_drive(env, ep["drive"], seed=3)
    assert again["final_heading_deg"] == pytest.approx(ep["summary"]["final_heading_deg"], abs=1e-9)
    env.close()

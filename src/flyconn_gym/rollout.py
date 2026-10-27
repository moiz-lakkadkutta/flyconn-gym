"""Run one Level A episode and record trajectory, DN readouts and provenance."""

from __future__ import annotations

from typing import Any

import numpy as np

from flyconn_gym.coupling import LevelAPolicy
from flyconn_gym.env import LoomingWalkEnv


def run_episode(env: LoomingWalkEnv, policy: LevelAPolicy, seed: int = 0) -> dict[str, Any]:
    obs, info = env.reset(seed=seed)
    policy.reset()
    h0 = float(np.arctan2(info["heading"][1], info["heading"][0]))
    xy, drive, dn, dist = [info["fly_xy"]], [], [], [info["loom_distance"]]
    dn_raw: list[list[float]] = []
    terminated = truncated = False
    while not (terminated or truncated):
        action = policy.act(obs[:2])
        obs, _, terminated, truncated, info = env.step(action)
        xy.append(info["fly_xy"])
        drive.append(action)
        dn.append([policy.last_readouts[k] for k in sorted(policy.last_readouts)])
        dn_raw.append([policy.last_raw_readouts[k] for k in sorted(policy.last_readouts)])
        dist.append(info["loom_distance"])
    h1 = float(np.arctan2(info["heading"][1], info["heading"][0]))
    return {
        "provenance": {**policy.provenance(), "episode_seed": seed},
        "xy": np.array(xy),
        "drive": np.array(drive),
        "dn_names": sorted(policy.last_readouts),
        "dn": np.array(dn),
        "dn_raw": np.array(dn_raw),
        "loom_distance": np.array(dist),
        "summary": {
            "steps": len(drive),
            "duration_s": info["time_s"],
            "hit": info.get("hit", False),
            "flipped": info.get("flipped", False),
            "min_loom_distance_mm": float(np.min(dist)),
            "final_heading_deg": float(np.degrees(h1 - h0)),
            "path_length_mm": float(np.sum(np.linalg.norm(np.diff(np.array(xy), axis=0), axis=1))),
        },
    }


def replay_drive(env: LoomingWalkEnv, drive: np.ndarray, seed: int) -> dict[str, Any]:
    """Re-run ``drive`` (shape ``(steps, 2)``) open-loop on ``seed``; same summary keys.

    Physics is deterministic for a seed, so replaying an episode's own drive reproduces it.
    The replay stops early if the episode terminates (a hit or a flip).
    """
    _, info = env.reset(seed=seed)
    h0 = float(np.arctan2(info["heading"][1], info["heading"][0]))
    xy = [info["fly_xy"]]
    steps = 0
    for action in drive:
        _, _, terminated, truncated, info = env.step(np.asarray(action))
        xy.append(info["fly_xy"])
        steps += 1
        if terminated or truncated:
            break
    h1 = float(np.arctan2(info["heading"][1], info["heading"][0]))
    return {
        "steps": steps,
        "hit": info.get("hit", False),
        "flipped": info.get("flipped", False),
        "final_heading_deg": float(np.degrees(h1 - h0)),
        "path_length_mm": float(np.sum(np.linalg.norm(np.diff(np.array(xy), axis=0), axis=1))),
    }


def symmetric(drive: np.ndarray) -> np.ndarray:
    """Each step's mean of (left, right) on both sides: same speed, no steering."""
    mean = np.asarray(drive).mean(axis=1, keepdims=True)
    return np.repeat(mean, 2, axis=1)

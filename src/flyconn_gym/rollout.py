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
    terminated = truncated = False
    while not (terminated or truncated):
        action = policy.act(obs[:2])
        obs, _, terminated, truncated, info = env.step(action)
        xy.append(info["fly_xy"])
        drive.append(action)
        dn.append([policy.last_readouts[k] for k in sorted(policy.last_readouts)])
        dist.append(info["loom_distance"])
    h1 = float(np.arctan2(info["heading"][1], info["heading"][0]))
    return {
        "provenance": {**policy.provenance(), "episode_seed": seed},
        "xy": np.array(xy),
        "drive": np.array(drive),
        "dn_names": sorted(policy.last_readouts),
        "dn": np.array(dn),
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

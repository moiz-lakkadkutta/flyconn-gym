"""Render the toy Level A loop for every brain variant, side by side, plus a provenance JSON.

Usage: ``uv run python scripts/render_toy.py [--steps 800] [--out runs/toy]``. Local files only.
The toy connectome is synthetic: the video shows the loop and the controls, not fly biology.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw

from flyconn_gym.brain import V2_VARIANTS
from flyconn_gym.coupling import LevelAPolicy
from flyconn_gym.env import LoomingWalkEnv
from flyconn_gym.rollout import run_episode
from flyconn_gym.toy import toy_connectome, toy_iomap


def _label(frame: np.ndarray, text: str) -> np.ndarray:
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, img.width, 30], fill=(0, 0, 0))
    draw.text((6, 2), text, fill=(255, 255, 255))
    draw.text((6, 16), "LEVEL A - toy synthetic connectome", fill=(255, 210, 90))
    return np.asarray(img)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=800, help="control steps of 1 ms")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("runs/toy"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    m = toy_connectome()
    io = toy_iomap(m)
    columns, record = [], {}
    for variant in V2_VARIANTS:
        env = LoomingWalkEnv(max_steps=args.steps, loom_side="left", render_mode="rgb_array")
        policy = LevelAPolicy.build(m, io, variant, seed=args.seed)
        t0 = time.perf_counter()
        ep = run_episode(env, policy, seed=args.seed)
        wall = time.perf_counter() - t0
        frames = [_label(f, variant.value) for f in env.frames]
        columns.append(frames)
        record[variant.value] = {
            **ep["summary"],
            "wall_s": round(wall, 2),
            "physics_steps_per_s": round(ep["summary"]["steps"] * env.decimation / wall),
            "provenance": ep["provenance"],
        }
        print(variant.value, json.dumps(ep["summary"]), f"wall={wall:.1f}s")
        env.close()

    n = min(len(c) for c in columns)
    tiled = [np.concatenate([c[i] for c in columns], axis=1) for i in range(n)]
    gif = args.out / "toy_level_a.gif"
    iio.imwrite(gif, tiled, duration=40, loop=0)
    (args.out / "toy_level_a.json").write_text(json.dumps(record, indent=2, default=str))
    print(f"wrote {gif} ({n} frames) and toy_level_a.json")


if __name__ == "__main__":
    main()

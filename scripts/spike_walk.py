"""M0 spike: one headless FlyGym 2.x walking episode driven by the hybrid turning controller.

Prints physics steps per second. Usage: ``uv run python scripts/spike_walk.py [seconds] [l] [r]``.
"""

from __future__ import annotations

import sys
import time

import numpy as np
from flygym import Simulation
from flygym.compose import FlatGroundWorld
from flygym.utils.math import Rotation3D
from flygym_demo.complex_terrain import (
    HybridControllerObservation,
    HybridTurningController,
    apply_locomotion_action,
    make_locomotion_fly,
)


def main() -> None:
    duration = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    drive = np.array([float(a) for a in sys.argv[2:4]] or [1.0, 1.0])
    fly = make_locomotion_fly("nmf")
    world = FlatGroundWorld()
    world.add_fly(fly, np.array([0.0, 0.0, 0.7]), Rotation3D("quat", (1, 0, 0, 0)))  # pyright: ignore[reportArgumentType]
    sim = Simulation(world)
    ctrl = HybridTurningController(timestep=sim.timestep)
    sim.reset()
    ctrl.reset(seed=0)
    sim.warmup()
    start_xy = sim.get_body_positions("nmf")[0, :2].copy()
    n = int(duration / sim.timestep)
    t0 = time.perf_counter()
    for _ in range(n):
        obs = HybridControllerObservation.from_sim(sim, "nmf")
        apply_locomotion_action(sim, "nmf", ctrl.step(drive, obs))
        sim.step()
    dt = time.perf_counter() - t0
    end_xy = sim.get_body_positions("nmf")[0, :2]
    print(f"timestep={sim.timestep} steps={n} wall={dt:.2f}s steps/s={n / dt:.0f}")
    print(f"realtime factor={duration / dt:.3f} displacement_xy={end_xy - start_xy}")


if __name__ == "__main__":
    main()

"""Gymnasium env: a FlyGym 2.x fly on flat ground with a toy looming sphere.

Action: FlyGym's (left, right) descending drive for the hybrid turning controller (Level A).
Observation: ``[loom_left, loom_right, x, y, heading_x, heading_y]`` (rad/s, mm, unit vector).
There is no task reward; the env is a body, and the brain is the policy that acts on it.
"""

from __future__ import annotations

from typing import Any, Literal

import gymnasium as gym
import mujoco as mj
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

from flyconn_gym.coupling import COUPLING_LEVEL, looming_features

FLY = "nmf"
OBS_NAMES = ("loom_left", "loom_right", "x", "y", "heading_x", "heading_y")
LoomSide = Literal["left", "right", "none"]


def _rot(fmt: Literal["quat", "xyaxes"], *values: float) -> Rotation3D:
    # flygym annotates values as Sequence[numbers.Number], which pyright never matches for floats.
    return Rotation3D(fmt, values)  # pyright: ignore[reportArgumentType]


class LoomingWalkEnv(gym.Env[np.ndarray, np.ndarray]):
    metadata = {"render_modes": ["rgb_array"]}  # noqa: RUF012 (gymnasium convention)

    def __init__(
        self,
        *,
        max_steps: int = 1000,
        decimation: int = 10,
        loom_side: LoomSide = "left",
        loom_distance: float = 12.0,
        loom_radius: float = 1.0,
        loom_speed: float = 15.0,
        render_mode: Literal["rgb_array"] | None = None,
        camera_res: tuple[int, int] = (240, 320),
    ) -> None:
        self.max_steps = max_steps
        self.decimation = decimation
        self.loom_side: LoomSide = loom_side
        self.loom_distance = loom_distance
        self.loom_radius = loom_radius
        self.loom_speed = loom_speed
        self.render_mode = render_mode

        fly = make_locomotion_fly(FLY)
        camera = None
        if render_mode == "rgb_array":
            camera = fly.add_tracking_camera(
                name="topcam",
                pos_offset=np.array([0.0, 0.0, 40.0]),
                rotation=_rot("xyaxes", 1, 0, 0, 0, 1, 0),
            )
        world = FlatGroundWorld()
        loom = world.mjcf_root.worldbody.add_body(name="loom", mocap=True, pos=[0, 0, -10])
        loom.add_geom(
            type=mj.mjtGeom.mjGEOM_SPHERE,  # pyright: ignore[reportAttributeAccessIssue]
            size=[loom_radius, 0, 0],
            rgba=[0.85, 0.2, 0.2, 1],
            contype=0,
            conaffinity=0,
        )
        world.add_fly(fly, np.array([0.0, 0.0, 0.7]), _rot("quat", 1, 0, 0, 0))
        self.sim = Simulation(world)
        self._mocap = int(self.sim.mj_model.body("loom").mocapid[0])
        if camera is not None:
            self.sim.set_renderer(camera, camera_res=camera_res, playback_speed=0.2)
        self.controller = HybridTurningController(timestep=self.sim.timestep)
        self.dt = self.sim.timestep * decimation

        inf = np.inf
        self.observation_space = gym.spaces.Box(
            low=np.array([0, 0, -inf, -inf, -1, -1], dtype=np.float32),
            high=np.array([inf, inf, inf, inf, 1, 1], dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(-1.0, 1.5, shape=(2,), dtype=np.float32)
        self._t = 0
        self._obj_xy = np.zeros(2)

    # -- state readers -------------------------------------------------------------------------
    def _fly_pose(self) -> tuple[np.ndarray, np.ndarray, HybridControllerObservation]:
        cobs = HybridControllerObservation.from_sim(self.sim, FLY)
        fly = self.sim.world.fly_lookup[FLY]
        order = fly.get_bodysegs_order()
        thorax = order.index(type(fly).BODY_SEGMENT_CLASS("c_thorax"))
        xy = self.sim.get_body_positions(FLY)[thorax, :2].copy()
        heading = cobs.fly_heading[:2] / max(float(np.linalg.norm(cobs.fly_heading[:2])), 1e-9)
        return xy, heading, cobs

    def _loom(self, xy: np.ndarray, heading: np.ndarray) -> np.ndarray:
        if self.loom_side == "none":
            return np.zeros(2)
        return looming_features(xy, heading, self._obj_xy, self.loom_radius, self.loom_speed)

    def _obs(self, xy: np.ndarray, heading: np.ndarray) -> np.ndarray:
        return np.concatenate([self._loom(xy, heading), xy, heading]).astype(np.float32)

    def _info(self, xy: np.ndarray, heading: np.ndarray) -> dict[str, Any]:
        return {
            "coupling_level": COUPLING_LEVEL,
            "time_s": self._t * self.dt,
            "fly_xy": xy,
            "heading": heading,
            "loom_xy": self._obj_xy.copy(),
            "loom_distance": float(np.hypot(*(self._obj_xy - xy))),
        }

    # -- gymnasium API -------------------------------------------------------------------------
    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.sim.reset()
        self.controller.reset(seed=seed)
        self.sim.warmup()
        if self.sim.renderer is not None:
            self.sim.renderer.reset()
        self._t = 0
        xy, heading, _ = self._fly_pose()
        side = {"left": 1.0, "right": -1.0, "none": 0.0}[self.loom_side]
        normal = np.array([-heading[1], heading[0]])  # points to the fly's left
        self._obj_xy = xy + side * self.loom_distance * normal
        if self.loom_side == "none":
            self._obj_xy = xy + np.array([1e3, 0.0])
        self._place_loom()
        return self._obs(xy, heading), self._info(xy, heading)

    def _place_loom(self) -> None:
        z = self.loom_radius if self.loom_side != "none" else -10.0
        self.sim.mj_data.mocap_pos[self._mocap] = [*self._obj_xy, z]

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        drive = np.clip(np.asarray(action, dtype=float), -1.0, 1.5)
        cobs = None
        for _ in range(self.decimation):
            _, _, cobs = self._fly_pose()
            apply_locomotion_action(self.sim, FLY, self.controller.step(drive, cobs))
            self.sim.step()
            if self.sim.renderer is not None:
                self.sim.render_as_needed()
        self._t += 1
        xy, heading, cobs = self._fly_pose()
        if self.loom_side != "none":
            rel = xy - self._obj_xy
            d = float(np.hypot(*rel))
            if d > self.loom_radius:
                self._obj_xy = self._obj_xy + rel / d * min(self.loom_speed * self.dt, d)
            self._place_loom()
        info = self._info(xy, heading)
        hit = self.loom_side != "none" and info["loom_distance"] <= self.loom_radius
        flipped = cobs.thorax_z < 0.15
        terminated = bool(hit or flipped)
        info["hit"], info["flipped"] = bool(hit), bool(flipped)
        truncated = self._t >= self.max_steps and not terminated
        return self._obs(xy, heading), 0.0, terminated, truncated, info

    @property
    def frames(self) -> list[np.ndarray]:
        r = self.sim.renderer
        if r is None or r.frames is None:
            return []
        return next(iter(r.frames.values()))

    def render(self) -> np.ndarray | None:  # type: ignore[override]
        f = self.frames
        return f[-1] if f else None

    def close(self) -> None:
        self.sim.close()

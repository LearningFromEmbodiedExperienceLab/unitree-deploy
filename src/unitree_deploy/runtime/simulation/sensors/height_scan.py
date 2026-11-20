from copy import copy

import mujoco
import numpy as np

from unitree_deploy.runtime.transport.height_scan import HeightScanPublisher

from .base_sensor import MujocoSensor


class MujocoHeightScanSensor(MujocoSensor):
    """Sample static terrain with vertical rays in a yaw-aligned XY grid."""

    def __init__(
        self, *, name, body, shape, x_range, y_range, start_z, max_distance, rate_hz, geom_groups,
        debug_rgba=(0.25, 1.0, 0.35, 1.0),
    ):
        if len(shape) != 2 or min(shape) <= 0 or min(start_z, max_distance, rate_hz) <= 0:
            raise ValueError("height scan shape, ray limits, and update rate must be positive")
        self.debug_rgba = np.asarray(debug_rgba, dtype=np.float32)
        if self.debug_rgba.shape != (4,) or not np.all((0 <= self.debug_rgba) & (self.debug_rgba <= 1)):
            raise ValueError("height scan debug_rgba must contain four values between 0 and 1")
        super().__init__(rate_hz=rate_hz)
        self.name = name
        self.body = body
        self.start_z = start_z
        self.max_distance = max_distance
        self.geom_groups = geom_groups
        xs = np.linspace(*x_range, num=shape[0])
        ys = np.linspace(*y_range, num=shape[1])
        self.xy = np.stack(np.meshgrid(xs, ys, indexing="ij"), axis=-1).reshape(-1, 2)
        self.grid = np.empty(shape, dtype=np.float32)
        self.hit_points = np.empty((0, 3))
        self.ray_origin = np.empty(3)
        self.ray_down = np.array([0.0, 0.0, -1.0])
        self.ray_groups = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)
        self.ray_model = None
        self.publisher = None

    def configure_model(self, spec: mujoco.MjSpec) -> None:
        if spec.body(self.body) is None:
            raise ValueError(f"height scan body {self.body!r} not found in the MuJoCo model")

    def _capture_and_publish(
        self, model: mujoco.MjModel, data: mujoco.MjData, *, captured_at: float,
    ) -> bool:
        if self.ray_model is None:
            self.body_id = model.body(self.body).id
            static = model.body_weldid[model.geom_bodyid] == 0
            terrain = static & np.isin(model.geom_group, self.geom_groups)
            self.ray_model = copy(model)
            self.ray_model.geom_group[:] = np.where(terrain, 0, 1)
            self.publisher = HeightScanPublisher(self.name)

        mujoco.mj_kinematics(model, data)
        pos = data.xpos[self.body_id]
        rotation = data.xmat[self.body_id].reshape(3, 3)
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
        c, s = np.cos(yaw), np.sin(yaw)
        xy_world = self.xy @ np.array([[c, s], [-s, c]]) + pos[:2]
        self.ray_origin[2] = pos[2] + self.start_z
        heights = self.grid.reshape(-1)
        for i, xy in enumerate(xy_world):
            self.ray_origin[:2] = xy
            distance = mujoco.mj_ray(
                self.ray_model, data, self.ray_origin, self.ray_down, self.ray_groups, True, -1, None,
            )
            heights[i] = self.start_z - distance if 0 <= distance <= self.max_distance else np.nan

        hits = np.isfinite(heights)
        self.hit_points = np.column_stack((xy_world[hits], pos[2] + heights[hits]))
        self.publisher.publish(self.grid, captured_at=captured_at)
        return True

    def add_to_scene(self, model: mujoco.MjModel, data: mujoco.MjData, scene: mujoco.MjvScene) -> None:
        size = np.full(3, 0.012)
        transparent = self.debug_rgba[3] < 1.0
        identity = np.eye(3).reshape(9)
        for point in self.hit_points[:scene.maxgeom - scene.ngeom]:
            geom = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE, size, point, identity, self.debug_rgba)
            geom.category = mujoco.mjtCatBit.mjCAT_DECOR
            geom.transparent = transparent
            scene.ngeom += 1

    def close(self) -> None:
        if self.publisher is not None:
            self.publisher.close()

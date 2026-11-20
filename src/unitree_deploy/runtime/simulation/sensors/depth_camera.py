from copy import copy

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from unitree_deploy.runtime.transport.realsense import RealSensePublisher
from unitree_deploy.sensors.realsense import RealSenseCaptureConfig, RealSenseFrame

from .base_sensor import MujocoSensor


class MujocoDepthSensor(MujocoSensor):
    """Publish raw metric depth using the same frame format as RealSense."""

    def __init__(
        self, *, name, body, position, rpy, width, height, fovy, near, far, rate_hz,
        debug_rgba=(0.1, 0.8, 1.0, 1.0),
    ):
        self.debug_rgba = np.asarray(debug_rgba, dtype=np.float32)
        if self.debug_rgba.shape != (4,) or not np.all((0 <= self.debug_rgba) & (self.debug_rgba <= 1)):
            raise ValueError("depth camera debug_rgba must contain four values between 0 and 1")
        super().__init__(rate_hz=rate_hz)
        self.name = name
        self.body = body
        self.position = position
        self.quat = Rotation.from_euler("xyz", rpy, degrees=True).as_quat(scalar_first=True)
        self.debug_image = None
        self.width = width
        self.height = height
        self.fovy = fovy
        self.near = near
        self.far = far
        focal = height / (2 * np.tan(np.deg2rad(fovy) / 2))
        self.intrinsics = (focal, focal, (width - 1) / 2, (height - 1) / 2)
        # MuJoCo cameras look along local -Z with +Y up.
        corners = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1]])
        corners = corners * [width / (2 * focal), height / (2 * focal), 1]
        self.frustum = np.vstack((near * corners, far * corners))
        self.renderer = None
        self.publisher = None

    def configure_model(self, spec: mujoco.MjSpec) -> None:
        body = spec.body(self.body)
        if body is None:
            raise ValueError(f"depth camera body {self.body!r} not found in the MuJoCo model")
        body.add_camera(name=self.name, pos=self.position, quat=self.quat, fovy=self.fovy)

    def _capture_and_publish(
        self, model: mujoco.MjModel, data: mujoco.MjData, *, captured_at: float,
    ) -> bool:
        if self.renderer is None:
            # MuJoCo uses the model clipping planes for both rendering and depth conversion.
            render_model = copy(model)
            render_model.vis.map.znear = self.near / render_model.stat.extent
            render_model.vis.map.zfar = self.far / render_model.stat.extent
            self.renderer = mujoco.Renderer(render_model, height=self.height, width=self.width)
            self.renderer.enable_depth_rendering()
            self.publisher = RealSensePublisher(
                RealSenseCaptureConfig(
                    camera_name=self.name, depth_width=self.width, depth_height=self.height,
                    align_depth_to_color=False,
                ),
                include_color=False,
            )
        # Refresh camera and geometry transforms after mj_step integrated the positions.
        mujoco.mj_kinematics(model, data)
        mujoco.mj_camlight(model, data)
        self.renderer.update_scene(data, camera=self.name)
        depth = self.renderer.render()
        intensity = (255 * np.clip((depth - self.near) / (self.far - self.near), 0, 1)).astype(np.uint8)
        self.debug_image = np.repeat(intensity[..., None], 3, axis=2)
        return self.publisher.publish(RealSenseFrame(
            camera_name=self.name, serial_number=None, color=None,
            depth=depth, timestamp_sec=data.time, depth_intrinsics=self.intrinsics,
        ))

    def add_to_scene(self, model: mujoco.MjModel, data: mujoco.MjData, scene: mujoco.MjvScene) -> None:
        camera = model.camera(self.name).id
        corners = self.frustum @ data.cam_xmat[camera].reshape(3, 3).T + data.cam_xpos[camera]
        line = mujoco.mjtGeom.mjGEOM_LINE
        transparent = self.debug_rgba[3] < 1.0
        size = np.zeros(3)
        identity = np.eye(3).reshape(9)
        for i in range(4):
            j = (i + 1) % 4
            for start, end in ((i, j), (i + 4, j + 4), (i, i + 4)):
                if scene.ngeom >= scene.maxgeom:
                    return
                geom = scene.geoms[scene.ngeom]
                mujoco.mjv_initGeom(geom, line, size, corners[start], identity, self.debug_rgba)
                mujoco.mjv_connector(geom, line, 2.0, corners[start], corners[end])
                geom.category = mujoco.mjtCatBit.mjCAT_DECOR
                geom.transparent = transparent
                scene.ngeom += 1

    def close(self) -> None:
        if self.publisher is not None:
            self.publisher.close()
        if self.renderer is not None:
            self.renderer.close()

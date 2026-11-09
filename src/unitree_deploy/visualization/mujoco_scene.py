from __future__ import annotations

from collections.abc import Sequence
import copy
from dataclasses import dataclass
import io

import mujoco
from mjviser import ViserMujocoScene
import numpy as np
import trimesh
import viser
import viser.transforms as vtf

from unitree_deploy.runtime.transport.realsense import RealSenseReader
from unitree_deploy.sensors.realsense import RealSenseCaptureConfig, RealSenseFrame


@dataclass
class RealSenseCameraConfig(RealSenseCaptureConfig):
    pose_camera_name: str | None = None
    depth_visualization_min_m: float = 0.1
    depth_visualization_max_m: float = 3.0
    frustum_scale: float = 0.15
    jpeg_quality: int | None = 80


def _camera_id_by_name(mj_model: mujoco.MjModel, camera_name: str | None) -> int | None:
    if not camera_name:
        return None

    camera_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    if camera_id < 0:
        return None
    return int(camera_id)


def _opencv_pose_from_mujoco_matrix(position: np.ndarray, rotation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # MuJoCo cameras use +Y up and look along -Z, while viser frustums follow
    # the OpenCV convention of +Y down and +Z forward.
    rot_180_x = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.float32)
    adjusted_rotation = rotation @ rot_180_x
    wxyz = vtf.SO3.from_matrix(adjusted_rotation).wxyz.astype(np.float32)
    return position.astype(np.float32), wxyz


# ----- Image conversion helpers -----

def _camera_pose(mj_data: mujoco.MjData, camera_id: int) -> tuple[np.ndarray, np.ndarray]:
    cam_pos = np.asarray(mj_data.cam_xpos[camera_id], dtype=np.float32)
    cam_mat = np.asarray(mj_data.cam_xmat[camera_id], dtype=np.float32).reshape(3, 3)
    return _opencv_pose_from_mujoco_matrix(cam_pos, cam_mat)


def _depth_to_display_image(
    depth_m: np.ndarray,
    *,
    min_depth_m: float,
    max_depth_m: float,
) -> np.ndarray:
    if depth_m.size == 0:
        return np.zeros((*depth_m.shape, 3), dtype=np.uint8)

    if max_depth_m <= min_depth_m:
        max_depth_m = min_depth_m + 1e-3

    valid = np.isfinite(depth_m) & (depth_m > 0.0)
    normalized = np.clip((depth_m - min_depth_m) / (max_depth_m - min_depth_m), 0.0, 1.0)
    normalized = 1.0 - normalized

    red = np.clip(1.5 - np.abs(4.0 * normalized - 3.0), 0.0, 1.0)
    green = np.clip(1.5 - np.abs(4.0 * normalized - 2.0), 0.0, 1.0)
    blue = np.clip(1.5 - np.abs(4.0 * normalized - 1.0), 0.0, 1.0)
    image = np.stack([red, green, blue], axis=-1)
    image = (255.0 * image).astype(np.uint8)
    image[~valid] = 0
    return image


class RealSenseCameraStream:
    """Display shared RealSense frames without opening the camera device."""

    def __init__(
        self,
        config: RealSenseCameraConfig,
        *,
        server: viser.ViserServer | None = None,
        mj_model: mujoco.MjModel | None = None,
        show_frustum: bool = True,
    ):
        self.config = config
        self.server = server
        self.mj_model = mj_model
        self.show_frustum = show_frustum
        self.latest_frame: RealSenseFrame | None = None
        self._reader = RealSenseReader(config.camera_name)
        self._pose_camera_id: int | None = None
        self._frustum_handle = None
        self._gui_folder = None
        self._gui_rgb_image_handle = None
        self._gui_depth_image_handle = None

        color_width = config.color_width
        color_height = config.color_height

        if self.mj_model is not None:
            # Prefer an explicitly configured camera, then common G1 head camera names.
            pose_camera_name = self.config.pose_camera_name or self.config.camera_name
            self._pose_camera_id = _camera_id_by_name(self.mj_model, pose_camera_name)
            if self._pose_camera_id is None:
                self._pose_camera_id = _camera_id_by_name(self.mj_model, "d435_head")
            if self._pose_camera_id is None and int(self.mj_model.ncam) == 1:
                self._pose_camera_id = 0

        if self.show_frustum and self.server is not None and self._pose_camera_id is not None:
            self._frustum_handle = self.server.scene.add_camera_frustum(
                name=f"/realsense/{self.config.camera_name}/frustum",
                fov=np.pi / 3.0,  # Replaced with measured intrinsics on the first shared frame.
                aspect=float(color_width) / color_height,
                visible=False,
                position=np.zeros(3, dtype=np.float32),
                wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                scale=self.config.frustum_scale,
                color=(80, 180, 255),
                jpeg_quality=self.config.jpeg_quality,
            )
        elif self.show_frustum and self.server is not None and self.mj_model is not None:
            pose_camera_name = self.config.pose_camera_name or self.config.camera_name
            print(
                f"[WARN] Failed to locate MuJoCo camera '{pose_camera_name}' for RealSense frustum "
                f"{self.config.camera_name}."
            )

        if self.server is not None:
            self._gui_folder = self.server.gui.add_folder(
                f"Camera: {self.config.camera_name}",
                expand_by_default=True,
            )
            self._rgb_placeholder = np.zeros((color_height, color_width, 3), dtype=np.uint8)
            with self._gui_folder:
                self._gui_rgb_image_handle = self.server.gui.add_image(
                    self._rgb_placeholder,
                    label="RGB",
                    jpeg_quality=self.config.jpeg_quality,
                )
                if self.config.enable_depth:
                    depth_height = color_height if self.config.align_depth_to_color else self.config.depth_height
                    depth_width = color_width if self.config.align_depth_to_color else self.config.depth_width
                    self._depth_placeholder = np.zeros((depth_height, depth_width, 3), dtype=np.uint8)
                    self._gui_depth_image_handle = self.server.gui.add_image(
                        self._depth_placeholder,
                        label="Depth",
                        jpeg_quality=self.config.jpeg_quality,
                    )

    def poll_frame(self) -> RealSenseFrame | None:
        self.latest_frame = self._reader.read()
        return self.latest_frame

    def update_visualization(self, mj_data: mujoco.MjData, *, scene_offset=None) -> None:
        frame = self.latest_frame
        if self._frustum_handle is not None and self._pose_camera_id is not None:
            position, wxyz = _camera_pose(mj_data, self._pose_camera_id)
            if scene_offset is not None:
                position = position + scene_offset
            self._frustum_handle.visible = frame is not None and frame.color_intrinsics is not None
            if frame is not None and frame.color is not None and frame.color_intrinsics is not None:
                height, width = frame.color.shape[:2]
                self._frustum_handle.fov = float(2.0 * np.arctan2(height, 2.0 * frame.color_intrinsics[1]))
                self._frustum_handle.aspect = float(width) / height
            self._frustum_handle.position = position
            self._frustum_handle.wxyz = wxyz

        if frame is None:
            # Do not leave a stale image displayed as a live camera feed.
            if self._gui_rgb_image_handle is not None:
                self._gui_rgb_image_handle.image = self._rgb_placeholder
            if self._gui_depth_image_handle is not None:
                self._gui_depth_image_handle.image = self._depth_placeholder
            return
        if self._gui_rgb_image_handle is not None and frame.color is not None:
            self._gui_rgb_image_handle.image = frame.color
        if self._gui_depth_image_handle is not None and frame.depth is not None:
            self._gui_depth_image_handle.image = _depth_to_display_image(
                frame.depth,
                min_depth_m=self.config.depth_visualization_min_m,
                max_depth_m=self.config.depth_visualization_max_m,
            )

    def close(self) -> None:
        self._reader.close()
        if self._gui_folder is not None:
            self._gui_folder.remove()


class RobotScene(ViserMujocoScene):
    """mjviser scene with robot layers, tracking, and optional RealSense feeds."""

    def __init__(
        self,
        server: viser.ViserServer,
        mj_model: mujoco.MjModel,
        *,
        visual_rgba: tuple[float, float, float, float] | None = None,
        show_sites: bool = True,
        show_camera_frustums: bool = True,
        real_sense_configs: Sequence[RealSenseCameraConfig] | None = None,
    ):
        # Layer styling belongs to the viewer; leave the publisher/simulator model intact.
        model = copy.copy(mj_model)
        robot = (model.geom_bodyid != 0) & (model.geom_type != mujoco.mjtGeom.mjGEOM_PLANE)
        collision = robot & ((model.geom_contype != 0) | (model.geom_conaffinity != 0))
        visual = robot & ~collision
        model.geom_group[:] = 2
        model.geom_group[visual] = 0
        model.geom_group[collision] = 1
        if visual_rgba is not None:
            model.geom_matid[visual] = -1
            model.geom_rgba[visual] = visual_rgba
        model.geom_matid[collision] = -1
        model.geom_rgba[collision] = (40 / 255, 200 / 255, 90 / 255, 120 / 255)
        super().__init__(server, model, num_envs=1)
        self.real_sense_cameras: list[RealSenseCameraStream] = []
        # Match any4hdmi: recenter the scene without writing client camera poses.
        self.camera_tracking_enabled = True
        self.geom_groups_visible = [bool(visual.any()), not visual.any(), True, False, False, False]
        self.site_groups_visible = [show_sites] * 6
        self._sync_visibilities()

        # Like any4hdmi, use BLEND materials for translucent geometry.
        translucent_groups = (0, 1) if visual_rgba is not None and visual_rgba[3] < 1 else (1,)
        handles = [group.handle for group in self._mesh_groups if group.group_id in translucent_groups]
        handles += [handle for (_, group, _), handle in self._fixed_geom_handles.items() if group in translucent_groups]
        for handle in handles:
            meshes = trimesh.load(io.BytesIO(handle.glb_data), file_type="glb", force="scene")
            for mesh in meshes.geometry.values():
                material = trimesh.visual.material.PBRMaterial(
                    baseColorFactor=mesh.visual.vertex_colors[0], alphaMode="BLEND", doubleSided=True,
                )
                mesh.visual = trimesh.visual.TextureVisuals(material=material)
            handle.glb_data = meshes.export(file_type="glb")

        if not (model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE).any():
            server.scene.add_grid(
                "/fixed_bodies/ground", infinite_grid=True, fade_distance=50.0,
                shadow_opacity=0.2, plane_opacity=0.4,
            )
        with server.gui.add_folder("Robot"):
            visual_checkbox = server.gui.add_checkbox("Visual", initial_value=self.geom_groups_visible[0])
            collision_checkbox = server.gui.add_checkbox("Collision", initial_value=self.geom_groups_visible[1])

            def update_layers(_):
                def mutate():
                    self.geom_groups_visible[:2] = [visual_checkbox.value, collision_checkbox.value]
                    self._sync_visibilities()

                self._apply_visualization_change(mutate)

            visual_checkbox.on_update(update_layers)
            collision_checkbox.on_update(update_layers)
        self.create_scene_gui(camera_distance=1.5 * model.stat.extent)
        # any4hdmi's mjviser has no fixed-camera presets; keep the orbit camera.
        if self._camera_dropdown is not None:
            self._camera_dropdown.remove()
            self._camera_dropdown = None

        for config in real_sense_configs or []:
            try:
                camera = RealSenseCameraStream(
                    config, server=server, mj_model=model, show_frustum=show_camera_frustums,
                )
            except Exception as exc:
                print(f"[WARN] Failed to create RealSense camera {config.camera_name}: {exc}")
                continue
            self.real_sense_cameras.append(camera)
        mujoco.mj_forward(model, self.mj_data)
        self.update_from_mjdata(self.mj_data)

    def update_from_mjdata(self, mj_data: mujoco.MjData) -> None:
        with self._update_lock, self.server.atomic():
            super().update_from_mjdata(mj_data)
            for camera in self.real_sense_cameras:
                camera.poll_frame()
                camera.update_visualization(mj_data, scene_offset=self._scene_offset)

    def close(self) -> None:
        for camera in self.real_sense_cameras:
            camera.close()

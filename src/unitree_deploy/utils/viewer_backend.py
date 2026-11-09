from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol
from unittest.mock import patch

import glfw
import mujoco
import mujoco.viewer
import numpy as np

from unitree_deploy.config.defaults import sim_key_for_button
from unitree_deploy.robot_model.robot_config import RobotModel
from unitree_deploy.utils.yaml_utils import load_yaml
from unitree_deploy.visualization.policy_debug_viz import PolicyDebugViz
from unitree_deploy.visualization.video import VideoConfig, VideoRecorder


class ViewerBackend(Protocol):
    """Small lifecycle adapter so SimBridge does not branch on viewer type."""

    def run(self, simulate: Callable[[], None]) -> None:
        ...

    def sync(self) -> bool:
        ...


@dataclass(frozen=True)
class ViewerCameraConfig:
    lookat: tuple[float, float, float] = (0.0, 0.0, 0.85)
    distance: float = 2.0
    elevation: float = -15.0
    azimuth: float = 200.0
    track_body: str | None = None
    track_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)


def load_viewer_camera_config(robot: RobotModel) -> ViewerCameraConfig:
    path = robot.config_dir / "visualizer.yaml"
    if not path.exists():
        return ViewerCameraConfig()

    config = load_yaml(path)
    camera = config.get("viewer", {}).get("camera", {})
    if not isinstance(camera, dict):
        return ViewerCameraConfig()

    defaults = ViewerCameraConfig()
    lookat = camera.get("lookat", defaults.lookat)
    if len(lookat) != 3:
        raise ValueError(f"{path} viewer.camera.lookat must contain 3 values")

    track_offset = camera.get("track_offset", defaults.track_offset)
    if len(track_offset) != 3:
        raise ValueError(f"{path} viewer.camera.track_offset must contain 3 values")

    return ViewerCameraConfig(
        lookat=tuple(float(value) for value in lookat),
        distance=float(camera.get("distance", defaults.distance)),
        elevation=float(camera.get("elevation", defaults.elevation)),
        azimuth=float(camera.get("azimuth", defaults.azimuth)),
        track_body=camera.get("track_body"),
        track_offset=tuple(float(value) for value in track_offset),
    )


class MujocoViewerBackend:
    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        camera: ViewerCameraConfig,
        *,
        sim_hz: int,
        render_hz: int,
        log: Callable[[str], None],
        policy_debug_viz: PolicyDebugViz | None = None,
        video: VideoConfig | None = None,
        simulation_speed: Callable[[], float] | None = None,
        controller_state: Callable[[], str] | None = None,
        scene_overlay: Callable[[mujoco.MjvScene], None] | None = None,
        set_sim_keyboard_enabled: Callable[[bool], None] | None = None,
    ) -> None:
        self.model = model
        self.policy_debug_viz = policy_debug_viz
        self.scene_overlay = scene_overlay
        self.set_sim_keyboard_enabled = set_sim_keyboard_enabled
        self.data = data
        self.camera = camera
        self.log = log
        self.simulation_speed = simulation_speed or (lambda: 1.0)
        self.controller_state = controller_state or (lambda: "waiting")
        free_joints = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
        self.light_root_qpos = int(model.jnt_qposadr[free_joints[0]]) if len(free_joints) else None
        self.world_light_ids = np.flatnonzero(model.light_bodyid == 0)
        self.light_positions = model.light_pos[self.world_light_ids].copy()
        self.track_body_id = self.resolve_track_body_id()
        self.tracking_enabled = self.track_body_id is not None
        self.keyboard_masked = False
        self._viewer_window = None
        self._mujoco_key_callback = None
        self._masked_key_callback = glfw._GLFWkeyfun(self.on_masked_key)
        self.viewer = None
        self.keyboard_help_visible = True
        self._keyboard_help = None
        self.viewer_tick = 0
        self.viewer_decim = max(1, sim_hz // render_hz)
        self.video = VideoRecorder(model, video) if video is not None else None

    def resolve_track_body_id(self) -> int | None:
        if not self.camera.track_body:
            return None
        body_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            self.camera.track_body,
        )
        if body_id < 0:
            raise ValueError(f"viewer.camera.track_body not found: {self.camera.track_body}")
        return int(body_id)

    def run(self, simulate: Callable[[], None]) -> None:
        backend = self

        class MaskedSimulate(mujoco.viewer._Simulate):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                # The constructor runs on the GLFW thread with the new window's context current.
                backend.enable_keyboard_mask()

        with patch.object(mujoco.viewer, "_Simulate", MaskedSimulate):
            viewer = mujoco.viewer.launch_passive(
                self.model,
                self.data,
                key_callback=self.on_key,
                show_left_ui=False,
                show_right_ui=False,
            )
        with viewer:
            self.viewer = viewer
            viewer.cam.lookat[:] = self.camera.lookat
            viewer.cam.distance = self.camera.distance
            viewer.cam.elevation = self.camera.elevation
            viewer.cam.azimuth = self.camera.azimuth
            self.update_keyboard_help()
            self.log("MuJoCo native keyboard shortcuts are masked by default; F12 toggles them")
            self.log("MuJoCo key hints: press ? to show or hide")
            if self.track_body_id is not None:
                self.log("MuJoCo camera: press T to toggle robot tracking")
            if self.video is not None:
                self.log(f"MP4 export: {self.video.config.output} (simulation time)")
            try:
                simulate()
            finally:
                if self.video is not None:
                    self.video.close()
                    self.log(f"MP4 export finished: {self.video.frames} frames")
                self.viewer = None

    def on_key(self, key: int) -> None:
        if key == glfw.KEY_F12:
            self.enable_keyboard_mask()
        elif key == glfw.KEY_T:
            self.toggle_tracking()
        elif key == glfw.KEY_SLASH and (
            glfw.get_key(self._viewer_window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
            or glfw.get_key(self._viewer_window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
        ):
            self.toggle_keyboard_help()

    def on_masked_key(
        self,
        _window,
        key: int,
        _scancode: int,
        action: int,
        mods: int,
    ) -> None:
        if action != glfw.PRESS:
            return
        if key == glfw.KEY_F12:
            self.disable_keyboard_mask()
        elif key == glfw.KEY_T:
            self.toggle_tracking()
        elif key == glfw.KEY_SLASH and mods & glfw.MOD_SHIFT:
            self.toggle_keyboard_help()

    def enable_keyboard_mask(self) -> None:
        if self.keyboard_masked:
            return
        window = glfw.get_current_context()
        if not window:
            self.log("camera keyboard mask unavailable: no current GLFW window")
            return

        # MuJoCo's Python key callback cannot consume events. Replacing the
        # underlying GLFW callback is the only way to block its native keys.
        previous = glfw._glfw.glfwSetKeyCallback(window, self._masked_key_callback)
        if not previous:
            self.log("camera keyboard mask unavailable: MuJoCo callback not found")
            return
        self._viewer_window = window
        self._mujoco_key_callback = previous
        self.keyboard_masked = True
        if self.set_sim_keyboard_enabled is not None:
            self.set_sim_keyboard_enabled(True)
        self.log("MuJoCo keyboard input=masked; sim2sim controls=enabled (F12: switch, T: camera tracking)")

    def disable_keyboard_mask(self) -> None:
        if not self.keyboard_masked:
            return
        if self._viewer_window is None or self._mujoco_key_callback is None:
            return
        glfw._glfw.glfwSetKeyCallback(self._viewer_window, self._mujoco_key_callback)
        self.keyboard_masked = False
        if self.set_sim_keyboard_enabled is not None:
            self.set_sim_keyboard_enabled(False)
        self.log("MuJoCo keyboard input=enabled; sim2sim controls=disabled")

    def toggle_tracking(self) -> None:
        if self.track_body_id is None:
            return
        self.tracking_enabled = not self.tracking_enabled
        self.log(f"camera tracking={'on' if self.tracking_enabled else 'off'}")

    def sync(self) -> bool:
        if self.viewer is None:
            return True
        if not self.viewer.is_running():
            return False
        self.viewer_tick += 1
        if self.viewer_tick % self.viewer_decim == 0 or (self.video is not None and not self.video.done):
            with self.viewer.lock():
                self.update_scene_lights()
        if self.video is not None and not self.video.done:
            with self.viewer.lock():
                self.update_tracked_lookat()
                self.video.capture(self.data, self.viewer.cam, self.viewer.opt,
                                   self.policy_debug_viz, self.status_texts(), self.scene_overlay)
        if self.viewer_tick % self.viewer_decim == 0:
            self.update_tracked_lookat()
            self.update_keyboard_help()
            if self.policy_debug_viz is not None or self.scene_overlay is not None:
                with self.viewer.lock():
                    self.viewer.user_scn.ngeom = 0
                    if self.policy_debug_viz is not None:
                        self.policy_debug_viz.add_to_scene(self.viewer.user_scn)
                    if self.scene_overlay is not None:
                        self.scene_overlay(self.viewer.user_scn)
            self.viewer.sync()
        return True

    def toggle_keyboard_help(self) -> None:
        self.keyboard_help_visible = not self.keyboard_help_visible

    def update_keyboard_help(self) -> None:
        texts = self.status_texts() + self.keyboard_help_texts()
        if texts != self._keyboard_help:
            # Replace the complete overlay in one call: clearing first lets the
            # independent viewer thread render an empty frame between updates.
            self.viewer.set_texts(texts)
            self._keyboard_help = texts

    def status_texts(self) -> list:
        return [(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                 "State\nSpeed", f"{self.controller_state()}\n{self.simulation_speed():g}x")]

    def update_scene_lights(self) -> None:
        if self.light_root_qpos is None or not len(self.world_light_ids):
            return
        root = self.light_root_qpos
        offset = self.data.qpos[root:root + 3] - self.model.qpos0[root:root + 3]
        self.model.light_pos[self.world_light_ids] = self.light_positions + offset
        mujoco.mj_camlight(self.model, self.data)

    def keyboard_help_texts(self) -> list:
        if not self.keyboard_help_visible:
            return []
        rows = [
            ("", "CONTROL"),
            ("WASD / QE", "Move / turn"),
            (f'{sim_key_for_button("A").capitalize()} / {sim_key_for_button("Start").upper()}', "Pose / run"),
            (f'{sim_key_for_button("B").upper()} / {sim_key_for_button("X").upper()}', "Switch / damping"),
            ("", "SIMULATION"),
            ("Space", "Pause / resume"),
            ("= / -", "Faster / slower"),
            ("R", "Reset"),
            ("N / Up Down", "Bands / height"),
            ("", "VIEWER"),
        ]
        viewer_keys, viewer_actions = [], []
        if self.track_body_id is not None:
            viewer_keys.append("T")
            viewer_actions.append("Follow")
        if self.policy_debug_viz is not None and self.policy_debug_viz.frame is not None:
            viewer_keys.append(self.policy_debug_viz.toggle_key.upper())
            viewer_actions.append("Debug Viz")
        if viewer_keys:
            rows.append((" / ".join(viewer_keys), " / ".join(viewer_actions)))
        rows.extend((("F12", "Sim / MuJoCo keys"), ("?", "Show / hide keys"), ("Esc", "Quit")))
        texts = ("\n".join(key for key, _ in rows), "\n".join(tip for _, tip in rows))
        return [(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPRIGHT, *texts)]

    def update_tracked_lookat(self) -> None:
        if self.viewer is None or self.track_body_id is None or not self.tracking_enabled:
            return
        self.viewer.cam.lookat[:] = self.data.xpos[self.track_body_id] + self.camera.track_offset


def create_viewer_backend(
    viewer: str,
    robot: RobotModel,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    sim_hz: int,
    render_hz: int,
    log: Callable[[str], None],
    policy_debug_viz: PolicyDebugViz | None = None,
    video: VideoConfig | None = None,
    simulation_speed: Callable[[], float] | None = None,
    controller_state: Callable[[], str] | None = None,
    scene_overlay: Callable[[mujoco.MjvScene], None] | None = None,
    set_sim_keyboard_enabled: Callable[[bool], None] | None = None,
) -> ViewerBackend:
    camera = load_viewer_camera_config(robot)
    if viewer == "mujoco":
        return MujocoViewerBackend(
            model,
            data,
            camera,
            sim_hz=sim_hz,
            render_hz=render_hz,
            log=log,
            policy_debug_viz=policy_debug_viz,
            video=video,
            simulation_speed=simulation_speed,
            controller_state=controller_state,
            scene_overlay=scene_overlay,
            set_sim_keyboard_enabled=set_sim_keyboard_enabled,
        )
    raise ValueError(f"unsupported viewer backend: {viewer}")

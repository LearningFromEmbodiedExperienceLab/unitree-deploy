from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from mujoco.glfw import glfw
from unitree_deploy.config.defaults import RENDER_HZ
from unitree_deploy.robot_model.robot_config import (
    DEFAULT_ROBOT,
    DEFAULT_TERRAIN,
    RobotModel,
    load_robot_model,
)
from unitree_deploy.utils.terminal_status import ComponentConsole
from unitree_deploy.utils.viewer_backend import load_viewer_camera_config


console = ComponentConsole("replay", "bright_blue")
DEFAULT_REPLAY_VIEWER = "viser"
REPLAY_VIEWER_CHOICES = ("viser", "mujoco")


def log(message: str) -> None:
    console.log(message)


@dataclass(frozen=True)
class RuntimeConfig:
    trajectory: Path
    robot: RobotModel | None
    speed: float
    loop: bool
    viewer: str = DEFAULT_REPLAY_VIEWER


def _load_metadata(trajectory_path: Path) -> dict:
    metadata_path = trajectory_path.with_name("metadata.json")
    if metadata_path.exists():
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    return {}


def _load_arrays(trajectory_path: Path) -> dict[str, np.ndarray]:
    with np.load(trajectory_path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


class TrajectoryReplayBase:
    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config
        self.metadata = _load_metadata(config.trajectory)
        self.arrays = _load_arrays(config.trajectory)
        self.time = np.asarray(self.arrays.get("time", np.zeros(0)), dtype=np.float64)
        if self.time.ndim != 1:
            raise ValueError("trajectory time must be 1D")
        if len(self.time) == 0:
            raise ValueError("trajectory is empty")

        self.robot = config.robot or self._load_robot_from_metadata()
        self.model = mujoco.MjModel.from_xml_path(str(self.robot.xml_path))
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetData(self.model, self.data)

        self.lock = threading.Lock()
        self.alive = True
        self.paused = True
        self.speed = max(float(config.speed), 0.0)
        self.loop = bool(config.loop)
        self.frame = 0
        self.play_start_wall_t = time.perf_counter()
        self.play_start_traj_t = float(self.time[0])
        self.last_render_frame = -1

    def _load_robot_from_metadata(self) -> RobotModel:
        """Prefer the original source XML so replay does not re-compose terrain twice."""
        robot_name = self.metadata.get("robot", DEFAULT_ROBOT)
        terrain = self.metadata.get("terrain", DEFAULT_TERRAIN)
        source_xml = self.metadata.get("source_xml_path") or self.metadata.get("source_xml")
        if source_xml:
            return load_robot_model(robot_name, Path(source_xml), terrain)

        model_xml = self.metadata.get("model_xml")
        model_xml_path = Path(model_xml) if model_xml else None
        return load_robot_model(robot_name, model_xml_path, terrain)

    def status_markdown(self) -> str:
        t = float(self.time[self.frame])
        return (
            f"**Frame**: {self.frame + 1}/{len(self.time)}  \n"
            f"**t**: {t:.3f}s  \n"
            f"**Speed**: {self.speed:.2f}x"
        )

    def status_line(self) -> str:
        t = float(self.time[self.frame])
        state = "paused" if self.paused else "playing"
        return f"{state} frame={self.frame + 1}/{len(self.time)} t={t:.3f}s speed={self.speed:.2f}x"

    def set_frame(self, frame: int) -> None:
        frame = int(np.clip(frame, 0, len(self.time) - 1))
        with self.lock:
            self.frame = frame
            self.play_start_wall_t = time.perf_counter()
            self.play_start_traj_t = float(self.time[frame])
        self.sync_frame_control(frame)
        self.apply_frame(frame)

    def set_paused(self, paused: bool) -> None:
        with self.lock:
            was_paused = self.paused
            self.paused = bool(paused)
            if was_paused and not self.paused:
                self.play_start_wall_t = time.perf_counter()
                self.play_start_traj_t = float(self.time[self.frame])
        self.sync_paused_control(self.paused)

    def toggle_paused(self) -> None:
        with self.lock:
            paused = not self.paused
        self.set_paused(paused)

    def set_speed(self, speed: float) -> None:
        with self.lock:
            self.speed = max(float(speed), 0.0)
            self.play_start_wall_t = time.perf_counter()
            self.play_start_traj_t = float(self.time[self.frame])
        self.sync_speed_control(self.speed)

    def set_loop(self, loop: bool) -> None:
        with self.lock:
            self.loop = bool(loop)

    def sync_frame_control(self, _frame: int) -> None:
        pass

    def sync_paused_control(self, _paused: bool) -> None:
        pass

    def sync_speed_control(self, _speed: float) -> None:
        pass

    def apply_frame(self, frame: int) -> None:
        qpos = np.asarray(self.arrays["qpos"][frame], dtype=np.float64)
        qvel = np.asarray(self.arrays["qvel"][frame], dtype=np.float64)
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        if "ctrl" in self.arrays and self.arrays["ctrl"].ndim == 2:
            ctrl = np.asarray(self.arrays["ctrl"][frame], dtype=np.float64)
            self.data.ctrl[:] = 0.0
            width = min(self.data.ctrl.shape[0], ctrl.shape[0])
            self.data.ctrl[:width] = ctrl[:width]
        mujoco.mj_forward(self.model, self.data)

    def tick(self) -> None:
        with self.lock:
            paused = self.paused
            speed = self.speed
            frame = self.frame
            start_wall = self.play_start_wall_t
            start_traj_t = self.play_start_traj_t

        if paused or speed <= 0.0:
            if self.last_render_frame != frame:
                self.apply_frame(frame)
                self.last_render_frame = frame
            return

        elapsed = (time.perf_counter() - start_wall) * speed
        target_t = start_traj_t + elapsed
        idx = int(np.searchsorted(self.time, target_t, side="right") - 1)
        if idx < 0:
            idx = 0
        if idx >= len(self.time):
            if self.loop:
                self.set_frame(0)
                return
            idx = len(self.time) - 1
            self.set_paused(True)

        if idx != frame:
            with self.lock:
                self.frame = idx
            self.sync_frame_control(idx)
            self.apply_frame(idx)
            self.last_render_frame = idx
        elif self.last_render_frame != idx:
            self.apply_frame(idx)
            self.last_render_frame = idx

    def log_startup(self) -> None:
        log(f"trajectory={self.config.trajectory}")
        log(
            f"samples={len(self.time)} robot={self.robot.name} terrain={self.robot.terrain} "
            f"viewer={self.config.viewer}"
        )

    def close(self, *_args) -> None:
        self.alive = False
        console.stop()


class ViserTrajectoryReplayApp(TrajectoryReplayBase):
    def __init__(self, config: RuntimeConfig) -> None:
        super().__init__(config)
        import viser
        from unitree_deploy.visualization.scene_config import StandaloneMujocoScene

        self.syncing_gui = False
        self.follow_robot = True  # Keep the camera centered unless the user opts out.
        self.follow_offset = np.array([2.8, -2.8, 1.8], dtype=np.float64)
        self.follow_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)

        self.server = viser.ViserServer()
        self.scene = StandaloneMujocoScene.create(
            self.server,
            self.model,
            show_camera_frustums=False,
            real_sense_configs=None,
        )
        self.controls = self.build_controls()
        self.server.on_client_connect(self.on_client_connect)
        self.apply_frame(0)
        self._apply_camera_follow()

    def build_controls(self) -> dict[str, object]:
        controls: dict[str, object] = {}
        panel = self.server.gui.add_folder("Playback", expand_by_default=True)
        with panel:
            controls["paused"] = self.server.gui.add_checkbox(
                "Paused",
                True,
                hint="Pause or resume playback",
            )
            controls["speed"] = self.server.gui.add_slider(
                "Speed",
                0.0,
                3.0,
                0.1,
                self.speed,
                hint="Playback speed multiplier",
            )
            controls["follow"] = self.server.gui.add_checkbox(
                "Follow",
                True,
                hint="Keep the camera centered on the robot",
            )
            controls["frame"] = self.server.gui.add_slider(
                "Frame",
                0,
                int(len(self.time) - 1),
                1,
                0,
                hint="Seek to an exact frame",
            )
            controls["step_back"] = self.server.gui.add_button("Prev", hint="Step backward one frame")
            controls["step_next"] = self.server.gui.add_button("Next", hint="Step forward one frame")
            controls["restart"] = self.server.gui.add_button("Restart", hint="Jump to the first frame")
            controls["status_text"] = self.server.gui.add_markdown(self.status_markdown())

        controls["paused"].on_update(self.on_paused_changed)
        controls["speed"].on_update(self.on_speed_changed)
        controls["follow"].on_update(self.on_follow_changed)
        controls["frame"].on_update(self.on_frame_changed)
        controls["step_back"].on_click(self.on_step_back)
        controls["step_next"].on_click(self.on_step_next)
        controls["restart"].on_click(self.on_restart)
        return controls

    def on_paused_changed(self, event) -> None:
        if self.syncing_gui:
            return
        self.set_paused(bool(event.target.value))

    def on_speed_changed(self, event) -> None:
        if self.syncing_gui:
            return
        self.set_speed(float(event.target.value))

    def on_follow_changed(self, event) -> None:
        if self.syncing_gui:
            return
        with self.lock:
            self.follow_robot = bool(event.target.value)
        self._apply_camera_follow()

    def on_frame_changed(self, event) -> None:
        if self.syncing_gui:
            return
        self.set_frame(int(event.target.value))

    def on_step_back(self, _event) -> None:
        self.set_frame(max(self.frame - 1, 0))

    def on_step_next(self, _event) -> None:
        self.set_frame(min(self.frame + 1, len(self.time) - 1))

    def on_restart(self, _event) -> None:
        self.set_frame(0)

    def _set_gui_value(self, handle, value) -> None:
        # GUI writes re-trigger update callbacks in viser, so guard programmatic syncs.
        self.syncing_gui = True
        try:
            handle.value = value
        finally:
            self.syncing_gui = False

    def sync_frame_control(self, frame: int) -> None:
        self._set_gui_value(self.controls["frame"], frame)

    def sync_paused_control(self, paused: bool) -> None:
        self._set_gui_value(self.controls["paused"], paused)

    def sync_speed_control(self, speed: float) -> None:
        self._set_gui_value(self.controls["speed"], speed)

    def apply_frame(self, frame: int) -> None:
        super().apply_frame(frame)
        self.scene.update_from_mjdata(self.data)
        self._apply_camera_follow()
        self.controls["status_text"].content = self.status_markdown()

    def on_client_connect(self, client) -> None:
        # New clients should start on the robot instead of an arbitrary default camera.
        self._set_client_camera(client)

    def _base_position(self) -> np.ndarray:
        if self.data.qpos.shape[0] < 3:
            return np.zeros(3, dtype=np.float64)
        return np.asarray(self.data.qpos[:3], dtype=np.float64)

    def _apply_camera_follow(self) -> None:
        if not self.follow_robot:
            return
        for client in self.server.get_clients().values():
            self._set_client_camera(client)

    def _set_client_camera(self, client) -> None:
        if not self.follow_robot:
            return
        target = self._base_position()
        # Fixed offset keeps the robot centered without requiring per-user camera logic.
        position = target + self.follow_offset
        client.camera.position = position
        client.camera.look_at = target
        client.camera.up_direction = self.follow_up

    def run(self) -> None:
        # Keep the UI responsive by ticking the replay in a short loop.
        self.log_startup()
        while self.alive:
            self.tick()
            time.sleep(0.02)

    def close(self, *_args) -> None:
        super().close(*_args)
        try:
            self.scene.close()
        except Exception:
            pass
        try:
            self.server.stop()
        except Exception:
            pass


class MujocoTrajectoryReplayApp(TrajectoryReplayBase):
    def __init__(self, config: RuntimeConfig) -> None:
        super().__init__(config)
        self.camera = load_viewer_camera_config(self.robot)
        self.track_body_id = self.resolve_track_body_id()
        self.viewer = None
        self.last_status_log = 0.0

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

    def run(self) -> None:
        self.log_startup()
        log('MuJoCo controls: space play/pause, left/right step, r restart, +/- speed, l loop')
        self.apply_frame(0)
        with mujoco.viewer.launch_passive(
            self.model,
            self.data,
            key_callback=self.on_key,
            show_left_ui=True,
            show_right_ui=False,
        ) as viewer:
            self.viewer = viewer
            self.configure_camera()
            while self.alive and viewer.is_running():
                self.tick()
                self.sync_viewer()
                self.maybe_log_status()
                time.sleep(1.0 / RENDER_HZ)
            self.viewer = None

    def configure_camera(self) -> None:
        if self.viewer is None:
            return
        self.viewer.cam.lookat[:] = self.camera.lookat
        self.viewer.cam.distance = self.camera.distance
        self.viewer.cam.elevation = self.camera.elevation
        self.viewer.cam.azimuth = self.camera.azimuth

    def sync_viewer(self) -> None:
        if self.viewer is None:
            return
        self.update_tracked_lookat()
        self.viewer.sync()

    def update_tracked_lookat(self) -> None:
        if self.viewer is None or self.track_body_id is None:
            return
        self.viewer.cam.lookat[:] = self.data.xpos[self.track_body_id] + self.camera.track_offset

    def maybe_log_status(self) -> None:
        now = time.perf_counter()
        if now - self.last_status_log < 1.0:
            return
        log(self.status_line())
        self.last_status_log = now

    def on_key(self, key: int) -> None:
        if key == glfw.KEY_SPACE:
            self.toggle_paused()
        elif key == glfw.KEY_RIGHT:
            self.set_frame(min(self.frame + 1, len(self.time) - 1))
        elif key == glfw.KEY_LEFT:
            self.set_frame(max(self.frame - 1, 0))
        elif key in (glfw.KEY_HOME, glfw.KEY_R):
            self.set_frame(0)
        elif key in (glfw.KEY_EQUAL, glfw.KEY_KP_ADD):
            self.set_speed(self.speed + 0.1)
        elif key in (glfw.KEY_MINUS, glfw.KEY_KP_SUBTRACT):
            self.set_speed(max(self.speed - 0.1, 0.0))
        elif key == glfw.KEY_L:
            self.set_loop(not self.loop)
            log(f"loop={'on' if self.loop else 'off'}")

        if key in {
            glfw.KEY_SPACE,
            glfw.KEY_RIGHT,
            glfw.KEY_LEFT,
            glfw.KEY_HOME,
            glfw.KEY_R,
            glfw.KEY_EQUAL,
            glfw.KEY_KP_ADD,
            glfw.KEY_MINUS,
            glfw.KEY_KP_SUBTRACT,
        }:
            log(self.status_line())

    def close(self, *_args) -> None:
        super().close(*_args)
        self.viewer = None


TrajectoryReplayApp = ViserTrajectoryReplayApp

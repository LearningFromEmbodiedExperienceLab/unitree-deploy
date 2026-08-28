import argparse
import math
import signal
import struct
import threading
import time
from dataclasses import dataclass

import mujoco
import numpy as np
from pynput import keyboard
from unitree_deploy.config.defaults import (
    ACC_SENSOR_NAMES,
    BAND_CLEARANCE,
    BAND_DAMPING,
    BAND_MAX_FORCE,
    BAND_MAX_Z,
    BAND_MIN_Z,
    BAND_SITES,
    BAND_STEP,
    BAND_STIFFNESS,
    BASE_HEIGHT,
    BASE_QUAT,
    DEFAULT_NET,
    GYRO_SENSOR_NAMES,
    HG_MODE_MACHINE,
    HG_MODE_PR,
    ODOM_TOPIC,
    RENDER_HZ,
    SIM_HZ,
    SIM_REMOTE_BUTTON_KEYS,
    STATE_HZ,
    Z1_LOWCMD_TOPIC,
    Z1_LOWSTATE_TOPIC,
)
from unitree_deploy.robot_model.robot_config import (
    DEFAULT_ROBOT,
    DEFAULT_TERRAIN,
    DEFAULT_VIEWER,
    VIEWER_CHOICES,
    RobotModel,
    load_robot_model,
)
from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize,
    ChannelPublisher,
    ChannelSubscriber,
)
from unitree_sdk2py.idl.default import (
    unitree_go_msg_dds__MotorState_,
    unitree_go_msg_dds__SportModeState_,
)
from unitree_sdk2py.idl.unitree_go.msg.dds_ import (
    MotorCmds_,
    MotorStates_,
    SportModeState_,
)
from unitree_sdk2py.utils.crc import CRC
from unitree_deploy.runtime.unitree_dds import resolve_low_level_dds
from unitree_deploy.utils.terminal_status import ComponentConsole
from unitree_deploy.utils.viewer_backend import create_viewer_backend


console = ComponentConsole("sim_bridge", "cyan")


def log(message: str) -> None:
    console.log(message)


def status(fields) -> None:
    console.status(fields)


STICK_KEYS = frozenset("wasdqe")


class KeyboardState:
    def __init__(self) -> None:
        self.pressed: set[str] = set()
        self.lock = threading.Lock()
        self.listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
        self.control_handler = None

    def _on_press(self, key) -> None:
        try:
            if hasattr(key, 'char') and key.char:
                k = key.char.lower()
            else:
                k = str(key).replace('Key.', '').lower()
        except AttributeError:
            k = str(key).replace('Key.', '').lower()

        with self.lock:
            if k in STICK_KEYS or k in SIM_REMOTE_BUTTON_KEYS:
                self.pressed.add(k)

        if self.control_handler:
            self.control_handler(k)

    def _on_release(self, key) -> None:
        try:
            if hasattr(key, 'char') and key.char:
                k = key.char.lower()
            else:
                k = str(key).replace('Key.', '').lower()
        except AttributeError:
            k = str(key).replace('Key.', '').lower()

        with self.lock:
            self.pressed.discard(k)

    def start(self) -> None:
        self.listener.start()

    def stop(self) -> None:
        self.listener.stop()

    def stick_keys(self) -> set[str]:
        with self.lock:
            return self.pressed & STICK_KEYS

    def active_keys(self) -> set[str]:
        with self.lock:
            return self.pressed.copy()

    def set_control_handler(self, handler) -> None:
        self.control_handler = handler


@dataclass(frozen=True)
class RuntimeConfig:
    robot: RobotModel
    net: str
    viewer: str
    band_sites: tuple[str, ...]
    band_enabled: bool


class LoopTimer:
    def __init__(self, hz: int):
        self.dt = 1.0 / hz
        self.next_t = time.perf_counter() + self.dt

    def sleep(self) -> None:
        now = time.perf_counter()
        if self.next_t > now:
            time.sleep(self.next_t - now)
            self.next_t += self.dt
        else:
            self.next_t = now + self.dt


def quat_to_rpy(q: np.ndarray) -> list[float]:
    w, x, y, z = [float(v) for v in q]
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return [roll, pitch, yaw]


class SimBridge:
    """MuJoCo <-> Unitree DDS bridge used for sim2sim validation.

    Data flow:
      LowCmd/MotorCmds -> MuJoCo PD control -> physics step
      -> LowState/MotorStates/Odom DDS topics
      keyboard -> wireless_remote bytes and optional suspension-band controls
    """

    def __init__(self, config: RuntimeConfig):
        self.config = config
        self.alive = True
        self.base_command_received = False
        self.z1_command_received = False
        self.simulation_paused = True
        self.tick = 1
        self.mode_machine = HG_MODE_MACHINE
        self.mode_pr = HG_MODE_PR
        self.dds = resolve_low_level_dds(self.config.robot.name)
        self.keyboard = KeyboardState()
        self.lock = threading.Lock()
        self.cmd_lock = threading.Lock()

        self.model = mujoco.MjModel.from_xml_path(str(self.config.robot.xml_path))
        self.model.opt.timestep = 1.0 / SIM_HZ
        self.data = mujoco.MjData(self.model)
        # Viewer backend owns GUI/server lifecycle; the physics loop only calls sync().
        self.viewer_backend = create_viewer_backend(
            self.config.viewer,
            self.config.robot,
            self.model,
            self.data,
            sim_hz=SIM_HZ,
            render_hz=RENDER_HZ,
            log=log,
        )
        self.num_motor = int(self.model.nu)
        self.motor_names = self.actuator_names()
        self.z1_motor_indices = np.asarray(
            [i for i, name in enumerate(self.motor_names) if name.startswith("z1_")],
            dtype=np.int64,
        )
        self.base_motor_indices = np.asarray(
            [i for i, name in enumerate(self.motor_names) if not name.startswith("z1_")],
            dtype=np.int64,
        )
        self.motor_joint_ids = self.actuator_joint_ids()
        self.motor_qposadr = self.model.jnt_qposadr[self.motor_joint_ids].astype(np.int64)
        self.motor_dofadr = self.model.jnt_dofadr[self.motor_joint_ids].astype(np.int64)
        # Actuator ctrlrange is the final safety clamp before writing data.ctrl.
        self.ctrl_lower = self.model.actuator_ctrlrange[:, 0].copy()
        self.ctrl_upper = self.model.actuator_ctrlrange[:, 1].copy()

        self.initial_qpos = self.make_initial_qpos()
        self.initial_joint_qpos = self.initial_qpos[self.motor_qposadr].copy()
        self.reset_sim(print_log=False)

        self.target_q = self.initial_joint_qpos.copy()
        self.target_dq = np.zeros(self.num_motor, dtype=np.float64)
        self.kp = np.zeros(self.num_motor, dtype=np.float64)
        self.kd = np.zeros(self.num_motor, dtype=np.float64)
        self.tau_ff = np.zeros(self.num_motor, dtype=np.float64)
        self.motor_enable = np.zeros(self.num_motor, dtype=bool)
        self.ctrl = np.zeros(self.num_motor, dtype=np.float64)

        self.imu_gyro = self.sensor_slice(
            GYRO_SENSOR_NAMES,
            mujoco.mjtSensor.mjSENS_GYRO,
            label="IMU gyro",
            required=False,
        )
        self.imu_acc = self.sensor_slice(
            ACC_SENSOR_NAMES,
            mujoco.mjtSensor.mjSENS_ACCELEROMETER,
            label="IMU accelerometer",
            required=False,
        )
        # secondary imu is specially for G1
        self.secondary_imu_quat = None
        self.secondary_imu_gyro = None
        self.secondary_imu_acc = None
        if self.dds.secondary_imu_topic is not None:
            self.secondary_imu_quat = self.named_sensor_slice(
                "secondary_imu_quat",
                mujoco.mjtSensor.mjSENS_FRAMEQUAT,
                label="secondary IMU quaternion",
                min_dim=4,
            )
            self.secondary_imu_gyro = self.named_sensor_slice(
                "secondary_imu_gyro",
                mujoco.mjtSensor.mjSENS_GYRO,
                label="secondary IMU gyro",
                min_dim=3,
            )
            self.secondary_imu_acc = self.named_sensor_slice(
                "secondary_imu_acc",
                mujoco.mjtSensor.mjSENS_ACCELEROMETER,
                label="secondary IMU accelerometer",
                min_dim=3,
            )
        self.crc = CRC()

        self.band_site_ids = (
            self.site_ids(self.config.band_sites) if self.config.band_enabled else []
        )
        self.band_on = self.config.band_enabled and bool(self.band_site_ids)
        self.band_jacp = np.zeros((3, self.model.nv), dtype=np.float64)
        self.band_zero = np.zeros(3, dtype=np.float64)
        self.band_anchors = self.make_band_anchors()
        self.band_start_anchors = self.band_anchors.copy()
        self.band_z = float(np.mean(self.band_anchors[:, 2])) if self.band_on else 0.0
        self.band_start_z = self.band_z

        self.lowstate_pub = ChannelPublisher(self.dds.lowstate_topic, self.dds.lowstate_type)
        self.lowstate_pub.Init()
        self.secondary_imu_pub = None
        if self.dds.secondary_imu_topic is not None and self.dds.secondary_imu_type is not None:
            self.secondary_imu_pub = ChannelPublisher(
                self.dds.secondary_imu_topic,
                self.dds.secondary_imu_type,
            )
            self.secondary_imu_pub.Init()
        self.odom_pub = ChannelPublisher(ODOM_TOPIC, SportModeState_)
        self.odom_pub.Init()
        self.lowcmd_sub = ChannelSubscriber(self.dds.lowcmd_topic, self.dds.lowcmd_type)
        self.lowcmd_sub.Init(self.on_lowcmd)
        self.z1_lowstate_pub = None
        self.z1_lowcmd_sub = None
        if self.z1_motor_indices.size:
            self.z1_lowstate_pub = ChannelPublisher(Z1_LOWSTATE_TOPIC, MotorStates_)
            self.z1_lowstate_pub.Init()
            self.z1_lowcmd_sub = ChannelSubscriber(Z1_LOWCMD_TOPIC, MotorCmds_)
            self.z1_lowcmd_sub.Init(self.on_z1_lowcmd)

        self.state_thread = threading.Thread(target=self.publish_state_loop, daemon=False)

        self.keyboard.set_control_handler(self.handle_control_key)
        self.keyboard.start()

        signal.signal(signal.SIGINT, self.close)
        signal.signal(signal.SIGTERM, self.close)

    # ----- MuJoCo model lookup helpers -----

    def actuator_names(self) -> list[str]:
        return [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or ""
            for i in range(self.num_motor)
        ]

    def actuator_joint_ids(self) -> np.ndarray:
        joint_ids = np.zeros(self.num_motor, dtype=np.int32)
        for i in range(self.num_motor):
            trn_type = int(self.model.actuator_trntype[i])
            if trn_type != int(mujoco.mjtTrn.mjTRN_JOINT):
                name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                raise ValueError(f"actuator {name or i!r} must use joint transmission")
            joint_id = int(self.model.actuator_trnid[i, 0])
            joint_type = int(self.model.jnt_type[joint_id])
            if joint_type not in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)):
                joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
                raise ValueError(f"actuated joint {joint_name or joint_id!r} must be 1-DoF")
            joint_ids[i] = joint_id
        return joint_ids

    def make_initial_qpos(self) -> np.ndarray:
        home_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if home_id < 0 and self.model.nkey == 1:
            home_id = 0
        if home_id >= 0:
            return self.model.key_qpos[home_id].copy()

        qpos = np.zeros(self.model.nq, dtype=np.float64)
        qpos[:7] = np.array([0.0, 0.0, BASE_HEIGHT, *BASE_QUAT], dtype=np.float64)
        return qpos

    def sensor_slice_by_id(self, sid: int, label: str, required: bool):
        adr = int(self.model.sensor_adr[sid])
        dim = int(self.model.sensor_dim[sid])
        if required and dim < 3:
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SENSOR, sid)
            raise ValueError(f"MuJoCo {label} sensor {name} must have dim >= 3")
        if dim < 3:
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SENSOR, sid)
            log(f"ignoring {label} sensor {name}: dim={dim} < 3")
            return None
        return adr, dim

    def sensor_slice(
        self,
        names: tuple[str, ...],
        sensor_type: mujoco.mjtSensor,
        label: str,
        required: bool,
    ):
        for name in names:
            sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
            if sid >= 0:
                return self.sensor_slice_by_id(int(sid), label, required)

        for sid in range(self.model.nsensor):
            if int(self.model.sensor_type[sid]) == int(sensor_type):
                return self.sensor_slice_by_id(int(sid), label, required)

        if required:
            raise ValueError(
                f"MuJoCo XML missing required {label} sensor; tried names: {', '.join(names)}"
            )
        log(f"MuJoCo XML has no {label} sensor; using fallback data")
        return None

    def named_sensor_slice(
        self,
        name: str,
        sensor_type: mujoco.mjtSensor,
        *,
        label: str,
        min_dim: int,
    ) -> tuple[int, int]:
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        if sid < 0:
            raise ValueError(f"MuJoCo XML missing required {label} sensor {name!r}")
        if int(self.model.sensor_type[sid]) != int(sensor_type):
            raise ValueError(f"MuJoCo {label} sensor {name!r} has the wrong sensor type")
        adr = int(self.model.sensor_adr[sid])
        dim = int(self.model.sensor_dim[sid])
        if dim < min_dim:
            raise ValueError(f"MuJoCo {label} sensor {name!r} has dim={dim}, expected >= {min_dim}")
        return adr, dim

    def site_ids(self, names: tuple[str, ...]) -> list[int]:
        site_ids = []
        missing = []
        for name in names:
            site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
            if site_id < 0:
                missing.append(name)
            else:
                site_ids.append(int(site_id))

        if missing:
            log(f"MuJoCo XML missing suspension band site(s), skipping: {', '.join(missing)}")
        if names and not site_ids:
            log("suspension bands disabled for this XML")
        return site_ids

    # ----- Reset and suspension-band setup -----

    def make_band_anchors(self) -> np.ndarray:
        if not self.band_site_ids:
            return np.zeros((0, 3), dtype=np.float64)

        anchors = np.zeros((len(self.band_site_ids), 3), dtype=np.float64)
        mujoco.mj_forward(self.model, self.data)
        for i, site_id in enumerate(self.band_site_ids):
            anchors[i] = self.data.site_xpos[site_id]
            anchors[i, 2] += BAND_CLEARANCE
        return anchors

    def reset_sim(self, print_log: bool = True) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.initial_qpos
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        if hasattr(self, "target_q"):
            with self.cmd_lock:
                self.target_q[:] = self.initial_joint_qpos
                self.target_dq[:] = 0.0
                self.kp[:] = 0.0
                self.kd[:] = 0.0
                self.tau_ff[:] = 0.0
                self.motor_enable[:] = False
            if self.band_on:
                self.band_z = self.band_start_z
                self.band_anchors[:] = self.band_start_anchors

        if print_log:
            log("reset robot state to initial pose")

    # ----- DDS input: LowCmd from controller.py -----

    def update_motor_command(self, motor_index: int, cmd) -> None:
        self.target_q[motor_index] = float(cmd.q)
        self.target_dq[motor_index] = float(cmd.dq)
        self.kp[motor_index] = float(cmd.kp)
        self.kd[motor_index] = float(cmd.kd)
        self.tau_ff[motor_index] = float(cmd.tau)
        self.motor_enable[motor_index] = int(getattr(cmd, "mode", 1)) != 0

    def on_lowcmd(self, msg) -> None:
        if msg is None:
            return

        with self.cmd_lock:
            self.mode_machine = int(getattr(msg, "mode_machine", self.mode_machine))
            self.mode_pr = int(getattr(msg, "mode_pr", self.mode_pr))
            motor_count = min(self.base_motor_indices.size, len(msg.motor_cmd))
            for message_index in range(motor_count):
                motor_index = self.base_motor_indices[message_index]
                cmd = msg.motor_cmd[message_index]
                self.update_motor_command(motor_index, cmd)
            self.base_command_received = True

    def on_z1_lowcmd(self, msg) -> None:
        if msg is None:
            return

        with self.cmd_lock:
            motor_count = min(self.z1_motor_indices.size, len(msg.cmds))
            for message_index in range(motor_count):
                motor_index = self.z1_motor_indices[message_index]
                cmd = msg.cmds[message_index]
                self.update_motor_command(motor_index, cmd)
            self.z1_command_received = True

    # ----- Keyboard controls and virtual wireless remote -----

    def handle_control_key(self, key: str) -> None:
        if key == "space":
            self.toggle_simulation_pause()
        elif key == "up":
            self.move_band(BAND_STEP)
        elif key == "down":
            self.move_band(-BAND_STEP)
        elif key == "n":
            self.toggle_band()
        elif key == "r":
            with self.lock:
                self.reset_sim()
        elif key == "esc":
            self.close()

    def toggle_simulation_pause(self) -> None:
        self.simulation_paused = not self.simulation_paused
        state = "paused" if self.simulation_paused else "resumed"
        log(f"simulation {state} by space")

    def move_band(self, dz: float) -> None:
        if not self.band_on:
            return
        self.band_z = float(np.clip(self.band_z + dz, BAND_MIN_Z, BAND_MAX_Z))
        self.band_anchors[:, 2] = self.band_z
        log(f"band height -> {self.band_z:.3f} m")

    def toggle_band(self) -> None:
        if not self.band_site_ids:
            log("suspension bands unavailable")
            return
        self.band_on = not self.band_on
        if self.band_on:
            self.band_anchors[:, 2] = self.band_z
            log(f"suspension bands restored at z={self.band_z:.3f} m")
        else:
            log("suspension bands released")

    def remote_bytes(self) -> list[int]:
        remote = bytearray(40)
        keys = self.current_keys()
        stick_keys = keys & STICK_KEYS

        # Match Unitree wireless_remote layout consumed by controller.RemoteCommand.
        for offset, value in zip(
            (4, 8, 12, 20),
            (
                float("d" in stick_keys) - float("a" in stick_keys),
                float("e" in stick_keys) - float("q" in stick_keys),
                0.0,
                float("w" in stick_keys) - float("s" in stick_keys),
            ),
        ):
            struct.pack_into("<f", remote, offset, value)

        for key, (byte_i, bit_i) in SIM_REMOTE_BUTTON_KEYS.items():
            if key in keys:
                remote[byte_i] |= 1 << bit_i
        return list(remote)

    def current_keys(self) -> set[str]:
        return self.keyboard.active_keys()

    def current_stick_keys(self) -> set[str]:
        return self.keyboard.stick_keys()

    def current_command(self) -> tuple[float, float, float]:
        stick_keys = self.current_stick_keys()
        lx = float("a" in stick_keys) - float("d" in stick_keys)
        rx = float("q" in stick_keys) - float("e" in stick_keys)
        ly = float("w" in stick_keys) - float("s" in stick_keys)
        return ly, -lx, -rx

    # ----- Physics step helpers -----

    def apply_band(self) -> None:
        if not self.band_on:
            return

        for site_id, anchor in zip(self.band_site_ids, self.band_anchors):
            pos = self.data.site_xpos[site_id]
            self.band_jacp[:] = 0.0
            mujoco.mj_jacSite(self.model, self.data, self.band_jacp, None, site_id)
            vel = self.band_jacp @ self.data.qvel
            force = BAND_STIFFNESS * (anchor - pos) - BAND_DAMPING * vel
            force = np.clip(force, -BAND_MAX_FORCE, BAND_MAX_FORCE)
            body_id = int(self.model.site_bodyid[site_id])
            mujoco.mj_applyFT(
                self.model,
                self.data,
                force,
                self.band_zero,
                pos,
                body_id,
                self.data.qfrc_applied,
            )

    def compute_ctrl(self) -> np.ndarray:
        # PD target arrays are written by the DDS callback; this runs inside the MuJoCo step loop.
        q = self.data.qpos[self.motor_qposadr]
        dq = self.data.qvel[self.motor_dofadr]
        with self.cmd_lock:
            self.ctrl[:] = (
                self.kp * (self.target_q - q)
                + self.kd * (self.target_dq - dq)
                + self.tau_ff
            )
            self.ctrl[~self.motor_enable] = 0.0
        np.clip(self.ctrl, self.ctrl_lower, self.ctrl_upper, out=self.ctrl)
        return self.ctrl

    # ----- DDS output: simulated robot state -----

    def state_snapshot(self):
        with self.lock:
            qpos = self.data.qpos.copy()
            qvel = self.data.qvel.copy()
            ctrl = self.data.ctrl[: self.num_motor].copy()
            sensordata = self.data.sensordata.copy()

        gyro = qvel[3:6]
        if self.imu_gyro is not None:
            adr, dim = self.imu_gyro
            if dim >= 3:
                gyro = sensordata[adr : adr + 3]
        acc = np.zeros(3, dtype=np.float64)
        if self.imu_acc is not None:
            acc_adr, dim = self.imu_acc
            if dim >= 3:
                acc = sensordata[acc_adr : acc_adr + 3]
        secondary_imu = None
        if self.secondary_imu_quat is not None:
            quat_adr, _ = self.secondary_imu_quat
            gyro_adr, _ = self.secondary_imu_gyro
            acc_adr, _ = self.secondary_imu_acc
            secondary_imu = (
                sensordata[quat_adr : quat_adr + 4],
                sensordata[gyro_adr : gyro_adr + 3],
                sensordata[acc_adr : acc_adr + 3],
            )
        return qpos, qvel, ctrl, gyro, acc, secondary_imu

    @staticmethod
    def fill_imu_state(imu_state, quat, gyro, acc) -> None:
        imu_state.quaternion = quat.tolist()
        imu_state.gyroscope = gyro.tolist()
        imu_state.accelerometer = acc.tolist()
        if hasattr(imu_state, "rpy"):
            imu_state.rpy = quat_to_rpy(quat)

    @classmethod
    def fill_imu(cls, msg, quat, gyro, acc) -> None:
        cls.fill_imu_state(msg.imu_state, quat, gyro, acc)

    def make_lowstate(self, qpos, qvel, ctrl, gyro, acc):
        msg = self.dds.make_lowstate()
        if self.dds.has_mode_fields:
            msg.mode_pr = int(self.mode_pr)
            msg.mode_machine = int(self.mode_machine)
        msg.tick = int(self.tick)

        motor_count = min(self.base_motor_indices.size, len(msg.motor_state))
        for message_index in range(motor_count):
            motor_index = self.base_motor_indices[message_index]
            msg.motor_state[message_index].q = float(qpos[self.motor_qposadr[motor_index]])
            msg.motor_state[message_index].dq = float(qvel[self.motor_dofadr[motor_index]])
            msg.motor_state[message_index].tau_est = float(ctrl[motor_index])

        self.fill_imu(msg, qpos[3:7], gyro, acc)
        msg.wireless_remote = self.remote_bytes()
        msg.crc = self.crc.Crc(msg)

        self.tick += 1
        return msg

    def make_secondary_imu(self, quat, gyro, acc):
        if self.dds.make_secondary_imu is None:
            raise RuntimeError("secondary IMU is not supported by this robot")
        msg = self.dds.make_secondary_imu()
        self.fill_imu_state(msg, quat, gyro, acc)
        return msg

    def make_z1_lowstate(self, qpos, qvel, ctrl) -> MotorStates_:
        msg = MotorStates_()
        for motor_index in self.z1_motor_indices:
            state = unitree_go_msg_dds__MotorState_()
            state.mode = 1
            state.q = float(qpos[self.motor_qposadr[motor_index]])
            state.dq = float(qvel[self.motor_dofadr[motor_index]])
            state.tau_est = float(ctrl[motor_index])
            state.temperature = 0
            state.lost = 0
            msg.states.append(state)
        return msg

    def make_odom(self, qpos, qvel, gyro, acc) -> SportModeState_:
        msg = unitree_go_msg_dds__SportModeState_()
        msg.position = qpos[:3].tolist()
        msg.velocity = qvel[:3].tolist()
        self.fill_imu(msg, qpos[3:7], gyro, acc)
        return msg

    def publish_state_loop(self) -> None:
        timer = LoopTimer(STATE_HZ)
        while self.alive:
            qpos, qvel, ctrl, gyro, acc, secondary_imu = self.state_snapshot()
            self.lowstate_pub.Write(self.make_lowstate(qpos, qvel, ctrl, gyro, acc))
            if self.secondary_imu_pub is not None and secondary_imu is not None:
                self.secondary_imu_pub.Write(self.make_secondary_imu(*secondary_imu))
            if self.z1_lowstate_pub is not None:
                self.z1_lowstate_pub.Write(self.make_z1_lowstate(qpos, qvel, ctrl))
            self.odom_pub.Write(self.make_odom(qpos, qvel, gyro, acc))
            timer.sleep()

    # ----- Simulation loop and cleanup -----

    def simulate(self) -> None:
        timer = LoopTimer(SIM_HZ)
        last_log = time.perf_counter()
        steps = 0

        while self.alive:
            if not self.simulation_paused:
                with self.lock:
                    self.data.qfrc_applied[:] = 0.0
                    self.apply_band()
                    self.data.ctrl[:] = self.compute_ctrl()
                    mujoco.mj_step(self.model, self.data)

            if not self.viewer_backend.sync():
                self.alive = False
                break

            if not self.simulation_paused:
                steps += 1
            now = time.perf_counter()
            if now - last_log >= 1.0:
                if self.simulation_paused:
                    status(
                        [
                            ("state", "paused", "yellow"),
                            ("hint", "press space", "white"),
                            ("band", "on" if self.band_on else "off", "green" if self.band_on else "red"),
                        ]
                    )
                else:
                    command = self.current_command()
                    fields = [
                        ("state", "running", "green"),
                        ("t", f"{steps / SIM_HZ:6.2f}s", "cyan"),
                        ("height", f"{self.data.qpos[2]:.3f}m", "magenta"),
                        (
                            "remote",
                            f"{command[0]:+.2f} {command[1]:+.2f} {command[2]:+.2f}",
                            "white",
                        ),
                        (
                            "base_cmd",
                            "yes" if self.base_command_received else "no",
                            "green" if self.base_command_received else "yellow",
                        ),
                    ]
                    if self.z1_motor_indices.size:
                        fields.append(
                            (
                                "z1_cmd",
                                "yes" if self.z1_command_received else "no",
                                "green" if self.z1_command_received else "yellow",
                            )
                        )
                    fields.append(
                        (
                            "band",
                            "on" if self.band_on else "off",
                            "green" if self.band_on else "red",
                        )
                    )
                    status(fields)
                last_log = now
            timer.sleep()

    def run(self) -> None:
        log(
            f"robot={self.config.robot.name} terrain={self.config.robot.terrain} "
            f"model={self.config.robot.xml_path}"
        )
        log(
            f"dds={self.dds.type} "
            f"topics: lowcmd={self.dds.lowcmd_topic}, lowstate={self.dds.lowstate_topic}, "
            f"secondary_imu={self.dds.secondary_imu_topic or 'none'}, odom={ODOM_TOPIC}"
        )
        log(
            f"actuators: base={self.base_motor_indices.size}, "
            f"z1={self.z1_motor_indices.size}"
        )
        if self.z1_motor_indices.size:
            log(f"z1 topics: lowcmd={Z1_LOWCMD_TOPIC}, lowstate={Z1_LOWSTATE_TOPIC}")
        log(f"sim={SIM_HZ}Hz state_pub={STATE_HZ}Hz viewer={self.config.viewer}")
        log(f"simulation starts paused; press \"space\" to continue")
        if self.band_on:
            log(f"suspension bands enabled at z={self.band_z:.3f} m")

        self.state_thread.start()
        # Blocks until the selected viewer or simulation loop exits.
        self.viewer_backend.run(self.simulate)

    def close(self, *_args) -> None:
        if self.alive:
            log("shutting down...")
        self.alive = False
        console.stop()
        self.keyboard.stop()

        if self.state_thread.is_alive() and threading.current_thread() is not self.state_thread:
            self.state_thread.join(timeout=1.0)

    def cleanup(self) -> None:
        self.alive = False
        for obj in (
            self.lowcmd_sub,
            self.lowstate_pub,
            self.secondary_imu_pub,
            self.odom_pub,
            self.z1_lowcmd_sub,
            self.z1_lowstate_pub,
        ):
            if obj is None:
                continue
            try:
                obj.Close()
            except Exception:
                pass


def parse_args() -> RuntimeConfig:
    parser = argparse.ArgumentParser(description="MuJoCo to Unitree DDS bridge.")
    parser.add_argument("--robot", default=DEFAULT_ROBOT, help="Robot folder under robot_model/.")
    parser.add_argument("--model-xml", help="Override robot XML path.")
    parser.add_argument(
        "--terrain",
        default=DEFAULT_TERRAIN,
        help="Terrain name under robot_model/scene or XML path.",
    )
    parser.add_argument("--net", default=DEFAULT_NET, help="DDS network interface.")
    parser.add_argument(
        "--viewer",
        choices=VIEWER_CHOICES,
        default=DEFAULT_VIEWER,
        help="Visualization backend.",
    )
    parser.add_argument(
        "--band-sites",
        default=",".join(BAND_SITES),
        help="Comma-separated MuJoCo site names used by the suspension bands.",
    )
    parser.add_argument(
        "--band",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable suspension bands.",
    )
    args = parser.parse_args()
    band_sites = tuple(site.strip() for site in args.band_sites.split(",") if site.strip())
    return RuntimeConfig(
        robot=load_robot_model(args.robot, args.model_xml, args.terrain),
        net=args.net,
        viewer=args.viewer,
        band_sites=band_sites,
        band_enabled=bool(args.band),
    )


def main() -> None:
    config = parse_args()
    ChannelFactoryInitialize(0, config.net)
    bridge = SimBridge(config)
    try:
        bridge.run()
    finally:
        bridge.close()
        bridge.cleanup()


if __name__ == "__main__":
    main()

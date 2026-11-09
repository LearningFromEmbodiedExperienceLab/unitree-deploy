import signal
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from unitree_deploy.config.defaults import (
    CONTROLLER_STATE_TOPIC,
    DAMPING_STATE,
    HG_MODE_MACHINE,
    HG_MODE_PR,
    ODOM_TOPIC,
    RUN_POLICY_STATE,
    SIM_PAUSE_TOPIC,
    SIM_SPEED_TOPIC,
    WIRELESS_REMOTE_BUTTON_BITS,
    sim_key_for_button,
)
from unitree_deploy.obs.base_obs import ObservationContext
from unitree_deploy.robot_model.robot_config import DEFAULT_ROBOT
from unitree_deploy.visualization.policy_debug_viz import POLICY_DEBUG_VIZ_TOPIC, policy_debug_viz_packet
from unitree_deploy.utils.yaml_utils import load_yaml
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
from unitree_deploy.runtime.controllers.checkpoints import PolicyManager
from unitree_deploy.runtime.timing import LoopTimer
from unitree_deploy.runtime.transport.remote import RemoteCommand
from unitree_deploy.runtime.state_machine import (
    ControllerStateMachine,
    load_state_machine_config,
    resolve_state_machine_path,
)
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.core.channel import (
    ChannelPublisher,
    ChannelSubscriber,
)
from unitree_sdk2py.utils.crc import CRC
from unitree_deploy.runtime.transport.dds import resolve_low_level_dds
from unitree_deploy.utils.terminal_status import ComponentConsole


console = ComponentConsole("controller", "bright_blue")


def log(message: str) -> None:
    console.log(message)


def status(fields) -> None:
    console.status(fields)


@dataclass(frozen=True)
class RuntimeConfig:
    mode: str
    net: str | None
    ckpt_dir: Path
    robot: str | None
    multi_ckpt: Path | None = None
    state_machine: Path | None = None
    motion_path: Path | None = None


class Controller:
    """Real-time Unitree controller shared by sim and real deployment.

    Data flow:
      LowState -> policy joint order -> observation -> ONNX policy -> raw joint order -> LowCmd
    """

    def __init__(self, config: RuntimeConfig):
        self.config = config
        self.ckpt_dir = config.ckpt_dir.expanduser().resolve()
        self.policy_manager = PolicyManager.load(self.ckpt_dir, config.multi_ckpt, motion_path=config.motion_path)
        if config.multi_ckpt is None:
            self.ckpt_dir = self.active_profile.policy_yaml_path.parent
        self.robot = config.robot or self.active_profile.policy.config.get("robot", DEFAULT_ROBOT)
        self.dds = resolve_low_level_dds(self.robot)

        self.lowcmd_topic = self.dds.lowcmd_topic
        self.lowstate_topic = self.dds.lowstate_topic
        self.sdk_joint_order = list(self.active_profile.sdk_joint_order)
        self.obs_joint_order = list(self.active_profile.obs_joint_order)
        self.num_joints = len(self.sdk_joint_order)
        self.raw_command = self.active_profile.command_default.copy()
        self.zero = np.zeros(self.num_joints, dtype=np.float64)
        self.target_sdk = np.zeros(self.num_joints, dtype=np.float32)

        self.lock = threading.Lock()
        self.alive = True
        self.cleanup_done = False
        self.has_low_state = False
        self.mode_machine = HG_MODE_MACHINE
        self.mode_pr = HG_MODE_PR
        self.state = DAMPING_STATE
        self.state_enter_t = time.perf_counter()

        self.q = np.zeros(self.num_joints, dtype=np.float64)
        self.dq = np.zeros(self.num_joints, dtype=np.float64)
        self.quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.gyro = np.zeros(3, dtype=np.float64)
        self.torso_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.torso_gyro = np.zeros(3, dtype=np.float64)
        self.has_torso_imu = False
        self.base_pos_w = np.zeros(3)
        self.base_lin_vel_w = np.zeros(3)
        self.odom_received_at = None
        self.command = self.active_profile.command_default.copy()
        self.last_policy_command = self.active_profile.command_default.copy()
        self.remote = RemoteCommand()
        self.log = log

        if config.mode == "real":
            self.enter_debug_mode()

        state_machine_base = config.multi_ckpt.parent if config.multi_ckpt else self.ckpt_dir
        state_machine_path = resolve_state_machine_path(state_machine_base, config.state_machine)
        state_machine_config = load_state_machine_config(state_machine_path)
        self.state_machine_path = state_machine_path

        self.low_cmd = self.dds.make_lowcmd()
        self.crc = CRC()
        self.lowstate_sub = ChannelSubscriber(self.lowstate_topic, self.dds.lowstate_type)
        self.lowstate_sub.Init(self.on_lowstate, 1)
        self.secondary_imu_sub = None
        if any(profile.uses_secondary_imu for profile in self.policy_manager.profiles.values()):
            if self.dds.secondary_imu_topic is None or self.dds.secondary_imu_type is None:
                raise ValueError(f"robot {self.robot!r} does not provide a secondary torso IMU")
            self.secondary_imu_sub = ChannelSubscriber(
                self.dds.secondary_imu_topic,
                self.dds.secondary_imu_type,
            )
            self.secondary_imu_sub.Init(self.on_secondary_imu, 1)
        self.odom_sub = None
        # Sim references need the entry position even when the actor does not use odometry.
        if config.mode == "sim" or any(
            getattr(profile.policy, "requires_odometry", False)
            for profile in self.policy_manager.profiles.values()
        ):
            from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_

            self.odom_sub = ChannelSubscriber(ODOM_TOPIC, SportModeState_)
            self.odom_sub.Init(self.on_odometry, 1)
        self.lowcmd_pub = ChannelPublisher(self.lowcmd_topic, self.dds.lowcmd_type)
        self.lowcmd_pub.Init()

        self.controller_state_pub = None
        if config.mode == "sim":
            self.controller_state_pub = ChannelPublisher(CONTROLLER_STATE_TOPIC, String_)
            self.controller_state_pub.Init()

        self.policy_debug_viz_pub = None
        self.motion_ghost_settings = {}
        if config.mode == "sim":
            if config.multi_ckpt is not None:
                self.motion_ghost_settings = load_yaml(config.multi_ckpt).get("motion_ghost") or {}
            self.policy_debug_viz_pub = ChannelPublisher(POLICY_DEBUG_VIZ_TOPIC, String_)
            self.policy_debug_viz_pub.Init()

        signal.signal(signal.SIGINT, self.close)
        signal.signal(signal.SIGTERM, self.close)

        self.state_machine = ControllerStateMachine(self, state_machine_config)
        self.state = self.state_machine.current_name
        self.simulation_paused = config.mode == "sim"
        self.simulation_pause_started = time.perf_counter() if self.simulation_paused else None
        self.simulation_speed = 1.0
        self.simulation_speed_sub = None
        self.simulation_pause_sub = None
        if config.mode == "sim":
            self.simulation_pause_sub = ChannelSubscriber(SIM_PAUSE_TOPIC, String_)
            self.simulation_pause_sub.Init(self.on_simulation_pause, 1)
            self.simulation_speed_sub = ChannelSubscriber(SIM_SPEED_TOPIC, String_)
            self.simulation_speed_sub.Init(self.on_simulation_speed, 1)

    @property
    def active_profile(self):
        return self.policy_manager.active

    @property
    def active_profile_name(self) -> str:
        return self.policy_manager.active_name

    def reorder_policy_to_sdk(self, value: np.ndarray) -> np.ndarray:
        np.take(value, self.active_profile.obs_to_sdk, out=self.target_sdk)
        return self.target_sdk

    def update_command_from_remote(self) -> None:
        profile = self.active_profile
        self.raw_command[:] = profile.command_default
        joystick_command = np.asarray(
            [self.remote.ly, -self.remote.lx, -self.remote.rx],
            dtype=np.float64,
        )
        copy_dim = min(self.raw_command.size, joystick_command.size)
        self.raw_command[:copy_dim] = joystick_command[:copy_dim]
        np.clip(self.raw_command, profile.command_min, profile.command_max, out=self.command)

    # ----- Real-robot setup -----

    def enter_debug_mode(self) -> None:
        log("real mode: releasing current motion mode...")
        msc = MotionSwitcherClient()
        msc.SetTimeout(5.0)
        msc.Init()

        _, result = msc.CheckMode()
        while result["name"]:
            msc.ReleaseMode()
            _, result = msc.CheckMode()
            time.sleep(1.0)
        log("real mode: motion mode released")

    # ----- DDS input and controller state snapshot -----

    def on_simulation_speed(self, msg) -> None:
        with self.lock:
            self.simulation_speed = float(msg.data)

    def on_simulation_pause(self, msg) -> None:
        paused = msg.data == "paused"
        with self.lock:
            if paused and not self.simulation_paused:
                self.simulation_pause_started = time.perf_counter()
            elif not paused and self.simulation_paused:
                self.state_enter_t += time.perf_counter() - self.simulation_pause_started
                self.simulation_pause_started = None
            self.simulation_paused = paused

    def on_lowstate(self, msg) -> None:
        with self.lock:
            self.mode_machine = int(getattr(msg, "mode_machine", HG_MODE_MACHINE))
            self.mode_pr = int(getattr(msg, "mode_pr", HG_MODE_PR))

            for i in range(self.num_joints):
                state = msg.motor_state[i]
                self.q[i] = float(state.q)
                self.dq[i] = float(state.dq)

            self.quat[:] = np.asarray(msg.imu_state.quaternion[:4], dtype=np.float64)
            self.gyro[:] = np.asarray(msg.imu_state.gyroscope[:3], dtype=np.float64)
            self.remote.set(msg.wireless_remote)
            self.update_command_from_remote()
            self.has_low_state = True

    def on_secondary_imu(self, msg) -> None:
        with self.lock:
            self.torso_quat[:] = np.asarray(msg.quaternion[:4], dtype=np.float64)
            self.torso_gyro[:] = np.asarray(msg.gyroscope[:3], dtype=np.float64)
            self.has_torso_imu = True

    def on_odometry(self, msg) -> None:
        # rt/odommodestate: pelvis position and linear velocity in the IMU world frame.
        with self.lock:
            self.base_pos_w[:] = msg.position[:3]
            self.base_lin_vel_w[:] = msg.velocity[:3]
            self.odom_received_at = time.perf_counter()

    def observation(self) -> ObservationContext:
        with self.lock:
            profile = self.active_profile
            q = self.q[profile.sdk_to_obs].copy()
            dq = self.dq[profile.sdk_to_obs].copy()
            if profile.uses_secondary_imu:
                quat = self.torso_quat.copy()
                gyro = self.torso_gyro.copy()
            else:
                quat = self.quat.copy()
                gyro = self.gyro.copy()
            # Allow five policy steps of latency, with a 100 ms minimum window.
            odom_timeout = max(0.1, 5 * profile.policy.policy_step_dt)
            odom_fresh = (
                self.odom_received_at is not None
                and time.perf_counter() - self.odom_received_at < odom_timeout
            )
            return ObservationContext(
                q=q,
                dq=dq,
                quat=quat,
                gyro=gyro,
                command=self.command.copy(),
                base_pos_w=self.base_pos_w.copy() if odom_fresh else None,
                base_lin_vel_w=self.base_lin_vel_w.copy() if odom_fresh else None,
            )

    # ----- State-machine controls -----

    def button_pressed(self, name: str) -> bool:
        with self.lock:
            return self.remote.button_pressed(name)

    def transition(self, state: str, *, force: bool = False) -> None:
        self.state_machine.transition(state, force=force)

    def on_state_transition(self) -> None:
        if self.state_machine.current.reset_policy_on_enter:
            self.active_profile.policy.reset()

    def activate_policy(self, name: str) -> None:
        """Select a profile from a configured state, without manual switch restrictions."""
        if name == self.active_profile_name:
            return
        with self.lock:
            profile = self.policy_manager.switch_to(name)
            self.obs_joint_order = list(profile.obs_joint_order)
            self.update_command_from_remote()
        log(f"policy -> {name} ({profile.policy_yaml_path})")

    def switch_to_policy(self, name: str) -> bool:
        if name == self.active_profile_name:
            return False
        if self.state not in self.policy_manager.switch.only_when:
            allowed = ", ".join(sorted(self.policy_manager.switch.only_when))
            log(f"policy switch ignored in state={self.state}; allowed states: {allowed}")
            return False

        self.activate_policy(name)
        if self.policy_manager.switch.on_switch:
            self.transition(self.policy_manager.switch.on_switch, force=True)
        return True

    def switch_to_next_policy(self) -> bool:
        if not self.policy_manager.switch_allowed(self.state):
            allowed = ", ".join(sorted(self.policy_manager.switch.only_when))
            log(f"policy switch ignored in state={self.state}; allowed states: {allowed}")
            return False

        with self.lock:
            profile = self.policy_manager.switch_next()
            self.obs_joint_order = list(profile.obs_joint_order)
            self.update_command_from_remote()
        log(f"policy -> {profile.name} ({profile.policy_yaml_path})")
        if self.policy_manager.switch.on_switch:
            self.transition(self.policy_manager.switch.on_switch, force=True)
        return True

    # ----- DDS output -----

    def send_joint_cmd(
        self,
        target_q: np.ndarray,
        kp: np.ndarray,
        kd: np.ndarray,
        *,
        enable: bool = True,
        target_dq: np.ndarray | None = None,
        tau_ff: np.ndarray | None = None,
    ) -> None:
        target_dq = self.zero if target_dq is None else target_dq
        tau_ff = self.zero if tau_ff is None else tau_ff
        if self.dds.has_mode_fields:
            with self.lock:
                self.low_cmd.mode_pr = int(self.mode_pr)
                self.low_cmd.mode_machine = int(self.mode_machine)

        # Clear all motors first so any joints outside num_joints stay disabled.
        for cmd in self.low_cmd.motor_cmd:
            cmd.mode = 0
            cmd.q = cmd.dq = cmd.tau = cmd.kp = cmd.kd = 0.0

        for i in range(self.num_joints):
            cmd = self.low_cmd.motor_cmd[i]
            cmd.mode = 1 if enable else 0
            cmd.q = float(target_q[i])
            cmd.dq = float(target_dq[i])
            cmd.tau = float(tau_ff[i])
            cmd.kp = float(kp[i])
            cmd.kd = float(kd[i])

        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.lowcmd_pub.Write(self.low_cmd)

    # ----- State dispatch -----

    def step(self) -> None:
        policy = self.active_profile.policy
        set_timing = getattr(policy, "set_simulation_timing", None)
        with self.lock:
            paused = self.simulation_paused
            if set_timing is not None:
                speed = self.simulation_speed
        if set_timing is not None:
            set_timing(paused, speed)
        if not paused:
            if self.policy_manager.switch.enabled and self.button_pressed(self.policy_manager.switch.button):
                self.switch_to_next_policy()
            else:
                self.state_machine.step()
        if self.policy_debug_viz_pub is not None:
            policy = self.active_profile.policy if self.state_machine.current.state_type == "policy" else None
            self.policy_debug_viz_pub.Write(String_(policy_debug_viz_packet(
                policy, self.motion_ghost_settings, self.last_policy_command,
            )))

    # ----- Main loop and cleanup -----

    def spin(self) -> None:
        log(
            f"robot={self.robot} mode={self.config.mode} joints={self.num_joints} "
            f"dds={self.dds.type} "
            f"topics: lowstate={self.lowstate_topic}, lowcmd={self.lowcmd_topic}"
        )
        log(
            f"policy={self.active_profile_name} "
            f"imu={self.active_profile.imu_source or 'lowstate'} "
            f"available={','.join(self.policy_manager.profiles)}"
        )
        log(f"state_machine={self.state_machine_path or 'default'} initial={self.state_machine.initial}")
        if self.config.mode == "sim":
            switch_hint = (
                f", {sim_key_for_button(self.policy_manager.switch.button)} -> switch policy"
                if self.policy_manager.switch.enabled
                else ""
            )
            log(
                f"sim keymap: {sim_key_for_button('A')} -> A, "
                f"{sim_key_for_button('Start')} -> Start, "
                f"{sim_key_for_button('X')} -> Damping{switch_hint}, R -> reset sim"
            )
        if self.state_machine_path is None:
            control_hint = "A: damping/policy -> default pose, Start: default pose -> run policy, X: damping"
            if self.policy_manager.switch.enabled:
                control_hint += f", {self.policy_manager.switch.button}: switch policy"
            log(control_hint)
        log("waiting for lowstate...")

        timer = LoopTimer(float(self.active_profile.policy.policy_step_dt))
        last_log = time.perf_counter()

        while self.alive:
            with self.lock:
                lowstate_ready = self.has_low_state
                imu_ready = not self.active_profile.uses_secondary_imu or self.has_torso_imu
                ready = lowstate_ready and imu_ready
            if ready:
                self.step()
            if self.controller_state_pub is not None:
                self.controller_state_pub.Write(String_(self.state))

            now = time.perf_counter()
            if now - last_log >= 1.0:
                command = self.last_policy_command
                command_text = " ".join(f"{value:+.2f}" for value in command)
                state_style = "green" if self.state == RUN_POLICY_STATE else "yellow"
                status(
                    [
                        ("state", self.state, state_style),
                        ("policy", self.active_profile_name, "cyan"),
                        ("cmd", command_text, "white"),
                        (
                            "lowstate",
                            "yes" if lowstate_ready else "no",
                            "green" if lowstate_ready else "red",
                        ),
                        (
                            "imu",
                            (self.active_profile.imu_source or "lowstate")
                            if imu_ready
                            else "waiting for torso",
                            "green" if imu_ready else "red",
                        ),
                    ]
                )
                last_log = now
            # Each tracking inference advances one reference frame at this same rate.
            timer.sleep(self.simulation_speed)

    def close(self, *_args) -> None:
        self.alive = False

    def cleanup(self) -> None:
        if self.cleanup_done:
            return
        self.cleanup_done = True
        self.alive = False
        console.stop()
        self.lowstate_sub.Close()
        if self.simulation_pause_sub is not None:
            self.simulation_pause_sub.Close()
        if self.simulation_speed_sub is not None:
            self.simulation_speed_sub.Close()
        if self.secondary_imu_sub is not None:
            self.secondary_imu_sub.Close()
        if self.odom_sub is not None:
            self.odom_sub.Close()
        for profile in self.policy_manager.profiles.values():
            close = getattr(profile.policy, "close", None)
            if close is not None:
                close()
        self.lowcmd_pub.Close()
        if self.controller_state_pub is not None:
            self.controller_state_pub.Close()
        if self.policy_debug_viz_pub is not None:
            self.policy_debug_viz_pub.Write(String_(policy_debug_viz_packet(None, self.motion_ghost_settings)))
            self.policy_debug_viz_pub.Close()

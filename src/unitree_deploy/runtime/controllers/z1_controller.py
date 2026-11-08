from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from unitree_deploy.config.defaults import (
    DAMPING_STATE,
    LOWSTATE_TOPIC,
    Z1_CONTROLLER_STATE_TOPIC,
    Z1_LOCAL_PORT,
    Z1_LOWCMD_TOPIC,
    Z1_LOWSTATE_TOPIC,
    Z1_ROBOT_IP,
    Z1_UDP_NAMESPACE,
    Z1_UDP_SERVICE_PATH,
)
from unitree_deploy.runtime.timing import LoopTimer
from unitree_deploy.runtime.state_machine import ControllerStateMachine
from unitree_deploy.runtime.transport.remote import RemoteCommand
from unitree_deploy.utils.terminal_status import ComponentConsole
from unitree_deploy.utils.yaml_utils import load_yaml
from unitree_sdk2py.core.channel import (
    ChannelPublisher,
    ChannelSubscriber,
)
from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import (
    LowState_ as GoLowState_,
    MotorCmds_,
    MotorStates_,
)
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_


console = ComponentConsole("z1_controller", "magenta")


def log(message: str) -> None:
    console.log(message)


@dataclass(frozen=True)
class RuntimeConfig:
    mode: str
    net: str | None
    ckpt_dir: Path


@dataclass(frozen=True)
class Z1ControlProfile:
    sdk_joint_order: list[str]
    sdk_to_obs: np.ndarray
    obs_to_sdk: np.ndarray
    default_q_obs: np.ndarray
    kp_fixed_stand: np.ndarray
    kd_fixed_stand: np.ndarray
    kp_impedance: np.ndarray
    kd_impedance: np.ndarray
    kd_damping: np.ndarray
    control_dt: float
    state_timeout: float
    state_machine: dict[str, Any]


def load_z1_profile(path: Path) -> Z1ControlProfile:
    config = load_yaml(path)
    joint_order = [str(name) for name in config["sdk_joint_order"]]
    identity = np.arange(len(joint_order), dtype=np.int64)
    return Z1ControlProfile(
        sdk_joint_order=joint_order,
        sdk_to_obs=identity,
        obs_to_sdk=identity,
        default_q_obs=np.asarray(config["default_qpos"], dtype=np.float64),
        kp_fixed_stand=np.asarray(config["kp_fixed_stand"], dtype=np.float64),
        kd_fixed_stand=np.asarray(config["kd_fixed_stand"], dtype=np.float64),
        kp_impedance=np.asarray(config["kp_impedance"], dtype=np.float64),
        kd_impedance=np.asarray(config["kd_impedance"], dtype=np.float64),
        kd_damping=np.asarray(config["kd_damping"], dtype=np.float64),
        control_dt=float(config["control_dt"]),
        state_timeout=float(config["state_timeout"]),
        state_machine=config["state_machine"],
    )


def z1_udp_service_command() -> list[str]:
    return [
        Z1_UDP_SERVICE_PATH,
        "--ns",
        Z1_UDP_NAMESPACE,
        "--ip",
        Z1_ROBOT_IP,
        "--localport",
        str(Z1_LOCAL_PORT),
    ]


class Z1Controller:
    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config
        self.profile_path = config.ckpt_dir.expanduser().resolve() / "policy.yaml"
        self.profile = load_z1_profile(self.profile_path)
        self.num_joints = len(self.profile.sdk_joint_order)
        self.zero = np.zeros(self.num_joints, dtype=np.float64)
        self.target_sdk = np.zeros(self.num_joints, dtype=np.float64)

        self.lock = threading.Lock()
        self.q = np.zeros(self.num_joints, dtype=np.float64)
        self.dq = np.zeros(self.num_joints, dtype=np.float64)
        self.tau_est = np.zeros(self.num_joints, dtype=np.float64)
        self.temperature = np.zeros(self.num_joints, dtype=np.uint8)
        self.lost = np.zeros(self.num_joints, dtype=np.uint32)
        self.last_state_t = 0.0
        self.has_low_state = False
        self.remote = RemoteCommand()
        self.state = DAMPING_STATE
        self.state_enter_t = time.perf_counter()
        self.alive = True
        self.cleanup_done = False
        self.log = log
        self.bridge_process: subprocess.Popen | None = None

        self.low_cmd = MotorCmds_()
        self.low_cmd.cmds.extend(
            unitree_go_msg_dds__MotorCmd_() for _ in range(self.num_joints)
        )
        self.lowstate_sub = ChannelSubscriber(Z1_LOWSTATE_TOPIC, MotorStates_)
        self.lowstate_sub.Init(self.on_z1_state, 1)
        self.lowcmd_pub = ChannelPublisher(Z1_LOWCMD_TOPIC, MotorCmds_)
        self.lowcmd_pub.Init()
        self.controller_state_pub = None
        if config.mode == "sim":
            self.controller_state_pub = ChannelPublisher(Z1_CONTROLLER_STATE_TOPIC, String_)
            self.controller_state_pub.Init()
        self.base_state_sub = ChannelSubscriber(LOWSTATE_TOPIC, GoLowState_)
        self.base_state_sub.Init(self.on_base_state, 1)
        self.state_machine = ControllerStateMachine(self, self.profile.state_machine)
        self.state = self.state_machine.current_name

        signal.signal(signal.SIGINT, self.close)
        signal.signal(signal.SIGTERM, self.close)

    @property
    def active_profile(self) -> Z1ControlProfile:
        return self.profile

    def start_udp_bridge(self) -> None:
        executable = Path(Z1_UDP_SERVICE_PATH)
        if not executable.is_file():
            raise FileNotFoundError(f"Z1 UDP service not found: {executable}")
        if not os.access(executable, os.X_OK):
            raise PermissionError(f"Z1 UDP service is not executable: {executable}")
        command = z1_udp_service_command()
        log(f"starting Z1 UDP bridge: {' '.join(command)}")
        self.bridge_process = subprocess.Popen(command)

    def on_z1_state(self, msg) -> None:
        if msg is None or len(msg.states) < self.num_joints:
            return
        states = msg.states[: self.num_joints]
        with self.lock:
            self.q[:] = [state.q for state in states]
            self.dq[:] = [state.dq for state in states]
            self.tau_est[:] = [state.tau_est for state in states]
            self.temperature[:] = [state.temperature for state in states]
            self.lost[:] = [state.lost for state in states]
            self.last_state_t = time.perf_counter()
            self.has_low_state = True

    def on_base_state(self, msg) -> None:
        with self.lock:
            self.remote.set(msg.wireless_remote)

    def state_ready(self, now: float | None = None) -> bool:
        now = time.perf_counter() if now is None else float(now)
        with self.lock:
            return (
                self.has_low_state
                and now - self.last_state_t <= self.profile.state_timeout
                and not np.any(self.lost)
            )

    def button_pressed(self, name: str) -> bool:
        with self.lock:
            return self.remote.button_pressed(name)

    def reorder_policy_to_sdk(self, value: np.ndarray) -> np.ndarray:
        np.take(value, self.profile.obs_to_sdk, out=self.target_sdk)
        return self.target_sdk

    def on_state_transition(self) -> None:
        pass

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
        target_q = np.asarray(target_q, dtype=np.float64)
        target_dq = self.zero if target_dq is None else target_dq
        tau_ff = self.zero if tau_ff is None else tau_ff
        for index, cmd in enumerate(self.low_cmd.cmds):
            cmd.mode = 1 if enable else 0
            cmd.q = float(target_q[index])
            cmd.dq = float(target_dq[index])
            cmd.tau = float(tau_ff[index])
            cmd.kp = float(kp[index])
            cmd.kd = float(kd[index])
        self.lowcmd_pub.Write(self.low_cmd)

    def step(self) -> None:
        self.state_machine.step()

    def spin(self) -> None:
        if self.config.mode == "real":
            self.start_udp_bridge()
        log(
            f"mode={self.config.mode} joints={self.num_joints} "
            f"topics: lowstate={Z1_LOWSTATE_TOPIC}, lowcmd={Z1_LOWCMD_TOPIC}"
        )
        log(f"profile={self.profile_path}")
        log("A: move to default pose, Start: impedance, X: damping")
        log("waiting for Z1 lowstate...")

        timer = LoopTimer(self.profile.control_dt)
        last_log = time.perf_counter()
        while self.alive:
            if self.bridge_process is not None and self.bridge_process.poll() is not None:
                raise RuntimeError(
                    f"Z1 UDP service exited with code {self.bridge_process.returncode}"
                )

            ready = self.state_ready()
            if ready:
                self.step()
            if self.controller_state_pub is not None:
                self.controller_state_pub.Write(String_(self.state))

            now = time.perf_counter()
            if now - last_log >= 1.0:
                with self.lock:
                    lost_count = int(np.count_nonzero(self.lost))
                    max_tau = float(np.max(np.abs(self.tau_est)))
                console.status(
                    [
                        ("state", self.state, "green" if ready else "yellow"),
                        ("lowstate", "yes" if ready else "no", "green" if ready else "red"),
                        ("lost", str(lost_count), "green" if lost_count == 0 else "red"),
                        ("max_tau", f"{max_tau:.2f}", "white"),
                    ]
                )
                last_log = now
            timer.sleep()

    def close(self, *_args) -> None:
        self.alive = False

    def cleanup(self) -> None:
        if self.cleanup_done:
            return
        self.cleanup_done = True
        self.alive = False
        console.stop()
        self.base_state_sub.Close()
        self.lowstate_sub.Close()
        self.lowcmd_pub.Close()
        if self.controller_state_pub is not None:
            self.controller_state_pub.Close()
        if self.bridge_process is not None and self.bridge_process.poll() is None:
            self.bridge_process.terminate()
            try:
                self.bridge_process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self.bridge_process.kill()
                self.bridge_process.wait(timeout=2.0)

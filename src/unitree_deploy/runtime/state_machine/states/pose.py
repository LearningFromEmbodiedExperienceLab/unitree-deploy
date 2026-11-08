"""PD states for damping, pose interpolation, and impedance holding."""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any

import numpy as np

from unitree_deploy.config.defaults import MOVE_TO_DEFAULT_TIME
from ..config import parse_event_set, resolve_target_q, resolve_kp, resolve_kd
from .base import ControllerState


class DampingState(ControllerState):
    state_type = "damping"

    def step(self, controller) -> str | None:
        with controller.lock:
            q = controller.q.copy()
            kd_damping = controller.active_profile.kd_damping
        controller.send_joint_cmd(q, controller.zero, kd_damping)
        return None


class MoveToTargetPosState(ControllerState):
    state_type = "move_to_target_pos"

    def __init__(self, name: str, config: Mapping[str, Any]) -> None:
        super().__init__(name, config)
        self.target_spec = config.get("target", "default_q")
        self.policy_name = config.get("policy")
        self.target_order = str(config.get("target_order", "sdk"))
        self.duration = float(config.get("duration", MOVE_TO_DEFAULT_TIME))
        self.kp_spec = config.get("kp", "fixed_stand")
        self.kd_spec = config.get("kd", "fixed_stand")
        self.allow_early_transitions = bool(config.get("allow_early_transitions", False))
        self.early_transition_events = parse_event_set(
            config.get("early_transition_events", ["X"]),
            self.name,
            "early_transition_events",
        )
        self.start_q: np.ndarray | None = None
        self.target_q_obs: np.ndarray | None = None
        self.command_q_obs: np.ndarray | None = None
        self.last_ratio = 0.0

    @property
    def is_done(self) -> bool:
        return self.start_q is not None and self.last_ratio >= 1.0

    def next_for_button(self, button: str) -> str | None:
        if self.allow_early_transitions or self.is_done or button in self.early_transition_events:
            return super().next_for_button(button)
        return None

    def on_enter(self, controller) -> None:
        if self.policy_name is not None:
            controller.activate_policy(self.policy_name)
        profile = controller.active_profile
        with controller.lock:
            self.start_q = controller.q[profile.sdk_to_obs].copy()
        self.target_q_obs = resolve_target_q(controller, self.target_spec, self.target_order)
        self.command_q_obs = np.zeros_like(self.target_q_obs)
        self.last_ratio = 0.0

    def step(self, controller) -> str | None:
        if self.start_q is None or self.target_q_obs is None or self.command_q_obs is None:
            self.on_enter(controller)

        elapsed = time.perf_counter() - controller.state_enter_t
        ratio = 1.0 if self.duration <= 0.0 else float(np.clip(elapsed / self.duration, 0.0, 1.0))
        self.last_ratio = ratio
        self.command_q_obs[:] = self.target_q_obs
        self.command_q_obs -= self.start_q
        self.command_q_obs *= ratio
        self.command_q_obs += self.start_q

        controller.send_joint_cmd(
            controller.reorder_policy_to_sdk(self.command_q_obs),
            resolve_kp(controller, self.kp_spec),
            resolve_kd(controller, self.kd_spec),
        )

        if ratio >= 1.0:
            return self.next_on_done()
        return None


class ImpedanceState(ControllerState):
    state_type = "impedance"

    def __init__(self, name: str, config: Mapping[str, Any]) -> None:
        super().__init__(name, config)
        self.target_spec = config.get("target", "default_q")
        self.target_order = str(config.get("target_order", "sdk"))
        self.kp_spec = config.get("kp", "impedance")
        self.kd_spec = config.get("kd", "impedance")
        self.target_q_obs: np.ndarray | None = None

    def on_enter(self, controller) -> None:
        self.target_q_obs = resolve_target_q(
            controller,
            self.target_spec,
            self.target_order,
        )

    def step(self, controller) -> str | None:
        if self.target_q_obs is None:
            self.on_enter(controller)
        controller.send_joint_cmd(
            controller.reorder_policy_to_sdk(self.target_q_obs),
            resolve_kp(controller, self.kp_spec),
            resolve_kd(controller, self.kd_spec),
        )
        return None

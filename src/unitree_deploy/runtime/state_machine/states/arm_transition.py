from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import numpy as np

from unitree_deploy.obs.base_obs import (
    JointPositionObservation,
    JointVelocityObservation,
    PreviousActionObservation,
)
from ..filters import TargetEMA
from .base import ControllerState


def _arm_history(policy, context, arm_indices, previous_action) -> None:
    """Restore arm history columns; previous_action contains processed observation values."""
    # Arm indices use observation order; each term may expose a different column order.
    arm_actions = [
        i for i, joint in enumerate(policy.action_to_obs_indices) if joint in arm_indices
    ]
    for term in policy.observation.observations:
        if isinstance(term, (JointPositionObservation, JointVelocityObservation)):
            columns = [
                i for i, joint in enumerate(term.controlled_joint_indices) if joint in arm_indices
            ]
            values = term._compute_processed_current_obs(context)
        elif isinstance(term, PreviousActionObservation):
            columns = arm_actions
            values = previous_action
        else:
            continue
        term.buffer[:, columns] = values[columns]


class ArmTransitionState(ControllerState):
    """Blend arm targets; mask arm feedback only while preparing motion tracking."""

    state_type = "arm_transition"

    def __init__(self, name: str, config: Mapping[str, Any]) -> None:
        super().__init__(name, config)
        self.reset_policy_on_enter = False
        self.locomotion_policy = str(config["locomotion_policy"])
        self.motion_policy = str(config["motion_policy"])
        self.direction = str(config["direction"])
        if self.direction not in ("to_motion", "to_locomotion"):
            raise ValueError("arm_transition direction must be to_motion or to_locomotion")
        self.enabled = bool(config.get("enabled", True))
        self.duration = float(config.get("duration", 2.0))
        if self.duration < 0:
            raise ValueError("arm_transition duration must be nonnegative")
        speed = config.get("max_joint_speed")
        self.max_joint_speed = float(speed) if speed is not None else None
        if self.max_joint_speed is not None and self.max_joint_speed <= 0:
            raise ValueError("arm_transition max_joint_speed must be positive or null")
        self.entry_ema = TargetEMA(config.get("entry_ema", {}))

    def on_enter(self, controller) -> None:
        controller.activate_policy(self.locomotion_policy)
        profile = controller.active_profile
        self.arm_indices = np.array([
            i for i, name in enumerate(profile.obs_joint_order)
            if any(part in name for part in ("shoulder", "elbow", "wrist"))
        ])
        if self.arm_indices.size == 0:
            raise ValueError("arm_transition requires arm joints in the locomotion policy")
        if self.direction == "to_motion":
            context = controller.observation()
            self.arm_q = context.q[self.arm_indices].copy()
            self.arm_dq = context.dq[self.arm_indices].copy()
            self.arm_previous_action = profile.policy.previous_action_observation.buffer[-1].copy()
        self.start_q = controller.target_sdk[profile.sdk_to_obs].copy()
        self.entry_ema.reset(self.start_q)
        self.last_arm_target = self.start_q[self.arm_indices].copy()
        motion = controller.policy_manager.profiles[self.motion_policy]
        first_q = dict(zip(motion.sdk_joint_order, motion.policy.config["motion_start_q"]))
        self.motion_q = np.array([first_q[name] for name in profile.obs_joint_order])

    def step(self, controller) -> str | None:
        if not self.enabled:
            return self.next_on_done()
        profile = controller.active_profile
        policy = profile.policy
        context = controller.observation()

        policy_context = context
        if self.direction == "to_motion":
            # Preparation freezes arm feedback; return needs live feedback as arms recover.
            q = context.q.copy()
            dq = context.dq.copy()
            q[self.arm_indices] = self.arm_q
            dq[self.arm_indices] = self.arm_dq
            policy_context = replace(context, q=q, dq=dq)
            # Prime before masking so initialization cannot clear the frozen action history.
            if policy._observation_needs_prime:
                policy.observation.prime(policy_context)
                policy._observation_needs_prime = False
            _arm_history(policy, policy_context, self.arm_indices, self.arm_previous_action)
        # Filter full-body velocity targets before blending the arms. Copy so arm
        # overrides and speed limiting cannot change the EMA's internal history.
        target = self.entry_ema.apply(policy.compute_target_q(policy_context)).copy()
        # Inference records a new action; restore the frozen arm columns for the next tick.
        if self.direction == "to_motion":
            _arm_history(policy, policy_context, self.arm_indices, self.arm_previous_action)

        elapsed = time.perf_counter() - controller.state_enter_t
        ratio = 1.0 if self.duration == 0 else float(np.clip(elapsed / self.duration, 0.0, 1.0))
        # Smoothstep has zero endpoint slope; the return target follows live policy inference.
        blend = ratio * ratio * (3.0 - 2.0 * ratio)
        arm_goal = self.motion_q[self.arm_indices] if self.direction == "to_motion" else target[self.arm_indices]
        target[self.arm_indices] = (
            (1.0 - blend) * self.start_q[self.arm_indices] + blend * arm_goal
        )
        if self.max_joint_speed is not None:
            max_step = self.max_joint_speed * policy.policy_step_dt
            target[self.arm_indices] = self.last_arm_target + np.clip(
                target[self.arm_indices] - self.last_arm_target, -max_step, max_step
            )
        self.last_arm_target = target[self.arm_indices].copy()
        controller.last_policy_command[:] = context.command
        controller.send_joint_cmd(
            controller.reorder_policy_to_sdk(target), profile.kp_policy, profile.kd_policy
        )

        # Both the EMA window and arm speed cap must finish before the handoff.
        handoff_ready = self.entry_ema.remaining == 0 and ratio >= 1.0 and np.allclose(
            target[self.arm_indices], arm_goal, atol=1e-6, rtol=0
        )
        if handoff_ready:
            return self.next_on_done()
        return None

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from unitree_deploy.config.defaults import DAMPING_STATE
from ..filters import TargetEMA
from .base import ControllerState


class PolicyState(ControllerState):
    state_type = "policy"

    def __init__(self, name: str, config: Mapping[str, Any]) -> None:
        super().__init__(name, config)
        self.policy_name = config.get("policy")
        entry_ema = config.get("entry_ema", {})
        self.entry_ema = TargetEMA(entry_ema)
        arm_speed = entry_ema.get("max_arm_speed")
        self.entry_arm_max_speed = float(arm_speed) if arm_speed is not None else None
        if self.entry_arm_max_speed is not None and (
            not np.isfinite(self.entry_arm_max_speed) or self.entry_arm_max_speed <= 0
        ):
            raise ValueError("entry_ema.max_arm_speed must be positive and finite or null")
        self.entry_arm_limited = False
        self.entry_arm_target: np.ndarray | None = None

    def on_enter(self, controller) -> None:
        if self.policy_name is not None:
            controller.activate_policy(self.policy_name)
        self.entry_arm_limited = self.entry_arm_max_speed is not None
        if self.entry_ema.frames or self.entry_arm_limited:
            # Seed from the last sent target, reordered for the incoming policy.
            profile = controller.active_profile
            seed = controller.target_sdk[profile.sdk_to_obs]
            self.entry_ema.reset(seed)
            if self.entry_arm_limited:
                self.entry_arm_indices = np.array([
                    i for i, name in enumerate(profile.obs_joint_order)
                    if any(part in name for part in ("shoulder", "elbow", "wrist"))
                ], dtype=np.int64)
                self.entry_arm_target = seed[self.entry_arm_indices].copy()

    def step(self, controller) -> str | None:
        profile = controller.active_profile
        observation = controller.observation()
        if getattr(profile.policy, "requires_odometry", False) and (
            observation.base_pos_w is None or observation.base_lin_vel_w is None
        ):
            controller.log("tracking odometry missing or stale; returning to damping")
            return DAMPING_STATE
        controller.last_policy_command[:] = observation.command
        policy_target_q = profile.policy.compute_target_q(observation)
        # Smooth joint targets only; actor history and the motion cursor advance normally.
        target_q = self.entry_ema.apply(policy_target_q)
        if self.entry_arm_limited:
            # Continue the arm-transition rate bound through the first tracking targets.
            max_step = self.entry_arm_max_speed * profile.policy.policy_step_dt
            arm_step = target_q[self.entry_arm_indices] - self.entry_arm_target
            target_q = target_q.copy()  # Keep the actor and EMA buffers unmodified.
            target_q[self.entry_arm_indices] = self.entry_arm_target + np.clip(
                arm_step, -max_step, max_step
            )
            self.entry_arm_target[:] = target_q[self.entry_arm_indices]
            # Ending the EMA window alone must not release a remaining target gap.
            caught_up = np.allclose(
                self.entry_arm_target, policy_target_q[self.entry_arm_indices], atol=1e-6, rtol=0
            )
            self.entry_arm_limited = self.entry_ema.remaining > 0 or not caught_up
        controller.send_joint_cmd(
            controller.reorder_policy_to_sdk(target_q),
            profile.kp_policy,
            profile.kd_policy,
        )
        if getattr(profile.policy, "motion_finished", False):
            return self.next_on_done()
        return None

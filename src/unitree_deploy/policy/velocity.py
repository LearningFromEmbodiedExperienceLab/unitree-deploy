"""Deployment runtime for velocity ONNX policies."""

import numpy as np

from .base_policy import BasePolicy


class VelocityPolicy(BasePolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        limits = self.config.get("action_target_clip")
        self._target_limits = None if limits is None else np.asarray(limits, dtype=np.float32)

    def _infer_action(self, obs_vector):
        action = super()._infer_action(obs_vector)
        # Isaac Lab's last_action sees the wrapper-clipped action, before target scaling.
        if self.action_clip is not None:
            np.clip(action, -self.action_clip, self.action_clip, out=action)
        return action

    def compute_target_q(self, context):
        target = super().compute_target_q(context)
        limits = self._target_limits
        if limits is not None:
            indices = self.action_to_obs_indices
            target[indices] = np.clip(target[indices], limits[:, 0], limits[:, 1])
        return target

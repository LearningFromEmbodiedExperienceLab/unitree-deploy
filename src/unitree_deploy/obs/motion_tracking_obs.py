"""LocoLab tracking terms; base quantities use the pelvis, anchor uses torso FK."""

import numpy as np
from scipy.spatial.transform import Rotation

from .base_obs import ObservationBase


def rotation(quat_wxyz):
    return Rotation.from_quat(np.asarray(quat_wxyz)[[1, 2, 3, 0]])


class MotionCommandObservation(ObservationBase):
    def __init__(self, *, history_len, motion_command_terms, num_joints, dtype=np.float32):
        dimensions = {"joint_pos": num_joints, "joint_vel": num_joints, "base_height": 1,
                      "base_lin_vel": 3, "base_ang_vel": 3, "projected_gravity": 3}
        self.motion_command_terms = tuple(motion_command_terms)
        if not self.motion_command_terms or len(set(self.motion_command_terms)) != len(self.motion_command_terms):
            raise ValueError("motion_command_terms must be non-empty and contain no duplicates")
        unknown = set(self.motion_command_terms) - dimensions.keys()
        if unknown:
            raise ValueError(f"Unknown motion_command_terms: {sorted(unknown)}")
        base_dim = sum(dimensions[name] for name in self.motion_command_terms)
        super().__init__(base_dim=base_dim, history_len=history_len, dtype=dtype)

    def compute(self, context):
        ref = self.policy.motion_reference
        pelvis = self.policy.pelvis_index
        values = []
        base_rotation = None
        for name in self.motion_command_terms:
            if name in ("joint_pos", "joint_vel"):
                value = ref[name]
            elif name == "base_height":
                value = ref["body_pos_w"][pelvis, 2:3]
            else:
                if base_rotation is None:
                    base_rotation = rotation(ref["body_quat_w"][pelvis]).inv()
                if name == "projected_gravity":
                    value = base_rotation.apply([0., 0., -1.])
                else:
                    field = "body_lin_vel_w" if name == "base_lin_vel" else "body_ang_vel_w"
                    value = base_rotation.apply(ref[field][pelvis])
            values.append(value)
        return np.concatenate(values)


class MotionAnchorPositionObservation(ObservationBase):
    def __init__(self, *, history_len, dtype=np.float32):
        super().__init__(base_dim=3, history_len=history_len, dtype=dtype)

    def compute(self, context):
        if context.base_pos_w is None:
            raise ValueError("This tracking actor requires world-frame pelvis position from odometry.")
        policy = self.policy
        ref_pos = policy.motion_reference["body_pos_w"][policy.anchor_index]
        return policy.anchor_rotation.inv().apply(ref_pos - policy.anchor_position)


class MotionAnchorOrientationObservation(ObservationBase):
    def __init__(self, *, history_len, dtype=np.float32):
        super().__init__(base_dim=6, history_len=history_len, dtype=dtype)

    def compute(self, context):
        policy = self.policy
        ref_quat = policy.motion_reference["body_quat_w"][policy.anchor_index]
        return (policy.anchor_rotation.inv() * rotation(ref_quat)).as_matrix()[:, :2].reshape(-1)


OBSERVATION_TYPES = {
    "motion_command": MotionCommandObservation,
    # these below are privileged and require odometry
    "motion_anchor_pos_b": MotionAnchorPositionObservation,
    "motion_anchor_ori_b": MotionAnchorOrientationObservation,
}

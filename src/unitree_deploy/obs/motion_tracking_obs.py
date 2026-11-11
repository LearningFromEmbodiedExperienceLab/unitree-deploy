"""Motion tracking observations for embedded references and sampled reference windows."""

import numpy as np
from scipy.spatial.transform import Rotation

from .base_obs import (
    BaseAngularVelocityObservation,
    JointPositionObservation,
    JointVelocityObservation,
    ObservationBase,
    ProjectedGravityObservation,
)


def rotation(quat_wxyz):
    return Rotation.from_quat(np.asarray(quat_wxyz)[..., [1, 2, 3, 0]])


def yaw_rotation(wxyz):
    return Rotation.from_euler("z", rotation(wxyz).as_euler("xyz")[2])


def reference_velocities(reference, dt):
    """Differentiate a sampled pose window, keeping held endpoints out of the stencil."""
    valid = np.flatnonzero(reference.get("valid", np.ones(len(reference["joint_pos"]), dtype=bool)))
    velocities = {name: np.zeros_like(reference[source]) for name, source in (
        ("joint_vel", "joint_pos"), ("body_lin_vel_w", "body_pos_w"),
    )}
    velocities["body_ang_vel_w"] = np.zeros_like(reference["body_pos_w"])
    if len(valid) < 2:
        return velocities
    for name, source in (("joint_vel", "joint_pos"), ("body_lin_vel_w", "body_pos_w")):
        velocities[name][valid] = np.gradient(reference[source][valid], dt, axis=0)
    for body in range(reference["body_quat_w"].shape[1]):
        rotations = rotation(reference["body_quat_w"][valid, body])
        if len(valid) == 2:
            omega = (rotations[1] * rotations[0].inv()).as_rotvec() / dt
        else:
            interior = (rotations[2:] * rotations[:-2].inv()).as_rotvec() / (2 * dt)
            omega = np.concatenate((interior[:1], interior, interior[-1:]))
        velocities["body_ang_vel_w"][valid, body] = omega
    return velocities


class MotionCommandObservation(ObservationBase):
    def __init__(self, *, history_len, motion_command_terms, num_joints, history_order="oldest_first", dtype=np.float32):
        dimensions = {"joint_pos": num_joints, "joint_vel": num_joints, "base_height": 1,
                      "base_lin_vel": 3, "base_ang_vel": 3, "projected_gravity": 3}
        self.motion_command_terms = tuple(motion_command_terms)
        if not self.motion_command_terms or len(set(self.motion_command_terms)) != len(self.motion_command_terms):
            raise ValueError("motion_command_terms must be non-empty and contain no duplicates")
        unknown = set(self.motion_command_terms) - dimensions.keys()
        if unknown:
            raise ValueError(f"Unknown motion_command_terms: {sorted(unknown)}")
        base_dim = sum(dimensions[name] for name in self.motion_command_terms)
        super().__init__(base_dim=base_dim, history_len=history_len, history_order=history_order, dtype=dtype)

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
    def __init__(self, *, history_len, history_order="oldest_first", dtype=np.float32):
        super().__init__(base_dim=3, history_len=history_len, history_order=history_order, dtype=dtype)

    def compute(self, context):
        if context.base_pos_w is None:
            raise ValueError("This tracking actor requires world-frame pelvis position from odometry.")
        policy = self.policy
        ref_pos = policy.motion_reference["body_pos_w"][policy.anchor_index]
        return policy.anchor_rotation.inv().apply(ref_pos - policy.anchor_position)


class MotionAnchorOrientationObservation(ObservationBase):
    def __init__(self, *, history_len, history_order="oldest_first", dtype=np.float32):
        super().__init__(base_dim=6, history_len=history_len, history_order=history_order, dtype=dtype)

    def compute(self, context):
        policy = self.policy
        if policy.reference_source == "external":
            current = np.searchsorted(policy.motion.steps, 0)
            reference = rotation(policy.reference_window["body_quat_w"][current, policy.anchor_index])
            robot = policy.robot_heading_offset * rotation(context.quat)
            return (robot.inv() * reference).as_matrix()[:, :2].reshape(-1)
        ref_quat = policy.motion_reference["body_quat_w"][policy.anchor_index]
        return (policy.anchor_rotation.inv() * rotation(ref_quat)).as_matrix()[:, :2].reshape(-1)


# ========== Mimic Lite style sparse history observations ========== 

class SparseHistory:
    """Training selects offsets from newest to oldest, including nonconsecutive steps."""

    def __init__(self, *, history_steps, history_len=1, **kwargs):
        self.history_steps = np.asarray(history_steps, dtype=int)
        if self.history_steps.ndim != 1 or not self.history_steps.size or np.any(self.history_steps < 0):
            raise ValueError("history_steps must be a nonempty list of nonnegative offsets")
        super().__init__(history_len=int(self.history_steps.max()) + 1, **kwargs)

    @property
    def size(self):
        return self.base_dim * len(self.history_steps)

    def ordered_buffer(self):
        return self.buffer[-1 - self.history_steps]


class SparseAngularVelocityObservation(SparseHistory, BaseAngularVelocityObservation):
    pass


class SparseGravityObservation(SparseHistory, ProjectedGravityObservation):
    pass


class SparseJointPositionObservation(SparseHistory, JointPositionObservation):
    pass


class SparseJointVelocityObservation(SparseHistory, JointVelocityObservation):
    pass


class MotionReferenceWindowObservation(ObservationBase):
    def __init__(self, *, field, future_steps, joint_names=(), body_names=(), history_len=1):
        self.field = field
        self.future_steps = list(future_steps)
        self.joint_names, self.body_names = list(joint_names), list(body_names)
        widths = {
            "ref_joint_pos_future": len(joint_names),
            "ref_body_pos_future_local": 3 * len(body_names),
            "ref_body_ori_future_local": 6 * len(body_names),
            "ref_root_ori_future_b": 6,
            "root_pos_future_z": 1,
            "root_pos_future_diff": 3,
        }
        if field not in widths or not future_steps or widths[field] == 0:
            raise ValueError(f"invalid reference field/selection: {field}")
        if history_len != 1:
            raise ValueError("Reference motion terms require history_len: 1")
        # root_pos_future_diff excludes the current step, as in the original deployment.
        count = len(future_steps) - (field == "root_pos_future_diff" and 0 in future_steps)
        if count <= 0:
            raise ValueError("reference term selected no future steps")
        super().__init__(base_dim=widths[field] * count, history_len=1)

    def compute(self, context):
        policy = self.policy
        indices = [policy.motion.steps.tolist().index(step) for step in self.future_steps]
        ref = policy.reference_window
        root = policy.root_index
        if self.field == "ref_joint_pos_future":
            joints = [policy.action_joint_order.index(name) for name in self.joint_names]
            return ref["joint_pos"][indices][:, joints].reshape(-1)
        if self.field == "ref_root_ori_future_b":
            robot = policy.robot_heading_offset * rotation(context.quat)
            relative = robot.inv() * rotation(ref["body_quat_w"][indices, root])
            # This feature encodes rotation using the first two rows.
            return relative.as_matrix()[:, :2, :].reshape(-1)
        current = policy.motion.steps.tolist().index(0)
        root_pos = ref["body_pos_w"][current, root]
        root_rot = rotation(ref["body_quat_w"][current, root])
        if self.field == "root_pos_future_z":
            return ref["body_pos_w"][indices, root, 2]
        if self.field == "root_pos_future_diff":
            indices = [i for i, step in zip(indices, self.future_steps) if step != 0]
            return root_rot.inv().apply(ref["body_pos_w"][indices, root] - root_pos).reshape(-1)
        bodies = [policy.body_names.index(name) for name in self.body_names]
        anchor = policy.anchor_index
        anchor_pos = ref["body_pos_w"][current, anchor].copy()
        anchor_pos[2] = 0.0
        anchor_yaw = yaw_rotation(ref["body_quat_w"][current, anchor])
        if self.field == "ref_body_pos_future_local":
            values = ref["body_pos_w"][indices][:, bodies] - anchor_pos
            return anchor_yaw.inv().apply(values.reshape(-1, 3)).reshape(-1)
        values = ref["body_quat_w"][indices][:, bodies].reshape(-1, 4)
        return (anchor_yaw.inv() * rotation(values)).as_matrix()[:, :2, :].reshape(-1)


class SuperMimicGoalObservation(MotionReferenceWindowObservation):
    """Future reference features in SuperMimic's training coordinates and order."""

    def __init__(self, *, num_joints, history_len=1):
        if history_len != 1:
            raise ValueError("SuperMimic goals contain a future window, not observation history.")
        # Neighbors at -1 and 11 support central differences for live pose streams.
        self.future_steps = list(range(-1, 12))
        ObservationBase.__init__(self, base_dim=10 * (15 + 2 * num_joints), history_len=1)

    def compute(self, context):
        policy = self.policy
        ref = policy.reference_window
        current = np.searchsorted(policy.motion.steps, 0)
        future = np.searchsorted(policy.motion.steps, np.arange(1, 11))
        root = policy.root_index
        heading = yaw_rotation(ref["body_quat_w"][current, root]).inv()
        future_rotation = rotation(ref["body_quat_w"][future, root])
        dp = heading.apply(ref["body_pos_w"][future, root] - ref["body_pos_w"][current, root])
        orientation = (heading * future_rotation).as_matrix()[:, :, :2].reshape(10, 6)
        velocity = ref if "joint_vel" in ref else reference_velocities(ref, policy.policy_step_dt)
        valid = np.asarray(ref.get("valid", np.ones(len(ref["joint_pos"]), dtype=bool)))[future, None]
        linear = future_rotation.inv().apply(velocity["body_lin_vel_w"][future, root])
        angular = future_rotation.inv().apply(velocity["body_ang_vel_w"][future, root])
        return np.concatenate((dp, orientation, np.where(valid, linear, 0), np.where(valid, angular, 0),
                               ref["joint_pos"][future], np.where(valid, velocity["joint_vel"][future] * .1, 0)), axis=-1).reshape(-1)


OBSERVATION_TYPES = {
    "motion_command": MotionCommandObservation,
    # Anchor position requires odometry; external anchor orientation uses the root IMU.
    "motion_anchor_pos_b": MotionAnchorPositionObservation,
    "motion_anchor_ori_b": MotionAnchorOrientationObservation,
    # mimic-lite style motion command observations with sparse history
    "motion_reference_window": MotionReferenceWindowObservation,
    "supermimic_goal": SuperMimicGoalObservation,
}

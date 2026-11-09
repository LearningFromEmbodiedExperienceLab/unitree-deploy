"""Pico/XRobot -> FRAX online IK, independent of the GMR runtime."""

import numpy as np
from scipy.spatial.transform import Rotation

from .retarget import RobotPose
from .xrobot_g1 import XROBOT_G1_TARGETS as _TARGETS


_HUMAN_NAMES = tuple(row[1] for row in _TARGETS)
_ROTATION_OFFSETS = Rotation.from_quat(np.array([row[5] for row in _TARGETS])[:, [1, 2, 3, 0]])
_FEET = np.array([3, 6])
# FRAX's G1 retarget example uses these GMR limits to avoid shoulder branch flips.
_ARM_LIMITS = {
    "left_shoulder_pitch_joint": (-3.089, 1.149),
    "left_shoulder_roll_joint": (-.6, 2.252),
    "left_shoulder_yaw_joint": (-1.4, 2.),
    "left_elbow_joint": (-1.047, 1.7),
    "right_shoulder_pitch_joint": (-3.089, 1.149),
    "right_shoulder_roll_joint": (-2.252, .6),
    "right_shoulder_yaw_joint": (-2., 1.4),
    "right_elbow_joint": (-1.047, 1.7),
}


def xrobot_reference(body_pose, human_height):
    """Scale about the pelvis, then apply the XRobot-to-G1 local frame offsets."""
    positions = np.array([body_pose[name][0] for name in _HUMAN_NAMES], dtype=np.float64)
    quaternions = np.array([body_pose[name][1] for name in _HUMAN_NAMES], dtype=np.float64)
    if positions.shape != (len(_TARGETS), 3) or quaternions.shape != (len(_TARGETS), 4) or not (
        np.isfinite(positions).all() and np.isfinite(quaternions).all()
    ):
        raise ValueError("FRAX requires finite XRobot positions [14, 3] and wxyz quaternions [14, 4]")
    rotations = Rotation.from_quat(quaternions[:, [1, 2, 3, 0]]) * _ROTATION_OFFSETS
    scales = np.array([row[4] for row in _TARGETS]) * human_height / 1.8
    root = positions[0]
    positions = root * scales[0] + (positions - root) * scales[:, None]
    foot_rotations = Rotation.from_matrix(rotations.as_matrix()[_FEET])
    positions[_FEET] += foot_rotations.apply(np.tile([.05, 0, 0], (2, 1)))
    return {"frame_pos": positions, "frame_rmat": rotations.as_matrix()}


class FraxRetargeter:
    def __init__(self, *, human_height: float, step_dt: float = .02):
        if not np.isfinite([human_height, step_dt]).all() or min(human_height, step_dt) <= 0:
            raise ValueError("human_height and step_dt must be finite and positive")
        import jax
        import jax.numpy as jnp

        # ElastiQP requires float64. JIT compilation finishes before DDS/XR starts.
        jax.config.update("jax_enable_x64", True)
        from frax.ik import DiffIKSolver, FloorConstraint, FrameTask, JointLimits, OnlineRetargeter, Retargeter
        from frax.robots.unitree_g1 import load_g1
        from frax.utils.rotation_utils import rmat_to_quat_wxyz

        self.robot = load_g1(floating_base="quaternion")
        self.joint_names = tuple(self.robot.joint_names[self.robot.nv_floating:])
        self.step_dt = step_dt
        self.human_height = human_height
        self.target_indices = [self.robot.joint_name_to_index[row[0]] for row in _TARGETS]
        offsets = np.tile(np.eye(4), (len(_TARGETS), 1, 1))
        offsets[_FEET, :3, 3] = [.1, 0, -.02]  # GMR's G1 toe relative to ankle_roll.
        gain = min(30., 1 / step_dt)
        task = FrameTask(
            self.robot, self.target_indices, gain_pos=gain, gain_rot=gain,
            weight_pos=[row[2] for row in _TARGETS], weight_rot=[row[3] for row in _TARGETS], offsets=offsets,
        )
        lower = self.robot.actuated_joint_lower_limits.copy()
        upper = self.robot.actuated_joint_upper_limits.copy()
        for name, (lo, hi) in _ARM_LIMITS.items():
            index = self.robot.joint_name_to_index[name] - self.robot.nv_floating
            lower[index], upper[index] = lo, hi
        limits = JointLimits(self.robot, q_min=lower, q_max=upper, gain=gain)
        self.lower, self.upper = limits.q_min, limits.q_max
        solver = DiffIKSolver(self.robot, [task], [limits, FloorConstraint(self.robot, gain=gain)],
                              regularization=10.)

        def initial_q(reference):
            q = jnp.asarray(self.robot.default_configuration)
            q = q.at[:3].set(reference["frame_pos"][0])
            return q.at[3:7].set(rmat_to_quat_wxyz(reference["frame_rmat"][0]))

        self.backend = OnlineRetargeter(Retargeter(solver, lambda reference: reference, initial_q))
        self.backend.compile({
            "frame_pos": jax.ShapeDtypeStruct((len(_TARGETS), 3), jnp.float64),
            "frame_rmat": jax.ShapeDtypeStruct((len(_TARGETS), 3, 3), jnp.float64),
        })

    def retarget(self, body_pose):
        q = np.asarray(self.backend.step(xrobot_reference(body_pose, self.human_height), self.step_dt))
        joints = q[self.robot.nq_floating:]
        if not np.isfinite(q).all() or np.any(joints < self.lower - 1e-4) or np.any(joints > self.upper + 1e-4):
            raise RuntimeError("FRAX returned nonfinite or out-of-limit joint positions")
        return RobotPose(q[:3].copy(), q[3:7].copy(), self.joint_names, joints.copy())

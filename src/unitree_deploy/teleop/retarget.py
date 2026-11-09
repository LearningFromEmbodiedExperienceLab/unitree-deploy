"""The backend boundary uses named joints, meters, and wxyz quaternions."""

from dataclasses import dataclass
import platform
from typing import Protocol

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class RobotPose:
    root_pos: np.ndarray
    root_quat: np.ndarray
    joint_names: tuple[str, ...]
    joint_pos: np.ndarray


class Retargeter(Protocol):
    def retarget(self, body_pose: dict[str, tuple[np.ndarray, np.ndarray]]) -> RobotPose:
        """Convert a Z-up human world pose into a named robot pose."""
        ...


class RetargetSession:
    """Hold the last pose on pause; preserve its XY/yaw when live input resumes."""

    def __init__(self, backend: Retargeter, initial_pose: RobotPose):
        self.backend = backend
        self.pose = initial_pose
        self.paused = True
        self.heading = None

    def toggle_pause(self):
        self.paused = not self.paused
        self.heading = None

    def update(self, body_pose):
        if self.paused:
            return self.pose
        pose = self.backend.retarget(body_pose)
        orientation = Rotation.from_quat(pose.root_quat[[1, 2, 3, 0]])
        if self.heading is None:
            previous = Rotation.from_quat(self.pose.root_quat[[1, 2, 3, 0]])
            yaw = previous.as_euler("xyz")[2] - orientation.as_euler("xyz")[2]
            self.heading = Rotation.from_euler("z", yaw)
            self.translation = self.pose.root_pos - self.heading.apply(pose.root_pos)
            self.translation[2] = 0.0  # Keep calibrated height, including crouches and jumps.
        quat = (self.heading * orientation).as_quat()[[3, 0, 1, 2]]
        self.pose = RobotPose(self.heading.apply(pose.root_pos) + self.translation,
                              quat, pose.joint_names, pose.joint_pos)
        return self.pose


class GMRRetargeter:
    def __init__(self, *, human_height: float):
        # Keep GMR and its native dependencies out of the controller process.
        if platform.machine() == "aarch64":
            import torch  # Load before MuJoCo to avoid Orin's static TLS import failure.
        import mujoco
        from general_motion_retargeting import GeneralMotionRetargeting
        from general_motion_retargeting.params import IK_CONFIG_DICT

        if "unitree_g1" not in IK_CONFIG_DICT.get("xrobot", {}):
            raise RuntimeError("GMR must include the xrobot -> unitree_g1 IK configuration")

        self.backend = GeneralMotionRetargeting(
            src_human="xrobot", tgt_robot="unitree_g1",
            actual_human_height=human_height, verbose=False,
        )
        model = self.backend.configuration.model
        self.joint_names = tuple(
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
            for i in range(model.njnt) if model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE
        )
        self.qpos_indices = [model.jnt_qposadr[model.joint(name).id] for name in self.joint_names]

    def retarget(self, body_pose):
        qpos = np.asarray(self.backend.retarget(body_pose, offset_to_ground=False), dtype=np.float32)
        return RobotPose(qpos[:3].copy(), qpos[3:7].copy(), self.joint_names,
                         qpos[self.qpos_indices].copy())

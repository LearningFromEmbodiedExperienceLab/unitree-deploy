"""Runtime for normalized, single-clip LocoLab tracking ONNX exports."""

import json

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .base_policy import BasePolicy
from ..obs.motion_tracking_obs import rotation
from ..robot_model.robot_config import ROBOT_MODEL_ROOT


class MotionTrackingPolicy(BasePolicy):
    """Align an embedded motion to the robot, then infer one action per reference frame.

    Observation terms read the aligned reference and the measured anchor pose
    prepared before BasePolicy assembles the observation vector.
    """

    def __init__(self, policy_yaml_path, *, providers=None, observation_types=None):
        super().__init__(policy_yaml_path, providers=providers, observation_types=observation_types)
        metadata = self.session.get_modelmeta().custom_metadata_map
        if metadata.get("locolab_tracking_version") != "2":
            raise ValueError("Tracking requires the latest LocoLab motion bundle (version 2).")
        if json.loads(metadata["motion_joint_names"]) != self.action_joint_order:
            raise ValueError("action_joint_order differs from the ONNX motion joint order")
        if json.loads(metadata["motion_body_names"]) != self.config["body_names"]:
            raise ValueError("body_names differs from the ONNX motion body order")
        if metadata["quaternion_order"] != "wxyz":
            raise ValueError("ONNX motion quaternions must use wxyz order")
        inputs = {item.name: item for item in self.session.get_inputs()}
        if inputs[self.input_name].shape != [1, self.observation.size]:
            raise ValueError("ONNX observation shape differs from policy.yaml")
        if inputs["time_step"].type != "tensor(int64)" or inputs["time_step"].shape != [1]:
            raise ValueError("LocoLab time_step must be int64[1]")
        if self.session.get_outputs()[0].shape != [1, self.action_dim]:
            raise ValueError("ONNX action shape differs from policy.yaml")
        self.reference_names = [
            "joint_pos", "joint_vel", "body_pos_w", "body_quat_w",
            "body_lin_vel_w", "body_ang_vel_w",
        ]
        self.num_frames = int(metadata["motion_num_frames"])
        if not np.isclose(float(metadata["motion_dt"]), self.policy_step_dt):
            raise ValueError("Motion and policy control timesteps must match")
        self.anchor_index = self.config["body_names"].index(self.config["anchor_body_name"])
        self.pelvis_index = self.config["body_names"].index("pelvis")
        self.requires_odometry = any(
            spec["type"] in ("motion_anchor_pos_b", "base_lin_vel")
            for spec in self.config["observations"]
        )
        self.model = mujoco.MjModel.from_xml_path(str(ROBOT_MODEL_ROOT / "g1/g1.xml"))
        self.data = mujoco.MjData(self.model)
        self.anchor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.config["anchor_body_name"])
        self.joint_qpos = []
        for name in self.obs_joint_order:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id < 0:
                raise ValueError(f"Tracking FK model is missing joint {name!r}")
            self.joint_qpos.append(self.model.jnt_qposadr[joint_id])
        for observation in self.observation.observations:
            observation.policy = self
        self._zero_obs = np.zeros((1, self.observation.size), np.float32)
        self.reset()

    def _build_observation(self, observation_spec):
        if observation_spec["type"] == "motion_command":
            observation_spec = dict(observation_spec, params=dict(
                observation_spec.get("params", {}),
                motion_command_terms=self.config["motion_command_terms"], num_joints=self.action_dim,
            ))
        return super()._build_observation(observation_spec)

    def reset(self):
        super().reset()
        self.time_step = 0
        self.motion_finished = False
        self.heading = None
        self.translation = None

    def _run(self, obs, outputs):
        inputs = {
            self.input_name: obs.reshape(1, -1).astype(np.float32),
            "time_step": np.array([self.time_step], np.int64),
        }
        return self.session.run(outputs, inputs)

    def _infer_action(self, obs_vector):
        return self._run(obs_vector, [self.action_output_name])[0].reshape(-1)

    def _update_motion_reference(self, context):
        """Read this frame and apply the heading/XY offset captured on the first step."""
        values = self._run(self._zero_obs, self.reference_names)
        self.motion_reference = {name: value[0].copy() for name, value in zip(self.reference_names, values)}
        ref = self.motion_reference
        if self.heading is None:
            ref_heading = rotation(ref["body_quat_w"][self.pelvis_index]).as_euler("xyz")[2]
            robot_heading = rotation(context.quat).as_euler("xyz")[2]
            self.heading = Rotation.from_euler("z", robot_heading - ref_heading)
            self.translation = np.zeros(3)
            if context.base_pos_w is not None:
                aligned_pelvis = self.heading.apply(ref["body_pos_w"][self.pelvis_index])
                self.translation[:2] = (context.base_pos_w - aligned_pelvis)[:2]
        ref["body_pos_w"][:] = self.heading.apply(ref["body_pos_w"]) + self.translation
        # MuJoCo uses wxyz; SciPy uses xyzw.
        xyzw = (self.heading * Rotation.from_quat(ref["body_quat_w"][:, [1, 2, 3, 0]])).as_quat()
        ref["body_quat_w"][:] = xyzw[:, [3, 0, 1, 2]]
        for name in ("body_lin_vel_w", "body_ang_vel_w"):
            ref[name][:] = self.heading.apply(ref[name])

    def _update_anchor_pose(self, context):
        """Use measured joints and base pose to compute the anchor's world pose."""
        self.data.qpos[:3] = context.base_pos_w if context.base_pos_w is not None else 0.0
        self.data.qpos[3:7] = context.quat
        self.data.qpos[self.joint_qpos] = context.q
        mujoco.mj_kinematics(self.model, self.data)
        self.anchor_position = self.data.xpos[self.anchor_id].copy()
        self.anchor_rotation = rotation(self.data.xquat[self.anchor_id])

    def compute_target_q(self, context):
        if self.requires_odometry and (context.base_pos_w is None or context.base_lin_vel_w is None):
            raise ValueError("160-D tracking requires fresh pelvis position/velocity odometry; no zero filling is used.")

        self._update_motion_reference(context)
        self._update_anchor_pose(context)
        target = super().compute_target_q(context)
        if self.config.get("use_reference_residual", False):
            target[self.action_to_obs_indices] = (
                self.motion_reference["joint_pos"] + self.action_scaling * self.action
            )
        # last_action in training sees the wrapper-clipped action.
        self.previous_action_observation.record_action(self.action)
        self.motion_finished = self.time_step >= self.num_frames - 1
        self.time_step = min(self.time_step + 1, self.num_frames - 1)
        return target

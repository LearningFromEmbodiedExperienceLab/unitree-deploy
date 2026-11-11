"""ONNX motion tracking with an embedded or external reference."""

import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from ...obs.motion_tracking_obs import MotionReferenceWindowObservation, rotation, yaw_rotation
from ...robot_model.robot_config import ROBOT_MODEL_ROOT
from ...runtime.transport.reference_motion import MotionSubscriber, MotionUnavailable, ReferenceStreamLost
from ..base_policy import BasePolicy
from .clip import Any4HDMIMotionClip
from .pose import ReferencePose


class MotionTrackingPolicy(BasePolicy):
    """Track a reference supplied by the ONNX bundle, DDS, or a local clip."""

    supports_motion_path = True

    def __init__(self, policy_yaml_path, *, providers=None, observation_types=None, motion_path=None):
        super().__init__(policy_yaml_path, providers=providers, observation_types=observation_types)
        self.reference_source = self.config["reference_source"]
        self.body_names = self.config["body_names"]
        self.root_index = self.body_names.index(self.config.get("root_body_name", "pelvis"))
        self.anchor_index = self.body_names.index(self.config["anchor_body_name"])
        if self.reference_source == "embedded":
            if motion_path is not None or self.config.get("motion_path") is not None:
                raise ValueError("An embedded reference cannot be overridden by a local motion path")
            self._init_embedded_reference()
        elif self.reference_source == "external":
            self._init_external_reference(motion_path)
        else:
            raise ValueError("reference_source must be 'embedded' or 'external'")
        self.reset()

    def _init_embedded_reference(self):
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
        self.pelvis_index = self.root_index
        self.requires_odometry = any(
            spec["type"] in ("motion_anchor_pos_b", "base_lin_vel")
            for spec in self.config["observations"]
        )
        robot = self.config["robot"]
        self.model = mujoco.MjModel.from_xml_path(str(ROBOT_MODEL_ROOT / robot / f"{robot}.xml"))
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

    def _init_external_reference(self, motion_path):
        if self.observation.obs_group_concat_mode != "term_major":
            raise ValueError("Reference tracking requires term_major observations")
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        # Original exports can have separate command/policy inputs. Each YAML term
        # names its input; single-vector exports simply omit that field.
        self.input_slices = {}
        offset = 0
        for spec, observation in zip(self.config["observations"], self.observation.observations):
            name = spec.get("input", self.input_name)
            self.input_slices.setdefault(name, []).append(slice(offset, offset + observation.size))
            offset += observation.size
        self.input_indices = {}
        for name, parts in self.input_slices.items():
            mode = self.config.get("obs_input_concat_modes", {}).get(name, "term_major")
            if mode == "history_major":
                terms = [obs for spec, obs in zip(self.config["observations"], self.observation.observations)
                         if spec.get("input", self.input_name) == name]
                if len({obs.history_len for obs in terms}) != 1:
                    raise ValueError("History-major inputs require equal term history lengths")
                indices = [np.arange(part.start, part.stop).reshape(obs.history_len, obs.base_dim)
                           for part, obs in zip(parts, terms)]
                self.input_indices[name] = np.concatenate(indices, axis=1).reshape(-1)
            elif mode == "term_major":
                self.input_indices[name] = np.concatenate([np.arange(part.start, part.stop) for part in parts])
            else:
                raise ValueError(f"Unknown observation input concatenation mode: {mode}")
        if set(self.input_slices) != {item.name for item in inputs}:
            raise ValueError("Reference tracking ONNX input names differ from policy.yaml")
        self.input_shapes = {item.name: item.shape for item in inputs}
        for item in inputs:
            size = sum(part.stop - part.start for part in self.input_slices[item.name])
            if item.type != "tensor(float)" or item.shape not in ([size], [1, size]):
                raise ValueError("Reference tracking ONNX observation shape/type differs from policy.yaml")
        output = next(item for item in outputs if item.name == self.action_output_name)
        if output.type != "tensor(float)" or output.shape not in ([self.action_dim], [1, self.action_dim]):
            raise ValueError("Reference tracking ONNX action shape differs from policy.yaml")
        future_steps = []
        for observation in self.observation.observations:
            observation.policy = self
            if isinstance(observation, MotionReferenceWindowObservation):
                future_steps.extend(observation.future_steps)
        if not future_steps:
            raise ValueError("Reference tracking requires reference observations")
        self.clip = None
        motion_path = motion_path if motion_path is not None else self.config.get("motion_path")
        if motion_path is None:
            self.motion = MotionSubscriber(
                self.action_joint_order, self.body_names, dt=self.policy_step_dt,
                future_steps=future_steps, **self.config.get("reference_motion", {}),
            )
        else:
            if len(self.body_names) != 1:
                raise ValueError("Local any4hdmi clips provide only the root body reference")
            path = (self.policy_yaml_path.parent / Path(motion_path).expanduser()).resolve()
            self.clip = self.motion = Any4HDMIMotionClip(
                path, self.action_joint_order, dt=self.policy_step_dt, future_steps=future_steps,
                include_endpoint=self.config.get("motion_include_endpoint", True),
            )
        self.reference_hold = None
        self.reference_session = None
        self.playback_requested = False
        self.playback_rate = 1.0

    def _build_observation(self, observation_spec):
        if observation_spec["type"] == "motion_command":
            observation_spec = dict(observation_spec, params=dict(
                observation_spec.get("params", {}),
                motion_command_terms=self.config["motion_command_terms"], num_joints=self.action_dim,
            ))
        return super()._build_observation(observation_spec)

    def reset(self):
        super().reset()
        self.motion_finished = False
        if self.reference_source == "embedded":
            self.time_step = 0
            self.heading = None
            self.translation = None
            return
        self._stop_playback()
        # Sensor histories are filled with the first measured frame; actions start at zero.
        self._observation_needs_prime = True
        self.robot_heading_offset = None
        self.ghost_translation = None
        self.reference_session = None
        self.reference_hold = None
        self.playback_requested = False
        self.preview = None
        self.reference_window = None
        self.replacement_reference = None
        self.reference_lost = False
        if self.clip is not None:
            self.clip.reset()

    def reference_pose(self) -> ReferencePose | None:
        if self.reference_source == "embedded":
            if self.heading is None:
                return None
            ref = self.motion_reference
            return ReferencePose(self.action_joint_order, ref["joint_pos"],
                                 ref["body_pos_w"][self.pelvis_index], ref["body_quat_w"][self.pelvis_index])
        if self.robot_heading_offset is None:
            return None
        ref = self.reference_window
        current = np.searchsorted(self.motion.steps, 0)
        heading = self.robot_heading_offset.inv()
        root_pos = heading.apply(ref["body_pos_w"][current, self.root_index])
        if self.ghost_translation is not None:
            root_pos[:2] += self.ghost_translation
        root_quat = heading * rotation(ref["body_quat_w"][current, self.root_index])
        return ReferencePose(self.action_joint_order, ref["joint_pos"][current],
                             root_pos, root_quat.as_quat()[[3, 0, 1, 2]])

    def _infer_action(self, obs_vector):
        if self.reference_source == "embedded":
            return self._run(obs_vector, [self.action_output_name])[0].reshape(-1)
        inputs = {name: obs_vector[indices].reshape(self.input_shapes[name])
                  for name, indices in self.input_indices.items()}
        return self.session.run([self.action_output_name], inputs)[0].reshape(-1)

    def compute_target_q(self, context):
        embedded = self.reference_source == "embedded"
        if embedded:
            if self.requires_odometry and (context.base_pos_w is None or context.base_lin_vel_w is None):
                raise ValueError("Tracking requires fresh pelvis position/velocity odometry.")
            self._update_motion_reference(context)
            self._update_anchor_pose(context)
        else:
            self._update_external_reference(context)

        if self._observation_needs_prime:
            self.observation.prime(context)
            self._observation_needs_prime = False
        else:
            self.observation.update(context)
        obs = self.observation.compute().astype(np.float32, copy=False)
        if not embedded and not np.isfinite(obs).all():
            raise MotionUnavailable("Reference tracking observation contains nonfinite values")
        self.action[:] = self._infer_action(obs)
        if not np.isfinite(self.action).all():
            raise MotionUnavailable("Reference tracking actor produced nonfinite actions")
        if self.action_clip is not None:
            np.clip(self.action, -self.action_clip, self.action_clip, out=self.action)
        previous = self.action * self.action_scaling if self.obs_use_scaled_prev_action else self.action
        self.previous_action_observation.record_action(previous)

        base = self.default_joint_pos_action
        if self.config.get("use_reference_residual", False):
            base = (self.motion_reference["joint_pos"] if embedded else
                    self.reference_window["joint_pos"][np.searchsorted(self.motion.steps, 0)])
        self.target_q[:] = self.default_joint_pos
        self.target_q[self.action_to_obs_indices] = base + self.action_scaling * self.action
        if embedded:
            self.motion_finished = self.time_step >= self.num_frames - 1
            self.time_step = min(self.time_step + 1, self.num_frames - 1)
        else:
            self.motion_finished = (self.clip.finished if self.clip is not None
                                    else self.reference_window.get("file_finished", False))
        return self.target_q

    def _run(self, obs, outputs, *, time_step=None):
        inputs = {
            self.input_name: obs.reshape(1, -1).astype(np.float32),
            "time_step": np.array([self.time_step if time_step is None else time_step], np.int64),
        }
        return self.session.run(outputs, inputs)

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

    def set_reference_hold(self, hold: bool) -> None:
        self.reference_hold = hold
        if hold and self.replacement_reference is not None:
            self._stop_playback()
            self.playback_requested = False
            self.preview = self.replacement_reference
            self.reference_session = self.preview["session"]
            self.replacement_reference = None
            self.robot_heading_offset = None
            self.ghost_translation = None

    def set_simulation_timing(self, paused: bool, speed: float) -> None:
        if self.reference_source == "embedded":
            return
        rate = 0.0 if paused else speed
        if rate != self.playback_rate:
            self.playback_rate = rate
            if self.playback_requested:
                self.motion.set_playback(self.reference_session, True, rate=rate)

    def _stop_playback(self) -> None:
        if self.playback_requested:
            self.motion.set_playback(self.reference_session, False)

    def _freeze_reference(self, window, *, first=False):
        frame = window.get("first_frame") if first else None
        current = np.searchsorted(self.motion.steps, 0)
        frozen = dict(window)
        for name in ("joint_pos", "body_pos_w", "body_quat_w"):
            value = getattr(frame, name) if frame is not None else window[name][current]
            frozen[name] = np.repeat(value[None], len(self.motion.steps), axis=0)
        for name in ("joint_vel", "body_lin_vel_w", "body_ang_vel_w"):
            if name in frozen:
                frozen[name] = np.zeros_like(frozen[name])
        frozen["valid"] = np.zeros(len(self.motion.steps), dtype=bool)
        return frozen

    def _sample_reference(self, *, hold):
        if self.clip is None:
            try:
                window = self.motion.sample()
            except ReferenceStreamLost:
                self.replacement_reference = None
                raise
        else:
            window = self.clip.sample(advance=not hold)
        session = window["session"]
        if self.reference_session is not None and session != self.reference_session:
            if window.get("file_playing"):
                self.motion.set_playback(session, False)
                self.replacement_reference = None
            elif self.replacement_reference is None or self.replacement_reference["session"] != session:
                self.replacement_reference = self._freeze_reference(window, first=True)
            raise ReferenceStreamLost("reference motion publisher changed")
        self.replacement_reference = None
        if self.clip is None:
            file_playing = window.get("file_playing")
            if hold:
                if file_playing:
                    self.motion.set_playback(session, False)
                    raise MotionUnavailable("waiting for the file source to rewind")
                if self.preview is None:
                    self.preview = self._freeze_reference(window, first=True)
                window = self.preview
            elif file_playing is not None and (not file_playing or not self.playback_requested):
                self.motion.set_playback(session, True, rate=self.playback_rate)
                self.playback_requested = True
        self.reference_session = session
        return window

    def first_reference_q(self) -> np.ndarray:
        """Read the frozen preview without running the actor or advancing playback."""
        if self.reference_source == "embedded":
            target = self.default_joint_pos.copy()
            target[self.action_to_obs_indices] = self._run(self._zero_obs, ["joint_pos"], time_step=0)[0][0]
            return target
        try:
            window = self._sample_reference(hold=True)
        except ReferenceStreamLost:
            if self.replacement_reference is None:
                raise
            window = self.replacement_reference
        current = np.searchsorted(self.motion.steps, 0)
        target = self.default_joint_pos.copy()
        target[self.action_to_obs_indices] = window["joint_pos"][current]
        return target

    def _update_external_reference(self, context):
        try:
            self.reference_window = self._sample_reference(hold=self.reference_hold)
        except ReferenceStreamLost:
            if self.reference_window is None:
                raise
            if not self.reference_lost:
                self.reference_window = self._freeze_reference(self.reference_window)
            self.reference_lost = True
        else:
            self.reference_lost = False
        window = self.reference_window
        if self.robot_heading_offset is None:
            current = self.motion.steps.tolist().index(0)
            ref_quat = self.reference_window["body_quat_w"][current, self.root_index]
            self.robot_heading_offset = yaw_rotation(ref_quat) * yaw_rotation(context.quat).inv()
        if self.ghost_translation is None and context.base_pos_w is not None:
            current = self.motion.steps.tolist().index(0)
            root = window["body_pos_w"][current, self.root_index]
            self.ghost_translation = (context.base_pos_w - self.robot_heading_offset.inv().apply(root))[:2]

    def close(self):
        if self.reference_source == "external" and self.clip is None:
            self._stop_playback()
            self.motion.close()

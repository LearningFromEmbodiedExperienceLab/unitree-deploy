"""Render policy references and velocity tracking with one shared visibility toggle."""

import json
import threading
import time
from dataclasses import dataclass

import mujoco
import numpy as np

from unitree_deploy.obs.base_obs import CommandObservation
from unitree_deploy.robot_model.robot_config import RobotModel
from unitree_deploy.utils.yaml_utils import load_yaml


POLICY_DEBUG_VIZ_TOPIC = 'rt/policy_debug_viz'


@dataclass(frozen=True)
class VelocityVizConfig:
    """Body-frame offset in meters; scale affects arrow lengths, not the offset."""

    offset: tuple[float, float, float] = (0., 0., .1)
    scale: float = .5
    width: float = .015

    def __post_init__(self):
        offset = np.asarray(self.offset, dtype=np.float64)
        if offset.shape != (3,) or not np.all(np.isfinite(offset)):
            raise ValueError('velocity.offset must contain 3 finite values')
        for field in ('scale', 'width'):
            value = getattr(self, field)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f'velocity.{field} must be positive and finite')


def load_velocity_viz_config(robot: RobotModel) -> VelocityVizConfig:
    """Read viewer.policy_debug_viz.velocity from the robot's visualizer.yaml."""
    defaults = VelocityVizConfig()
    path = robot.config_dir / 'visualizer.yaml'
    if not path.exists():
        return defaults
    settings = load_yaml(path).get('viewer', {}).get('policy_debug_viz', {}).get('velocity', {})
    return VelocityVizConfig(
        offset=tuple(settings.get('offset', defaults.offset)),
        scale=float(settings.get('scale', defaults.scale)),
        width=float(settings.get('width', defaults.width)),
    )


def policy_debug_viz_packet(policy, settings, command=None) -> str:
    """Send the active reference and unscaled velocity command in physical units."""
    color = [float(value) for value in settings.get('color', [.35, .65, 1.])]
    opacity = float(settings.get('opacity', .25))
    if len(color) != 3 or not all(0 <= value <= 1 for value in color):
        raise ValueError('motion_ghost.color must contain 3 RGB values between 0 and 1')
    if not 0 <= opacity <= 1:
        raise ValueError('motion_ghost.opacity must be between 0 and 1')
    packet = {'enabled': bool(settings.get('enabled', True)),
              'toggle_key': str(settings.get('toggle_key', 'g')).lower(), 'active': False,
              'color': color, 'opacity': opacity, 'velocity_command': None}
    reference_pose = getattr(policy, 'reference_pose', None)
    if reference_pose is not None:
        ref = reference_pose()
        if ref is not None:
            packet.update(active=True, joint_names=list(ref.joint_names),
                          joint_pos=ref.joint_pos.tolist(), root_pos=ref.root_pos.tolist(),
                          root_quat=ref.root_quat.tolist())
    elif policy is not None:
        for term in policy.observation.observations:
            if isinstance(term, CommandObservation) and term.base_dim == 3:
                packet['velocity_command'] = np.clip(
                    np.asarray(command)[:3], term.command_min, term.command_max,
                ).tolist()
                break
    return json.dumps(packet)


class PolicyDebugViz:
    def __init__(self, model, live_data, *, velocity_config: VelocityVizConfig | None = None):
        self.model = model
        self.live_data = live_data
        self.velocity_config = velocity_config or VelocityVizConfig()
        # Ghost kinematics use scratch data; arrows read the live simulation state.
        self.data = mujoco.MjData(model)
        self.scene = mujoco.MjvScene(model, maxgeom=model.ngeom + 20)
        self.option = mujoco.MjvOption()
        self.option.geomgroup[:] = 0
        self.option.geomgroup[2] = 1  # robot visual meshes; no collision or terrain geometry
        self.lock = threading.Lock()
        self.frame = None
        self.received_at = 0.0
        self.visible = None
        self.toggle_key = 'g'
        self.joint_indices = None
        self.joint_names = None
        free_joint = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)[0]
        self.root_qpos = int(model.jnt_qposadr[free_joint])
        self.root_body = int(model.jnt_bodyid[free_joint])

    def receive(self, payload: str) -> None:
        frame = json.loads(payload)
        with self.lock:
            self.frame = frame
            self.received_at = time.monotonic()
            self.toggle_key = frame['toggle_key']
            if self.visible is None:
                self.visible = frame['enabled']

    def handle_key(self, key: str) -> bool:
        with self.lock:
            if key != self.toggle_key:
                return False
            self.visible = not (True if self.visible is None else self.visible)
        return True

    def add_to_scene(self, scene) -> None:
        with self.lock:
            frame = self.frame
            visible = self.visible
            fresh = time.monotonic() - self.received_at < .5
        if not visible or not fresh or frame is None:
            return
        command = frame.get('velocity_command')
        if command is not None:
            self._add_velocity_arrows(scene, command)
        if frame['active']:
            self._add_motion_ghost(scene, frame)

    def _add_velocity_arrows(self, scene, command) -> None:
        # Match mjlab's body-frame arrows, including yaw-rate arrows along local Z.
        data = self.live_data
        rotation = data.xmat[self.root_body].reshape(3, 3)
        start = data.xpos[self.root_body] + rotation @ np.asarray(self.velocity_config.offset)
        velocity = np.zeros(6)
        # XBODY measures the base-link origin, not the displaced center of mass.
        mujoco.mj_objectVelocity(self.model, data, mujoco.mjtObj.mjOBJ_XBODY,
                                self.root_body, velocity, 1)
        arrows = (
            ([command[0], command[1], 0.], [.2, .2, .6, .6]),
            ([0., 0., command[2]], [.2, .6, .2, .6]),
            ([velocity[3], velocity[4], 0.], [0., .6, 1., .7]),
            ([0., 0., velocity[2]], [0., 1., .4, .7]),
        )
        for vector, color in arrows:
            delta = rotation @ np.asarray(vector) * self.velocity_config.scale
            if np.linalg.norm(delta) < 1e-6 or scene.ngeom >= scene.maxgeom:
                continue
            geom = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3),
                               start, np.eye(3).reshape(9), np.asarray(color, dtype=np.float32))
            mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_ARROW,
                                self.velocity_config.width, start, start + delta)
            geom.category = mujoco.mjtCatBit.mjCAT_DECOR
            geom.transparent = True
            scene.ngeom += 1

    def _add_motion_ghost(self, scene, frame) -> None:
        names = tuple(frame['joint_names'])
        if names != self.joint_names:
            ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in names]
            if -1 in ids:
                raise ValueError('Motion ghost joint names do not match the simulated robot')
            self.joint_indices = self.model.jnt_qposadr[ids]
            self.joint_names = names
        root = self.root_qpos
        self.data.qpos[root:root + 3] = frame['root_pos']
        self.data.qpos[root + 3:root + 7] = frame['root_quat']
        self.data.qpos[self.joint_indices] = frame['joint_pos']
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mjv_updateScene(self.model, self.data, self.option, None, mujoco.MjvCamera(),
                              mujoco.mjtCatBit.mjCAT_DYNAMIC, self.scene)
        rgba = np.array([*frame.get('color', [.35, .65, 1.]), frame.get('opacity', .25)], dtype=np.float32)
        for source in self.scene.geoms[:self.scene.ngeom]:
            if source.objtype != mujoco.mjtObj.mjOBJ_GEOM or self.model.geom_bodyid[source.objid] == 0:
                continue
            if scene.ngeom >= scene.maxgeom:
                break
            geom = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(geom, source.type, source.size.astype(np.float64),
                               source.pos.astype(np.float64), source.mat.astype(np.float64).reshape(9),
                               rgba)
            geom.dataid = source.dataid
            geom.modelrbound = source.modelrbound
            geom.matid = -1
            geom.category = mujoco.mjtCatBit.mjCAT_DECOR
            geom.transparent = rgba[3] < 1
            geom.emission = .2
            scene.ngeom += 1

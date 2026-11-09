"""Pico/XRobot to G1 using RoboKit's online humanoid solver."""

import xml.etree.ElementTree as ET

import numpy as np

from unitree_deploy.robot_model.robot_config import ROBOT_MODEL_ROOT
from .retarget import RobotPose
from .xrobot_g1 import XROBOT_G1_TARGETS


class RoboKitRetargeter:
    def __init__(self, *, human_height: float, step_dt: float = .02):
        if not np.isfinite([human_height, step_dt]).all() or min(human_height, step_dt) <= 0:
            raise ValueError("human_height and step_dt must be finite and positive")

        from robokit.helpers.humanoid_retargeting import HumanoidRetargeterOnline
        from robokit.helpers.humanoid_retargeting.config import HumanoidRetargeterOnlineConfig, LinkMapping
        from robokit.opt.lm_optimizer import LMOptimizerConfig
        from robokit.opt.population_solver import PopulationSolverConfig, Stage
        from robokit.robo import Robot

        model_path = ROBOT_MODEL_ROOT / "g1/g1.xml"
        model = ET.parse(model_path).getroot()
        pelvis = model.find("./worldbody/body[@name='pelvis']")
        # RoboKit optimizes a separate floating base; its robot model holds only the joints.
        pelvis.remove(pelvis.find("freejoint"))
        compiler = model.find("compiler")
        compiler.set("meshdir", str(model_path.parent / compiler.get("meshdir")))
        robot = Robot.load(ET.tostring(model, encoding="unicode"), base_link_name="pelvis")
        links_by_joint = {joint.get("name"): link.get("name")
                          for link in pelvis.iter("body") for joint in link.findall("joint")}

        config = HumanoidRetargeterOnlineConfig(
            human_root_name="Pelvis", velocity_limit_dt=step_dt,
            solver=PopulationSolverConfig(
                # Start with stronger damping so a large first pose can leave the zero seed.
                stages=[Stage(6, 4, LMOptimizerConfig(lm_lambda=1000.)), Stage(3, 1)],
                cuda_graph_mode="full",
            ),
        )
        position_weights, orientation_weights = {}, {}
        for joint, human, pos_weight, rot_weight, scale, rotation in XROBOT_G1_TARGETS:
            link = "pelvis" if joint == "yaw_joint" else links_by_joint[joint]
            offset = [0., 0., 0.]
            if human.endswith("_Foot"):
                link = link.removesuffix("_link") + "_toe_link"
                # The deployment toe is 2 cm above the calibrated GMR toe.
                offset = [.05, 0., .02]
            config.link_mapping[link] = LinkMapping(
                human, np.asarray(offset, dtype=np.float32), np.asarray(rotation, dtype=np.float32),
            )
            # Use RoboKit's default height input (unit multiplier); preserve the shared XRobot scale.
            config.scale_table[human] = scale * human_height / 1.8
            # Like RoboKit's G1 preset, use arm positions to disambiguate IK branches.
            if human.endswith(("_Shoulder", "_Elbow", "_Wrist")):
                pos_weight = 30
            elif human == "Pelvis":
                pos_weight = 2000
            position_weights[link], orientation_weights[link] = pos_weight, rot_weight
        config.task("target").position_weight = position_weights
        config.task("target").orientation_weight = orientation_weights

        self.backend = HumanoidRetargeterOnline(config, robot=robot, device="cpu")
        self.joint_names = tuple(robot.spec.actuated_joint_names)
        self.human_names = tuple(self.backend.human_joint_names)
        self._frame = np.zeros((1, len(self.human_names), 7), dtype=np.float32)
        self._frame[..., 2:4] = [.8, 1.]
        # Compile the CPU kernels before DDS/XR starts and discard the warmup pose.
        self.backend.warmup(batch_size=1)
        self.backend.solve_numpy(self._frame)
        self.backend.reset()

    def retarget(self, body_pose):
        for i, name in enumerate(self.human_names):
            self._frame[0, i, :3], self._frame[0, i, 3:] = body_pose[name]
        if not np.isfinite(self._frame).all():
            raise ValueError("RoboKit requires finite XRobot poses")
        q = self.backend.solve_numpy(self._frame)[0]
        if not np.isfinite(q).all():
            raise RuntimeError("RoboKit returned a nonfinite G1 pose")
        return RobotPose(q[:3], q[3:7], self.joint_names, q[7:])

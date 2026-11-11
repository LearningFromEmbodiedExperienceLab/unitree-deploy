"""Publish Pico or local any4hdmi robot references over DDS."""

import argparse
from itertools import count
from pathlib import Path
import sys
import time
import uuid

import numpy as np
from tqdm import tqdm

from unitree_deploy.policy.motion_tracking.clip import Any4HDMIMotionClip
from unitree_deploy.robot_model.robot_config import ROBOT_MODEL_ROOT
from unitree_deploy.runtime.timing import LoopTimer
from unitree_deploy.runtime.transport.reference_motion import REFERENCE_MOTION_TOPIC, MotionFrame
from unitree_deploy.teleop.retarget import RobotPose, RetargetSession
from unitree_deploy.teleop.xrobot import XRobotSource
from unitree_deploy.utils.import_utils import import_symbol
from unitree_deploy.utils.terminal_status import ComponentConsole
from unitree_deploy.utils.yaml_utils import load_yaml


console = ComponentConsole("motion-source", "magenta")


class ReferenceKinematics:
    """Use deployment FK and named joints; never assume GMR's qpos joint order."""

    def __init__(self, config):
        import mujoco

        if config["robot"] != "g1":
            raise ValueError("Pico deployment currently supports G1")
        spec = mujoco.MjSpec.from_file(str(ROBOT_MODEL_ROOT / "g1/g1.xml"))
        spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, .01])
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self.joint_names = tuple(config["action_joint_order"])
        self.body_names = tuple(config["body_names"])
        self.joint_indices = [self.model.jnt_qposadr[self.model.joint(name).id] for name in self.joint_names]
        self.body_indices = [self.model.body(name).id for name in self.body_names]
        sdk_names = config["sdk_joint_order"]
        self.default_pose = RobotPose(
            self.data.qpos[:3].copy(), self.data.qpos[3:7].copy(), self.joint_names,
            np.array([config["default_qpos"][sdk_names.index(name)] for name in self.joint_names]),
        )

    def set_pose(self, pose):
        import mujoco

        self.data.qpos[:3] = pose.root_pos
        self.data.qpos[3:7] = pose.root_quat
        joints = pose.joint_pos[[pose.joint_names.index(name) for name in self.joint_names]]
        self.data.qpos[self.joint_indices] = joints
        mujoco.mj_kinematics(self.model, self.data)

    def frame(self, pose, *, session, seq, time_s):
        self.set_pose(pose)
        return MotionFrame(session, seq, time_s, self.joint_names, self.body_names,
                           self.data.qpos[self.joint_indices].copy(),
                           self.data.xpos[self.body_indices].copy(), self.data.xquat[self.body_indices].copy())


def publish_clip(clip, joint_names, root_body_name, dt, publish, playback_rate, session):
    """Keep publishing frame zero until the controller starts playback."""
    timer = LoopTimer(dt)
    playing = False
    last = len(clip.joint_pos) - 1
    last_sent = 0
    motion_tick = 0
    progress = None
    console.log("Holding the first reference frame. Press B to start motion playback.")
    try:
        for seq in count():
            rate = playback_rate()
            requested = rate is not None
            if requested != playing:
                playing = requested
                if playing:
                    console.log("Playing motion.")
                    progress = tqdm(total=last + 1, desc="[motion-source]", unit="frame",
                                    file=sys.stdout, dynamic_ncols=True, mininterval=.2)
                else:
                    if progress is not None:
                        progress.close()
                        progress = None
                    clip.reset()
                    last_sent = 0
                    console.log("Holding the first reference frame. Press B to start motion playback.")
            index = last_sent if rate == 0 else clip.frame
            at_end = playing and index == last
            if rate != 0:
                motion_tick += 1
            publish(MotionFrame(session, seq, time.monotonic(), joint_names, (root_body_name,),
                                clip.joint_pos[index], clip.body_pos_w[index], clip.body_quat_w[index],
                                file_playing=playing, file_finished=at_end, file_time_s=motion_tick * dt))
            last_sent = index
            if progress is not None and rate != 0:
                progress.update()
                if at_end:
                    progress.close()
                    progress = None
                    clip.finished = True
                    console.log("Motion finished; holding the final reference frame. Press Ctrl+C to exit.")
                else:
                    clip.frame += 1
            timer.sleep(rate or 1.0)
    finally:
        if progress is not None:
            progress.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="Deployment YAML; no ONNX is loaded.")
    parser.add_argument("--net", default="lo", help="DDS network interface shared with controller.")
    parser.add_argument("--human-height", type=float, help="Pico operator height in meters (required for Pico).")
    parser.add_argument("--publish-hz", type=float, default=50)
    parser.add_argument("--no-visualizer", action="store_true", help="Disable the Viser reference preview.")
    retarget = parser.add_mutually_exclusive_group()
    retarget.add_argument("--retarget", choices=("gmr", "frax", "robokit"), default="gmr")
    retarget.add_argument("--retarget-class", help="Custom backend as module:Class; overrides the built-in selector.")
    retarget.add_argument("--motion-path", type=Path, help="Publish a local any4hdmi .npz instead of Pico input.")
    args = parser.parse_args()
    if not np.isfinite(args.publish_hz) or args.publish_hz <= 0:
        parser.error("publish-hz must be finite and positive")
    if args.motion_path is None and (
        args.human_height is None or not np.isfinite(args.human_height) or args.human_height <= 0
    ):
        parser.error("Pico requires a finite positive --human-height")
    config_path = args.config.expanduser().resolve()
    config = load_yaml(config_path)
    dt = 1 / args.publish_hz
    clip = None
    if args.motion_path is not None:
        root_body_name = config["root_body_name"]
        if config["body_names"] != [root_body_name]:
            parser.error("File publishing requires a policy with only the root body in its reference")
        joint_names = tuple(config["action_joint_order"])
        clip = Any4HDMIMotionClip(args.motion_path.expanduser().resolve(), joint_names, dt=dt, future_steps=[],
                                 include_endpoint=config.get("motion_include_endpoint", True))
    else:
        backends = {
            "gmr": "unitree_deploy.teleop.retarget:GMRRetargeter",
            "frax": "unitree_deploy.teleop.frax:FraxRetargeter",
            "robokit": "unitree_deploy.teleop.robokit:RoboKitRetargeter",
        }
        backend_class = import_symbol(args.retarget_class or backends[args.retarget], config_path.parent)
        backend_kwargs = {"human_height": args.human_height}
        if args.retarget_class is None and args.retarget in ("frax", "robokit"):
            backend_kwargs["step_dt"] = dt
        backend = backend_class(**backend_kwargs)
        fk = ReferenceKinematics(config)

    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
    from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_

    ChannelFactoryInitialize(0, args.net)
    topic = config.get("reference_motion", {}).get("topic", REFERENCE_MOTION_TOPIC)
    publisher = ChannelPublisher(topic, String_)
    publisher.Init()
    preview = None
    if clip is not None:
        session = uuid.uuid4().hex
        rate = None

        def on_playback(message):
            nonlocal rate
            command, _, address = message.data.partition(":")
            target, _, value = address.partition(":")
            if target != session:
                return
            if command == "hold":
                rate = None
            elif command == "play":
                try:
                    requested = float(value or 1.0)
                except ValueError:
                    return
                if np.isfinite(requested) and requested >= 0:
                    rate = requested

        playback = ChannelSubscriber(f"{topic}/playback", String_)
        playback.Init(on_playback, 1)
        try:
            if not args.no_visualizer:
                from unitree_deploy.visualization.motion_source import MotionSourceVisualizer

                fk = ReferenceKinematics(config)
                fk.set_pose(RobotPose(clip.body_pos_w[0, 0], clip.body_quat_w[0, 0],
                                      joint_names, clip.joint_pos[0]))
                preview = MotionSourceVisualizer(fk.model, fk.data.qpos)

            def publish(frame):
                publisher.Write(String_(frame.encode()))
                if preview is not None:
                    fk.set_pose(RobotPose(frame.body_pos_w[0], frame.body_quat_w[0],
                                          frame.joint_names, frame.joint_pos))
                    preview.update(fk.data.qpos)

            publish_clip(clip, joint_names, root_body_name, dt, publish, lambda: rate, session)
        except KeyboardInterrupt:
            pass
        finally:
            playback.Close()
            publisher.Close()
            if preview is not None:
                preview.close()
        return
    source = XRobotSource()
    session, seq = uuid.uuid4().hex, 0
    was_pressed = False
    retarget_session = RetargetSession(backend, fk.default_pose)
    timer = LoopTimer(dt)
    console.log("Pico reference paused. Press Pico X to pause or resume the reference.")
    try:
        if not args.no_visualizer:
            from unitree_deploy.visualization.motion_source import MotionSourceVisualizer

            fk.set_pose(retarget_session.pose)
            preview = MotionSourceVisualizer(fk.model, fk.data.qpos)
        while True:
            body_pose = source.read()
            captured_at = time.monotonic()
            pressed = bool(source.sdk.get_X_button())
            if pressed and not was_pressed:
                retarget_session.toggle_pause()
                console.log(f"Pico reference {'paused' if retarget_session.paused else 'live'}")
            was_pressed = pressed
            if body_pose is not None:
                pose = retarget_session.update(body_pose)
                frame = fk.frame(pose, session=session, seq=seq, time_s=captured_at)
                publisher.Write(String_(frame.encode()))
                if preview is not None:
                    preview.update(fk.data.qpos)
                seq += 1
            # No heartbeat pose on missing XR input: the subscriber watchdog must expire.
            timer.sleep()
    except KeyboardInterrupt:
        pass
    finally:
        source.close()
        publisher.Close()
        if preview is not None:
            preview.close()


if __name__ == "__main__":
    main()

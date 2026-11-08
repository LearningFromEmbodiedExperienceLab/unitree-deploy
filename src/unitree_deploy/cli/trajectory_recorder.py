import argparse
from pathlib import Path

from unitree_deploy.config.defaults import DEFAULT_NET
from unitree_deploy.robot_model.robot_config import DEFAULT_ROBOT, DEFAULT_TERRAIN, load_robot_model
from unitree_deploy.runtime.trajectory.recorder import DdsTrajectoryRecorder, RuntimeConfig
from unitree_sdk2py.core.channel import ChannelFactoryInitialize


def parse_args() -> RuntimeConfig:
    parser = argparse.ArgumentParser(description="Record Unitree DDS state into trajectory.npz for rendering.")
    parser.add_argument("--robot", default=DEFAULT_ROBOT)
    parser.add_argument("--model-xml", type=Path, help="Optional MuJoCo XML override.")
    parser.add_argument("--terrain", default=DEFAULT_TERRAIN)
    parser.add_argument("--net", default=DEFAULT_NET, help="DDS network interface.")
    parser.add_argument("--out", type=Path, required=True, help="Recording root directory.")
    parser.add_argument("--record-hz", type=float, default=60.0)
    args = parser.parse_args()
    if args.record_hz <= 0.0:
        parser.error("--record-hz must be positive")
    return RuntimeConfig(
        robot=load_robot_model(args.robot, args.model_xml, args.terrain),
        out=args.out,
        net=args.net,
        record_hz=float(args.record_hz),
    )


def main() -> None:
    config = parse_args()
    ChannelFactoryInitialize(0, config.net)
    recorder = DdsTrajectoryRecorder(config)
    try:
        recorder.run()
    finally:
        recorder.close()
        recorder.cleanup()


if __name__ == "__main__":
    main()

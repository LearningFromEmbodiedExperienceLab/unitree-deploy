import argparse
from pathlib import Path

from unitree_deploy.robot_model.robot_config import DEFAULT_ROBOT, DEFAULT_TERRAIN, load_robot_model
from unitree_deploy.runtime.trajectory.replay import (
    DEFAULT_REPLAY_VIEWER,
    REPLAY_VIEWER_CHOICES,
    MujocoTrajectoryReplayApp,
    RuntimeConfig,
    ViserTrajectoryReplayApp,
)


def parse_args() -> RuntimeConfig:
    parser = argparse.ArgumentParser(description="Replay a saved Unitree trajectory.")
    parser.add_argument("trajectory", type=Path, help="Path to trajectory.npz or its directory.")
    parser.add_argument("--robot", default=None, help="Optional robot folder override.")
    parser.add_argument("--model-xml", type=Path, help="Optional MuJoCo XML override.")
    parser.add_argument("--terrain", default=None, help="Optional terrain override.")
    parser.add_argument(
        "--viewer",
        "--frontend",
        dest="viewer",
        choices=REPLAY_VIEWER_CHOICES,
        default=DEFAULT_REPLAY_VIEWER,
        help="Replay frontend.",
    )
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier.")
    parser.add_argument("--loop", action="store_true", help="Loop playback at the end.")
    args = parser.parse_args()

    trajectory = args.trajectory.expanduser().resolve()
    if trajectory.is_dir():
        trajectory = trajectory / "trajectory.npz"
    if not trajectory.exists():
        parser.error(f"trajectory file not found: {trajectory}")

    robot = None
    if args.robot is not None or args.model_xml is not None or args.terrain is not None:
        robot = load_robot_model(
            args.robot or DEFAULT_ROBOT,
            args.model_xml,
            args.terrain or DEFAULT_TERRAIN,
        )

    return RuntimeConfig(
        trajectory=trajectory,
        robot=robot,
        viewer=args.viewer,
        speed=float(args.speed),
        loop=bool(args.loop),
    )


def main() -> None:
    config = parse_args()
    app_cls = {
        "viser": ViserTrajectoryReplayApp,
        "mujoco": MujocoTrajectoryReplayApp,
    }[config.viewer]
    app = app_cls(config)
    try:
        app.run()
    finally:
        app.close()


if __name__ == "__main__":
    main()

import argparse
from pathlib import Path

from unitree_deploy.config.defaults import BAND_SITES, DEFAULT_NET, SIM_HZ
from unitree_deploy.robot_model.robot_config import (
    DEFAULT_ROBOT,
    DEFAULT_TERRAIN,
    DEFAULT_VIEWER,
    VIEWER_CHOICES,
    load_robot_model,
)
from unitree_deploy.runtime.simulation.bridge import RuntimeConfig, SimBridge
from unitree_deploy.visualization.video import VideoConfig
from unitree_sdk2py.core.channel import ChannelFactoryInitialize


def parse_args() -> RuntimeConfig:
    parser = argparse.ArgumentParser(description="MuJoCo to Unitree DDS bridge.")
    parser.add_argument("--robot", default=DEFAULT_ROBOT, help="Robot folder under robot_model/.")
    parser.add_argument("--model-xml", help="Override robot XML path.")
    parser.add_argument(
        "--terrain",
        default=DEFAULT_TERRAIN,
        help="Terrain name under robot_model/scene or XML path.",
    )
    parser.add_argument("--net", default=DEFAULT_NET, help="DDS network interface.")
    parser.add_argument(
        "--viewer",
        choices=VIEWER_CHOICES,
        default=DEFAULT_VIEWER,
        help="Visualization backend.",
    )
    parser.add_argument(
        "--band-sites",
        default=",".join(BAND_SITES),
        help="Comma-separated MuJoCo site names used by the suspension bands.",
    )
    parser.add_argument(
        "--band",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable suspension bands.",
    )
    parser.add_argument("--video", type=Path, help="Export the live camera to a new MP4 file (requires ffmpeg).")
    parser.add_argument("--video-width", type=int, default=960)
    parser.add_argument("--video-height", type=int, default=540)
    parser.add_argument("--video-fps", type=int, default=25)
    parser.add_argument("--video-start", type=float, default=0.0, help="Start at this simulation time in seconds.")
    parser.add_argument("--video-seconds", type=float, help="Duration in simulation seconds; default: until viewer closes.")
    args = parser.parse_args()
    video = None
    if args.video is not None:
        try:
            video = VideoConfig(args.video.expanduser(), args.video_width, args.video_height,
                                args.video_fps, args.video_start, args.video_seconds)
            if video.fps > SIM_HZ:
                parser.error(f"--video-fps must not exceed the simulation rate ({SIM_HZ})")
        except ValueError as error:
            parser.error(str(error))
    band_sites = tuple(site.strip() for site in args.band_sites.split(",") if site.strip())
    return RuntimeConfig(
        robot=load_robot_model(args.robot, args.model_xml, args.terrain),
        net=args.net,
        viewer=args.viewer,
        band_sites=band_sites,
        band_enabled=bool(args.band),
        video=video,
    )


def main() -> None:
    config = parse_args()
    ChannelFactoryInitialize(0, config.net)
    bridge = SimBridge(config)
    try:
        bridge.run()
    finally:
        bridge.close()
        bridge.cleanup()


if __name__ == "__main__":
    main()

"""Run the sole RealSense SDK owner and publish frames for local consumers."""

import argparse
from contextlib import ExitStack
from pathlib import Path
import signal
import time

from unitree_deploy.robot_model.robot_config import DEFAULT_ROBOT, ROBOT_MODEL_ROOT
from unitree_deploy.runtime.transport.realsense import RealSensePublisher
from unitree_deploy.sensors.realsense import RealSenseCapture, RealSenseCaptureConfig
from unitree_deploy.utils.yaml_utils import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", default=DEFAULT_ROBOT)
    parser.add_argument("--visualizer-yaml", type=Path, help="Use the same camera config as the visualizer.")
    parser.add_argument("--camera-name", help="Capture only this camera; default: all configured cameras.")
    args = parser.parse_args()
    path = args.visualizer_yaml or ROBOT_MODEL_ROOT / args.robot / "visualizer.yaml"
    configs = [RealSenseCaptureConfig.from_dict(item) for item in load_yaml(path).get("cameras", [])]
    if args.camera_name:
        configs = [config for config in configs if config.camera_name == args.camera_name]
    if not configs:
        parser.error("no matching cameras configured")
    if len({config.camera_name for config in configs}) != len(configs):
        parser.error("camera names must be unique")

    # SIGTERM follows the same cleanup path as Ctrl+C.
    signal.signal(signal.SIGTERM, signal.default_int_handler)
    try:
        with ExitStack() as stack:
            streams = []
            for config in configs:
                # Claim the shared stream before opening the USB device.
                publisher = RealSensePublisher(config)
                stack.callback(publisher.close)
                capture = RealSenseCapture(config)
                stack.callback(capture.close)
                streams.append((capture, publisher))
                print(f"[realsense] publishing {config.camera_name} ({capture.serial_number})", flush=True)
            while True:
                for capture, publisher in streams:
                    frame = capture.poll_frame()
                    if frame is not None:
                        publisher.publish(frame)
                time.sleep(0.002)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

import argparse
from pathlib import Path

from unitree_deploy.config.defaults import DEFAULT_MODE, DEFAULT_NET
from unitree_deploy.runtime.controllers.z1_controller import RuntimeConfig, Z1Controller, console
from unitree_sdk2py.core.channel import ChannelFactoryInitialize


def parse_args() -> RuntimeConfig:
    parser = argparse.ArgumentParser(description="Unitree Z1 impedance controller over DDS.")
    parser.add_argument("--mode", choices=("real", "sim"), default=DEFAULT_MODE)
    parser.add_argument("--net", default=DEFAULT_NET, help="DDS network interface.")
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=Path("ckpt/b2z1/b2z1_manip"),
        help="Directory containing the Z1 policy.yaml control profile.",
    )
    args = parser.parse_args()
    return RuntimeConfig(mode=args.mode, net=args.net, ckpt_dir=args.ckpt)


def main() -> None:
    config = parse_args()
    if config.mode == "real":
        console.log(
            "WARNING: Z1 real mode will start the UDP bridge and publish motor commands.",
            style="bold red",
        )
        input("Press Enter to continue...")

    if config.net:
        ChannelFactoryInitialize(0, config.net)
    else:
        ChannelFactoryInitialize(0)

    controller = Z1Controller(config)
    try:
        controller.spin()
    finally:
        controller.cleanup()


if __name__ == "__main__":
    main()

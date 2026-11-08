import argparse
from pathlib import Path

from unitree_deploy.config.defaults import DEFAULT_MODE, DEFAULT_NET
from unitree_deploy.runtime.controllers.policy_controller import Controller, RuntimeConfig, console
from unitree_sdk2py.core.channel import ChannelFactoryInitialize


def parse_args() -> RuntimeConfig:
    parser = argparse.ArgumentParser(description="Reusable Unitree controller for sim or real robot.")
    parser.add_argument("--mode", choices=("real", "sim"), default=DEFAULT_MODE)
    parser.add_argument("--net", default=DEFAULT_NET, help="DDS network interface. Use lo for local sim.")
    parser.add_argument("--robot", help="Robot name for logs. Defaults to controller.yaml robot.")
    parser.add_argument(
        "--ckpt",
        type=Path,
        help="Policy YAML file or checkpoint directory containing policy.yaml.",
    )
    parser.add_argument(
        "--multi-ckpt",
        type=Path,
        help="YAML manifest containing multiple ckpt directories.",
    )
    parser.add_argument(
        "--state-machine",
        type=Path,
        help="Optional YAML state machine. Defaults to state_machine.yaml beside the ckpt or multi-ckpt YAML.",
    )
    args = parser.parse_args()
    if args.ckpt is None and args.multi_ckpt is None:
        parser.error("one of --ckpt or --multi-ckpt is required")

    ckpt_dir = args.ckpt or args.multi_ckpt.expanduser().resolve().parent
    return RuntimeConfig(
        mode=args.mode,
        net=args.net,
        ckpt_dir=ckpt_dir,
        robot=args.robot,
        multi_ckpt=args.multi_ckpt,
        state_machine=args.state_machine,
    )


def main() -> None:
    config = parse_args()

    if config.mode == "real":
        console.log(
            "WARNING: Please ensure there are no obstacles around the robot while running controller.py.",
            style="bold red",
        )
        input("Press Enter to continue...")

    if config.net:
        ChannelFactoryInitialize(0, config.net)
    else:
        ChannelFactoryInitialize(0)

    controller = Controller(config)
    try:
        controller.spin()
    finally:
        controller.cleanup()


if __name__ == "__main__":
    main()

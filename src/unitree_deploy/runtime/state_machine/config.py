"""Load state-machine YAML and resolve its events, poses, and gains."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from unitree_deploy.config.defaults import (
    DAMPING_STATE,
    MOVE_TO_DEFAULT_STATE,
    MOVE_TO_DEFAULT_TIME,
    RUN_POLICY_STATE,
    WIRELESS_REMOTE_BUTTON_BITS,
)
from unitree_deploy.utils.yaml_utils import load_yaml


STATE_MACHINE_FILE = "state_machine.yaml"


def default_state_machine_config() -> dict[str, Any]:
    return {
        "initial": DAMPING_STATE,
        "states": {
            DAMPING_STATE: {
                "type": "damping",
                "transitions": {
                    "A": MOVE_TO_DEFAULT_STATE,
                },
            },
            MOVE_TO_DEFAULT_STATE: {
                "type": "move_to_target_pos",
                "target": "default_q",
                "duration": MOVE_TO_DEFAULT_TIME,
                "kp": "fixed_stand",
                "kd": "fixed_stand",
                "transitions": {
                    "Start": RUN_POLICY_STATE,
                    "X": DAMPING_STATE,
                },
            },
            RUN_POLICY_STATE: {
                "type": "policy",
                "transitions": {
                    "A": MOVE_TO_DEFAULT_STATE,
                    "X": DAMPING_STATE,
                },
            },
        },
    }


def resolve_state_machine_path(base_dir: Path, explicit_path: Path | None = None) -> Path | None:
    if explicit_path is not None:
        return explicit_path.expanduser().resolve()

    candidate = base_dir / STATE_MACHINE_FILE
    return candidate.resolve() if candidate.exists() else None


def load_state_machine_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return default_state_machine_config()
    config = load_yaml(path)
    if not isinstance(config, dict):
        raise TypeError(f"{path} must contain a mapping")
    return config


def parse_transitions(value: Any, state_name: str) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(f"transitions for state {state_name!r} must be a mapping")
    transitions = {str(event): str(target) for event, target in value.items() if target is not None}
    known_events = set(WIRELESS_REMOTE_BUTTON_BITS) | {"done"}
    unknown_events = sorted(set(transitions) - known_events)
    if unknown_events:
        raise KeyError(f"state {state_name!r} has unknown transition events: {unknown_events}")
    return transitions


def parse_event_set(value: Any, state_name: str, field: str) -> set[str]:
    if isinstance(value, str):
        events = {value}
    else:
        try:
            events = {str(item) for item in value}
        except TypeError as exc:
            raise TypeError(f"{field} for state {state_name!r} must be a string or iterable") from exc

    known_events = set(WIRELESS_REMOTE_BUTTON_BITS) | {"done"}
    unknown_events = sorted(events - known_events)
    if unknown_events:
        raise KeyError(f"state {state_name!r} has unknown {field}: {unknown_events}")
    return events


def resolve_target_q(controller, spec: Any, target_order: str) -> np.ndarray:
    profile = controller.active_profile
    if isinstance(spec, str):
        if spec in ("default_q", "default_qpos"):
            return profile.default_q_obs.copy()
        if spec not in profile.policy.config:
            raise KeyError(f"unknown target {spec!r}; define it in policy.yaml or use default_q")
        values = profile.policy.config[spec]
    else:
        values = spec

    target = np.asarray(values, dtype=np.float32).reshape(-1)
    if target.size != controller.num_joints:
        raise ValueError(f"target {spec!r} has {target.size} joints, expected {controller.num_joints}")
    if target_order == "sdk":
        return target[profile.sdk_to_obs].copy()
    if target_order == "obs":
        return target.copy()
    raise ValueError(f"target_order must be 'sdk' or 'obs', got {target_order!r}")


def resolve_kp(controller, spec: Any) -> np.ndarray:
    profile = controller.active_profile
    if spec == "fixed_stand":
        return profile.kp_fixed_stand
    if spec == "policy":
        return profile.kp_policy
    if spec == "impedance":
        return profile.kp_impedance
    return gain_array(controller, spec, "kp")


def resolve_kd(controller, spec: Any) -> np.ndarray:
    profile = controller.active_profile
    if spec == "fixed_stand":
        return profile.kd_fixed_stand
    if spec == "policy":
        return profile.kd_policy
    if spec == "impedance":
        return profile.kd_impedance
    if spec == "damping":
        return profile.kd_damping
    return gain_array(controller, spec, "kd")


def gain_array(controller, spec: Any, label: str) -> np.ndarray:
    if isinstance(spec, str):
        config = controller.active_profile.policy.config
        if spec not in config:
            raise KeyError(f"unknown {label} gain {spec!r}; define it in policy.yaml or use a built-in gain")
        values = config[spec]
    else:
        values = spec

    gain = np.asarray(values, dtype=np.float64).reshape(-1)
    if gain.size != controller.num_joints:
        raise ValueError(f"{label} gain has {gain.size} joints, expected {controller.num_joints}")
    return gain

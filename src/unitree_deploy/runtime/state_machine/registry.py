"""Construct configured states and validate their transition targets."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .states.arm_transition import ArmTransitionState
from .states.base import ControllerState
from .states.policy import PolicyState
from .states.pose import DampingState, ImpedanceState, MoveToTargetPosState


STATE_TYPES = {
    DampingState.state_type: DampingState,
    MoveToTargetPosState.state_type: MoveToTargetPosState,
    PolicyState.state_type: PolicyState,
    ArmTransitionState.state_type: ArmTransitionState,
    ImpedanceState.state_type: ImpedanceState,
}


def build_states(config: Mapping[str, Any]) -> dict[str, ControllerState]:
    specs = config.get("states")
    if not isinstance(specs, dict) or not specs:
        raise ValueError("state_machine.yaml must define a non-empty 'states' mapping")

    states: dict[str, ControllerState] = {}
    for name, spec in specs.items():
        if not isinstance(spec, dict):
            raise TypeError(f"state {name!r} must be a mapping")
        state_type = str(spec.get("type", ""))
        try:
            state_cls = STATE_TYPES[state_type]
        except KeyError as exc:
            known = ", ".join(sorted(STATE_TYPES))
            raise KeyError(f"unknown state type {state_type!r}; known types: {known}") from exc
        states[str(name)] = state_cls(str(name), spec)
    return states


def validate_transitions(states: Mapping[str, ControllerState]) -> None:
    known_states = set(states)
    for state in states.values():
        unknown_targets = sorted(set(state.transitions.values()) - known_states)
        if unknown_targets:
            raise KeyError(f"state {state.name!r} transitions to unknown states: {unknown_targets}")

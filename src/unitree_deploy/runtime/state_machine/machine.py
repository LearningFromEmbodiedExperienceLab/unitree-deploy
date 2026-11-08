from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any, Protocol

from unitree_deploy.config.defaults import DAMPING_STATE, WIRELESS_REMOTE_BUTTON_BITS
from .registry import build_states, validate_transitions
from .states.base import ControllerState


class StateMachineHost(Protocol):
    """Capabilities a host provides to the scheduler, without requiring inheritance.

    Individual states may need additional capabilities, such as policy inference.
    """

    state: str
    state_enter_t: float

    def button_pressed(self, name: str) -> bool:
        ...

    def on_state_transition(self) -> None:
        ...

    def log(self, message: str) -> None:
        ...


class ControllerStateMachine:
    """Dispatch button transitions before state work, then apply completion transitions."""

    def __init__(self, controller: StateMachineHost, config: Mapping[str, Any]) -> None:
        self.controller = controller
        self.initial = str(config.get("initial", DAMPING_STATE))
        self.states = build_states(config)
        if self.initial not in self.states:
            raise KeyError(f"initial state {self.initial!r} is not defined")
        validate_transitions(self.states)
        self.current_name = self.initial
        controller.state = self.initial
        controller.state_enter_t = time.perf_counter()
        self.states[self.current_name].on_enter(controller)

    @property
    def current(self) -> ControllerState:
        return self.states[self.current_name]

    def transition(self, state: str, *, force: bool = False) -> None:
        if state not in self.states:
            raise KeyError(f"unknown controller state {state!r}")
        if self.current_name == state and not force:
            return
        self.current_name = state
        self.controller.state = state
        self.controller.state_enter_t = time.perf_counter()
        self.controller.on_state_transition()
        self.current.on_enter(self.controller)
        self.controller.log(f"state -> {state}")

    def step(self) -> None:
        state = self.current
        # Consume this tick's edges even when blocked, so they cannot trigger a later state.
        pressed_buttons = [
            button for button in WIRELESS_REMOTE_BUTTON_BITS
            if self.controller.button_pressed(button)
        ]
        for button in pressed_buttons:
            next_state = state.next_for_button(button)
            if next_state:
                self.transition(next_state)
                return

        next_state = state.step(self.controller)
        if next_state:
            self.transition(next_state)

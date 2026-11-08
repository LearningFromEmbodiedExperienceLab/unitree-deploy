from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

from ..config import parse_transitions


class ControllerState(ABC):
    """State lifecycle: on_enter per entry, step each tick, transitions by button or done."""

    state_type = "base"

    def __init__(self, name: str, config: Mapping[str, Any]) -> None:
        self.name = name
        self.reset_policy_on_enter = bool(config.get("reset_policy", True))
        self.transitions = parse_transitions(config.get("transitions", {}), name)

    def on_enter(self, controller) -> None:
        """Optional setup when entering the state."""

    @abstractmethod
    def step(self, controller) -> str | None:
        """Run one control tick; return a next state name or None to stay."""
        raise NotImplementedError

    def next_for_button(self, button: str) -> str | None:
        return self.transitions.get(button)

    def next_on_done(self) -> str | None:
        return self.transitions.get("done")

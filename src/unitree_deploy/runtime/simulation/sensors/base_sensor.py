from abc import ABC, abstractmethod
import time

import mujoco


class MujocoSensor(ABC):
    """Use simulation time while running and the host monotonic clock while paused."""

    def __init__(self, *, rate_hz: float):
        self.period = 1 / rate_hz
        self._last_sim_time = float("-inf")
        self._last_publish = float("-inf")

    @abstractmethod
    def configure_model(self, spec: mujoco.MjSpec) -> None:
        """Configure the sensor before the bridge compiles the model."""

    def update(self, model: mujoco.MjModel, data: mujoco.MjData, *, paused: bool) -> None:
        now = time.monotonic()
        elapsed = data.time - self._last_sim_time
        if paused:
            if elapsed == 0 and now - self._last_publish < self.period:
                return
        elif 0 <= elapsed < self.period - 1e-9:
            return

        if self._capture_and_publish(model, data, captured_at=now):
            self._last_sim_time = data.time
            self._last_publish = now

    @abstractmethod
    def _capture_and_publish(
        self, model: mujoco.MjModel, data: mujoco.MjData, *, captured_at: float,
    ) -> bool:
        """Capture and publish a frame. Return False to retry at the next update."""

    @abstractmethod
    def close(self) -> None:
        """Release sensor resources when the bridge exits."""

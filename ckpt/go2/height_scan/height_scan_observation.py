import time

import numpy as np

from unitree_deploy.obs import ObservationBase, ObservationContext, ObservationUnavailable
from unitree_deploy.runtime.transport.height_scan import HeightScanReader


class HeightScanObservation(ObservationBase):
    """Read a base-relative height grid, with history stacked at policy rate."""

    def __init__(
        self, *, shape, history_len, stream_name="go2_height_scan", source=None,
        max_age=0.5, offset=0.0, fill_miss=-1.0, history_order="oldest_first",
    ):
        self.shape = tuple(shape)
        if len(self.shape) != 2 or min(self.shape) <= 0 or history_len <= 0:
            raise ValueError("height scan needs a positive [X, Y] shape and history length")
        if not np.isfinite(max_age) or max_age <= 0:
            raise ValueError("height scan max_age must be finite and positive")
        super().__init__(
            base_dim=int(np.prod(self.shape)), history_len=history_len, history_order=history_order,
        )
        self.source = HeightScanReader(stream_name) if source is None else source
        self.max_age = max_age
        self.offset = offset
        self.fill_miss = fill_miss

    def compute(self, context: ObservationContext) -> np.ndarray:
        frame = self.source.read()
        if frame is None:
            raise ObservationUnavailable("height scan has no frame")
        grid, captured_at = frame
        age = time.monotonic() - captured_at
        if not 0 <= age <= self.max_age:
            raise ObservationUnavailable("height scan has no fresh frame")
        if grid.shape != self.shape:
            raise ObservationUnavailable(f"height scan shape {grid.shape} does not match {self.shape}")
        return np.where(np.isfinite(grid), grid, self.fill_miss) + self.offset

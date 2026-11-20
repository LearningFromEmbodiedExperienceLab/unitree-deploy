from collections import deque
import time

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

from unitree_deploy.obs import ObservationBase, ObservationContext, ObservationUnavailable


class DepthImageObservation(ObservationBase):
    """Rectify the depth field of view and sample history in camera time."""

    def __init__(
        self,
        *,
        history_len: int,
        history_interval_sec: float,
        open_source,
        width: int,
        height: int,
        fovx: float,
        fovy: float,
        crop: dict,
        clip_range: tuple[float, float],
        blur_sigma: float,
        blur_radius: int,
        max_age: float,
        history_order: str = "oldest_first",
    ) -> None:
        self.image_shape = (height - crop["top"] - crop["bottom"], width - crop["left"] - crop["right"])
        if min(self.image_shape) <= 0 or history_len < 1 or history_interval_sec <= 0 or max_age <= 0:
            raise ValueError("depth crop, history length, interval, and max_age must be valid")
        super().__init__(
            base_dim=int(np.prod(self.image_shape)), history_len=history_len, history_order=history_order,
        )
        self.clip_min, self.clip_max = clip_range
        if not 0 <= self.clip_min < self.clip_max:
            raise ValueError("depth clip_range must be increasing and nonnegative")
        self.blur_sigma = blur_sigma
        self.blur_radius = blur_radius
        self.max_age = max_age
        self.history_offsets = np.arange(history_len - 1, -1, -1) * history_interval_sec
        self.frames = deque()
        self.frame = None
        self.received_at = float("-inf")

        self.tan_half_fovx = np.tan(np.deg2rad(fovx) / 2)
        self.tan_half_fovy = np.tan(np.deg2rad(fovy) / 2)
        rows, columns = np.indices(self.image_shape, dtype=np.float64)
        self.rays_y = (rows + crop["top"] + 0.5 - height / 2) * (2 * self.tan_half_fovy / height)
        self.rays_x = (columns + crop["left"] + 0.5 - width / 2) * (2 * self.tan_half_fovx / width)
        self.source = open_source()

    def reset(self) -> None:
        super().reset()
        self.frames.clear()
        self.frame = None
        self.received_at = float("-inf")

    def close(self) -> None:
        self.source.close()

    def update(self, context: ObservationContext) -> None:
        frame = self.source.read()
        now = time.monotonic()
        if frame is not None and frame.depth is not None:
            published_at = now if frame.published_at is None else frame.published_at
            stamp = published_at if frame.timestamp_sec is None else frame.timestamp_sec
            if now - published_at <= self.max_age:
                if not self.frames or stamp != self.frames[-1][0]:
                    self._append_frame(context, frame, stamp, now)
                    self.received_at = published_at
                elif frame.published_at is not None:
                    self.received_at = published_at
        if now - self.received_at > self.max_age:
            raise ObservationUnavailable("depth camera has no fresh frame")

    def _append_frame(self, context, frame, stamp, now) -> None:
        if self.frames and (stamp < self.frames[-1][0] or now - self.received_at > self.max_age):
            self.frames.clear()
        self.frame = frame
        self.frames.append((stamp, self._compute_processed_current_obs(context)))
        oldest = stamp - self.history_offsets[0]
        # Keep one frame before the oldest requested sample for nearest-time lookup.
        while len(self.frames) > 1 and self.frames[1][0] <= oldest:
            self.frames.popleft()
        for i, offset in enumerate(self.history_offsets):
            _, values = min(self.frames, key=lambda item: abs(item[0] - (stamp - offset)))
            self.buffer[i] = values

    def prime(self, context: ObservationContext) -> None:
        self.reset()
        self.update(context)

    def compute(self, context: ObservationContext) -> np.ndarray:
        depth = np.asarray(self.frame.depth, dtype=np.float32)
        height, width = depth.shape
        intrinsics = self.frame.depth_intrinsics
        if intrinsics is None:
            intrinsics = (
                width / (2 * self.tan_half_fovx), height / (2 * self.tan_half_fovy),
                (width - 1) / 2, (height - 1) / 2,
            )
        fx, fy, cx, cy = intrinsics
        depth = np.where(np.isfinite(depth) & (depth > 0), depth, self.clip_max)
        image = map_coordinates(depth, (fy * self.rays_y + cy, fx * self.rays_x + cx), order=1, mode="nearest")
        if self.blur_sigma > 0:
            image = gaussian_filter(image, self.blur_sigma, radius=self.blur_radius, mode="mirror")
        np.clip(image, self.clip_min, self.clip_max, out=image)
        return (image - self.clip_min) / (self.clip_max - self.clip_min)

"""Latest RGB/depth frame in Linux shared memory, with independent process readers.

Use mmap on /dev/shm so exiting a reader never unlinks the publisher's memory
(including on Python 3.11/3.12, where SharedMemory readers are resource-tracked).

Layout: [4-byte JSON length][JSON metadata][RGB uint8][optional depth float32].
A zero JSON length marks an empty or unfinished frame.
"""

from contextlib import ExitStack
import fcntl
import json
import mmap
import os
from pathlib import Path
import re
import struct
import tempfile
import time

import numpy as np

from unitree_deploy.sensors.realsense import RealSenseCaptureConfig, RealSenseFrame


_SHM_ROOT = Path("/dev/shm") / f"unitree-deploy-{os.getuid()}"
_MAX_METADATA_BYTES = 4096
_LENGTH = struct.Struct("<I")


def _frame_path(camera_name: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_-][A-Za-z0-9_.-]*", camera_name):
        raise ValueError("camera_name must contain only letters, digits, underscores, dots or hyphens")
    return _SHM_ROOT / f"realsense-{camera_name}.frame"


class RealSensePublisher:
    """One writer per camera name. Readers cannot block acquisition."""

    def __init__(self, config: RealSenseCaptureConfig):
        self.config = config
        self.path = _frame_path(config.camera_name)
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.color_shape = (config.color_height, config.color_width, 3)
        self.depth_shape = None
        if config.enable_depth:
            if config.align_depth_to_color:
                self.depth_shape = (config.color_height, config.color_width)
            else:
                self.depth_shape = (config.depth_height, config.depth_width)
        size = _LENGTH.size + _MAX_METADATA_BYTES + int(np.prod(self.color_shape))
        if self.depth_shape is not None:
            size += int(np.prod(self.depth_shape)) * 4
        self._cleanup = ExitStack()
        try:
            # Hold this separate lock for the publisher's lifetime, even between frames.
            self._owner = self._cleanup.enter_context(self.path.with_suffix(".lock").open("a+b"))
            try:
                fcntl.flock(self._owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError(f"RealSense publisher already running: {config.camera_name}") from None
            # Replace the inode rather than resizing memory still mapped by old readers.
            self._file = self._cleanup.enter_context(
                tempfile.NamedTemporaryFile(dir=self.path.parent, delete=False),
            )
            temporary_path = Path(self._file.name)
            self._cleanup.callback(temporary_path.unlink, missing_ok=True)
            self._file.truncate(size)
            self._memory = mmap.mmap(self._file.fileno(), size)
            self._cleanup.callback(self._memory.close)
            os.replace(temporary_path, self.path)
            self._cleanup.callback(self.path.unlink, missing_ok=True)
        except BaseException:
            self._cleanup.close()
            raise

    def publish(self, frame: RealSenseFrame) -> bool:
        color = np.ascontiguousarray(frame.color, dtype=np.uint8)
        depth = np.ascontiguousarray(frame.depth, dtype=np.float32) if frame.depth is not None else None
        depth_shape = None if depth is None else depth.shape
        if (frame.camera_name != self.config.camera_name
                or color.shape != self.color_shape or depth_shape != self.depth_shape):
            raise ValueError("frame camera or image shapes differ from the capture config")
        metadata = json.dumps({
            "camera_name": frame.camera_name, "serial_number": frame.serial_number,
            "color_shape": color.shape, "depth_shape": depth_shape,
            "timestamp_sec": frame.timestamp_sec, "color_intrinsics": frame.color_intrinsics,
            "published_at": time.monotonic(),
        }, allow_nan=False).encode()
        if len(metadata) > _MAX_METADATA_BYTES:
            raise ValueError("RealSense frame metadata exceeds shared header size")
        try:
            # Skip a frame if a reader is copying; never wait on a slow consumer.
            fcntl.flock(self._file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        try:
            self._memory.seek(0)
            self._memory.write(_LENGTH.pack(0))  # Invalidate before writing, including if the writer crashes.
            self._memory.write(metadata)
            self._memory.write(memoryview(color).cast("B"))
            if depth is not None:
                self._memory.write(memoryview(depth).cast("B"))
            # Commit only after both images are complete.
            self._memory.seek(0)
            self._memory.write(_LENGTH.pack(len(metadata)))
        finally:
            fcntl.flock(self._file, fcntl.LOCK_UN)
        return True

    def close(self) -> None:
        self._cleanup.close()


class RealSenseReader:
    """Nonblocking snapshots; None means absent, busy, or older than max_age seconds.

    Each reader owns its copies. Start before the publisher, survive its restart,
    and close independently without affecting other readers or the camera.
    """

    def __init__(self, camera_name: str, *, max_age: float = 0.5):
        if not np.isfinite(max_age) or max_age <= 0:
            raise ValueError("max_age must be positive and finite")
        self.path = _frame_path(camera_name)
        self.max_age = max_age
        self._file = None
        self._memory = None

    def read(self) -> RealSenseFrame | None:
        try:
            inode = self.path.stat().st_ino
        except FileNotFoundError:
            self.close()
            return None
        # A restarted publisher replaces the file; existing mappings still refer to the old one.
        if self._file is None or os.fstat(self._file.fileno()).st_ino != inode:
            self.close()
            try:
                self._file = self.path.open("rb")
            except FileNotFoundError:
                return None
            self._memory = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            fcntl.flock(self._file, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            return None
        try:
            length = _LENGTH.unpack_from(self._memory)[0]
            if length == 0:
                return None
            offset = _LENGTH.size + length
            metadata = json.loads(self._memory[_LENGTH.size:offset])
            if time.monotonic() - metadata["published_at"] > self.max_age:
                return None

            # Copy while holding the read lock, so policy preprocessing owns its arrays.
            color = np.ndarray(metadata["color_shape"], dtype=np.uint8,
                               buffer=self._memory, offset=offset).copy()
            offset += color.nbytes
            depth = None
            if metadata["depth_shape"] is not None:
                depth = np.ndarray(metadata["depth_shape"], dtype=np.float32,
                                   buffer=self._memory, offset=offset).copy()
            intrinsics = metadata["color_intrinsics"]
            return RealSenseFrame(
                camera_name=metadata["camera_name"], serial_number=metadata["serial_number"],
                color=color, depth=depth, timestamp_sec=metadata["timestamp_sec"],
                color_intrinsics=None if intrinsics is None else tuple(intrinsics),
            )
        finally:
            fcntl.flock(self._file, fcntl.LOCK_UN)

    def close(self) -> None:
        if self._memory is not None:
            self._memory.close()
            self._memory = None
        if self._file is not None:
            self._file.close()
            self._file = None

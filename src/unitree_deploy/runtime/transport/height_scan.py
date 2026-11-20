"""Latest height grid in Linux shared memory."""

import fcntl
import os
import re
import struct
import tempfile
import time
from pathlib import Path

import numpy as np


_STREAM_DIR = Path("/dev/shm") / f"unitree-deploy-{os.getuid()}"
_HEADER = struct.Struct("<dII")  # Monotonic timestamp, X samples, Y samples.


def _stream_path(name: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_-][A-Za-z0-9_.-]*", name):
        raise ValueError("height scan stream name must contain only letters, digits, _, -, or .")
    return _STREAM_DIR / f"height-scan-{name}.frame"


class HeightScanPublisher:
    """Own one stream and atomically publish complete float32 grids."""

    def __init__(self, name: str):
        self.path = _stream_path(name)
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.owner = self.path.with_suffix(".lock").open("a+b")
        try:
            fcntl.flock(self.owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.owner.close()
            raise
        self.path.unlink(missing_ok=True)

    def publish(self, grid: np.ndarray, *, captured_at: float | None = None) -> None:
        values = np.ascontiguousarray(grid, dtype="<f4")
        if values.ndim != 2 or not values.size:
            raise ValueError("height scan must be a nonempty [X, Y] grid")
        stamp = time.monotonic() if captured_at is None else captured_at
        with tempfile.NamedTemporaryFile(dir=self.path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            try:
                stream.write(_HEADER.pack(stamp, *values.shape))
                stream.write(memoryview(values).cast("B"))
                stream.flush()
                os.replace(temporary, self.path)
            finally:
                temporary.unlink(missing_ok=True)

    def close(self) -> None:
        if not self.owner.closed:
            try:
                self.path.unlink(missing_ok=True)
            finally:
                self.owner.close()


class HeightScanReader:
    def __init__(self, name: str):
        self.path = _stream_path(name)

    def read(self) -> tuple[np.ndarray, float] | None:
        try:
            data = self.path.read_bytes()
        except FileNotFoundError:
            return None
        stamp, nx, ny = _HEADER.unpack_from(data)
        return np.frombuffer(data, dtype="<f4", offset=_HEADER.size).reshape(nx, ny), stamp

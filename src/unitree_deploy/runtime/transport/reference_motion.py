"""Versioned reference poses over DDS String_, matching the existing SDK transport."""

from collections import deque
from dataclasses import dataclass
import json
import threading
import time

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


REFERENCE_MOTION_TOPIC = "rt/reference_motion"


class MotionUnavailable(RuntimeError):
    """Missing, stale, or incomplete motion must not reach policy inference."""


class ReferenceStreamLost(MotionUnavailable):
    """The selected reference publisher stopped or was replaced."""


@dataclass(frozen=True)
class MotionFrame:
    session: str
    seq: int
    time_s: float
    joint_names: tuple[str, ...]
    body_names: tuple[str, ...]
    joint_pos: np.ndarray
    body_pos_w: np.ndarray
    body_quat_w: np.ndarray
    file_playing: bool | None = None
    file_finished: bool = False
    file_time_s: float | None = None

    def encode(self):
        return json.dumps({
            "version": 1, "session": self.session, "seq": self.seq,
            "time_s": self.time_s, "quaternion_order": "wxyz",
            "joint_names": self.joint_names, "body_names": self.body_names,
            "joint_pos": self.joint_pos.tolist(), "body_pos_w": self.body_pos_w.tolist(),
            "body_quat_w": self.body_quat_w.tolist(),
            "file_playing": self.file_playing,
            "file_finished": self.file_finished,
            "file_time_s": self.file_time_s,
        }, allow_nan=False, separators=(",", ":"))

    @classmethod
    def decode(cls, payload):
        value = json.loads(payload)
        if value["version"] != 1 or value["quaternion_order"] != "wxyz":
            raise ValueError("unsupported reference motion pose contract")
        joints, bodies = tuple(value["joint_names"]), tuple(value["body_names"])
        if not joints or not bodies or len(set(joints)) != len(joints) or len(set(bodies)) != len(bodies):
            raise ValueError("motion names must be nonempty and unique")
        arrays = [np.asarray(value[key], dtype=np.float32) for key in
                  ("joint_pos", "body_pos_w", "body_quat_w")]
        for array, shape in zip(arrays, ((len(joints),), (len(bodies), 3), (len(bodies), 4))):
            if array.shape != shape or not np.isfinite(array).all():
                raise ValueError(f"motion array must be finite with shape {shape}")
        if not np.allclose(np.linalg.norm(arrays[2], axis=-1), 1, atol=1e-3):
            raise ValueError("motion quaternions must be normalized")
        if not isinstance(value["session"], str) or not value["session"]:
            raise ValueError("motion session must be nonempty")
        if not np.isfinite(value["time_s"]) or value["seq"] < 0:
            raise ValueError("invalid motion time or sequence")
        file_playing = value.get("file_playing")
        if file_playing is not None and type(file_playing) is not bool:
            raise ValueError("file_playing must be boolean")
        file_finished = value.get("file_finished", False)
        if type(file_finished) is not bool:
            raise ValueError("file_finished must be boolean")
        file_time = value.get("file_time_s")
        if file_time is not None and not np.isfinite(file_time):
            raise ValueError("invalid file playback time")
        return cls(value["session"], int(value["seq"]), float(value["time_s"]), joints, bodies,
                   *arrays, file_playing, file_finished, file_time)

    @property
    def sample_time(self) -> float:
        return self.time_s if self.file_time_s is None else self.file_time_s


class MotionBuffer:
    """Delay playback for future samples; use local monotonic time, not host clock sync."""

    def __init__(self, joint_names, body_names, *, dt, future_steps, timeout=0.2, tolerance=0.04):
        self.joint_names, self.body_names = tuple(joint_names), tuple(body_names)
        self.steps = np.asarray(sorted(set([0, *future_steps])), dtype=int)
        if not np.isfinite([dt, timeout, tolerance]).all() or dt <= 0 or timeout <= 0 or tolerance < 0:
            raise ValueError("invalid motion timing")
        self.dt, self.timeout, self.tolerance = dt, timeout, tolerance
        self.delay = max(0, self.steps[-1]) * dt + tolerance
        self.frames = deque()
        self.lock = threading.Lock()
        self.received_at = None
        self.error = None

    def receive(self, payload, *, now=None):
        now = time.monotonic() if now is None else now
        try:
            frame = MotionFrame.decode(payload)
            if frame.joint_names != self.joint_names or frame.body_names != self.body_names:
                raise ValueError("reference motion joint/body order differs from policy.yaml")
        except (ValueError, KeyError, TypeError) as exc:
            with self.lock:
                self.error = str(exc)
            return
        with self.lock:
            if not self.frames or frame.session != self.frames[-1].session:
                self.frames.clear()
                self.clock_offset = now - frame.time_s
            elif frame.seq <= self.frames[-1].seq or frame.time_s <= self.frames[-1].time_s:
                return  # Replays must not refresh the watchdog.
            elif frame.sample_time < self.frames[-1].sample_time:
                return
            elif frame.sample_time == self.frames[-1].sample_time:
                # Paused file heartbeats keep the stream alive without changing the window.
                self.frames.pop()
            self.frames.append(frame)
            self.received_at, self.error = now, None
            history = self.delay + abs(self.steps[0]) * self.dt + self.timeout
            while len(self.frames) > 2 and frame.sample_time - self.frames[1].sample_time > history:
                self.frames.popleft()

    def sample(self, *, now=None):
        now = time.monotonic() if now is None else now
        with self.lock:
            if self.error:
                raise MotionUnavailable(self.error)
            if self.received_at is None or now - self.received_at > self.timeout:
                raise ReferenceStreamLost("reference motion stream missing or stale")
            frames = list(self.frames)
            latest = frames[-1]
            head = now - self.clock_offset if latest.file_time_s is None else latest.file_time_s
            targets = head - self.delay + self.steps * self.dt
        times = np.array([frame.sample_time for frame in frames])
        if len(times) < 2 or targets[0] < times[0] or targets[-1] > times[-1] + self.tolerance:
            raise ReferenceStreamLost("reference motion window is not ready")
        result = {"session": latest.session, "file_playing": latest.file_playing}
        if latest.file_playing is False:
            result["first_frame"] = latest
        elif latest.file_finished:
            # The latest packet can lead the frame currently tracked by the actor.
            current = targets[np.searchsorted(self.steps, 0)]
            index = np.searchsorted(times, current, side="right") - 1
            result["file_finished"] = frames[index].file_finished
        indices = np.clip(np.searchsorted(times, targets, side="right"), 1, len(times) - 1)
        if np.any(times[indices] - times[indices - 1] > self.timeout):
            raise ReferenceStreamLost("reference motion contains a data gap")
        for field in ("joint_pos", "body_pos_w"):
            values = np.stack([getattr(frame, field) for frame in frames])
            weight = np.clip((targets - times[indices - 1]) / (times[indices] - times[indices - 1]), 0, 1)
            weight = weight.reshape((-1,) + (1,) * (values.ndim - 1))
            result[field] = values[indices - 1] * (1 - weight) + values[indices] * weight
        quats = np.stack([frame.body_quat_w for frame in frames])
        result["body_quat_w"] = np.stack([
            Slerp(times, Rotation.from_quat(quats[:, i, [1, 2, 3, 0]]))(
                np.clip(targets, times[0], times[-1])
            ).as_quat()[:, [3, 0, 1, 2]] for i in range(len(self.body_names))
        ], axis=1)
        return result


class MotionSubscriber(MotionBuffer):
    def __init__(self, *args, topic=REFERENCE_MOTION_TOPIC, **kwargs):
        super().__init__(*args, **kwargs)
        from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
        from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_

        self.subscriber = ChannelSubscriber(topic, String_)
        self.subscriber.Init(lambda msg: self.receive(msg.data), 64)
        self.playback = ChannelPublisher(f"{topic}/playback", String_)
        self.playback.Init()
        self.message_type = String_

    def set_playback(self, session: str, playing: bool, *, rate: float = 1.0) -> None:
        command = f"play:{session}:{rate}" if playing else f"hold:{session}"
        self.playback.Write(self.message_type(command))

    def close(self):
        self.subscriber.Close()
        self.playback.Close()

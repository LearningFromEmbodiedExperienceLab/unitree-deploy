"""Export the live MuJoCo camera and policy debug viz using simulation time."""

from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess

import mujoco
import numpy as np


@dataclass(frozen=True)
class VideoConfig:
    output: Path
    width: int = 960
    height: int = 540
    fps: int = 25
    start: float = 0.0
    seconds: float | None = None

    def __post_init__(self):
        if self.width <= 0 or self.height <= 0 or self.width % 2 or self.height % 2:
            raise ValueError('video width and height must be positive even integers')
        if self.fps <= 0:
            raise ValueError('video fps must be positive')
        if not np.isfinite(self.start) or self.start < 0:
            raise ValueError('video start must be finite and nonnegative')
        if self.seconds is not None and (not np.isfinite(self.seconds) or self.seconds <= 0):
            raise ValueError('video seconds must be finite and positive')


class VideoRecorder:
    def __init__(self, model, config: VideoConfig):
        if shutil.which('ffmpeg') is None:
            raise RuntimeError('MP4 export requires ffmpeg on PATH')
        if config.output.exists():
            raise FileExistsError(f'video already exists: {config.output}')
        self.model = model
        self.config = config
        self.frames = 0
        self.last_time = None
        self.done = False
        self.gl = None
        self.context = None
        self.process = None

    def capture(self, data, camera, option, policy_debug_viz=None, texts=()):
        if self.done:
            return
        t = float(data.time)
        if self.last_time is not None and t < self.last_time:
            self.close()  # A simulator reset ends this recording.
            return
        self.last_time = t
        config = self.config
        if config.seconds is not None and t >= config.start + config.seconds - 1e-9:
            self.close()
            return
        if t + 1e-9 < config.start + self.frames / config.fps:
            return
        if self.gl is None:
            self.open()
        self.gl.make_current()
        mujoco.mjv_updateScene(self.model, data, option, None, camera,
                              mujoco.mjtCatBit.mjCAT_ALL, self.scene)
        if policy_debug_viz is not None:
            policy_debug_viz.add_to_scene(self.scene)
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self.context)
        mujoco.mjr_render(self.viewport, self.scene, self.context)
        for font, position, left, right in texts:
            mujoco.mjr_overlay(font, position, self.viewport, left, right, self.context)
        mujoco.mjr_readPixels(self.rgb, None, self.viewport, self.context)
        try:
            self.process.stdin.write(self.rgb[::-1].tobytes())
        except BrokenPipeError:
            self.close()
            raise
        self.frames += 1

    def open(self):
        config = self.config
        try:
            self.gl = mujoco.GLContext(config.width, config.height)
            self.gl.make_current()
            self.context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150)
            mujoco.mjr_resizeOffscreen(config.width, config.height, self.context)
            self.scene = mujoco.MjvScene(self.model, maxgeom=10000)
            self.viewport = mujoco.MjrRect(0, 0, config.width, config.height)
            self.rgb = np.empty((config.height, config.width, 3), dtype=np.uint8)
            config.output.parent.mkdir(parents=True, exist_ok=True)
            self.process = subprocess.Popen([
                'ffmpeg', '-n', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                '-s', f'{config.width}x{config.height}', '-r', str(config.fps), '-i', '-',
                '-an', '-c:v', 'libx264', '-threads', '2', '-pix_fmt', 'yuv420p',
                '-crf', '20', str(config.output),
            ], stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        except Exception:
            self.close()
            raise

    def close(self):
        if self.done:
            return
        self.done = True
        try:
            if self.process is not None:
                try:
                    self.process.stdin.close()
                except BrokenPipeError:
                    pass
                self.process.stdin = None
                _, error = self.process.communicate()
                if self.process.returncode:
                    raise RuntimeError(f'ffmpeg failed: {error.decode().strip()}')
        finally:
            if self.gl is not None:
                self.gl.make_current()
                if self.context is not None:
                    self.context.free()
                self.gl.free()

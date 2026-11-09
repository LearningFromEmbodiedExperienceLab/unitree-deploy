"""RealSense acquisition without viewer or controller dependencies."""

from dataclasses import dataclass

import numpy as np


@dataclass
class RealSenseFrame:
    camera_name: str
    serial_number: str | None
    color: np.ndarray | None  # uint8 RGB
    depth: np.ndarray | None  # float32 meters
    timestamp_sec: float | None  # SDK timestamp, not the host clock
    color_intrinsics: tuple[float, float, float, float] | None = None  # fx, fy, ppx, ppy


@dataclass
class RealSenseCaptureConfig:
    camera_name: str = "realsense"
    serial_number: str | None = None
    color_width: int = 640
    color_height: int = 480
    depth_width: int = 640
    depth_height: int = 480
    fps: int = 30
    enable_depth: bool = True
    align_depth_to_color: bool = True

    @classmethod
    def from_dict(cls, item: dict) -> "RealSenseCaptureConfig":
        # Keep the existing visualizer.yaml aliases for names, serials and image sizes.
        width = int(item.get("width", item.get("color_width", 640)))
        height = int(item.get("height", item.get("color_height", 480)))
        return cls(
            camera_name=item.get("camera_name", item.get("name", "realsense")),
            serial_number=item.get("serial_number", item.get("serial")),
            color_width=int(item.get("color_width", width)),
            color_height=int(item.get("color_height", height)),
            depth_width=int(item.get("depth_width", width)),
            depth_height=int(item.get("depth_height", height)),
            fps=int(item.get("fps", 30)),
            enable_depth=bool(item.get("enable_depth", True)),
            align_depth_to_color=bool(item.get("align_depth_to_color", True)),
        )


class RealSenseCapture:
    """Own one SDK pipeline. Only the acquisition process creates this object."""

    def __init__(self, config: RealSenseCaptureConfig):
        # Consumers can import the frame/config types without installing the SDK.
        import pyrealsense2 as rs

        self.config = config
        self.pipeline = rs.pipeline()
        pipeline_config = rs.config()
        if config.serial_number:
            pipeline_config.enable_device(config.serial_number)
        pipeline_config.enable_stream(
            rs.stream.color, config.color_width, config.color_height, rs.format.bgr8, config.fps,
        )
        if config.enable_depth:
            pipeline_config.enable_stream(
                rs.stream.depth, config.depth_width, config.depth_height, rs.format.z16, config.fps,
            )
        profile = self.pipeline.start(pipeline_config)
        try:
            self.serial_number = profile.get_device().get_info(rs.camera_info.serial_number)
            intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
            self.color_intrinsics = (intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy)
            self.depth_scale = 1.0
            self.align = None
            if config.enable_depth:
                self.depth_scale = float(profile.get_device().first_depth_sensor().get_depth_scale())
                if config.align_depth_to_color:
                    self.align = rs.align(rs.stream.color)
        except BaseException:
            self.pipeline.stop()
            raise

    def poll_frame(self) -> RealSenseFrame | None:
        frames = self.pipeline.poll_for_frames()
        if not frames:
            return None
        if self.align is not None:
            frames = self.align.process(frames)
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame() if self.config.enable_depth else None
        if not color_frame or (self.config.enable_depth and not depth_frame):
            return None

        # Copy SDK buffers before releasing the frameset; consumers use RGB and meters.
        color = np.asarray(color_frame.get_data())[:, :, ::-1].copy()
        depth = None
        if depth_frame:
            depth = np.asarray(depth_frame.get_data(), dtype=np.float32) * self.depth_scale
        return RealSenseFrame(
            camera_name=self.config.camera_name,
            serial_number=self.serial_number,
            color=color,
            depth=depth,
            timestamp_sec=float(frames.get_timestamp()) / 1000.0,
            color_intrinsics=self.color_intrinsics,
        )

    def close(self) -> None:
        self.pipeline.stop()

"""Pico body input; SDK poses are Unity xyz/xyzw, output is Z-up/wxyz."""

import numpy as np
from scipy.spatial.transform import Rotation


BODY_NAMES = (
    "Pelvis", "Left_Hip", "Right_Hip", "Spine1", "Left_Knee", "Right_Knee",
    "Spine2", "Left_Ankle", "Right_Ankle", "Spine3", "Left_Foot", "Right_Foot",
    "Neck", "Left_Collar", "Right_Collar", "Head", "Left_Shoulder",
    "Right_Shoulder", "Left_Elbow", "Right_Elbow", "Left_Wrist", "Right_Wrist",
    "Left_Hand", "Right_Hand",
)


def convert_body_poses(poses):
    poses = np.asarray(poses, dtype=np.float64)
    if poses.shape != (len(BODY_NAMES), 7) or not np.isfinite(poses).all():
        raise ValueError("XR body poses must be finite [24, 7]")
    transform = Rotation.from_matrix([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    positions = transform.apply(poses[:, :3])
    quaternions = (transform * Rotation.from_quat(poses[:, 3:])).as_quat()[:, [3, 0, 1, 2]]
    return {name: (pos, quat) for name, pos, quat in zip(BODY_NAMES, positions, quaternions)}


class XRobotSource:
    def __init__(self):
        import xrobotoolkit_sdk

        self.sdk = xrobotoolkit_sdk
        self.last_timestamp = None
        self.floor_offset = None
        self.sdk.init()

    def read(self):
        if not self.sdk.is_body_data_available():
            return None
        timestamp = int(self.sdk.get_body_timestamp_ns())
        if timestamp <= 0 or (self.last_timestamp is not None and timestamp <= self.last_timestamp):
            return None
        poses = self.sdk.get_body_joints_pose()
        # A racing SDK update is retried, rather than assigning it the wrong timestamp.
        if timestamp != int(self.sdk.get_body_timestamp_ns()):
            return None
        self.last_timestamp = timestamp
        body_pose = convert_body_poses(poses)
        if self.floor_offset is None:
            # Calibrate once while standing; per-frame grounding would erase jumps.
            self.floor_offset = .01 - min(pos[2] for pos, _ in body_pose.values())
        for pos, _ in body_pose.values():
            pos[2] += self.floor_offset
        return body_pose

    def close(self):
        self.sdk.close()

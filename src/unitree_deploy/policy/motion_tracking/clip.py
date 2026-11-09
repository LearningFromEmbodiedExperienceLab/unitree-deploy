"""Local any4hdmi v2 root and joint references."""

import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


class Any4HDMIMotionClip:
    def __init__(self, path: Path, joint_names, *, dt, future_steps):
        manifest_path = next(
            (parent / "manifest.json" for parent in path.parents if (parent / "manifest.json").is_file()),
            None,
        )
        if manifest_path is None:
            raise ValueError(f"Missing any4hdmi manifest.json above {path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("format_version") != 2:
            raise ValueError(f"Expected any4hdmi format_version 2 in {manifest_path}")
        names = manifest["qpos_names"]
        source_dt = float(manifest["timestep"])
        if not np.isfinite([dt, source_dt]).all() or min(dt, source_dt) <= 0:
            raise ValueError("Policy and any4hdmi timesteps must be finite and positive")
        if len(names) != len(set(names)):
            raise ValueError("Duplicate any4hdmi qpos names")
        # The suitcase datasets name the robot's free joint pelvis and add object DoFs.
        suffixes = ("tx", "ty", "tz", "qw", "qx", "qy", "qz")
        root = next(([f"{prefix}_{s}" for s in suffixes] for prefix in ("root", "pelvis")
                     if all(f"{prefix}_{s}" in names for s in suffixes)), None)
        if root is None or not set(joint_names).issubset(names):
            raise ValueError(f"Missing root or joint columns in {manifest_path}")
        with np.load(path, allow_pickle=False) as clip:
            qpos = clip["qpos"]
        if qpos.ndim != 2 or not len(qpos) or qpos.shape[1] != len(names):
            raise ValueError(f"Invalid any4hdmi qpos shape {qpos.shape} in {path}")
        qpos = qpos[:, [names.index(name) for name in [*root, *joint_names]]]
        if not np.isfinite(qpos).all() or not np.allclose(np.linalg.norm(qpos[:, 3:7], axis=1), 1, atol=1e-3):
            raise ValueError(f"Invalid joint pose or root quaternion in {path}")

        # Keep the endpoint even if the source duration is not a whole policy step.
        duration = (len(qpos) - 1) * source_dt
        num_steps = int(np.ceil(duration / dt - 1e-9)) + 1
        times = np.minimum(np.arange(num_steps) * dt, duration)
        source_times = np.arange(len(qpos)) * source_dt
        columns = [*range(3), *range(7, qpos.shape[1])]
        positions = np.stack([np.interp(times, source_times, qpos[:, i]) for i in columns], axis=1)
        rotations = Rotation.from_quat(qpos[:, [4, 5, 6, 3]])
        if len(qpos) > 1:
            rotations = Slerp(source_times, rotations)(times)
        self.joint_pos = positions[:, 3:].astype(np.float32)
        self.body_pos_w = positions[:, None, :3].astype(np.float32)
        self.body_quat_w = rotations.as_quat()[:, None, [3, 0, 1, 2]].astype(np.float32)
        self.steps = np.asarray(sorted(set([0, *future_steps])), dtype=int)
        self.session = str(path)
        self.reset()

    def reset(self):
        self.frame = 0
        self.finished = False

    def sample(self, *, advance=True):
        last = len(self.joint_pos) - 1
        if advance:
            indices = np.clip(self.frame + self.steps, 0, last)
            self.finished = self.frame == last
            self.frame = min(self.frame + 1, last)
        else:
            indices = np.zeros_like(self.steps)
        return {"session": self.session, "joint_pos": self.joint_pos[indices],
                "body_pos_w": self.body_pos_w[indices], "body_quat_w": self.body_quat_w[indices]}

"""Reference pose shared by tracking policies and simulation debug views."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ReferencePose:
    """Named joint positions and a world-frame root pose in wxyz order."""

    joint_names: Sequence[str]
    joint_pos: np.ndarray
    root_pos: np.ndarray
    root_quat: np.ndarray

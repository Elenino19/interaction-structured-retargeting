"""Coordinate transforms shared by source parsing and robot retargeting."""

from __future__ import annotations

import numpy as np

# CORE4D uses y-up coordinates. G1 URDFs use the conventional z-up frame.
# This proper rotation maps (x, y, z)_core4d to (x, -z, y)_robot.
CORE4D_TO_ROBOT = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)


def points_core4d_to_robot(points: np.ndarray) -> np.ndarray:
    """Convert points or vectors from CORE4D y-up to robot z-up coordinates."""
    array = np.asarray(points)
    if array.shape[-1] != 3:
        raise ValueError(f"Expected final dimension 3, got {array.shape}")
    return array @ CORE4D_TO_ROBOT.T


def points_robot_to_core4d(points: np.ndarray) -> np.ndarray:
    """Convert robot z-up points back to CORE4D's y-up coordinates."""
    array = np.asarray(points)
    if array.shape[-1] != 3:
        raise ValueError(f"Expected final dimension 3, got {array.shape}")
    return array @ CORE4D_TO_ROBOT


def poses_core4d_to_robot(poses: np.ndarray) -> np.ndarray:
    """Change the basis of homogeneous object poses to the robot world frame."""
    array = np.asarray(poses)
    if array.ndim != 3 or array.shape[1:] != (4, 4):
        raise ValueError(f"Expected poses shaped (T, 4, 4), got {array.shape}")
    basis = np.eye(4, dtype=array.dtype)
    basis[:3, :3] = CORE4D_TO_ROBOT.astype(array.dtype)
    inverse = basis.T
    return basis[None] @ array @ inverse[None]

"""Temporal smoothing of dual-G1 trajectories with FK refresh."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytorch_kinematics as pk
import torch
from scipy.ndimage import gaussian_filter1d

from ..robot_model import G1_29DOF_KEYPOINT_LINKS
from .independent import _apply_root_transform


ARM_JOINT_TOKENS = ("shoulder", "elbow", "wrist", "hand")
LEG_JOINT_TOKENS = ("hip", "knee", "ankle")
CLEARANCE_ARM_SIGMA = 0.35
CLEARANCE_ROOT_SIGMA = 0.0
CLEARANCE_LEG_SIGMA = 0.0


def resolve_smooth_sigmas(
    *,
    arm_sigma: float | None,
    root_sigma: float | None,
    clearance_recipe: bool,
) -> tuple[float | None, float | None]:
    """Apply clearance defaults (arm 0.2, root 0) without changing explicit flags."""
    if not clearance_recipe:
        return arm_sigma, root_sigma
    if arm_sigma is None:
        arm_sigma = CLEARANCE_ARM_SIGMA
    if root_sigma is None:
        root_sigma = CLEARANCE_ROOT_SIGMA
    return arm_sigma, root_sigma


def smooth_array(values: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return np.asarray(values, dtype=np.float64)
    return gaussian_filter1d(np.asarray(values, dtype=np.float64), sigma=sigma, axis=0, mode="nearest")


def smooth_joint_positions(
    positions: np.ndarray,
    joint_names: list[str] | np.ndarray,
    *,
    sigma: float,
    arm_sigma: float | None = None,
    leg_sigma: float | None = None,
) -> np.ndarray:
    """Smooth all joints with `sigma`; arm/leg joints can use a different sigma."""
    names = [str(name) for name in joint_names]
    smoothed = smooth_array(positions, sigma)
    out = np.array(smoothed, copy=True)
    if arm_sigma is not None and arm_sigma != sigma:
        arms = smooth_array(positions, arm_sigma)
        for index, name in enumerate(names):
            if any(token in name for token in ARM_JOINT_TOKENS):
                out[:, index] = arms[:, index]
    if leg_sigma is not None and leg_sigma != sigma:
        legs = smooth_array(positions, leg_sigma)
        for index, name in enumerate(names):
            if any(token in name for token in LEG_JOINT_TOKENS):
                out[:, index] = legs[:, index]
    return out


def recompute_keypoints(
    urdf_path: Path,
    joint_positions: np.ndarray,
    root_translation: np.ndarray,
    root_yaw: np.ndarray,
    keypoint_names: tuple[str, ...] | list[str],
) -> np.ndarray:
    chain = pk.build_chain_from_urdf(urdf_path.read_text()).to(dtype=torch.float64)
    joints = torch.tensor(joint_positions, dtype=torch.float64)
    transforms = chain.forward_kinematics(joints)
    local = torch.stack(
        [
            transforms[G1_29DOF_KEYPOINT_LINKS[name]].get_matrix()[:, :3, 3]
            for name in keypoint_names
        ],
        dim=1,
    )
    world = _apply_root_transform(
        local,
        torch.tensor(root_translation, dtype=torch.float64),
        torch.tensor(root_yaw, dtype=torch.float64),
    )
    return world.detach().cpu().numpy()


def nudge_root_z_to_ankle(
    data: dict[str, np.ndarray],
    urdf_path: Path,
    *,
    target_z: float = 0.055,
    max_abs_m: float = 0.025,
) -> dict[str, np.ndarray]:
    """Shift root z so mean ankle keypoints match the quality-gate height."""
    names = [str(name) for name in data["keypoint_names"]]
    keypoints = recompute_keypoints(
        urdf_path,
        data["joint_positions"],
        data["root_translation"],
        data["root_yaw"],
        names,
    )
    left = names.index("left_ankle")
    right = names.index("right_ankle")
    mean_z = keypoints[:, [left, right], 2].mean(axis=1)
    delta = np.clip(target_z - mean_z, -max_abs_m, max_abs_m)
    data["root_translation"] = np.array(data["root_translation"], copy=True, dtype=np.float64)
    data["root_translation"][:, 2] += delta
    data["robot_keypoints"] = recompute_keypoints(
        urdf_path,
        data["joint_positions"],
        data["root_translation"],
        data["root_yaw"],
        names,
    )
    return data


def lift_root_for_sole_clearance(
    data: dict[str, np.ndarray],
    urdf_path: Path,
    *,
    min_z: float = 0.0,
) -> dict[str, np.ndarray]:
    """Raise root z so ankle-roll volume samples are not underground."""
    from ..metrics import apply_sole_lift_to_root, rubber_foot_world_volume_points

    names = [str(name) for name in data["keypoint_names"]]
    feet = rubber_foot_world_volume_points(
        urdf_path,
        data["joint_positions"],
        data["root_translation"],
        data["root_yaw"],
    )
    zmin = feet[..., 2].min(axis=1)
    lifted, lift = apply_sole_lift_to_root(
        data["root_translation"],
        zmin,
        min_z=min_z,
    )
    if len(lift) > 3:
        lift = gaussian_filter1d(lift, sigma=1.0, mode="nearest")
        lifted = np.array(data["root_translation"], copy=True, dtype=np.float64)
        lifted[:, 2] += lift
    data["root_translation"] = lifted
    data["robot_keypoints"] = recompute_keypoints(
        urdf_path,
        data["joint_positions"],
        data["root_translation"],
        data["root_yaw"],
        names,
    )
    return data


def smooth_robot_npz(
    path: Path,
    urdf_path: Path,
    output_path: Path,
    *,
    sigma: float = 1.0,
    arm_sigma: float | None = None,
    root_sigma: float | None = None,
    leg_sigma: float | None = None,
    ankle_target_z: float | None = None,
    ankle_nudge_max_m: float = 0.025,
    lift_soles: bool = False,
    sole_min_z: float = 0.0,
) -> None:
    """Smooth joints with `sigma`; optional `arm_sigma` / `root_sigma` (None = `sigma`)."""
    with np.load(path) as payload:
        data = {name: payload[name] for name in payload.files}
    names = [str(name) for name in data["keypoint_names"]]
    joint_names = [str(name) for name in data["joint_names"]]
    data["joint_positions"] = smooth_joint_positions(
        data["joint_positions"],
        joint_names,
        sigma=sigma,
        arm_sigma=arm_sigma,
        leg_sigma=leg_sigma,
    )
    applied_root_sigma = sigma if root_sigma is None else root_sigma
    data["root_translation"] = smooth_array(data["root_translation"], applied_root_sigma)
    data["root_yaw"] = smooth_array(data["root_yaw"], applied_root_sigma)
    if ankle_target_z is not None:
        data = nudge_root_z_to_ankle(
            data,
            urdf_path,
            target_z=ankle_target_z,
            max_abs_m=ankle_nudge_max_m,
        )
    else:
        data["robot_keypoints"] = recompute_keypoints(
            urdf_path,
            data["joint_positions"],
            data["root_translation"],
            data["root_yaw"],
            names,
        )
    if lift_soles:
        data = lift_root_for_sole_clearance(data, urdf_path, min_z=sole_min_z)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **data)

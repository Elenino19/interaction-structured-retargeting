"""Independent, temporally smoothed SMPL-X to Unitree G1 baseline."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pytorch_kinematics as pk
import torch

from ..coordinates import points_core4d_to_robot
from ..robot_model import (
    G1_29DOF_KEYPOINT_LINKS,
    SMPLX_KEYPOINT_INDICES,
    load_actuated_joint_limits,
)


@dataclass(frozen=True)
class IndependentRetargetConfig:
    iterations: int = 400
    learning_rate: float = 0.03
    nominal_pelvis_height: float = 0.793
    keypoint_weight: float = 1.0
    root_regularization_weight: float = 10.0
    velocity_weight: float = 0.2
    acceleration_weight: float = 0.5


@dataclass(frozen=True)
class IndependentRetargetResult:
    joint_names: tuple[str, ...]
    keypoint_names: tuple[str, ...]
    joint_positions: np.ndarray
    root_translation: np.ndarray
    root_yaw: np.ndarray
    robot_keypoints: np.ndarray
    target_keypoints: np.ndarray
    scale: float
    loss_history: np.ndarray
    config_json: str


def _yaw_from_shoulders(joints: np.ndarray) -> np.ndarray:
    left = joints[:, SMPLX_KEYPOINT_INDICES["left_shoulder"]]
    right = joints[:, SMPLX_KEYPOINT_INDICES["right_shoulder"]]
    lateral = left - right
    forward = np.stack([lateral[:, 1], -lateral[:, 0]], axis=-1)
    return np.unwrap(np.arctan2(forward[:, 1], forward[:, 0]))


def _prepare_targets(
    source_joints: np.ndarray,
    nominal_pelvis_height: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    joints = points_core4d_to_robot(source_joints)
    pelvis = joints[:, SMPLX_KEYPOINT_INDICES["pelvis"]]
    ankles = (
        joints[:, SMPLX_KEYPOINT_INDICES["left_ankle"]]
        + joints[:, SMPLX_KEYPOINT_INDICES["right_ankle"]]
    ) / 2
    human_leg_length = float(np.median(np.linalg.norm(pelvis - ankles, axis=-1)))
    if human_leg_length < 0.2:
        raise ValueError(f"Implausible median human leg length: {human_leg_length:.3f} m")
    robot_leg_length = nominal_pelvis_height - 0.03
    scale = robot_leg_length / human_leg_length

    root = pelvis.copy()
    root[:, 2] = nominal_pelvis_height + scale * (pelvis[:, 2] - np.median(pelvis[:, 2]))
    keypoint_names = tuple(G1_29DOF_KEYPOINT_LINKS)
    source_indices = [SMPLX_KEYPOINT_INDICES[name] for name in keypoint_names]
    local = (joints[:, source_indices] - pelvis[:, None]) * scale
    targets = root[:, None] + local
    yaw = _yaw_from_shoulders(joints)
    return targets, root, yaw, scale


def _apply_root_transform(
    local_points: torch.Tensor,
    root_translation: torch.Tensor,
    root_yaw: torch.Tensor,
) -> torch.Tensor:
    cosine = torch.cos(root_yaw)
    sine = torch.sin(root_yaw)
    zeros = torch.zeros_like(cosine)
    ones = torch.ones_like(cosine)
    rotation = torch.stack(
        [
            cosine,
            -sine,
            zeros,
            sine,
            cosine,
            zeros,
            zeros,
            zeros,
            ones,
        ],
        dim=-1,
    ).reshape(-1, 3, 3)
    return local_points @ rotation.transpose(1, 2) + root_translation[:, None]


def retarget_independent(
    source_joints: np.ndarray,
    urdf_path: Path,
    config: IndependentRetargetConfig | None = None,
) -> IndependentRetargetResult:
    """Fit one G1 trajectory without awareness of the object or other robot."""
    if config is None:
        config = IndependentRetargetConfig()
    dtype = torch.float64
    chain = pk.build_chain_from_urdf(urdf_path.read_text()).to(dtype=dtype)
    joint_names = tuple(chain.get_joint_parameter_names())
    keypoint_names = tuple(G1_29DOF_KEYPOINT_LINKS)
    targets_np, base_root_np, initial_yaw_np, scale = _prepare_targets(
        source_joints,
        config.nominal_pelvis_height,
    )
    frame_count = len(source_joints)

    lower = []
    upper = []
    joint_map = {joint.name: joint for joint in load_actuated_joint_limits(urdf_path)}
    for name in joint_names:
        lower.append(joint_map[name].lower)
        upper.append(joint_map[name].upper)
    lower_tensor = torch.tensor(lower, dtype=dtype)
    upper_tensor = torch.tensor(upper, dtype=dtype)
    midpoint = (lower_tensor + upper_tensor) / 2
    half_range = (upper_tensor - lower_tensor) / 2
    normalized_zero = torch.clamp(-midpoint / half_range, -0.999, 0.999)
    raw_initial = torch.atanh(normalized_zero).repeat(frame_count, 1)

    raw_joints = torch.nn.Parameter(raw_initial)
    root_delta = torch.nn.Parameter(torch.zeros(frame_count, 3, dtype=dtype))
    root_yaw = torch.nn.Parameter(torch.tensor(initial_yaw_np, dtype=dtype))
    base_root = torch.tensor(base_root_np, dtype=dtype)
    targets = torch.tensor(targets_np, dtype=dtype)
    optimizer = torch.optim.Adam([raw_joints, root_delta, root_yaw], lr=config.learning_rate)

    keypoint_weights = torch.ones(len(keypoint_names), dtype=dtype)
    for name in ("left_elbow", "right_elbow"):
        keypoint_weights[keypoint_names.index(name)] = 2.0
    for name in ("left_wrist", "right_wrist"):
        keypoint_weights[keypoint_names.index(name)] = 4.0

    history: list[float] = []
    best: tuple[float, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None
    for _ in range(config.iterations):
        joint_positions = midpoint + half_range * torch.tanh(raw_joints)
        transforms = chain.forward_kinematics(joint_positions)
        local_keypoints = torch.stack(
            [
                transforms[G1_29DOF_KEYPOINT_LINKS[name]].get_matrix()[:, :3, 3]
                for name in keypoint_names
            ],
            dim=1,
        )
        root = base_root + root_delta
        robot_keypoints = _apply_root_transform(local_keypoints, root, root_yaw)

        squared_error = torch.sum((robot_keypoints - targets) ** 2, dim=-1)
        keypoint_loss = torch.mean(squared_error * keypoint_weights[None])
        root_loss = torch.mean(root_delta**2)
        velocity_loss = torch.mean((joint_positions[1:] - joint_positions[:-1]) ** 2)
        acceleration_loss = torch.mean(
            (joint_positions[2:] - 2 * joint_positions[1:-1] + joint_positions[:-2]) ** 2
        )
        loss = (
            config.keypoint_weight * keypoint_loss
            + config.root_regularization_weight * root_loss
            + config.velocity_weight * velocity_loss
            + config.acceleration_weight * acceleration_loss
        )
        optimizer.zero_grad()
        loss.backward()
        value = float(loss.detach())
        history.append(value)
        if best is None or value < best[0]:
            best = (
                value,
                joint_positions.detach().clone(),
                root.detach().clone(),
                root_yaw.detach().clone(),
                robot_keypoints.detach().clone(),
            )
        optimizer.step()

    if best is None:
        raise RuntimeError("Optimizer produced no result")
    _, best_joints, best_root, best_yaw, best_keypoints = best
    return IndependentRetargetResult(
        joint_names=joint_names,
        keypoint_names=keypoint_names,
        joint_positions=best_joints.cpu().numpy(),
        root_translation=best_root.cpu().numpy(),
        root_yaw=best_yaw.cpu().numpy(),
        robot_keypoints=best_keypoints.cpu().numpy(),
        target_keypoints=targets_np,
        scale=scale,
        loss_history=np.asarray(history),
        config_json=json.dumps(asdict(config), sort_keys=True),
    )


def save_independent_result(result: IndependentRetargetResult, path: Path) -> None:
    """Write a baseline trajectory in a stable, non-pickle NPZ schema."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        joint_names=np.asarray(result.joint_names),
        keypoint_names=np.asarray(result.keypoint_names),
        joint_positions=result.joint_positions,
        root_translation=result.root_translation,
        root_yaw=result.root_yaw,
        robot_keypoints=result.robot_keypoints,
        target_keypoints=result.target_keypoints,
        scale=np.asarray(result.scale),
        loss_history=result.loss_history,
        config_json=np.asarray(result.config_json),
    )

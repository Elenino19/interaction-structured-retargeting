"""Differentiable balance residuals for bipedal G1 retargeting."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

from ..coordinates import points_core4d_to_robot
from ..robot_model import SMPLX_KEYPOINT_INDICES

COM_KEYPOINT_MASSES = {
    "pelvis": 3.813,
    "left_hip": 4.572,
    "right_hip": 4.572,
    "left_knee": 1.932,
    "right_knee": 1.932,
    "left_ankle": 0.682,
    "right_ankle": 0.682,
    "neck": 8.118,
    "left_shoulder": 2.095,
    "right_shoulder": 2.095,
    "left_elbow": 0.6,
    "right_elbow": 0.6,
    "left_wrist": 0.824,
    "right_wrist": 0.824,
}


def com_mass_vector(keypoint_names: tuple[str, ...] | list[str]) -> np.ndarray:
    masses = np.asarray(
        [COM_KEYPOINT_MASSES[str(name)] for name in keypoint_names],
        dtype=np.float64,
    )
    return masses / masses.sum()


def weighted_com(keypoints: np.ndarray, masses: np.ndarray) -> np.ndarray:
    return np.einsum("tkd,k->td", keypoints, masses)


def human_foot_contact(joints_core4d: np.ndarray, height_margin: float = 0.04) -> np.ndarray:
    """Approximate left/right stance from source ankle height in the robot frame."""
    ankles = points_core4d_to_robot(joints_core4d)[
        :,
        [SMPLX_KEYPOINT_INDICES["left_ankle"], SMPLX_KEYPOINT_INDICES["right_ankle"]],
        2,
    ]
    floor = np.percentile(ankles, 15, axis=0)
    return ankles < (floor + height_margin)


def source_plant_mask(
    source_slow: np.ndarray,
    source_contact: np.ndarray,
    phase_ge2: np.ndarray,
    *,
    require_contact: bool = True,
    slow_only: bool = False,
) -> np.ndarray:
    """Lock feet from the source: slow XY, or lift/carry.

    If ``slow_only``, lock only when the source ankle XY is below the slow
    threshold (OmniRetarget stance). Carry never welds a sliding foot.
    If ``require_contact``, carry only locks a foot that the human also has down.
    Otherwise carry locks both feet so a skating source cannot teach ice-skating.
    Robot height is never used, so the optimizer cannot escape by jumping.
    """
    slow = np.asarray(source_slow, dtype=bool)
    if slow_only:
        return slow
    contact = np.asarray(source_contact, dtype=bool)
    carry = np.asarray(phase_ge2, dtype=bool)[:, None]
    if require_contact:
        return slow | (carry & contact)
    return slow | np.broadcast_to(carry, slow.shape)


def ankle_xy_speed_m_s(
    joints_core4d: np.ndarray,
    *,
    fps: float = 15.0,
) -> np.ndarray:
    """Per-frame left/right source-ankle XY speed in m/s."""
    ankles = points_core4d_to_robot(joints_core4d)[
        :,
        [SMPLX_KEYPOINT_INDICES["left_ankle"], SMPLX_KEYPOINT_INDICES["right_ankle"]],
        :2,
    ]
    speed = np.linalg.norm(np.diff(ankles, axis=0), axis=-1) * fps
    return np.concatenate([speed[:1], speed], axis=0)


def pelvis_xy_speed_m_s(
    joints_core4d: np.ndarray,
    *,
    fps: float = 15.0,
) -> np.ndarray:
    """Per-frame source pelvis XY speed in m/s."""
    pelvis = points_core4d_to_robot(np.asarray(joints_core4d, dtype=np.float64))[
        :, SMPLX_KEYPOINT_INDICES["pelvis"], :2
    ]
    speed = np.linalg.norm(np.diff(pelvis, axis=0), axis=-1) * fps
    return np.concatenate([speed[:1], speed], axis=0)


def human_knee_flexion_rad(joints_core4d: np.ndarray) -> np.ndarray:
    """Left/right anatomical knee flexion, where zero is a straight leg."""
    joints = np.asarray(joints_core4d, dtype=np.float64)
    flexion = []
    for side in ("left", "right"):
        hip = joints[:, SMPLX_KEYPOINT_INDICES[f"{side}_hip"]]
        knee = joints[:, SMPLX_KEYPOINT_INDICES[f"{side}_knee"]]
        ankle = joints[:, SMPLX_KEYPOINT_INDICES[f"{side}_ankle"]]
        thigh = hip - knee
        shin = ankle - knee
        cosine = np.sum(thigh * shin, axis=-1) / np.maximum(
            np.linalg.norm(thigh, axis=-1) * np.linalg.norm(shin, axis=-1),
            1e-8,
        )
        flexion.append(np.pi - np.arccos(np.clip(cosine, -1.0, 1.0)))
    return np.stack(flexion, axis=-1)


def source_ankle_body_y(joints_core4d: np.ndarray) -> np.ndarray:
    """Source left/right ankle +y (left) in the pelvis heading frame."""
    joints = points_core4d_to_robot(np.asarray(joints_core4d, dtype=np.float64))
    pelvis = joints[:, SMPLX_KEYPOINT_INDICES["pelvis"]]
    left_dir = (
        joints[:, SMPLX_KEYPOINT_INDICES["left_hip"]]
        - joints[:, SMPLX_KEYPOINT_INDICES["right_hip"]]
    )
    left_dir[:, 2] = 0.0
    forward = np.stack([left_dir[:, 1], -left_dir[:, 0]], axis=-1)
    yaw = np.arctan2(forward[:, 1], forward[:, 0])
    out = []
    for name in ("left_ankle", "right_ankle"):
        delta = joints[:, SMPLX_KEYPOINT_INDICES[name]] - pelvis
        out.append(-np.sin(yaw) * delta[:, 0] + np.cos(yaw) * delta[:, 1])
    return np.stack(out, axis=-1)


def gait_support_plant(slow: np.ndarray, ankle_speed_m_s: np.ndarray) -> np.ndarray:
    """Keep a support foot: slow feet, otherwise the slower-moving ankle."""
    plant = np.asarray(slow, dtype=bool).copy()
    speed = np.asarray(ankle_speed_m_s, dtype=np.float64)
    neither = ~plant[:, 0] & ~plant[:, 1]
    left_slower = speed[:, 0] <= speed[:, 1]
    plant[neither & left_slower, 0] = True
    plant[neither & ~left_slower, 1] = True
    return plant


def putdown_tail_mask(
    phases: np.ndarray,
    *,
    phase_min: int,
    phase_max: int,
    tail_frames: int,
) -> np.ndarray:
    """True on the last ``tail_frames`` of the put-down phase window.

    ``tail_frames <= 0`` keeps the whole window. Joint-carry is a walk; the
    actual box lowering is only the end of that phase.
    """
    phases = np.asarray(phases)
    window = (phases >= phase_min) & (phases <= phase_max)
    if tail_frames <= 0:
        return window
    index = np.flatnonzero(window)
    out = np.zeros(len(phases), dtype=bool)
    if index.size == 0:
        return out
    out[index[-min(int(tail_frames), index.size) :]] = True
    return out


def torch_knee_min_bend_loss(
    knees: torch.Tensor,
    mask: torch.Tensor,
    min_rad: float,
) -> torch.Tensor:
    """Require a visible squat: G1 knee 0 is straight."""
    active = mask.double()
    if not torch.any(mask):
        return knees.new_zeros(())
    deficit = torch.relu(min_rad - knees) ** 2
    return torch.sum(deficit.mean(dim=-1) * active) / torch.clamp(active.sum(), min=1.0)


def torch_swing_knee_bend_loss(
    knees: torch.Tensor,
    swing: torch.Tensor,
    min_rad: float,
) -> torch.Tensor:
    """Bend each swing knee without forcing the stance knee to squat."""
    active = swing.double()
    if not torch.any(swing):
        return knees.new_zeros(())
    deficit = torch.relu(min_rad - knees) ** 2
    return torch.sum(deficit * active) / torch.clamp(active.sum(), min=1.0)


def torch_swing_knee_target_loss(
    knees: torch.Tensor,
    swing: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Match the source knee profile only on the currently swinging leg."""
    active = swing.double()
    if not torch.any(swing):
        return knees.new_zeros(())
    return torch.sum((knees - target) ** 2 * active) / torch.clamp(
        active.sum(), min=1.0
    )


def exclusive_stance_plant(plant: np.ndarray, ankle_z: np.ndarray) -> np.ndarray:
    """If both feet would swing, plant the lower source ankle (never unlock both)."""
    plant = np.asarray(plant, dtype=bool).copy()
    ankle_z = np.asarray(ankle_z, dtype=np.float64)
    neither = ~plant[:, 0] & ~plant[:, 1]
    left_stance = ankle_z[:, 0] <= ankle_z[:, 1]
    plant[neither & left_stance, 0] = True
    plant[neither & ~left_stance, 1] = True
    return plant


def torch_exclusive_stance(
    plant: torch.Tensor,
    left_z: torch.Tensor,
    right_z: torch.Tensor,
) -> torch.Tensor:
    """Keep at least one planted foot, preferring the lower sole."""
    plant = plant.clone()
    neither = ~plant[:, 0] & ~plant[:, 1]
    left_stance = left_z <= right_z
    plant[neither & left_stance, 0] = True
    plant[neither & ~left_stance, 1] = True
    return plant


def torch_knee_squat_loss(
    knees: torch.Tensor,
    plant: torch.Tensor,
    max_rad: float,
) -> torch.Tensor:
    """Penalize stance-leg knee bend past ``max_rad`` (G1 knee 0 ≈ straight)."""
    active = plant.double()
    if not torch.any(plant):
        return knees.new_zeros(())
    excess = torch.relu(knees - max_rad) ** 2
    return torch.sum(excess * active) / torch.clamp(active.sum(), min=1.0)


def torch_knee_symmetry_loss(knees: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Keep left/right knee bend similar during a two-foot squat."""
    active = mask.double()
    if not torch.any(mask):
        return knees.new_zeros(())
    return torch.sum((knees[:, 0] - knees[:, 1]) ** 2 * active) / torch.clamp(
        active.sum(), min=1.0
    )


def torch_both_soles_down_loss(
    left_zmin: torch.Tensor,
    right_zmin: torch.Tensor,
    mask: torch.Tensor,
    height_m: float = 0.02,
) -> torch.Tensor:
    """Both soles must be on the floor when the mask is active."""
    active = mask.double()
    if not torch.any(mask):
        return left_zmin.new_zeros(())
    lift = torch.relu(left_zmin - height_m) ** 2 + torch.relu(right_zmin - height_m) ** 2
    return torch.sum(lift * active) / torch.clamp(active.sum(), min=1.0)


def torch_stance_width_loss(
    left_xy: torch.Tensor,
    right_xy: torch.Tensor,
    mask: torch.Tensor,
    target_m: float,
) -> torch.Tensor:
    """Pull the feet to a hip-width squat stance instead of a lunge."""
    active = mask.double()
    if not torch.any(mask):
        return left_xy.new_zeros(())
    width = torch.linalg.norm(left_xy - right_xy, dim=-1)
    return torch.sum((width - target_m) ** 2 * active) / torch.clamp(active.sum(), min=1.0)


def torch_pair_laterality_loss(
    left_xyz: torch.Tensor,
    right_xyz: torch.Tensor,
    yaw: torch.Tensor,
    min_width_m: float,
) -> torch.Tensor:
    """Keep the left point on the robot's left of the right point.

    Body-frame +y is left: ``y = -sin(yaw) * dx + cos(yaw) * dy``.
    Mean+max so a single crossing frame is not averaged away.
    """
    delta = left_xyz - right_xyz
    lateral = -torch.sin(yaw) * delta[:, 0] + torch.cos(yaw) * delta[:, 1]
    penalty = torch.relu(min_width_m - lateral) ** 2
    return torch.mean(penalty) + penalty.max()


def torch_limb_pair_clearance_loss(
    left_pts: torch.Tensor,
    right_pts: torch.Tensor,
    min_m: float,
) -> torch.Tensor:
    """Penalize left-chain vs right-chain samples closer than ``min_m``."""
    distances = torch.cdist(left_pts, right_pts)
    nearest = distances.amin(dim=(-1, -2))
    penalty = torch.relu(min_m - nearest) ** 2
    return torch.mean(penalty) + penalty.max()


def torch_body_lateral_y(
    points: torch.Tensor,
    origin: torch.Tensor,
    yaw: torch.Tensor,
) -> torch.Tensor:
    """Body-frame +y (left) of world points about ``origin`` with heading ``yaw``."""
    delta = points - origin
    return -torch.sin(yaw) * delta[:, 0] + torch.cos(yaw) * delta[:, 1]


def torch_swing_outward_loss(
    robot_y: torch.Tensor,
    source_y: torch.Tensor,
    swing: torch.Tensor,
    *,
    slack_m: float,
    left: bool,
) -> torch.Tensor:
    """Stop a swing foot circling wider than the source (circumduction)."""
    active = swing.double()
    if not torch.any(swing):
        return robot_y.new_zeros(())
    if left:
        excess = torch.relu(robot_y - source_y - slack_m)
    else:
        excess = torch.relu(source_y - robot_y - slack_m)
    weighted = excess**2 * active
    return weighted.sum() / torch.clamp(active.sum(), min=1.0)


def torch_segment_samples(
    start: torch.Tensor,
    end: torch.Tensor,
    count: int = 4,
) -> torch.Tensor:
    """Linear samples along each frame's segment, shape (T, count, 3)."""
    weights = torch.linspace(0.0, 1.0, count, dtype=start.dtype, device=start.device)
    return start[:, None, :] * (1.0 - weights)[None, :, None] + end[:, None, :] * weights[
        None, :, None
    ]


def torch_abs_limit_loss(values: torch.Tensor, max_abs: float) -> torch.Tensor:
    """Soft-cap joint angles so a hip cannot wind behind the other leg."""
    penalty = torch.relu(torch.abs(values) - max_abs) ** 2
    return torch.mean(penalty) + penalty.max()


def torch_xy_speed_cap_loss(
    xy: torch.Tensor,
    cap_m_s: torch.Tensor,
    *,
    fps: float = 15.0,
) -> torch.Tensor:
    """Penalize XY steps faster than a per-frame cap (blocks teleport strides)."""
    speed = torch.linalg.norm(xy[1:] - xy[:-1], dim=-1) * fps
    cap = cap_m_s if cap_m_s.ndim == 0 else cap_m_s[1:]
    penalty = torch.relu(speed - cap) ** 2
    return torch.mean(penalty) + penalty.max()


def torch_one_foot_down_loss(
    left_zmin: torch.Tensor,
    right_zmin: torch.Tensor,
    mask: torch.Tensor,
    height_m: float = 0.03,
) -> torch.Tensor:
    """At least one sole stays near the floor so the robot cannot hop/fly."""
    active = mask.double()
    if not torch.any(mask):
        return left_zmin.new_zeros(())
    hover = torch.relu(torch.minimum(left_zmin, right_zmin) - height_m) ** 2
    return torch.sum(hover * active) / torch.clamp(active.sum(), min=1.0)


def putdown_foot_targets(
    root_xy: np.ndarray,
    yaw: float,
    half_width_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Left/right foot XY beside a frozen root, along the robot's left axis."""
    lateral = np.array([-np.sin(yaw), np.cos(yaw)], dtype=np.float64)
    mid = np.asarray(root_xy, dtype=np.float64)
    return mid - half_width_m * lateral, mid + half_width_m * lateral


def narrow_stance_to_width(
    left_xy: np.ndarray,
    right_xy: np.ndarray,
    width_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep the stance midpoint, only shrink/expand left-right distance."""
    left = np.asarray(left_xy, dtype=np.float64).reshape(2)
    right = np.asarray(right_xy, dtype=np.float64).reshape(2)
    mid = 0.5 * (left + right)
    delta = right - left
    span = float(np.linalg.norm(delta))
    if span < 1e-6:
        return left.copy(), right.copy()
    along = delta / span
    half = 0.5 * width_m
    return mid - half * along, mid + half * along


def human_foot_xy_slow(
    joints_core4d: np.ndarray,
    *,
    fps: float = 15.0,
    slow_m_s: float = 0.05,
) -> np.ndarray:
    """Per-frame left/right source-ankle XY speed below ``slow_m_s`` (OmniRetarget-style)."""
    ankles = points_core4d_to_robot(joints_core4d)[
        :,
        [SMPLX_KEYPOINT_INDICES["left_ankle"], SMPLX_KEYPOINT_INDICES["right_ankle"]],
        :2,
    ]
    speed = np.linalg.norm(np.diff(ankles, axis=0), axis=-1) * fps
    speed = np.concatenate([speed[:1], speed], axis=0)
    return speed < slow_m_s


def support_excess_xy(
    com_xy: np.ndarray,
    left_xy: np.ndarray,
    right_xy: np.ndarray,
    left_contact: np.ndarray,
    right_contact: np.ndarray,
    yaw: np.ndarray,
    *,
    half_length: float = 0.05,
    half_width: float = 0.05,
    forward: float = 0.035,
    stance_radius: float = 0.08,
) -> np.ndarray:
    """Unsigned 2D overflow of COM beyond the stance support, in metres."""
    forward_xy = np.stack([np.cos(yaw), np.sin(yaw)], axis=-1)
    left_c = left_xy + forward * forward_xy
    right_c = right_xy + forward * forward_xy
    mid = 0.5 * (left_c + right_c)
    delta = right_c - left_c
    stance = np.linalg.norm(delta, axis=-1, keepdims=True)
    along = delta / np.maximum(stance, 1e-8)
    lateral = np.stack([-along[..., 1], along[..., 0]], axis=-1)
    local = com_xy - mid
    along_proj = np.abs((local * along).sum(axis=-1))
    lat_proj = np.abs((local * lateral).sum(axis=-1))
    double = np.maximum(
        along_proj - 0.5 * stance[..., 0] - half_length,
        0,
    ) + np.maximum(lat_proj - half_width, 0)

    left_err = np.maximum(np.linalg.norm(com_xy - left_c, axis=-1) - stance_radius, 0)
    right_err = np.maximum(np.linalg.norm(com_xy - right_c, axis=-1) - stance_radius, 0)
    both = left_contact & right_contact
    only_left = left_contact & ~right_contact
    only_right = right_contact & ~left_contact
    return np.where(both, double, np.where(only_left, left_err, np.where(only_right, right_err, double)))


def _yaw_rotate_xy(points: torch.Tensor, yaw: torch.Tensor, forward: float) -> torch.Tensor:
    direction = torch.stack([torch.cos(yaw), torch.sin(yaw)], dim=-1)
    return points + forward * direction


def torch_support_excess(
    com_xy: torch.Tensor,
    left_xy: torch.Tensor,
    right_xy: torch.Tensor,
    left_contact: torch.Tensor,
    right_contact: torch.Tensor,
    yaw: torch.Tensor,
    *,
    half_length: float = 0.05,
    half_width: float = 0.05,
    forward: float = 0.035,
    stance_radius: float = 0.08,
) -> torch.Tensor:
    left_c = _yaw_rotate_xy(left_xy, yaw, forward)
    right_c = _yaw_rotate_xy(right_xy, yaw, forward)
    mid = 0.5 * (left_c + right_c)
    delta = right_c - left_c
    stance = torch.linalg.norm(delta, dim=-1, keepdim=True)
    along = delta / torch.clamp(stance, min=1e-8)
    lateral = torch.stack([-along[..., 1], along[..., 0]], dim=-1)
    local = com_xy - mid
    along_proj = torch.abs((local * along).sum(dim=-1))
    lat_proj = torch.abs((local * lateral).sum(dim=-1))
    double = torch.relu(along_proj - 0.5 * stance[..., 0] - half_length) + torch.relu(
        lat_proj - half_width
    )
    left_err = torch.relu(torch.linalg.norm(com_xy - left_c, dim=-1) - stance_radius)
    right_err = torch.relu(torch.linalg.norm(com_xy - right_c, dim=-1) - stance_radius)
    both = left_contact & right_contact
    only_left = left_contact & ~right_contact
    only_right = right_contact & ~left_contact
    single = torch.where(only_left, left_err, torch.where(only_right, right_err, double))
    return torch.where(both, double, single)


def torch_com_center_loss(
    com_xy: torch.Tensor,
    left_xy: torch.Tensor,
    right_xy: torch.Tensor,
    left_contact: torch.Tensor,
    right_contact: torch.Tensor,
    yaw: torch.Tensor,
    *,
    forward: float = 0.035,
) -> torch.Tensor:
    left_c = _yaw_rotate_xy(left_xy, yaw, forward)
    right_c = _yaw_rotate_xy(right_xy, yaw, forward)
    mid = 0.5 * (left_c + right_c)
    only_left = left_contact & ~right_contact
    only_right = right_contact & ~left_contact
    target = torch.where(only_left[:, None], left_c, torch.where(only_right[:, None], right_c, mid))
    return torch.mean((com_xy - target) ** 2)


def torch_foot_height_loss(
    foot_z: torch.Tensor,
    contact: torch.Tensor,
    target_height: float,
) -> torch.Tensor:
    stance_error = (foot_z - target_height) ** 2
    swing_error = torch.relu(target_height - foot_z) ** 2
    return torch.mean(torch.where(contact, stance_error, swing_error))


def torch_foot_slip_loss(
    foot_xy: torch.Tensor,
    contact: torch.Tensor,
    frame_weight: torch.Tensor | None = None,
    *,
    stop_grad_prev: bool = False,
) -> torch.Tensor:
    prev = foot_xy[:-1].detach() if stop_grad_prev else foot_xy[:-1]
    velocity = foot_xy[1:] - prev
    active = contact[1:].double()
    if frame_weight is not None:
        weight = frame_weight[1:]
        while weight.ndim < active.ndim:
            weight = weight.unsqueeze(-1)
        active = active * weight
    if not torch.any(active > 0):
        return foot_xy.new_zeros(())
    return torch.sum((velocity**2).sum(dim=-1) * active) / torch.clamp(
        active.sum(),
        min=1.0,
    )


def torch_below_ground_loss(points_z: torch.Tensor) -> torch.Tensor:
    """Positive when world-z samples go under the floor."""
    depth = torch.relu(-points_z)
    return torch.mean(depth**2) + torch.mean(depth.max(dim=-1).values ** 2)


def torch_forced_swing_loss(
    sole_zmin: torch.Tensor,
    moving: torch.Tensor,
    lift_m: float = 0.04,
) -> torch.Tensor:
    """When the source foot is moving, the sole must leave the floor (step, not skate)."""
    active = moving.double()
    if not torch.any(moving):
        return sole_zmin.new_zeros(())
    return torch.sum(torch.relu(lift_m - sole_zmin) ** 2 * active) / torch.clamp(
        active.sum(), min=1.0
    )


def torch_near_ground_slide_loss(
    points: torch.Tensor,
    *,
    height_m: float = 0.03,
    frame_weight: torch.Tensor | None = None,
    hard: bool = False,
    stop_grad_prev: bool = False,
    plant_per_foot: torch.Tensor | None = None,
    n_left: int | None = None,
) -> torch.Tensor:
    """Penalize XY motion of samples that are near or below the floor.

    If ``plant_per_foot`` is ``(T, 2)``, only the planted foot's sole samples
    are penalized so a swing foot can step.
    """
    prev = points[:-1, :, :2].detach() if stop_grad_prev else points[:-1, :, :2]
    velocity = points[1:, :, :2] - prev
    height = 0.5 * (points[1:, :, 2] + points[:-1, :, 2])
    if hard:
        near = (height < height_m).to(dtype=points.dtype)
    else:
        near = torch.relu(height_m - height)
    if plant_per_foot is not None:
        split = points.shape[1] // 2 if n_left is None else int(n_left)
        sample_plant = torch.cat(
            [
                plant_per_foot[:, 0:1].to(dtype=points.dtype).expand(-1, split),
                plant_per_foot[:, 1:2]
                .to(dtype=points.dtype)
                .expand(-1, points.shape[1] - split),
            ],
            dim=1,
        )
        near = near * sample_plant[1:]
    if frame_weight is not None:
        near = near * frame_weight[1:, None]
    weight = near.sum()
    if float(weight.detach()) <= 0:
        return points.new_zeros(())
    return torch.sum((velocity**2).sum(dim=-1) * near) / torch.clamp(weight, min=1e-8)


def torch_planted_root_slip(
    root_xy: torch.Tensor,
    both_feet_contact: torch.Tensor,
    frame_weight: torch.Tensor | None = None,
    *,
    stop_grad_prev: bool = False,
) -> torch.Tensor:
    prev = root_xy[:-1].detach() if stop_grad_prev else root_xy[:-1]
    velocity = root_xy[1:] - prev
    active = both_feet_contact[1:].double()
    if frame_weight is not None:
        active = active * frame_weight[1:]
    if not torch.any(active > 0):
        return root_xy.new_zeros(())
    return torch.sum((velocity**2).sum(dim=-1) * active) / torch.clamp(
        active.sum(),
        min=1.0,
    )


COLLISION_KEYPOINT_NAMES = (
    "pelvis",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "neck",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
)

HAND_KEYPOINT_NAMES = ("left_wrist", "right_wrist")

# Palmar face of the official rubber-hand STL (left −y, right +y).
LEFT_PALM_CONTACT_OFFSET_M = (0.05, -0.03, 0.0)
RIGHT_PALM_CONTACT_OFFSET_M = (0.05, 0.03, 0.0)
LEFT_PALM_NORMAL_LOCAL = (0.0, -1.0, 0.0)
RIGHT_PALM_NORMAL_LOCAL = (0.0, 1.0, 0.0)


def freeze_stance_xy(centroids: torch.Tensor, plant: torch.Tensor) -> torch.Tensor:
    """Copy each stance foot's first-frame XY across that planted segment."""
    frozen = centroids.clone()
    plant_np = plant.detach().cpu().numpy().astype(bool)
    for foot in range(plant_np.shape[1]):
        index = 0
        n_frames = plant_np.shape[0]
        while index < n_frames:
            if not plant_np[index, foot]:
                index += 1
                continue
            end = index + 1
            while end < n_frames and plant_np[end, foot]:
                end += 1
            frozen[index:end, foot] = centroids[index, foot]
            index = end
    return frozen


def torch_palm_facing_loss(
    origins: torch.Tensor,
    palm_axes: torch.Tensor,
    anchors: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Penalize a palm whose outward palmar axis does not point at the contact."""
    if not torch.any(weights > 0):
        return origins.new_zeros(())
    direction = anchors - origins
    direction = direction / torch.clamp(
        torch.linalg.norm(direction, dim=-1, keepdim=True), min=1e-8
    )
    cosine = (palm_axes * direction).sum(dim=-1)
    return torch.sum(torch.relu(0.2 - cosine) ** 2 * weights) / torch.clamp(
        weights.sum(), min=1e-8
    )


def torch_finger_along_face_loss(
    finger_axes: torch.Tensor,
    toward_box: torch.Tensor,
    weights: torch.Tensor,
    *,
    up: torch.Tensor | None = None,
) -> torch.Tensor:
    """Keep fingers off the box normal and preferably along world-up."""
    if not torch.any(weights > 0):
        return finger_axes.new_zeros(())
    stab = torch.relu((finger_axes * toward_box).sum(dim=-1)) ** 2
    loss = torch.sum(stab * weights)
    if up is not None:
        up_dir = up.reshape(*((1,) * (finger_axes.ndim - 1)), 3)
        up_align = torch.relu(0.35 - torch.abs((finger_axes * up_dir).sum(dim=-1))) ** 2
        loss = loss + torch.sum(up_align * weights)
    return loss / torch.clamp(weights.sum(), min=1e-8)


# G1 rubber-hand mesh extends along +x from the palm-joint origin (~4 cm
# after wrist yaw). Wrist-only AABB clearance cannot see fingertips.
HAND_SURFACE_OFFSETS_M = (
    (0.06, 0.0, 0.0),
    (0.09, 0.0, 0.0),
    (0.07, 0.03, 0.0),
    (0.07, -0.03, 0.0),
)

def _aabb_grid(
    lower: tuple[float, float, float],
    upper: tuple[float, float, float],
    counts: tuple[int, int, int] = (6, 4, 4),
) -> tuple[tuple[float, float, float], ...]:
    axes = [np.linspace(lower[i], upper[i], counts[i]) for i in range(3)]
    xx, yy, zz = np.meshgrid(*axes, indexing="ij")
    points = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)
    return tuple((float(x), float(y), float(z)) for x, y, z in points)


def _g1_mesh_path(name: str) -> Path:
    """Rubber-hand STL. Override with UNITREE_G1_ROOT or ISR_G1_DESCRIPTION.

    The default stays the repository-relative description used by the original
    layout. The mesh is not vendored in the public tree.
    """
    override = os.environ.get("UNITREE_G1_ROOT") or os.environ.get("ISR_G1_DESCRIPTION")
    if override:
        return Path(override) / "meshes" / name
    return (
        Path(__file__).resolve().parents[3]
        / "third_party/unitree_ros/robots/g1_description/meshes"
        / name
    )


def _farthest_point_sample(points: np.ndarray, count: int) -> np.ndarray:
    """Spread samples over mesh vertices, starting at the distal (+x) extreme."""
    if len(points) <= count:
        return np.asarray(points, dtype=np.float64)
    chosen = np.empty((count, 3), dtype=np.float64)
    dist = np.full(len(points), np.inf, dtype=np.float64)
    idx = int(np.argmax(points[:, 0]))
    for i in range(count):
        chosen[i] = points[idx]
        delta = points - points[idx]
        dist = np.minimum(dist, np.einsum("ij,ij->i", delta, delta))
        idx = int(np.argmax(dist))
    return chosen


def _rubber_hand_mesh_offsets(
    stl_name: str, *, count: int = 512
) -> tuple[tuple[float, float, float], ...]:
    """Farthest-point samples on the official rubber-hand STL (no AABB padding)."""
    import trimesh

    vertices = np.asarray(
        trimesh.load_mesh(_g1_mesh_path(stl_name), process=False).vertices,
        dtype=np.float64,
    )
    samples = _farthest_point_sample(vertices, count)
    return tuple((float(x), float(y), float(z)) for x, y, z in samples)


# Official G1 rubber-hand STL vertices in the palm-link frame (~13.2×6.7×10.6 cm).
# Do not pad: the 1.5 cm AABB pad was reporting corners that are not on the mesh.
LEFT_HAND_VOLUME_OFFSETS_M = _rubber_hand_mesh_offsets("left_rubber_hand.STL")
RIGHT_HAND_VOLUME_OFFSETS_M = _rubber_hand_mesh_offsets("right_rubber_hand.STL")

# Front of torso_link.STL (~8 cm forward, 30 cm up). Padded so the chest
# mesh cannot sit inside the box while hands still reach.
TORSO_VOLUME_OFFSETS_M = _aabb_grid(
    (0.02, -0.12, 0.04),
    (0.10, 0.12, 0.30),
    (4, 3, 4),
)

# G1 ankle_roll STL (~21×8×6 cm). Ankle-pitch keypoints sit ~5 cm above
# the sole, so a 5.5 cm ankle gate still lets the toe punch through the
# floor and the box. Sample the roll-link AABB, padded 1 cm.
LEFT_FOOT_VOLUME_OFFSETS_M = _aabb_grid(
    (-0.076, -0.048, -0.045),
    (0.152, 0.048, 0.034),
    (5, 3, 3),
)
RIGHT_FOOT_VOLUME_OFFSETS_M = LEFT_FOOT_VOLUME_OFFSETS_M


def _stl_aabb_grid(
    stl_name: str, counts: tuple[int, int, int] = (4, 3, 3)
) -> tuple[tuple[float, float, float], ...]:
    """Unpadded AABB samples of an official G1 link mesh."""
    import trimesh

    vertices = np.asarray(
        trimesh.load_mesh(_g1_mesh_path(stl_name), process=False).vertices,
        dtype=np.float64,
    )
    lower = tuple(float(v) for v in vertices.min(axis=0))
    upper = tuple(float(v) for v in vertices.max(axis=0))
    return _aabb_grid(lower, upper, counts)


# Pelvis / thigh (hip_yaw) / shin (knee) so the box cannot occupy empty space
# between the sparse hip/knee keypoints.
PELVIS_VOLUME_OFFSETS_M = _stl_aabb_grid("pelvis.STL", (3, 3, 4))
LEFT_HIP_YAW_VOLUME_OFFSETS_M = _stl_aabb_grid("left_hip_yaw_link.STL")
RIGHT_HIP_YAW_VOLUME_OFFSETS_M = _stl_aabb_grid("right_hip_yaw_link.STL")
LEFT_KNEE_VOLUME_OFFSETS_M = _stl_aabb_grid("left_knee_link.STL")
RIGHT_KNEE_VOLUME_OFFSETS_M = _stl_aabb_grid("right_knee_link.STL")
LEG_VOLUME_LINKS = (
    ("pelvis", PELVIS_VOLUME_OFFSETS_M),
    ("left_hip_yaw_link", LEFT_HIP_YAW_VOLUME_OFFSETS_M),
    ("right_hip_yaw_link", RIGHT_HIP_YAW_VOLUME_OFFSETS_M),
    ("left_knee_link", LEFT_KNEE_VOLUME_OFFSETS_M),
    ("right_knee_link", RIGHT_KNEE_VOLUME_OFFSETS_M),
)
ARM_HULL_LINKS = (
    ("left_elbow_link", _rubber_hand_mesh_offsets("left_elbow_link.STL", count=96)),
    ("right_elbow_link", _rubber_hand_mesh_offsets("right_elbow_link.STL", count=96)),
    ("left_wrist_yaw_link", _rubber_hand_mesh_offsets("left_wrist_yaw_link.STL", count=96)),
    ("right_wrist_yaw_link", _rubber_hand_mesh_offsets("right_wrist_yaw_link.STL", count=96)),
)


def _transform_offsets(
    matrix: torch.Tensor,
    offsets: torch.Tensor,
) -> torch.Tensor:
    rotation = matrix[:, :3, :3]
    origin = matrix[:, :3, 3]
    return origin[:, None, :] + torch.einsum("tij,oj->toi", rotation, offsets)


def _transform_offsets_orient_only(
    matrix: torch.Tensor,
    offsets: torch.Tensor,
) -> torch.Tensor:
    """Same as `_transform_offsets` but the link origin has no gradient.

    Hull / AABB clearance can then rotate a rigid palm or sole out of the
    object without backing the wrist or ankle away from the contact face.
    """
    rotation = matrix[:, :3, :3]
    origin = matrix[:, :3, 3].detach()
    return origin[:, None, :] + torch.einsum("tij,oj->toi", rotation, offsets)


def torch_hand_surface_points(
    left_matrix: torch.Tensor,
    right_matrix: torch.Tensor,
    offsets: torch.Tensor,
) -> torch.Tensor:
    """Sample palm/finger points in each rubber-hand frame. Matrices are (T, 4, 4)."""
    return torch.cat(
        [_transform_offsets(left_matrix, offsets), _transform_offsets(right_matrix, offsets)],
        dim=1,
    )


def torch_hand_volume_points(
    left_matrix: torch.Tensor,
    right_matrix: torch.Tensor,
    left_offsets: torch.Tensor,
    right_offsets: torch.Tensor,
) -> torch.Tensor:
    """Sample the full rubber-hand mesh in each palm-link frame. Matrices are (T, 4, 4)."""
    return torch.cat(
        [
            _transform_offsets(left_matrix, left_offsets),
            _transform_offsets(right_matrix, right_offsets),
        ],
        dim=1,
    )


def torch_hand_volume_points_orient_only(
    left_matrix: torch.Tensor,
    right_matrix: torch.Tensor,
    left_offsets: torch.Tensor,
    right_offsets: torch.Tensor,
) -> torch.Tensor:
    return torch.cat(
        [
            _transform_offsets_orient_only(left_matrix, left_offsets),
            _transform_offsets_orient_only(right_matrix, right_offsets),
        ],
        dim=1,
    )


def numpy_transform_offsets(matrix: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    rotation = matrix[:, :3, :3]
    origin = matrix[:, :3, 3]
    return origin[:, None, :] + np.einsum("tij,oj->toi", rotation, offsets)


def numpy_hand_volume_points(
    left_matrix: np.ndarray,
    right_matrix: np.ndarray,
    left_offsets: np.ndarray | None = None,
    right_offsets: np.ndarray | None = None,
) -> np.ndarray:
    left_offsets = (
        np.asarray(LEFT_HAND_VOLUME_OFFSETS_M, dtype=np.float64)
        if left_offsets is None
        else np.asarray(left_offsets, dtype=np.float64)
    )
    right_offsets = (
        np.asarray(RIGHT_HAND_VOLUME_OFFSETS_M, dtype=np.float64)
        if right_offsets is None
        else np.asarray(right_offsets, dtype=np.float64)
    )
    return np.concatenate(
        [
            numpy_transform_offsets(left_matrix, left_offsets),
            numpy_transform_offsets(right_matrix, right_offsets),
        ],
        axis=1,
    )


def torch_softmin_sq_error(
    points: torch.Tensor,
    anchors: torch.Tensor,
    temperature: float = 0.01,
) -> torch.Tensor:
    """Per-frame squared distance from an anchor to the nearest of `points`."""
    delta = points - anchors[:, None, :]
    squared = torch.sum(delta * delta, dim=-1)
    weights = torch.softmax(-squared / temperature, dim=-1)
    return torch.sum(weights * squared, dim=-1)


def numpy_apply_root_yaw(
    local_points: np.ndarray,
    root_translation: np.ndarray,
    root_yaw: np.ndarray,
) -> np.ndarray:
    cosine = np.cos(root_yaw)
    sine = np.sin(root_yaw)
    zeros = np.zeros_like(cosine)
    ones = np.ones_like(cosine)
    rotation = np.stack(
        [cosine, -sine, zeros, sine, cosine, zeros, zeros, zeros, ones],
        axis=-1,
    ).reshape(-1, 3, 3)
    return local_points @ np.transpose(rotation, (0, 2, 1)) + root_translation[:, None]


def aabb_from_vertices(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(vertices, dtype=np.float64)
    lower = points.min(axis=0)
    upper = points.max(axis=0)
    center = 0.5 * (lower + upper)
    half = 0.5 * (upper - lower)
    return center, half


def points_in_object_frame(
    points_core4d: np.ndarray,
    object_poses: np.ndarray,
) -> np.ndarray:
    rotations = object_poses[:, :3, :3]
    translations = object_poses[:, :3, 3]
    return np.einsum(
        "tki,tij->tkj",
        np.asarray(points_core4d) - translations[:, None],
        rotations,
    )


def aabb_inside_depth(
    local_points: np.ndarray,
    center: np.ndarray,
    half: np.ndarray,
    margin: float,
) -> np.ndarray:
    """Positive depth when a point is inside the expanded AABB."""
    slack = (half + margin) - np.abs(local_points - center)
    return np.maximum(slack.min(axis=-1), 0.0)


def torch_aabb_inside_depth(
    local_points: torch.Tensor,
    center: torch.Tensor,
    half: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    slack = (half + margin) - (local_points - center).abs()
    return torch.relu(slack.min(dim=-1).values)


def convex_hull_equations(vertices: np.ndarray) -> np.ndarray:
    """Scipy half-spaces ``n·x + d <= 0`` for points inside the convex hull."""
    from scipy.spatial import ConvexHull

    return np.asarray(ConvexHull(np.asarray(vertices, dtype=np.float64)).equations, dtype=np.float64)


def torch_convex_inside_depth(
    local_points: torch.Tensor,
    equations: torch.Tensor,
    margin_m: float = 0.0,
) -> torch.Tensor:
    """Positive depth when a point is inside the expanded convex hull."""
    values = local_points @ equations[:, :3].T + equations[:, 3]
    return torch.relu(margin_m - values.max(dim=-1).values)


def torch_noncontact_sample_weights(
    points: torch.Tensor,
    anchor: torch.Tensor,
    contact_weight: torch.Tensor,
    temperature: float = 0.002,
) -> torch.Tensor:
    """Per-sample weight: ~0 on the contacting face, ~1 on the rest of the grid.

    ``contact_weight`` is (T,). While it is 0, every sample is fully penalized.
    """
    squared = torch.sum((points - anchor[:, None, :]) ** 2, dim=-1)
    contact_mass = torch.softmax(-squared / temperature, dim=-1)
    return 1.0 - contact_weight[:, None] * contact_mass


def torch_max_inside_loss(depth: torch.Tensor, slack_m: float = 0.0) -> torch.Tensor:
    """Mean-square plus per-frame and global max, so a few 3 cm spikes are not averaged away."""
    if depth.numel() == 0:
        return depth.new_zeros(())
    excess = torch.relu(depth - slack_m) if slack_m > 0 else depth
    per_frame = excess.max(dim=-1).values
    return torch.mean(excess**2) + torch.mean(per_frame**2) + per_frame.max() ** 2


def torch_object_clearance_loss(
    local_points: torch.Tensor,
    center: torch.Tensor,
    half: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    return torch.mean(torch_aabb_inside_depth(local_points, center, half, margin) ** 2)

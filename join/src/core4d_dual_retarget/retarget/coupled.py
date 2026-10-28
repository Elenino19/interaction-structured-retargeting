"""Object-mediated, phase-aware coupled retargeting for two G1 humanoids."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pytorch_kinematics as pk
import torch
import trimesh
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree

from ..audit import load_npz_mapping
from ..contact import (
    fingertips_in_object_space,
    hysteresis_contact,
    resolve_object_model,
)
from ..coordinates import CORE4D_TO_ROBOT, points_core4d_to_robot
from ..robot_model import (
    G1_29DOF_KEYPOINT_LINKS,
    SMPLX_KEYPOINT_INDICES,
    load_actuated_joint_limits,
)
from .balance import (
    ARM_HULL_LINKS,
    COLLISION_KEYPOINT_NAMES,
    HAND_KEYPOINT_NAMES,
    HAND_SURFACE_OFFSETS_M,
    LEFT_FOOT_VOLUME_OFFSETS_M,
    LEFT_HAND_VOLUME_OFFSETS_M,
    LEFT_PALM_CONTACT_OFFSET_M,
    LEFT_PALM_NORMAL_LOCAL,
    LEG_VOLUME_LINKS,
    RIGHT_FOOT_VOLUME_OFFSETS_M,
    RIGHT_HAND_VOLUME_OFFSETS_M,
    RIGHT_PALM_CONTACT_OFFSET_M,
    RIGHT_PALM_NORMAL_LOCAL,
    TORSO_VOLUME_OFFSETS_M,
    _transform_offsets,
    _transform_offsets_orient_only,
    aabb_from_vertices,
    ankle_xy_speed_m_s,
    com_mass_vector,
    convex_hull_equations,
    exclusive_stance_plant,
    freeze_stance_xy,
    gait_support_plant,
    human_foot_contact,
    human_foot_xy_slow,
    human_knee_flexion_rad,
    pelvis_xy_speed_m_s,
    putdown_tail_mask,
    source_ankle_body_y,
    source_plant_mask,
    torch_aabb_inside_depth,
    torch_abs_limit_loss,
    torch_below_ground_loss,
    torch_body_lateral_y,
    torch_both_soles_down_loss,
    torch_com_center_loss,
    torch_convex_inside_depth,
    torch_exclusive_stance,
    torch_finger_along_face_loss,
    torch_foot_height_loss,
    torch_foot_slip_loss,
    torch_forced_swing_loss,
    torch_hand_surface_points,
    torch_hand_volume_points,
    torch_hand_volume_points_orient_only,
    torch_knee_min_bend_loss,
    torch_knee_squat_loss,
    torch_knee_symmetry_loss,
    torch_limb_pair_clearance_loss,
    torch_max_inside_loss,
    torch_near_ground_slide_loss,
    torch_noncontact_sample_weights,
    torch_object_clearance_loss,
    torch_one_foot_down_loss,
    torch_pair_laterality_loss,
    torch_palm_facing_loss,
    torch_planted_root_slip,
    torch_segment_samples,
    torch_softmin_sq_error,
    torch_stance_width_loss,
    torch_support_excess,
    torch_swing_knee_bend_loss,
    torch_swing_knee_target_loss,
    torch_swing_outward_loss,
    torch_xy_speed_cap_loss,
)
from .independent import _apply_root_transform


@dataclass(frozen=True)
class CoupledRetargetConfig:
    iterations: int = 600
    learning_rate: float = 0.015
    pose_fidelity_weight: float = 8.0
    contact_weight: float = 40.0
    root_fidelity_weight: float = 25.0
    velocity_weight: float = 2.0
    acceleration_weight: float = 10.0
    jerk_weight: float = 10.0
    inter_robot_clearance: float = 0.08
    inter_robot_weight: float = 0.1
    anchor_smoothing_sigma: float = 1.5
    contact_ramp_sigma: float = 0.5
    com_support_weight: float = 0.0
    foot_height_weight: float = 0.0
    foot_slip_weight: float = 0.0
    torso_upright_weight: float = 0.0
    com_center_weight: float = 0.0
    object_clearance_weight: float = 0.0
    object_clearance_margin_m: float = 0.12
    hand_clearance_margin_m: float = 0.0
    hand_surface_clearance_margin_m: float = 0.0
    hand_volume_clearance_weight: float = 0.0
    hand_volume_clearance_margin_m: float = 0.008
    torso_volume_clearance_weight: float = 0.0
    torso_volume_clearance_margin_m: float = 0.04
    palm_offset_m: float = 0.03
    planted_root_slip_weight: float = 0.0
    foot_volume_clearance_weight: float = 0.0
    foot_volume_clearance_margin_m: float = 0.02
    foot_ground_weight: float = 0.0
    foot_slide_weight: float = 0.0
    hard_plant_weight: float = 0.0
    plant_height_m: float = 0.02
    source_slow_xy_m_s: float = 0.05
    plant_require_contact: bool = True
    plant_slow_only: bool = False
    object_hull_clearance_weight: float = 0.0
    hull_orient_only: bool = False
    hull_min_local_x_m: float = 0.0
    hull_mask_aabb: bool = True
    hull_use_contact_mask: bool = True
    hull_slack_m: float = 0.0
    hull_clearance_margin_m: float = 0.0
    hull_include_feet: bool = True
    leg_volume_clearance_weight: float = 0.0
    leg_volume_clearance_margin_m: float = 0.0
    leg_hull_weight: float = 0.0
    target_foot_height: float = 0.055
    swing_lift_weight: float = 0.0
    swing_height_m: float = 0.04
    swing_knee_weight: float = 0.0
    swing_knee_min_rad: float = 0.55
    gait_knee_fidelity_weight: float = 0.0
    gait_knee_bias_rad: float = 0.08
    gait_knee_min_rad: float = 0.30
    gait_knee_max_rad: float = 1.10
    grasp_up_frac: float = 1.0
    plant_exclusive: bool = False
    foot_slide_hard: bool = False
    squat_knee_weight: float = 0.0
    squat_knee_max_rad: float = 1.1
    palmar_contact: bool = False
    palm_facing_weight: float = 0.0
    plant_stop_grad: bool = False
    stance_lock_weight: float = 0.0
    finger_up_weight: float = 0.0
    putdown_dual_plant: bool = False
    putdown_phase_min: int = 4
    putdown_phase_max: int = 4
    putdown_tail_frames: int = 0
    squat_stance_width_m: float = 0.24
    putdown_root_lock_weight: float = 0.0
    knee_symmetry_weight: float = 0.0
    squat_knee_min_weight: float = 0.0
    squat_knee_min_rad: float = 0.8
    both_soles_weight: float = 0.0
    stance_width_weight: float = 0.0
    plant_gait_support: bool = False
    leg_laterality_weight: float = 0.0
    leg_laterality_ankle_m: float = 0.06
    leg_laterality_knee_m: float = 0.10
    self_leg_clearance_weight: float = 0.0
    self_leg_clearance_m: float = 0.08
    hip_yaw_limit_weight: float = 0.0
    hip_yaw_max_rad: float = 0.80
    hip_roll_limit_weight: float = 0.0
    hip_roll_max_rad: float = 0.45
    swing_outward_weight: float = 0.0
    swing_outward_slack_m: float = 0.04
    root_speed_cap_weight: float = 0.0
    root_speed_max_m_s: float = 1.6
    root_speed_slack_m_s: float = 0.35
    foot_speed_cap_weight: float = 0.0
    foot_speed_max_m_s: float = 2.0
    approach_ground_weight: float = 0.0
    arm_hull_weight: float = 0.0
    source_approach_root_weight: float = 0.0
    swing_lift_min_root_m_s: float = 0.0
    contact_transition_phase_weight: float = 1.5
    slip_boost_start_phase: int = 1
    slip_boost_factor: float = 5.0
    putdown_slip_boost_factor: float = 10.0
    # --- Evaluation-only controls (defaults reproduce the historical solver) ---
    # Which robots are optimized.  (1, 2) is the joint solver; (1,) or (2,)
    # solves one robot alone with exactly the same per-robot terms and its own
    # gradient clipping (the OA-Independent baseline).
    solve_robots: tuple[int, ...] = (1, 2)
    # Historical solver clips the gradient norm over BOTH robots' parameters
    # jointly (the only cross-robot numerical operation).  True clips each
    # robot's parameters separately (diagnostic control).
    per_robot_grad_clip: bool = False
    # --- RESEARCH EXTENSION (not part of frozen v41; weight 0 disables) ---
    # Joint-contact continuity: during source joint-carry frames, penalize
    # 1 - s_1(t) * s_2(t) where s_r is a differentiable contact indicator of
    # robot r.  The product couples both robots' states.  The separable
    # surrogate replaces the partner indicator by the constant 1.
    joint_contact_continuity_weight: float = 0.0
    joint_contact_tau_m: float = 0.04
    joint_contact_beta_m: float = 0.01
    joint_contact_separable_surrogate: bool = False
    # --- EXT-JC VISUAL REFINEMENT (default-off; EXT-JC w10 and v41 unchanged) ---
    # P1 surface-aware indicator: the per-frame soft-min hull distance is
    # clamped at >= 0 (smooth softplus) before the sigmoid, so the indicator
    # saturates AT the surface and has no gradient rewarding penetration.
    joint_contact_surface_aware: bool = False
    joint_contact_surface_eps_m: float = 0.002
    # Separate soft-min temperature over hand samples (0 -> joint_contact_beta_m).
    # Smaller values reduce the soft-min bias (beta * ln N) below the true minimum.
    joint_contact_softmin_beta_m: float = 0.0
    # P2 temporal: linear ramp of the joint-carry activation E_J over this
    # many frames at each window edge (0 -> hard 0/1 activation).
    joint_contact_edge_taper_frames: int = 0
    # P2 temporal: squared third difference of the wrist keypoints (world frame).
    hand_cartesian_jerk_weight: float = 0.0
    # P3 trust region to a frozen reference trajectory (e.g. v41) for the root,
    # yaw and non-arm joints.  Dead-band squared penalty; arms are free.
    reference_trust_weight: float = 0.0
    reference_trust_root: str = ""
    reference_trust_root_m: float = 0.02
    reference_trust_yaw_rad: float = 0.05
    reference_trust_joint_rad: float = 0.05
    # P1 explicit surface barrier evaluated on the trajectory that the frozen
    # post-hoc smoothing recipe (`smooth-trajectories --clearance-recipe`:
    # body sigma 1.0, arm sigma 0.35, root unsmoothed) will produce.  The raw
    # solve already satisfies the per-frame barriers; penetration appears only
    # after smoothing (object track has duplicated frames), so the barrier
    # must see the smoothed hands.  Same hull depth / margin as `hull`.
    smoothed_hand_barrier_weight: float = 0.0
    smoothed_hand_barrier_sigma: float = 1.0
    smoothed_hand_barrier_arm_sigma: float = 0.35
    smoothed_hand_barrier_margin_m: float = 0.008


def _contact_anchors(
    vertices: np.ndarray,
    object_poses: np.ndarray,
    object_vertices: np.ndarray,
    *,
    palm_offset: float = 0.03,
    smoothing_sigma: float = 0.0,
    grasp_up_frac: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    local_fingertips = fingertips_in_object_space(vertices, object_poses)
    tree = cKDTree(object_vertices)
    distances, indices = tree.query(local_fingertips.reshape(-1, 3), k=1)
    distances = distances.reshape(len(vertices), 10)
    indices = indices.reshape(len(vertices), 10)

    anchors = np.zeros((len(vertices), 2, 3), dtype=np.float64)
    masks = np.zeros((len(vertices), 2), dtype=bool)
    for hand in range(2):
        hand_slice = slice(hand * 5, (hand + 1) * 5)
        hand_distances = distances[:, hand_slice]
        minimum_finger = np.argmin(hand_distances, axis=1)
        selected_indices = indices[:, hand_slice][np.arange(len(vertices)), minimum_finger]
        surface_points = object_vertices[selected_indices]
        selected_fingertips = local_fingertips[:, hand_slice][
            np.arange(len(vertices)),
            minimum_finger,
        ]
        outward = selected_fingertips - surface_points
        norm = np.linalg.norm(outward, axis=-1, keepdims=True)
        outward = outward / np.maximum(norm, 1e-8)
        anchors[:, hand] = surface_points + palm_offset * outward
        if 0.0 < grasp_up_frac < 1.0:
            up_axis = int(np.argmax(np.abs(object_poses[:, 1, :3]).mean(axis=0)))
            lower = object_vertices.min(axis=0)[up_axis]
            upper = object_vertices.max(axis=0)[up_axis]
            mid = 0.5 * (lower + upper)
            half = 0.5 * (upper - lower)
            limit = grasp_up_frac * half
            clamped = np.array(anchors[:, hand], copy=True)
            clamped[:, up_axis] = np.clip(clamped[:, up_axis], mid - limit, mid + limit)
            _, snapped = tree.query(clamped, k=1)
            snapped_pts = object_vertices[snapped]
            outward = selected_fingertips - snapped_pts
            norm = np.linalg.norm(outward, axis=-1, keepdims=True)
            outward = outward / np.maximum(norm, 1e-8)
            anchors[:, hand] = snapped_pts + palm_offset * outward
        masks[:, hand] = hysteresis_contact(hand_distances.min(axis=1))
        if smoothing_sigma > 0:
            anchors[:, hand] = gaussian_filter1d(
                anchors[:, hand],
                sigma=smoothing_sigma,
                axis=0,
                mode="nearest",
            )

    rotations = object_poses[:, :3, :3]
    translations = object_poses[:, :3, 3]
    anchors_world_core = np.einsum("tki,tji->tkj", anchors, rotations) + translations[:, None]
    return points_core4d_to_robot(anchors_world_core), masks


def torch_soft_contact_indicator(
    local_points: torch.Tensor,
    hull_equations: torch.Tensor,
    *,
    tau_m: float,
    beta_m: float,
) -> torch.Tensor:
    """Differentiable per-frame contact indicator in (0, 1) from hull distances.

    ``local_points`` are (T, N, 3) object-frame samples; the signed plane
    distance ``max_j (a_j . x + b_j)`` is negative inside the convex hull and a
    lower bound of the Euclidean distance outside.  A soft-min over samples
    gives a per-frame hand-object distance ``d_t``; the indicator is
    ``sigmoid((tau - d_t) / beta)``.
    """
    signed = (local_points @ hull_equations[:, :3].T + hull_equations[:, 3]).max(dim=-1).values
    soft_min = -beta_m * torch.logsumexp(-signed / beta_m, dim=-1)
    return torch.sigmoid((tau_m - soft_min) / beta_m)


def torch_soft_contact_indicator_surface_aware(
    local_points: torch.Tensor,
    hull_equations: torch.Tensor,
    *,
    tau_m: float,
    beta_m: float,
    softmin_beta_m: float | None = None,
    eps_m: float = 0.002,
) -> torch.Tensor:
    """Surface-aware variant of :func:`torch_soft_contact_indicator`.

    The per-frame soft-min distance ``d_t`` is replaced by the smooth clamp
    ``eps * softplus(d_t / eps) >= 0`` so that the indicator is constant
    (saturated) for any point at or inside the hull surface: being deeper
    inside the object never increases the score and therefore never produces
    a gradient that fights the penetration barriers.  Outside the hull the
    indicator is unchanged up to ``eps``.
    """
    beta_sm = softmin_beta_m if softmin_beta_m else beta_m
    signed = (local_points @ hull_equations[:, :3].T + hull_equations[:, 3]).max(dim=-1).values
    soft_min = -beta_sm * torch.logsumexp(-signed / beta_sm, dim=-1)
    clamped = eps_m * torch.nn.functional.softplus(soft_min / eps_m)
    return torch.sigmoid((tau_m - clamped) / beta_m)


def joint_carry_activation(joint_frames: np.ndarray, taper_frames: int) -> np.ndarray:
    """Per-frame activation of the joint-carry term.

    ``taper_frames == 0`` returns the hard 0/1 mask.  Otherwise each contiguous
    joint-carry window is linearly ramped from ``1/(k+1)`` to ``k/(k+1)`` over
    its first and last ``k`` frames so the term switches on and off smoothly.
    """
    mask = np.asarray(joint_frames, dtype=bool)
    weights = mask.astype(np.float64)
    if taper_frames <= 0 or not mask.any():
        return weights
    k = int(taper_frames)
    padded = np.concatenate([[False], mask, [False]])
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    for start, stop in zip(edges[0::2], edges[1::2], strict=True):
        length = stop - start
        ramp = np.ones(length, dtype=np.float64)
        for i in range(min(k, length)):
            value = (i + 1) / (k + 1)
            ramp[i] = min(ramp[i], value)
            ramp[length - 1 - i] = min(ramp[length - 1 - i], value)
        weights[start:stop] = ramp
    return weights


ARM_JOINT_KEYWORDS = ("shoulder", "elbow", "wrist")


def gaussian_smoothing_matrix(num_frames: int, sigma: float) -> np.ndarray:
    """(T, T) linear operator equal to ``scipy.ndimage.gaussian_filter1d(., sigma, mode='nearest')``."""
    if sigma <= 0:
        return np.eye(num_frames, dtype=np.float64)
    return gaussian_filter1d(np.eye(num_frames, dtype=np.float64), sigma=sigma, axis=0, mode="nearest")


def _load_baseline(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as result:
        return {name: result[name] for name in result.files}


def _joint_bounds(urdf_path: Path, joint_names: tuple[str, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    limits = {joint.name: joint for joint in load_actuated_joint_limits(urdf_path)}
    lower = torch.tensor([limits[name].lower for name in joint_names], dtype=torch.float64)
    upper = torch.tensor([limits[name].upper for name in joint_names], dtype=torch.float64)
    return lower, upper


def _raw_from_bounded(
    positions: np.ndarray,
    midpoint: torch.Tensor,
    half_range: torch.Tensor,
) -> torch.Tensor:
    normalized = (torch.tensor(positions, dtype=torch.float64) - midpoint) / half_range
    return torch.atanh(torch.clamp(normalized, -0.999999, 0.999999))


def retarget_coupled(
    sequence_root: Path,
    sequence_id: str,
    object_models_root: Path,
    baseline_root: Path,
    urdf_path: Path,
    phase_path: Path,
    config: CoupledRetargetConfig | None = None,
    *,
    initial_root_offsets_xy: dict[int, tuple[float, float]] | None = None,
    term_log: list[dict[str, float]] | None = None,
    term_log_grad_every: int = 25,
    probe_callback: Callable[..., None] | None = None,
) -> tuple[dict[int, dict[str, np.ndarray]], np.ndarray]:
    """Jointly refine two independent trajectories around the fixed object.

    Keyword-only arguments are evaluation hooks that default to no-ops:
    ``initial_root_offsets_xy`` shifts a robot's initial root XY by a constant
    (stress tests), ``term_log`` receives one dict per iteration with every
    weighted term plus cross-agent term value / gradient norm / activation,
    and ``probe_callback(evaluate_objective, raw_joints, root_translation,
    root_yaw)`` is invoked once before optimization (separability probes).
    """
    if config is None:
        config = CoupledRetargetConfig()
    persons = tuple(int(p) for p in config.solve_robots)
    if not persons or any(p not in (1, 2) for p in persons):
        raise ValueError(f"solve_robots must be a subset of (1, 2), got {persons}")
    sequence_dir = sequence_root.joinpath(*sequence_id.split("/"))
    object_pose_path = sequence_dir / "smooth_objposes_so3.npy"
    if not object_pose_path.exists():
        object_pose_path = sequence_dir / "smooth_objposes.npy"
    object_poses = np.load(object_pose_path)
    object_model = resolve_object_model(sequence_dir, object_models_root)
    object_mesh = trimesh.load_mesh(object_model, process=False)
    object_vertices = np.asarray(object_mesh.vertices)
    with np.load(phase_path) as phase_data:
        phases = phase_data["phase"]

    baseline_dir = baseline_root / sequence_id.replace("/", "_")
    baselines = {person: _load_baseline(baseline_dir / f"robot{person}.npz") for person in (1, 2)}
    source = {
        person: load_npz_mapping(sequence_dir / f"person{person}_poses.npz")
        for person in (1, 2)
    }
    anchors_and_masks = {
        person: _contact_anchors(
            np.asarray(source[person]["vertices"]),
            object_poses,
            object_vertices,
            smoothing_sigma=config.anchor_smoothing_sigma,
            palm_offset=config.palm_offset_m,
            grasp_up_frac=config.grasp_up_frac,
        )
        for person in (1, 2)
    }

    chain = pk.build_chain_from_urdf(urdf_path.read_text()).to(dtype=torch.float64)
    joint_names = tuple(chain.get_joint_parameter_names())
    keypoint_names = tuple(str(name) for name in baselines[1]["keypoint_names"])
    lower, upper = _joint_bounds(urdf_path, joint_names)
    midpoint = (lower + upper) / 2
    half_range = (upper - lower) / 2

    raw_joints = {
        person: torch.nn.Parameter(
            _raw_from_bounded(baselines[person]["joint_positions"], midpoint, half_range)
        )
        for person in (1, 2)
    }
    initial_roots = {
        person: np.array(baselines[person]["root_translation"], dtype=np.float64, copy=True)
        for person in (1, 2)
    }
    if initial_root_offsets_xy:
        for person, (dx, dy) in initial_root_offsets_xy.items():
            initial_roots[int(person)][:, 0] += float(dx)
            initial_roots[int(person)][:, 1] += float(dy)
    root_translation = {
        person: torch.nn.Parameter(torch.tensor(initial_roots[person], dtype=torch.float64))
        for person in (1, 2)
    }
    root_yaw = {
        person: torch.nn.Parameter(torch.tensor(baselines[person]["root_yaw"], dtype=torch.float64))
        for person in (1, 2)
    }
    # Historical parameter order (all joints, then roots, then yaws) is kept for
    # the joint solve so that gradient clipping sees the identical vector.
    parameters = (
        [raw_joints[p] for p in persons]
        + [root_translation[p] for p in persons]
        + [root_yaw[p] for p in persons]
    )
    optimizer = torch.optim.Adam(parameters, lr=config.learning_rate)
    cross_agent_enabled = len(persons) == 2

    baseline_keypoints = {
        person: torch.tensor(baselines[person]["robot_keypoints"], dtype=torch.float64)
        for person in (1, 2)
    }
    baseline_roots = {
        person: torch.tensor(baselines[person]["root_translation"], dtype=torch.float64)
        for person in (1, 2)
    }
    contact_anchors = {
        person: torch.tensor(anchors_and_masks[person][0], dtype=torch.float64)
        for person in (1, 2)
    }
    contact_weights = {
        person: torch.tensor(
            gaussian_filter1d(
                anchors_and_masks[person][1].astype(np.float64),
                sigma=config.contact_ramp_sigma,
                axis=0,
                mode="nearest",
            ),
            dtype=torch.float64,
        )
        for person in (1, 2)
    }
    # Source-defined simultaneous-contact frames (per-person OR of hand masks).
    joint_frames_np = anchors_and_masks[1][1].any(axis=1) & anchors_and_masks[2][1].any(axis=1)
    joint_frames = torch.tensor(joint_frames_np, dtype=torch.bool)
    joint_activation = torch.tensor(
        joint_carry_activation(joint_frames_np, config.joint_contact_edge_taper_frames),
        dtype=torch.float64,
    )
    # P3 trust region: frozen reference trajectories (same layout as a result root).
    reference_trust: dict[int, dict[str, torch.Tensor]] = {}
    if config.reference_trust_weight > 0:
        if not config.reference_trust_root:
            raise ValueError("reference_trust_weight > 0 requires reference_trust_root")
        stem = sequence_id.replace("/", "_")
        for person in persons:
            ref = _load_baseline(Path(config.reference_trust_root) / stem / f"robot{person}.npz")
            if tuple(str(n) for n in ref["joint_names"]) != joint_names:
                raise ValueError("reference_trust_root joint order differs from the URDF")
            if len(ref["root_translation"]) != len(baselines[person]["root_translation"]):
                raise ValueError("reference_trust_root frame count differs from the baseline")
            reference_trust[person] = {
                "joints": torch.tensor(ref["joint_positions"], dtype=torch.float64),
                "root": torch.tensor(ref["root_translation"], dtype=torch.float64),
                "yaw": torch.tensor(ref["root_yaw"], dtype=torch.float64),
            }
    trust_joint_mask = torch.tensor(
        [not any(k in name for k in ARM_JOINT_KEYWORDS) for name in joint_names],
        dtype=torch.float64,
    )
    # P1 smoothing-aware barrier: per-joint-group smoothing operators.
    smoothed_barrier_enabled = config.smoothed_hand_barrier_weight > 0
    if smoothed_barrier_enabled:
        num_frames = len(baselines[1]["root_translation"])
        smooth_body = torch.tensor(
            gaussian_smoothing_matrix(num_frames, config.smoothed_hand_barrier_sigma),
            dtype=torch.float64,
        )
        smooth_arm = torch.tensor(
            gaussian_smoothing_matrix(num_frames, config.smoothed_hand_barrier_arm_sigma),
            dtype=torch.float64,
        )
        arm_column_mask = (1.0 - trust_joint_mask)[None, :]
    phase_weights = torch.ones(len(phases), dtype=torch.float64)
    phase_weights[phases == 3] = config.contact_transition_phase_weight
    # Walk-up can still step; lock skating once anyone is in single-carry/approach.
    slip_boost = torch.ones(len(phases), dtype=torch.float64)
    slip_boost[np.asarray(phases) >= config.slip_boost_start_phase] = (
        config.slip_boost_factor
    )
    palm_indices = [
        keypoint_names.index("left_wrist"),
        keypoint_names.index("right_wrist"),
    ]
    ankle_indices = [
        keypoint_names.index("left_ankle"),
        keypoint_names.index("right_ankle"),
    ]
    collision_indices = [
        keypoint_names.index(name)
        for name in COLLISION_KEYPOINT_NAMES
        if name in keypoint_names
    ]
    hand_indices = [
        keypoint_names.index(name)
        for name in HAND_KEYPOINT_NAMES
        if name in keypoint_names
    ]
    object_center, object_half = aabb_from_vertices(object_vertices)
    object_center_t = torch.tensor(object_center, dtype=torch.float64)
    object_half_t = torch.tensor(object_half, dtype=torch.float64)
    hull_equations_t = torch.tensor(convex_hull_equations(object_vertices), dtype=torch.float64)
    object_rotations = torch.tensor(object_poses[:, :3, :3], dtype=torch.float64)
    object_translations = torch.tensor(object_poses[:, :3, 3], dtype=torch.float64)
    core4d_from_robot = torch.tensor(CORE4D_TO_ROBOT, dtype=torch.float64)
    hand_surface_offsets = torch.tensor(HAND_SURFACE_OFFSETS_M, dtype=torch.float64)
    left_hand_volume_offsets = torch.tensor(LEFT_HAND_VOLUME_OFFSETS_M, dtype=torch.float64)
    right_hand_volume_offsets = torch.tensor(RIGHT_HAND_VOLUME_OFFSETS_M, dtype=torch.float64)
    left_palm_contact_offsets = torch.tensor(
        [LEFT_PALM_CONTACT_OFFSET_M], dtype=torch.float64
    )
    right_palm_contact_offsets = torch.tensor(
        [RIGHT_PALM_CONTACT_OFFSET_M], dtype=torch.float64
    )
    left_palm_normal = torch.tensor(LEFT_PALM_NORMAL_LOCAL, dtype=torch.float64)
    right_palm_normal = torch.tensor(RIGHT_PALM_NORMAL_LOCAL, dtype=torch.float64)
    finger_axis_local = torch.tensor((1.0, 0.0, 0.0), dtype=torch.float64)
    robot_world_up = torch.tensor((0.0, 0.0, 1.0), dtype=torch.float64)
    n_hand_volume = left_hand_volume_offsets.shape[0]
    torso_volume_offsets = torch.tensor(TORSO_VOLUME_OFFSETS_M, dtype=torch.float64)
    left_foot_volume_offsets = torch.tensor(LEFT_FOOT_VOLUME_OFFSETS_M, dtype=torch.float64)
    right_foot_volume_offsets = torch.tensor(RIGHT_FOOT_VOLUME_OFFSETS_M, dtype=torch.float64)
    leg_volume_parts = [
        (name, torch.tensor(offsets, dtype=torch.float64))
        for name, offsets in LEG_VOLUME_LINKS
    ]
    arm_hull_parts = [
        (name, torch.tensor(offsets, dtype=torch.float64))
        for name, offsets in ARM_HULL_LINKS
    ]
    volume_enabled = config.hand_volume_clearance_weight > 0
    torso_volume_enabled = config.torso_volume_clearance_weight > 0
    foot_volume_enabled = config.foot_volume_clearance_weight > 0
    foot_ground_enabled = config.foot_ground_weight > 0 or config.foot_slide_weight > 0
    hard_plant_enabled = config.hard_plant_weight > 0
    hull_enabled = config.object_hull_clearance_weight > 0
    leg_volume_enabled = config.leg_volume_clearance_weight > 0 or config.leg_hull_weight > 0
    arm_hull_enabled = config.arm_hull_weight > 0
    swing_enabled = config.swing_lift_weight > 0
    foot_samples_enabled = (
        foot_volume_enabled
        or foot_ground_enabled
        or hard_plant_enabled
        or hull_enabled
        or swing_enabled
        or config.both_soles_weight > 0
        or config.stance_width_weight > 0
        or config.stance_lock_weight > 0
    )
    clearance_enabled = config.object_clearance_weight > 0
    planted_root_enabled = config.planted_root_slip_weight > 0
    com_masses = torch.tensor(com_mass_vector(keypoint_names), dtype=torch.float64)
    source_foot_contact = {
        person: torch.tensor(
            human_foot_contact(np.asarray(source[person]["joints"])),
            dtype=torch.bool,
        )
        for person in (1, 2)
    }
    source_foot_slow = {
        person: torch.tensor(
            human_foot_xy_slow(
                np.asarray(source[person]["joints"]),
                fps=15.0,
                slow_m_s=config.source_slow_xy_m_s,
            ),
            dtype=torch.bool,
        )
        for person in (1, 2)
    }
    phase_ge2 = torch.tensor(np.asarray(phases) >= 2, dtype=torch.bool)
    source_plant = {
        person: torch.tensor(
            source_plant_mask(
                source_foot_slow[person].numpy(),
                source_foot_contact[person].numpy(),
                phase_ge2.numpy(),
                require_contact=config.plant_require_contact,
                slow_only=config.plant_slow_only,
            ),
            dtype=torch.bool,
        )
        for person in (1, 2)
    }
    if config.plant_exclusive:
        exclusive = {}
        for person in (1, 2):
            ankle_z = points_core4d_to_robot(np.asarray(source[person]["joints"]))[
                :,
                [
                    SMPLX_KEYPOINT_INDICES["left_ankle"],
                    SMPLX_KEYPOINT_INDICES["right_ankle"],
                ],
                2,
            ]
            exclusive[person] = torch.tensor(
                exclusive_stance_plant(source_plant[person].numpy(), ankle_z),
                dtype=torch.bool,
            )
        source_plant = exclusive
    phase_arr = np.asarray(phases)
    putdown = torch.tensor(
        putdown_tail_mask(
            phase_arr,
            phase_min=config.putdown_phase_min,
            phase_max=config.putdown_phase_max,
            tail_frames=config.putdown_tail_frames,
        ),
        dtype=torch.bool,
    )
    carry_walk = torch.tensor(phase_arr == 4, dtype=torch.bool) & ~putdown
    approach = torch.tensor(
        (phase_arr >= 1) & (phase_arr <= 3), dtype=torch.bool
    )
    step_walk = carry_walk | approach
    swing_mask: dict[int, torch.Tensor] = {}
    if config.plant_gait_support:
        supported = {}
        for person in (1, 2):
            speed = ankle_xy_speed_m_s(
                np.asarray(source[person]["joints"]), fps=15.0
            )
            supported[person] = torch.tensor(
                gait_support_plant(source_foot_slow[person].numpy(), speed),
                dtype=torch.bool,
            )
        source_plant = supported
    if config.putdown_dual_plant:
        updated = {}
        for person in (1, 2):
            plant = source_plant[person].clone()
            plant[putdown] = True
            if bool(putdown.any()):
                start = int(torch.argmax(putdown.to(dtype=torch.int64)))
                if start > 0:
                    plant[start - 1] = False
            updated[person] = plant
        source_plant = updated
    if config.swing_lift_min_root_m_s > 0:
        idle_planted = {}
        approach_np = approach.detach().cpu().numpy()
        for person in (1, 2):
            plant = source_plant[person].clone()
            idle = (
                pelvis_xy_speed_m_s(np.asarray(source[person]["joints"]))
                < config.swing_lift_min_root_m_s
            ) & approach_np
            plant[idle] = True
            idle_planted[person] = plant
        source_plant = idle_planted
    if bool(putdown.any()):
        slip_boost = slip_boost.clone()
        slip_boost[putdown] = config.putdown_slip_boost_factor
    for person in (1, 2):
        swing_mask[person] = ~source_plant[person]
    gait_knee_targets = {
        person: torch.tensor(
            np.clip(
                human_knee_flexion_rad(np.asarray(source[person]["joints"]))
                + config.gait_knee_bias_rad,
                config.gait_knee_min_rad,
                config.gait_knee_max_rad,
            ),
            dtype=torch.float64,
        )
        for person in (1, 2)
    }
    source_swing_y = {
        person: torch.tensor(
            source_ankle_body_y(np.asarray(source[person]["joints"])),
            dtype=torch.float64,
        )
        for person in (1, 2)
    }
    source_root_speed = {
        person: torch.tensor(config.root_speed_max_m_s, dtype=torch.float64)
        for person in (1, 2)
    }
    source_foot_speed_cap = {
        person: torch.tensor(config.foot_speed_max_m_s, dtype=torch.float64)
        for person in (1, 2)
    }
    source_pelvis_xy = {
        person: torch.tensor(
            points_core4d_to_robot(np.asarray(source[person]["joints"]))[
                :, SMPLX_KEYPOINT_INDICES["pelvis"], :2
            ],
            dtype=torch.float64,
        )
        for person in (1, 2)
    }
    source_pelvis_speed = {
        person: torch.tensor(
            pelvis_xy_speed_m_s(np.asarray(source[person]["joints"])),
            dtype=torch.float64,
        )
        for person in (1, 2)
    }
    knee_indices = tuple(
        joint_names.index(name)
        for name in ("left_knee_joint", "right_knee_joint")
        if name in joint_names
    )
    hip_yaw_indices = tuple(
        joint_names.index(name)
        for name in ("left_hip_yaw_joint", "right_hip_yaw_joint")
        if name in joint_names
    )
    hip_roll_indices = tuple(
        joint_names.index(name)
        for name in ("left_hip_roll_joint", "right_hip_roll_joint")
        if name in joint_names
    )
    hip_kp = (
        keypoint_names.index("left_hip"),
        keypoint_names.index("right_hip"),
    )
    knee_kp = (
        keypoint_names.index("left_knee"),
        keypoint_names.index("right_knee"),
    )
    stance_lock_xy: dict[int, torch.Tensor] = {}
    putdown_root_xy: dict[int, torch.Tensor] = {}
    putdown_start = (
        int(torch.argmax(putdown.to(dtype=torch.int64)))
        if config.putdown_dual_plant and bool(putdown.any())
        else None
    )
    if putdown_start is not None:
        for person in (1, 2):
            putdown_root_xy[person] = torch.tensor(
                np.asarray(
                    baselines[person]["root_translation"][putdown_start, :2],
                    dtype=np.float64,
                ),
                dtype=torch.float64,
            )
    if config.stance_lock_weight > 0:
        with torch.no_grad():
            for person in (1, 2):
                joints0 = torch.tensor(
                    baselines[person]["joint_positions"], dtype=torch.float64
                )
                tf0 = chain.forward_kinematics(joints0)
                local_feet = torch.cat(
                    [
                        _transform_offsets(
                            tf0["left_ankle_roll_link"].get_matrix(),
                            left_foot_volume_offsets,
                        ),
                        _transform_offsets(
                            tf0["right_ankle_roll_link"].get_matrix(),
                            right_foot_volume_offsets,
                        ),
                    ],
                    dim=1,
                )
                feet0 = _apply_root_transform(
                    local_feet,
                    torch.tensor(
                        baselines[person]["root_translation"], dtype=torch.float64
                    ),
                    torch.tensor(baselines[person]["root_yaw"], dtype=torch.float64),
                )
                n_lock = left_foot_volume_offsets.shape[0]
                centroids0 = torch.stack(
                    [
                        feet0[:, :n_lock, :2].mean(dim=1),
                        feet0[:, n_lock:, :2].mean(dim=1),
                    ],
                    dim=1,
                )
                stance_lock_xy[person] = freeze_stance_xy(
                    centroids0, source_plant[person]
                )
    balance_enabled = (
        config.com_support_weight
        + config.foot_height_weight
        + config.foot_slip_weight
        + config.torso_upright_weight
        + config.com_center_weight
        + config.planted_root_slip_weight
        + config.hard_plant_weight
        > 0
    )

    hand_volume_local: dict[int, torch.Tensor] = {}

    def evaluate_objective() -> tuple[
        torch.Tensor,
        dict[int, tuple[torch.Tensor, torch.Tensor]],
        dict[str, torch.Tensor],
    ]:
        """Build the full objective from the current parameters.

        Returns the scalar loss, per-robot (joints, keypoints), and a dict of
        *weighted* term contributions plus the unweighted cross-agent terms.
        """
        current: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        velocity_loss = torch.zeros((), dtype=torch.float64)
        acceleration_loss = torch.zeros((), dtype=torch.float64)
        jerk_loss = torch.zeros((), dtype=torch.float64)
        hand_jerk_loss = torch.zeros((), dtype=torch.float64)
        trust_loss = torch.zeros((), dtype=torch.float64)
        smoothed_barrier_loss = torch.zeros((), dtype=torch.float64)
        pose_loss = torch.zeros((), dtype=torch.float64)
        root_loss = torch.zeros((), dtype=torch.float64)
        contact_loss = torch.zeros((), dtype=torch.float64)
        support_loss = torch.zeros((), dtype=torch.float64)
        foot_height_loss = torch.zeros((), dtype=torch.float64)
        foot_slip_loss = torch.zeros((), dtype=torch.float64)
        torso_loss = torch.zeros((), dtype=torch.float64)
        center_loss = torch.zeros((), dtype=torch.float64)
        clearance_loss = torch.zeros((), dtype=torch.float64)
        hand_volume_loss = torch.zeros((), dtype=torch.float64)
        torso_volume_loss = torch.zeros((), dtype=torch.float64)
        foot_volume_loss = torch.zeros((), dtype=torch.float64)
        leg_volume_loss = torch.zeros((), dtype=torch.float64)
        foot_ground_loss = torch.zeros((), dtype=torch.float64)
        foot_slide_loss = torch.zeros((), dtype=torch.float64)
        planted_root_loss = torch.zeros((), dtype=torch.float64)
        hull_loss = torch.zeros((), dtype=torch.float64)
        leg_hull_loss = torch.zeros((), dtype=torch.float64)
        arm_hull_loss = torch.zeros((), dtype=torch.float64)
        swing_loss = torch.zeros((), dtype=torch.float64)
        swing_knee_loss = torch.zeros((), dtype=torch.float64)
        gait_knee_loss = torch.zeros((), dtype=torch.float64)
        squat_loss = torch.zeros((), dtype=torch.float64)
        knee_min_loss = torch.zeros((), dtype=torch.float64)
        palm_facing_loss = torch.zeros((), dtype=torch.float64)
        finger_up_loss = torch.zeros((), dtype=torch.float64)
        stance_lock_loss = torch.zeros((), dtype=torch.float64)
        knee_sym_loss = torch.zeros((), dtype=torch.float64)
        both_soles_loss = torch.zeros((), dtype=torch.float64)
        stance_width_loss = torch.zeros((), dtype=torch.float64)
        putdown_root_loss = torch.zeros((), dtype=torch.float64)
        laterality_loss = torch.zeros((), dtype=torch.float64)
        self_leg_loss = torch.zeros((), dtype=torch.float64)
        hip_yaw_loss = torch.zeros((), dtype=torch.float64)
        hip_roll_loss = torch.zeros((), dtype=torch.float64)
        outward_loss = torch.zeros((), dtype=torch.float64)
        speed_cap_loss = torch.zeros((), dtype=torch.float64)
        approach_ground_loss = torch.zeros((), dtype=torch.float64)
        approach_root_loss = torch.zeros((), dtype=torch.float64)
        hand_volume_local.clear()
        for person in persons:
            joints = midpoint + half_range * torch.tanh(raw_joints[person])
            transforms = chain.forward_kinematics(joints)
            local_keypoints = torch.stack(
                [
                    transforms[G1_29DOF_KEYPOINT_LINKS[name]].get_matrix()[:, :3, 3]
                    for name in keypoint_names
                ],
                dim=1,
            )
            keypoints = _apply_root_transform(
                local_keypoints,
                root_translation[person],
                root_yaw[person],
            )
            current[person] = (joints, keypoints)
            pose_loss = pose_loss + torch.mean((keypoints - baseline_keypoints[person]) ** 2)
            root_loss = root_loss + torch.mean(
                (root_translation[person] - baseline_roots[person]) ** 2
            )
            velocity_loss = velocity_loss + torch.mean((joints[1:] - joints[:-1]) ** 2)
            acceleration_loss = acceleration_loss + torch.mean(
                (joints[2:] - 2 * joints[1:-1] + joints[:-2]) ** 2
            )
            jerk_loss = jerk_loss + torch.mean(
                (
                    joints[3:]
                    - 3 * joints[2:-1]
                    + 3 * joints[1:-2]
                    - joints[:-3]
                )
                ** 2
            )
            palms = keypoints[:, palm_indices]
            if config.hand_cartesian_jerk_weight > 0:
                hand_jerk_loss = hand_jerk_loss + torch.mean(
                    (palms[3:] - 3 * palms[2:-1] + 3 * palms[1:-2] - palms[:-3]) ** 2
                )
            if person in reference_trust:
                ref = reference_trust[person]
                joint_excess = torch.relu(
                    (joints - ref["joints"]).abs() - config.reference_trust_joint_rad
                )
                root_excess = torch.relu(
                    (root_translation[person] - ref["root"]).abs() - config.reference_trust_root_m
                )
                yaw_delta = torch.atan2(
                    torch.sin(root_yaw[person] - ref["yaw"]),
                    torch.cos(root_yaw[person] - ref["yaw"]),
                )
                yaw_excess = torch.relu(yaw_delta.abs() - config.reference_trust_yaw_rad)
                trust_loss = (
                    trust_loss
                    + torch.mean((joint_excess * trust_joint_mask) ** 2)
                    + torch.mean(root_excess**2)
                    + torch.mean(yaw_excess**2)
                )
            weights = contact_weights[person] * phase_weights[:, None]
            left_hand_mat = transforms["left_rubber_hand"].get_matrix()
            right_hand_mat = transforms["right_rubber_hand"].get_matrix()
            palmar_contact_enabled = (
                config.palmar_contact
                or config.palm_facing_weight > 0
                or config.finger_up_weight > 0
            )
            palmar = None
            palm_origins = None
            palm_axes = None
            finger_axes = None
            if palmar_contact_enabled:
                palmar_local = torch.cat(
                    [
                        _transform_offsets(left_hand_mat, left_palm_contact_offsets),
                        _transform_offsets(right_hand_mat, right_palm_contact_offsets),
                    ],
                    dim=1,
                )
                palmar = _apply_root_transform(
                    palmar_local,
                    root_translation[person],
                    root_yaw[person],
                )
                origin_local = torch.stack(
                    [left_hand_mat[:, :3, 3], right_hand_mat[:, :3, 3]],
                    dim=1,
                )
                palm_origins = _apply_root_transform(
                    origin_local,
                    root_translation[person],
                    root_yaw[person],
                )
                axis_local = torch.stack(
                    [
                        torch.einsum("tij,j->ti", left_hand_mat[:, :3, :3], left_palm_normal),
                        torch.einsum("tij,j->ti", right_hand_mat[:, :3, :3], right_palm_normal),
                    ],
                    dim=1,
                )
                axis_tips = _apply_root_transform(
                    origin_local + axis_local,
                    root_translation[person],
                    root_yaw[person],
                )
                palm_axes = axis_tips - palm_origins
                palm_axes = palm_axes / torch.clamp(
                    torch.linalg.norm(palm_axes, dim=-1, keepdim=True), min=1e-8
                )
                finger_local = torch.stack(
                    [
                        torch.einsum("tij,j->ti", left_hand_mat[:, :3, :3], finger_axis_local),
                        torch.einsum("tij,j->ti", right_hand_mat[:, :3, :3], finger_axis_local),
                    ],
                    dim=1,
                )
                finger_tips = _apply_root_transform(
                    origin_local + finger_local,
                    root_translation[person],
                    root_yaw[person],
                )
                finger_axes = finger_tips - palm_origins
                finger_axes = finger_axes / torch.clamp(
                    torch.linalg.norm(finger_axes, dim=-1, keepdim=True), min=1e-8
                )
            local_volume = None
            volume = None
            if smoothed_barrier_enabled:
                # Hands of the trajectory the frozen smoothing recipe will output
                # (body joints sigma 1.0, arm joints sigma 0.35, root unsmoothed).
                joints_smoothed = (smooth_body @ joints) * (1.0 - arm_column_mask) + (
                    smooth_arm @ joints
                ) * arm_column_mask
                transforms_smoothed = chain.forward_kinematics(joints_smoothed)
                # The recipe then lifts the root so no sole sample is underground
                # (per-frame relu lift, Gaussian-smoothed with sigma 1).
                smoothed_feet = _apply_root_transform(
                    torch.cat(
                        [
                            _transform_offsets(
                                transforms_smoothed["left_ankle_roll_link"].get_matrix(),
                                left_foot_volume_offsets,
                            ),
                            _transform_offsets(
                                transforms_smoothed["right_ankle_roll_link"].get_matrix(),
                                right_foot_volume_offsets,
                            ),
                        ],
                        dim=1,
                    ),
                    root_translation[person],
                    root_yaw[person],
                )
                sole_lift = smooth_body @ torch.relu(-smoothed_feet[..., 2].min(dim=1).values)
                lifted_root = root_translation[person] + torch.stack(
                    [torch.zeros_like(sole_lift), torch.zeros_like(sole_lift), sole_lift], dim=-1
                )
                smoothed_volume = _apply_root_transform(
                    torch_hand_volume_points(
                        transforms_smoothed["left_rubber_hand"].get_matrix(),
                        transforms_smoothed["right_rubber_hand"].get_matrix(),
                        left_hand_volume_offsets,
                        right_hand_volume_offsets,
                    ),
                    lifted_root,
                    root_yaw[person],
                )
                smoothed_local = torch.einsum(
                    "tki,tij->tkj",
                    smoothed_volume @ core4d_from_robot - object_translations[:, None],
                    object_rotations,
                )
                smoothed_barrier_loss = smoothed_barrier_loss + torch_max_inside_loss(
                    torch_convex_inside_depth(
                        smoothed_local,
                        hull_equations_t,
                        margin_m=config.smoothed_hand_barrier_margin_m,
                    )
                )
            if volume_enabled:
                local_volume = torch_hand_volume_points(
                    left_hand_mat,
                    right_hand_mat,
                    left_hand_volume_offsets,
                    right_hand_volume_offsets,
                )
                volume = _apply_root_transform(
                    local_volume,
                    root_translation[person],
                    root_yaw[person],
                )
                if config.palmar_contact and palmar is not None:
                    weighted_error = (
                        torch.sum((palmar - contact_anchors[person]) ** 2, dim=-1) * weights
                    )
                else:
                    left_volume = volume[:, :n_hand_volume]
                    right_volume = volume[:, n_hand_volume:]
                    left_err = torch_softmin_sq_error(
                        left_volume, contact_anchors[person][:, 0]
                    )
                    right_err = torch_softmin_sq_error(
                        right_volume, contact_anchors[person][:, 1]
                    )
                    weighted_error = torch.stack([left_err, right_err], dim=-1) * weights
            elif config.palmar_contact and palmar is not None:
                weighted_error = (
                    torch.sum((palmar - contact_anchors[person]) ** 2, dim=-1) * weights
                )
            else:
                weighted_error = (
                    torch.sum((palms - contact_anchors[person]) ** 2, dim=-1)
                    * weights
                )
            if torch.any(weights > 0):
                contact_loss = contact_loss + torch.sum(weighted_error) / torch.sum(weights)
            if config.palm_facing_weight > 0 and palm_origins is not None:
                palm_facing_loss = palm_facing_loss + torch_palm_facing_loss(
                    palm_origins,
                    palm_axes,
                    contact_anchors[person],
                    weights,
                )
            if config.finger_up_weight > 0 and finger_axes is not None:
                toward = contact_anchors[person] - palm_origins
                toward = toward / torch.clamp(
                    torch.linalg.norm(toward, dim=-1, keepdim=True), min=1e-8
                )
                finger_up_loss = finger_up_loss + torch_finger_along_face_loss(
                    finger_axes,
                    toward,
                    weights,
                    up=robot_world_up,
                )

            if clearance_enabled and collision_indices:
                body = keypoints[:, collision_indices]
                body_core4d = body @ core4d_from_robot
                local = torch.einsum(
                    "tki,tij->tkj",
                    body_core4d - object_translations[:, None],
                    object_rotations,
                )
                clearance_loss = clearance_loss + torch_object_clearance_loss(
                    local,
                    object_center_t,
                    object_half_t,
                    config.object_clearance_margin_m,
                )
                if config.hand_clearance_margin_m > 0 and hand_indices:
                    hands = keypoints[:, hand_indices]
                    hands_core4d = hands @ core4d_from_robot
                    hand_local = torch.einsum(
                        "tki,tij->tkj",
                        hands_core4d - object_translations[:, None],
                        object_rotations,
                    )
                    clearance_loss = clearance_loss + torch_object_clearance_loss(
                        hand_local,
                        object_center_t,
                        object_half_t,
                        config.hand_clearance_margin_m,
                    )
                if config.hand_surface_clearance_margin_m > 0:
                    local_surface = torch_hand_surface_points(
                        transforms["left_rubber_hand"].get_matrix(),
                        transforms["right_rubber_hand"].get_matrix(),
                        hand_surface_offsets,
                    )
                    surface = _apply_root_transform(
                        local_surface,
                        root_translation[person],
                        root_yaw[person],
                    )
                    surface_core4d = surface @ core4d_from_robot
                    surface_local = torch.einsum(
                        "tki,tij->tkj",
                        surface_core4d - object_translations[:, None],
                        object_rotations,
                    )
                    clearance_loss = clearance_loss + torch_object_clearance_loss(
                        surface_local,
                        object_center_t,
                        object_half_t,
                        config.hand_surface_clearance_margin_m,
                    )

            if volume_enabled and volume is not None:
                volume_core4d = volume @ core4d_from_robot
                volume_local = torch.einsum(
                    "tki,tij->tkj",
                    volume_core4d - object_translations[:, None],
                    object_rotations,
                )
                hand_volume_local[person] = volume_local
                aabb_local = volume_local
                if hull_enabled and config.hull_orient_only:
                    local_orient = torch_hand_volume_points_orient_only(
                        transforms["left_rubber_hand"].get_matrix(),
                        transforms["right_rubber_hand"].get_matrix(),
                        left_hand_volume_offsets,
                        right_hand_volume_offsets,
                    )
                    volume_orient = _apply_root_transform(
                        local_orient,
                        root_translation[person].detach(),
                        root_yaw[person].detach(),
                    )
                    aabb_local = torch.einsum(
                        "tki,tij->tkj",
                        volume_orient @ core4d_from_robot
                        - object_translations[:, None],
                        object_rotations,
                    )
                depth = torch_aabb_inside_depth(
                    aabb_local,
                    object_center_t,
                    object_half_t,
                    config.hand_volume_clearance_margin_m,
                )
                if hull_enabled:
                    hull_local = aabb_local if config.hull_orient_only else volume_local
                    hull_depth = torch_convex_inside_depth(
                        hull_local,
                        hull_equations_t,
                        margin_m=config.hull_clearance_margin_m,
                    )
                    if config.hull_use_contact_mask:
                        keep = torch.cat(
                            [
                                torch_noncontact_sample_weights(
                                    volume[:, :n_hand_volume],
                                    contact_anchors[person][:, 0],
                                    weights[:, 0],
                                ),
                                torch_noncontact_sample_weights(
                                    volume[:, n_hand_volume:],
                                    contact_anchors[person][:, 1],
                                    weights[:, 1],
                                ),
                            ],
                            dim=1,
                        )
                        if config.hull_min_local_x_m > 0:
                            left_distal = (
                                left_hand_volume_offsets[:, 0] >= config.hull_min_local_x_m
                            ).to(dtype=keep.dtype)
                            right_distal = (
                                right_hand_volume_offsets[:, 0] >= config.hull_min_local_x_m
                            ).to(dtype=keep.dtype)
                            keep = keep * torch.cat(
                                [
                                    left_distal.expand(keep.shape[0], -1),
                                    right_distal.expand(keep.shape[0], -1),
                                ],
                                dim=1,
                            )
                        hull_depth = hull_depth * keep
                        if config.hull_mask_aabb and not config.hull_orient_only:
                            depth = depth * keep
                    hull_loss = hull_loss + torch_max_inside_loss(
                        hull_depth, slack_m=config.hull_slack_m
                    )
                hand_volume_loss = (
                    hand_volume_loss
                    + torch.mean(depth**2)
                    + torch.mean(depth.max(dim=-1).values ** 2)
                )

            if torso_volume_enabled:
                local_torso = _transform_offsets(
                    transforms["torso_link"].get_matrix(),
                    torso_volume_offsets,
                )
                torso_pts = _apply_root_transform(
                    local_torso,
                    root_translation[person],
                    root_yaw[person],
                )
                torso_core4d = torso_pts @ core4d_from_robot
                torso_local = torch.einsum(
                    "tki,tij->tkj",
                    torso_core4d - object_translations[:, None],
                    object_rotations,
                )
                torso_depth = torch_aabb_inside_depth(
                    torso_local,
                    object_center_t,
                    object_half_t,
                    config.torso_volume_clearance_margin_m,
                )
                torso_volume_loss = (
                    torso_volume_loss
                    + torch.mean(torso_depth**2)
                    + torch.mean(torso_depth.max(dim=-1).values ** 2)
                )

            if leg_volume_enabled:
                local_legs = torch.cat(
                    [
                        _transform_offsets(transforms[name].get_matrix(), offsets)
                        for name, offsets in leg_volume_parts
                    ],
                    dim=1,
                )
                leg_pts = _apply_root_transform(
                    local_legs,
                    root_translation[person],
                    root_yaw[person],
                )
                leg_local = torch.einsum(
                    "tki,tij->tkj",
                    leg_pts @ core4d_from_robot - object_translations[:, None],
                    object_rotations,
                )
                if config.leg_volume_clearance_weight > 0:
                    leg_depth = torch_aabb_inside_depth(
                        leg_local,
                        object_center_t,
                        object_half_t,
                        config.leg_volume_clearance_margin_m,
                    )
                    leg_volume_loss = (
                        leg_volume_loss
                        + torch.mean(leg_depth**2)
                        + torch.mean(leg_depth.max(dim=-1).values ** 2)
                    )
                if config.leg_hull_weight > 0:
                    leg_hull_loss = leg_hull_loss + torch_max_inside_loss(
                        torch_convex_inside_depth(leg_local, hull_equations_t)
                    )

            if arm_hull_enabled:
                local_arms = torch.cat(
                    [
                        _transform_offsets(transforms[name].get_matrix(), offsets)
                        for name, offsets in arm_hull_parts
                    ],
                    dim=1,
                )
                arm_pts = _apply_root_transform(
                    local_arms,
                    root_translation[person],
                    root_yaw[person],
                )
                arm_local = torch.einsum(
                    "tki,tij->tkj",
                    arm_pts @ core4d_from_robot - object_translations[:, None],
                    object_rotations,
                )
                arm_hull_loss = arm_hull_loss + torch_max_inside_loss(
                    torch_convex_inside_depth(
                        arm_local,
                        hull_equations_t,
                        margin_m=config.hull_clearance_margin_m,
                    ),
                    slack_m=0.008,
                )

            foot_pts = None
            if foot_samples_enabled:
                local_feet = torch.cat(
                    [
                        _transform_offsets(
                            transforms["left_ankle_roll_link"].get_matrix(),
                            left_foot_volume_offsets,
                        ),
                        _transform_offsets(
                            transforms["right_ankle_roll_link"].get_matrix(),
                            right_foot_volume_offsets,
                        ),
                    ],
                    dim=1,
                )
                foot_pts = _apply_root_transform(
                    local_feet,
                    root_translation[person],
                    root_yaw[person],
                )
            foot_local = None
            if foot_pts is not None and (
                foot_volume_enabled or hull_enabled or hard_plant_enabled
            ):
                foot_core4d = foot_pts @ core4d_from_robot
                foot_local = torch.einsum(
                    "tki,tij->tkj",
                    foot_core4d - object_translations[:, None],
                    object_rotations,
                )
            if foot_volume_enabled and foot_local is not None:
                foot_depth = torch_aabb_inside_depth(
                    foot_local,
                    object_center_t,
                    object_half_t,
                    config.foot_volume_clearance_margin_m,
                )
                foot_volume_loss = (
                    foot_volume_loss
                    + torch.mean(foot_depth**2)
                    + torch.mean(foot_depth.max(dim=-1).values ** 2)
                )
            foot_hull_depth = None
            if (hull_enabled or hard_plant_enabled) and foot_local is not None:
                foot_hull_depth = torch_convex_inside_depth(foot_local, hull_equations_t)
                if hull_enabled and config.hull_include_feet:
                    if config.hull_orient_only:
                        local_feet_orient = torch.cat(
                            [
                                _transform_offsets_orient_only(
                                    transforms["left_ankle_roll_link"].get_matrix(),
                                    left_foot_volume_offsets,
                                ),
                                _transform_offsets_orient_only(
                                    transforms["right_ankle_roll_link"].get_matrix(),
                                    right_foot_volume_offsets,
                                ),
                            ],
                            dim=1,
                        )
                        foot_orient = _apply_root_transform(
                            local_feet_orient,
                            root_translation[person].detach(),
                            root_yaw[person].detach(),
                        )
                        foot_orient_local = torch.einsum(
                            "tki,tij->tkj",
                            foot_orient @ core4d_from_robot
                            - object_translations[:, None],
                            object_rotations,
                        )
                        hull_loss = hull_loss + torch_max_inside_loss(
                            torch_convex_inside_depth(
                                foot_orient_local, hull_equations_t
                            ),
                            slack_m=config.hull_slack_m,
                        )
                    else:
                        hull_loss = hull_loss + torch_max_inside_loss(
                            foot_hull_depth, slack_m=config.hull_slack_m
                        )
            plant_mask = None
            left_zmin = right_zmin = None
            if foot_pts is not None:
                n_foot = left_foot_volume_offsets.shape[0]
                left_zmin = foot_pts[:, :n_foot, 2].min(dim=-1).values
                right_zmin = foot_pts[:, n_foot:, 2].min(dim=-1).values
                plant_mask = source_plant[person]
                if foot_hull_depth is not None:
                    inside = torch.stack(
                        [
                            foot_hull_depth[:, :n_foot].max(dim=-1).values > 0.005,
                            foot_hull_depth[:, n_foot:].max(dim=-1).values > 0.005,
                        ],
                        dim=-1,
                    )
                    plant_mask = plant_mask & ~inside
                if config.putdown_dual_plant:
                    plant_mask = plant_mask | putdown[:, None]
                if config.plant_exclusive and left_zmin is not None:
                    plant_mask = torch_exclusive_stance(plant_mask, left_zmin, right_zmin)
                if swing_enabled:
                    moving = swing_mask[person]
                    if config.swing_lift_min_root_m_s > 0:
                        moving = moving & (
                            source_pelvis_speed[person][:, None]
                            > config.swing_lift_min_root_m_s
                        )
                    swing_loss = swing_loss + torch_forced_swing_loss(
                        left_zmin, moving[:, 0], lift_m=config.swing_height_m
                    )
                    swing_loss = swing_loss + torch_forced_swing_loss(
                        right_zmin, moving[:, 1], lift_m=config.swing_height_m
                    )
            if foot_ground_enabled and foot_pts is not None:
                if config.foot_ground_weight > 0:
                    foot_ground_loss = foot_ground_loss + torch_below_ground_loss(foot_pts[..., 2])
                if config.foot_slide_weight > 0:
                    foot_slide_loss = foot_slide_loss + torch_near_ground_slide_loss(
                        foot_pts,
                        frame_weight=slip_boost,
                        hard=config.foot_slide_hard,
                        stop_grad_prev=config.plant_stop_grad,
                        plant_per_foot=plant_mask,
                        n_left=left_foot_volume_offsets.shape[0],
                    )
            if hard_plant_enabled and foot_pts is not None and plant_mask is not None:
                n_foot = left_foot_volume_offsets.shape[0]
                centroids = torch.stack(
                    [
                        foot_pts[:, :n_foot, :2].mean(dim=1),
                        foot_pts[:, n_foot:, :2].mean(dim=1),
                    ],
                    dim=1,
                )
                foot_slip_loss = foot_slip_loss + torch_foot_slip_loss(
                    centroids,
                    plant_mask,
                    slip_boost,
                    stop_grad_prev=config.plant_stop_grad,
                )
                if config.stance_lock_weight > 0 and person in stance_lock_xy:
                    delta = centroids - stance_lock_xy[person]
                    active = plant_mask.double()
                    stance_lock_loss = stance_lock_loss + torch.sum(
                        (delta**2).sum(dim=-1) * active
                    ) / torch.clamp(active.sum(), min=1.0)
                if config.stance_width_weight > 0:
                    stance_width_loss = stance_width_loss + torch_stance_width_loss(
                        centroids[:, 0],
                        centroids[:, 1],
                        putdown,
                        config.squat_stance_width_m,
                    )

            if config.squat_knee_weight > 0 and len(knee_indices) == 2:
                stance = plant_mask if plant_mask is not None else source_plant[person]
                squat_loss = squat_loss + torch_knee_squat_loss(
                    joints[:, list(knee_indices)],
                    stance,
                    config.squat_knee_max_rad,
                )
            if config.swing_knee_weight > 0 and len(knee_indices) == 2:
                swing_knee_loss = swing_knee_loss + torch_swing_knee_bend_loss(
                    joints[:, list(knee_indices)],
                    swing_mask[person] & step_walk[:, None],
                    config.swing_knee_min_rad,
                )
            if config.gait_knee_fidelity_weight > 0 and len(knee_indices) == 2:
                gait_knee_loss = gait_knee_loss + torch_swing_knee_target_loss(
                    joints[:, list(knee_indices)],
                    swing_mask[person] & step_walk[:, None],
                    gait_knee_targets[person],
                )

            if config.knee_symmetry_weight > 0 and len(knee_indices) == 2:
                knee_sym_loss = knee_sym_loss + torch_knee_symmetry_loss(
                    joints[:, list(knee_indices)],
                    putdown,
                )
            if config.squat_knee_min_weight > 0 and len(knee_indices) == 2:
                knee_min_loss = knee_min_loss + torch_knee_min_bend_loss(
                    joints[:, list(knee_indices)],
                    putdown,
                    config.squat_knee_min_rad,
                )
            if config.both_soles_weight > 0 and left_zmin is not None:
                both_soles_loss = both_soles_loss + torch_both_soles_down_loss(
                    left_zmin, right_zmin, putdown
                )
            if config.approach_ground_weight > 0 and left_zmin is not None:
                approach_ground_loss = approach_ground_loss + torch_one_foot_down_loss(
                    left_zmin, right_zmin, approach
                )
            if config.putdown_root_lock_weight > 0 and person in putdown_root_xy:
                delta = root_translation[person][:, :2] - putdown_root_xy[person]
                active = putdown.double()
                putdown_root_loss = putdown_root_loss + torch.sum(
                    (delta**2).sum(dim=-1) * active
                ) / torch.clamp(active.sum(), min=1.0)

            if config.leg_laterality_weight > 0:
                laterality_loss = laterality_loss + torch_pair_laterality_loss(
                    keypoints[:, ankle_indices[0]],
                    keypoints[:, ankle_indices[1]],
                    root_yaw[person],
                    config.leg_laterality_ankle_m,
                )
                laterality_loss = laterality_loss + torch_pair_laterality_loss(
                    keypoints[:, knee_kp[0]],
                    keypoints[:, knee_kp[1]],
                    root_yaw[person],
                    config.leg_laterality_knee_m,
                )
            if config.self_leg_clearance_weight > 0:
                left_chain = torch.cat(
                    [
                        torch_segment_samples(
                            keypoints[:, hip_kp[0]], keypoints[:, knee_kp[0]]
                        ),
                        torch_segment_samples(
                            keypoints[:, knee_kp[0]],
                            keypoints[:, ankle_indices[0]],
                        ),
                    ],
                    dim=1,
                )
                right_chain = torch.cat(
                    [
                        torch_segment_samples(
                            keypoints[:, hip_kp[1]], keypoints[:, knee_kp[1]]
                        ),
                        torch_segment_samples(
                            keypoints[:, knee_kp[1]],
                            keypoints[:, ankle_indices[1]],
                        ),
                    ],
                    dim=1,
                )
                self_leg_loss = self_leg_loss + torch_limb_pair_clearance_loss(
                    left_chain,
                    right_chain,
                    config.self_leg_clearance_m,
                )
            if config.hip_yaw_limit_weight > 0 and len(hip_yaw_indices) == 2:
                hip_yaw_loss = hip_yaw_loss + torch_abs_limit_loss(
                    joints[:, list(hip_yaw_indices)],
                    config.hip_yaw_max_rad,
                )
            if config.hip_roll_limit_weight > 0 and len(hip_roll_indices) == 2:
                hip_roll_loss = hip_roll_loss + torch_abs_limit_loss(
                    joints[:, list(hip_roll_indices)],
                    config.hip_roll_max_rad,
                )
            if config.swing_outward_weight > 0:
                pelvis_xy = keypoints[:, keypoint_names.index("pelvis")]
                left_y = torch_body_lateral_y(
                    keypoints[:, ankle_indices[0]],
                    pelvis_xy,
                    root_yaw[person],
                )
                right_y = torch_body_lateral_y(
                    keypoints[:, ankle_indices[1]],
                    pelvis_xy,
                    root_yaw[person],
                )
                swing = swing_mask[person] & carry_walk[:, None]
                outward_loss = outward_loss + torch_swing_outward_loss(
                    left_y,
                    source_swing_y[person][:, 0],
                    swing[:, 0],
                    slack_m=config.swing_outward_slack_m,
                    left=True,
                )
                outward_loss = outward_loss + torch_swing_outward_loss(
                    right_y,
                    source_swing_y[person][:, 1],
                    swing[:, 1],
                    slack_m=config.swing_outward_slack_m,
                    left=False,
                )

            if config.root_speed_cap_weight > 0:
                speed_cap_loss = speed_cap_loss + config.root_speed_cap_weight * torch_xy_speed_cap_loss(
                    root_translation[person][:, :2],
                    source_root_speed[person],
                )
            if config.foot_speed_cap_weight > 0:
                speed_cap_loss = speed_cap_loss + config.foot_speed_cap_weight * torch_xy_speed_cap_loss(
                    keypoints[:, ankle_indices[0], :2],
                    source_foot_speed_cap[person],
                )
                speed_cap_loss = speed_cap_loss + config.foot_speed_cap_weight * torch_xy_speed_cap_loss(
                    keypoints[:, ankle_indices[1], :2],
                    source_foot_speed_cap[person],
                )

            if config.source_approach_root_weight > 0:
                delta = root_translation[person][:, :2] - source_pelvis_xy[person]
                active = approach.double()
                approach_root_loss = approach_root_loss + torch.sum(
                    (delta**2).sum(dim=-1) * active
                ) / torch.clamp(active.sum(), min=1.0)

            if balance_enabled:
                feet = keypoints[:, ankle_indices]
                contact = source_foot_contact[person]
                if plant_mask is not None:
                    contact = plant_mask
                com_xy = torch.einsum("tkd,k->td", keypoints, com_masses)[:, :2]
                support_loss = support_loss + torch.mean(
                    torch_support_excess(
                        com_xy,
                        feet[:, 0, :2],
                        feet[:, 1, :2],
                        contact[:, 0],
                        contact[:, 1],
                        root_yaw[person],
                    )
                    ** 2
                )
                if hard_plant_enabled and plant_mask is not None and left_zmin is not None:
                    sole_z = torch.stack([left_zmin, right_zmin], dim=-1)
                    foot_height_loss = foot_height_loss + torch_foot_height_loss(
                        sole_z,
                        plant_mask,
                        0.0,
                    )
                else:
                    foot_height_loss = foot_height_loss + torch_foot_height_loss(
                        feet[:, :, 2],
                        contact,
                        config.target_foot_height,
                    )
                if not hard_plant_enabled:
                    foot_slip_loss = foot_slip_loss + torch_foot_slip_loss(
                        feet[:, :, :2],
                        contact,
                        slip_boost,
                        stop_grad_prev=config.plant_stop_grad,
                    )
                center_loss = center_loss + torch_com_center_loss(
                    com_xy,
                    feet[:, 0, :2],
                    feet[:, 1, :2],
                    contact[:, 0],
                    contact[:, 1],
                    root_yaw[person],
                )
                if planted_root_enabled:
                    if plant_mask is not None:
                        both = plant_mask[:, 0] & plant_mask[:, 1]
                    else:
                        both = contact[:, 0] & contact[:, 1]
                    planted_root_loss = planted_root_loss + torch_planted_root_slip(
                        root_translation[person][:, :2],
                        both,
                        slip_boost,
                        stop_grad_prev=config.plant_stop_grad,
                    )
                torso_z = transforms["torso_link"].get_matrix()[:, :3, 2]
                yaw = root_yaw[person]
                world_up = torch.stack(
                    [
                        torch.cos(yaw) * torso_z[:, 0] - torch.sin(yaw) * torso_z[:, 1],
                        torch.sin(yaw) * torso_z[:, 0] + torch.cos(yaw) * torso_z[:, 1],
                        torso_z[:, 2],
                    ],
                    dim=-1,
                )
                torso_loss = torso_loss + torch.mean((world_up[:, 2] - 1.0) ** 2)

        if cross_agent_enabled:
            inter_distances = torch.cdist(current[1][1], current[2][1])
            inter_penalty = torch.relu(config.inter_robot_clearance - inter_distances) ** 2
            inter_loss = torch.mean(inter_penalty)
            inter_active = (inter_penalty > 0).sum()
        else:
            inter_loss = torch.zeros((), dtype=torch.float64)
            inter_active = torch.zeros((), dtype=torch.int64)
        # RESEARCH EXTENSION: joint-contact continuity (weight 0 in frozen v41).
        joint_contact_loss = torch.zeros((), dtype=torch.float64)
        joint_contact_active = torch.zeros((), dtype=torch.int64)
        if config.joint_contact_continuity_weight > 0 and joint_frames.any():
            indicators = {}
            for person in persons:
                if person not in hand_volume_local:
                    raise RuntimeError(
                        "joint_contact_continuity requires hand_volume_clearance_weight > 0"
                    )
                if config.joint_contact_surface_aware:
                    indicators[person] = torch_soft_contact_indicator_surface_aware(
                        hand_volume_local[person],
                        hull_equations_t,
                        tau_m=config.joint_contact_tau_m,
                        beta_m=config.joint_contact_beta_m,
                        softmin_beta_m=config.joint_contact_softmin_beta_m or None,
                        eps_m=config.joint_contact_surface_eps_m,
                    )
                else:
                    indicators[person] = torch_soft_contact_indicator(
                        hand_volume_local[person],
                        hull_equations_t,
                        tau_m=config.joint_contact_tau_m,
                        beta_m=config.joint_contact_beta_m,
                    )
            if config.joint_contact_separable_surrogate or not cross_agent_enabled:
                product = sum(indicators[p] for p in persons) / len(persons)
            else:
                product = indicators[1] * indicators[2]
            active = joint_activation
            joint_contact_loss = torch.sum((1.0 - product) * active) / active.sum()
            joint_contact_active = (active > 0).sum()
        loss = (
            config.pose_fidelity_weight * pose_loss
            + config.contact_weight * contact_loss
            + config.root_fidelity_weight * root_loss
            + config.velocity_weight * velocity_loss
            + config.acceleration_weight * acceleration_loss
            + config.jerk_weight * jerk_loss
            + config.inter_robot_weight * inter_loss
            + config.com_support_weight * support_loss
            + config.foot_height_weight * foot_height_loss
            + (config.foot_slip_weight + config.hard_plant_weight) * foot_slip_loss
            + config.torso_upright_weight * torso_loss
            + config.com_center_weight * center_loss
            + config.object_clearance_weight * clearance_loss
            + config.hand_volume_clearance_weight * hand_volume_loss
            + config.torso_volume_clearance_weight * torso_volume_loss
            + config.foot_volume_clearance_weight * foot_volume_loss
            + config.object_hull_clearance_weight * hull_loss
            + config.leg_volume_clearance_weight * leg_volume_loss
            + config.leg_hull_weight * leg_hull_loss
            + config.arm_hull_weight * arm_hull_loss
            + config.foot_ground_weight * foot_ground_loss
            + config.foot_slide_weight * foot_slide_loss
            + config.planted_root_slip_weight * planted_root_loss
            + config.swing_lift_weight * swing_loss
            + config.swing_knee_weight * swing_knee_loss
            + config.gait_knee_fidelity_weight * gait_knee_loss
            + config.squat_knee_weight * squat_loss
            + config.squat_knee_min_weight * knee_min_loss
            + config.palm_facing_weight * palm_facing_loss
            + config.finger_up_weight * finger_up_loss
            + config.stance_lock_weight * stance_lock_loss
            + config.knee_symmetry_weight * knee_sym_loss
            + config.both_soles_weight * both_soles_loss
            + config.stance_width_weight * stance_width_loss
            + config.putdown_root_lock_weight * putdown_root_loss
            + config.leg_laterality_weight * laterality_loss
            + config.self_leg_clearance_weight * self_leg_loss
            + config.hip_yaw_limit_weight * hip_yaw_loss
            + config.hip_roll_limit_weight * hip_roll_loss
            + config.swing_outward_weight * outward_loss
            + speed_cap_loss
            + config.approach_ground_weight * approach_ground_loss
            + config.source_approach_root_weight * approach_root_loss
            + config.joint_contact_continuity_weight * joint_contact_loss
            + config.hand_cartesian_jerk_weight * hand_jerk_loss
            + config.reference_trust_weight * trust_loss
            + config.smoothed_hand_barrier_weight * smoothed_barrier_loss
        )
        terms = {
            "pose": config.pose_fidelity_weight * pose_loss,
            "contact": config.contact_weight * contact_loss,
            "root": config.root_fidelity_weight * root_loss,
            "velocity": config.velocity_weight * velocity_loss,
            "acceleration": config.acceleration_weight * acceleration_loss,
            "jerk": config.jerk_weight * jerk_loss,
            "inter_robot": config.inter_robot_weight * inter_loss,
            "support": config.com_support_weight * support_loss,
            "foot_height": config.foot_height_weight * foot_height_loss,
            "foot_slip": (config.foot_slip_weight + config.hard_plant_weight) * foot_slip_loss,
            "torso": config.torso_upright_weight * torso_loss,
            "center": config.com_center_weight * center_loss,
            "clearance": config.object_clearance_weight * clearance_loss,
            "hand_volume": config.hand_volume_clearance_weight * hand_volume_loss,
            "torso_volume": config.torso_volume_clearance_weight * torso_volume_loss,
            "foot_volume": config.foot_volume_clearance_weight * foot_volume_loss,
            "hull": config.object_hull_clearance_weight * hull_loss,
            "leg_volume": config.leg_volume_clearance_weight * leg_volume_loss,
            "leg_hull": config.leg_hull_weight * leg_hull_loss,
            "arm_hull": config.arm_hull_weight * arm_hull_loss,
            "foot_ground": config.foot_ground_weight * foot_ground_loss,
            "foot_slide": config.foot_slide_weight * foot_slide_loss,
            "planted_root": config.planted_root_slip_weight * planted_root_loss,
            "swing": config.swing_lift_weight * swing_loss,
            "swing_knee": config.swing_knee_weight * swing_knee_loss,
            "gait_knee": config.gait_knee_fidelity_weight * gait_knee_loss,
            "squat": config.squat_knee_weight * squat_loss,
            "knee_min": config.squat_knee_min_weight * knee_min_loss,
            "palm_facing": config.palm_facing_weight * palm_facing_loss,
            "finger_up": config.finger_up_weight * finger_up_loss,
            "stance_lock": config.stance_lock_weight * stance_lock_loss,
            "knee_sym": config.knee_symmetry_weight * knee_sym_loss,
            "both_soles": config.both_soles_weight * both_soles_loss,
            "stance_width": config.stance_width_weight * stance_width_loss,
            "putdown_root": config.putdown_root_lock_weight * putdown_root_loss,
            "laterality": config.leg_laterality_weight * laterality_loss,
            "self_leg": config.self_leg_clearance_weight * self_leg_loss,
            "hip_yaw": config.hip_yaw_limit_weight * hip_yaw_loss,
            "hip_roll": config.hip_roll_limit_weight * hip_roll_loss,
            "outward": config.swing_outward_weight * outward_loss,
            "speed_cap": speed_cap_loss,
            "approach_ground": config.approach_ground_weight * approach_ground_loss,
            "approach_root": config.source_approach_root_weight * approach_root_loss,
            "joint_contact": config.joint_contact_continuity_weight * joint_contact_loss,
            "hand_jerk": config.hand_cartesian_jerk_weight * hand_jerk_loss,
            "reference_trust": config.reference_trust_weight * trust_loss,
            "smoothed_barrier": config.smoothed_hand_barrier_weight * smoothed_barrier_loss,
            # unweighted cross-agent diagnostics
            "inter_robot_raw": inter_loss,
            "inter_robot_active_pairs": inter_active,
            "joint_contact_raw": joint_contact_loss,
            "joint_contact_active_frames": joint_contact_active,
        }
        return loss, current, terms

    if probe_callback is not None:
        probe_callback(evaluate_objective, raw_joints, root_translation, root_yaw)
        if config.iterations == 0:
            return {}, np.zeros(0)

    cross_terms = ("inter_robot_raw", "joint_contact_raw")
    history: list[float] = []
    best: tuple[float, dict[int, tuple[torch.Tensor, ...]]] | None = None
    for iteration in range(config.iterations):
        loss, current, terms = evaluate_objective()
        if term_log is not None:
            record: dict[str, float] = {"iteration": float(iteration), "loss": float(loss.detach())}
            for name, tensor in terms.items():
                record[name] = float(tensor.detach())
            if term_log_grad_every > 0 and iteration % term_log_grad_every == 0:
                for name in cross_terms:
                    tensor = terms[name]
                    if tensor.requires_grad:
                        grads = torch.autograd.grad(
                            tensor, parameters, retain_graph=True, allow_unused=True
                        )
                        norm_sq = sum(
                            float((g.detach() ** 2).sum()) for g in grads if g is not None
                        )
                        record[f"{name}_grad_norm"] = float(np.sqrt(norm_sq))
                    else:
                        record[f"{name}_grad_norm"] = 0.0
            term_log.append(record)
        optimizer.zero_grad()
        loss.backward()
        if config.per_robot_grad_clip:
            for person in persons:
                torch.nn.utils.clip_grad_norm_(
                    [raw_joints[person], root_translation[person], root_yaw[person]], 5.0
                )
        else:
            torch.nn.utils.clip_grad_norm_(parameters, 5.0)
        value = float(loss.detach())
        history.append(value)
        if best is None or value < best[0]:
            best = (
                value,
                {
                    person: (
                        current[person][0].detach().clone(),
                        root_translation[person].detach().clone(),
                        root_yaw[person].detach().clone(),
                        current[person][1].detach().clone(),
                    )
                    for person in persons
                },
            )
        optimizer.step()

    if best is None:
        raise RuntimeError("Coupled optimizer produced no result")
    outputs: dict[int, dict[str, np.ndarray]] = {}
    for person in persons:
        joints, root, yaw, keypoints = best[1][person]
        outputs[person] = {
            "joint_names": np.asarray(joint_names),
            "keypoint_names": np.asarray(keypoint_names),
            "joint_positions": joints.cpu().numpy(),
            "root_translation": root.cpu().numpy(),
            "root_yaw": yaw.cpu().numpy(),
            "robot_keypoints": keypoints.cpu().numpy(),
            "target_keypoints": baselines[person]["target_keypoints"],
            "contact_anchors": anchors_and_masks[person][0],
            "contact_mask": anchors_and_masks[person][1],
            "config_json": np.asarray(json.dumps(asdict(config), sort_keys=True)),
        }
    return outputs, np.asarray(history)


def save_coupled_results(
    outputs: dict[int, dict[str, np.ndarray]],
    history: np.ndarray,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for person, arrays in outputs.items():
        np.savez_compressed(output_dir / f"robot{person}.npz", **arrays, loss_history=history)

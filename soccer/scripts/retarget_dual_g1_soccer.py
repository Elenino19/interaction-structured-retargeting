#!/usr/bin/env python3
"""G1 retarget that copies kick-foot pose: ankle roll link + toe, planted support."""

from __future__ import annotations

import sys as _fg_sys
from pathlib import Path as _FgPath
_fg_sys.path.insert(0, str(_FgPath(__file__).resolve().parent))
from frozen_guard import guard_release_writes


import json
import shutil
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pytorch_kinematics as pk
import torch

from isr_assets import ensure_join_on_path, g1_urdf, g1_xml
ensure_join_on_path()

from core4d_dual_retarget.coordinates import points_core4d_to_robot
from core4d_dual_retarget.retarget.independent import (
    IndependentRetargetConfig,
    IndependentRetargetResult,
    _apply_root_transform,
    _yaw_from_shoulders,
)
from core4d_dual_retarget.robot_model import load_actuated_joint_limits

# ankle_pitch_link origin does not move with ankle joints. Use the roll link
# and a toe offset so pitch/roll actually copy the kick.
SOCCER_LINKS = {
    "pelvis": "pelvis",
    "left_hip": "left_hip_pitch_link",
    "right_hip": "right_hip_pitch_link",
    "left_knee": "left_knee_link",
    "right_knee": "right_knee_link",
    "left_ankle": "left_ankle_roll_link",
    "right_ankle": "right_ankle_roll_link",
    "left_toe": "left_ankle_roll_link",
    "right_toe": "right_ankle_roll_link",
    "neck": "torso_link",
    "left_shoulder": "left_shoulder_pitch_link",
    "right_shoulder": "right_shoulder_pitch_link",
    "left_elbow": "left_elbow_link",
    "right_elbow": "right_elbow_link",
    "left_wrist": "left_rubber_hand",
    "right_wrist": "right_rubber_hand",
}
SMPLX_IDX = {
    "pelvis": 0,
    "left_hip": 1,
    "right_hip": 2,
    "left_knee": 4,
    "right_knee": 5,
    "left_ankle": 7,
    "right_ankle": 8,
    "left_toe": 10,
    "right_toe": 11,
    "neck": 12,
    "left_shoulder": 16,
    "right_shoulder": 17,
    "left_elbow": 18,
    "right_elbow": 19,
    "left_wrist": 20,
    "right_wrist": 21,
}
TOE_OFFSET_LOCAL = torch.tensor([0.12, 0.0, -0.03], dtype=torch.float64)
BASE_WEIGHTS = {
    "pelvis": 6.0,
    "left_hip": 5.0,
    "right_hip": 5.0,
    "left_knee": 8.0,
    "right_knee": 8.0,
    "left_ankle": 8.0,
    "right_ankle": 8.0,
    "left_toe": 9.0,
    "right_toe": 9.0,
    "neck": 0.4,
    "left_shoulder": 2.0,
    "right_shoulder": 2.0,
    "left_elbow": 2.2,
    "right_elbow": 2.2,
    "left_wrist": 1.2,
    "right_wrist": 1.2,
}
KEYPOINT_NAMES = tuple(SOCCER_LINKS)
SOLE_Z = 0.035
CONE_COUNT = 5
CONE_SPACING_M = 0.50


@dataclass(frozen=True)
class SoccerRetargetWeights:
    """Terms layered on top of the shared keypoint/root IK objective.

    The defaults reproduce the accepted full model. Ablations only zero or
    disable terms declared here; shared pose, root, yaw, smoothness, pelvis,
    and upper-body terms remain identical across variants.
    """

    support_plant: float = 6.0
    support_floor: float = 6.0
    kick_velocity: float = 24.0
    event_keypoint_boost: bool = True
    event_smooth_relaxation: bool = True


def generic_support_windows(source_joints: np.ndarray, fps: float = 24.0) -> np.ndarray:
    """Estimate soft left/right support from source foot height and speed.

    This deliberately uses no ball event or kick-foot label, so the support
    ablation can be separated from foot-role event semantics.
    """
    feet = np.stack(
        [
            0.5
            * (
                source_joints[:, SMPLX_IDX["left_ankle"]]
                + source_joints[:, SMPLX_IDX["left_toe"]]
            ),
            0.5
            * (
                source_joints[:, SMPLX_IDX["right_ankle"]]
                + source_joints[:, SMPLX_IDX["right_toe"]]
            ),
        ],
        axis=1,
    )
    height = feet[..., 1]
    ground = np.percentile(height, 8, axis=0, keepdims=True)
    height_score = np.clip(1.0 - (height - ground) / 0.10, 0.0, 1.0)
    horizontal = feet[..., [0, 2]]
    speed = np.zeros_like(height)
    speed[1:] = np.linalg.norm(np.diff(horizontal, axis=0), axis=-1) * fps
    speed[0] = speed[1]
    speed_score = np.exp(-((speed / 0.45) ** 2))
    support = height_score * speed_score
    support[support < 0.10] = 0.0
    return support
def near_return_lanes(n_passes: int) -> tuple[int, ...]:
    """Near-camera ladder, lane 1 = farthest interior gap, lane 5 = outside the nearest cone.

    Order from the 3:23-3:29 clip: gap between near cones 2 and 3, gap between
    near cones 1 and 2, outside the nearest cone, back through near cones 1-2,
    then the next adjacent gap toward the far end on each later pass.
    """
    head = (3, 4, 5, 4)
    lanes = list(head[:n_passes])
    nxt = 3
    while len(lanes) < n_passes:
        lanes.append(nxt)
        nxt = max(nxt - 1, 1)
    return tuple(lanes)


def ladder_geometry(n_cones: int = CONE_COUNT, spacing: float = CONE_SPACING_M):
    half = 0.5 * spacing * (n_cones - 1)
    cone_y = np.linspace(half, -half, n_cones)
    gap_lanes = np.concatenate(
        [
            [cone_y[0] + 0.5 * spacing],
            0.5 * (cone_y[:-1] + cone_y[1:]),
            [cone_y[-1] - 0.5 * spacing],
        ]
    )
    return cone_y, gap_lanes


def _smoothstep(u: np.ndarray | float) -> np.ndarray | float:
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def _contact_stations(contacts: list[dict], gap_lanes: np.ndarray) -> list[dict]:
    """Each touch meets the ball on the gap that pass just crossed.

    The player is already waiting on that line. The next pass is the redirect.
    """
    lanes = near_return_lanes(max(len(contacts) - 1, 1))
    events = []
    for index, event in enumerate(contacts):
        receive = lanes[0] if index == 0 else lanes[index - 1]
        send = lanes[min(index, len(lanes) - 1)]
        events.append(
            {
                "frame": int(event["frame"]),
                "player": event["player"],
                "side": event.get("kick_side", event["side"]),
                "y": float(gap_lanes[send]),
                "send_y": float(gap_lanes[send]),
            }
        )
    return events


def _settle_count(total: int, prev: float, gap: float, has_prev: bool) -> int:
    """Frames spent taking the new line at the kicker before the straight roll.

    The step is long enough that the body stays near a walking speed. The roll
    through the cones is the rest of the pass and does not change Y.
    """
    if not has_prev or abs(gap - prev) < 0.08 or total <= 1:
        return 0
    need = int(np.ceil(1.5 * abs(gap - prev) * 24.0 / 2.0))
    return int(min(max(need, 1), max(total - 8, 1)))


def _pass_lines(n: int, events: list[dict], gap_lanes: np.ndarray) -> np.ndarray:
    """Robot-Y shared by both players and the ball.

    Take the new line beside the kicking foot, then hold it straight through the
    cones so the ball reaches the receiving foot on that same line.
    """
    lanes = near_return_lanes(max(len(events) - 1, 1))
    frames = [int(event["frame"]) for event in events]
    desired = np.full(n, float(gap_lanes[lanes[0]]), dtype=np.float64)
    for index, lane in enumerate(lanes):
        gap = float(gap_lanes[lane])
        start = frames[index]
        end = frames[index + 1] if index + 1 < len(frames) else n - 1
        prev = float(gap_lanes[lanes[index - 1]]) if index else gap
        total = end - start + 1
        step = _settle_count(total, prev, gap, index > 0)
        if step:
            weights = np.asarray(
                _smoothstep(np.linspace(0.0, 1.0, step + 1)[1:]), dtype=np.float64
            )
            desired[start : start + step] = prev + weights * (gap - prev)
            desired[start + step : end + 1] = gap
        else:
            desired[start : end + 1] = gap
    if frames:
        desired[frames[-1] :] = desired[min(frames[-1], n - 1)]
    return desired


def _ready_shift(n: int, player: str, events: list[dict], joints: np.ndarray, gap_lanes: np.ndarray) -> np.ndarray:
    """Keep the video hop. Put the striking toe, not the stance midpoint, on the ball line.

    Returns the robot-Y offset added to every joint. Relative footwork stays.
    """
    desired = _pass_lines(n, events, gap_lanes)
    mid_z = 0.5 * (
        joints[:n, SMPLX_IDX["left_ankle"], 2] + joints[:n, SMPLX_IDX["right_ankle"], 2]
    )
    # Constant stance bias so the toe sits on the line at the touch.
    # Tracking the swinging toe each frame would drag the whole body with the kick.
    frames = [int(event["frame"]) for event in events]
    extra = np.zeros(n, dtype=np.float64)
    for index, event in enumerate(events):
        if event["player"] != player:
            continue
        frame = int(event["frame"])
        side = event["side"]
        # Put this player's striking toe on the live pass line for the whole
        # approach, using that frame's own toe offset so the ball can meet it.
        arrive_from = frames[index - 1] if index else 0
        hold_from = max(0, arrive_from + 1)
        toe_series = joints[:n, SMPLX_IDX["left_toe" if side == "left" else "right_toe"], 2]
        for t in range(hold_from, min(n, frame + 1)):
            extra[t] = float(toe_series[t] - mid_z[t])
        lo, hi = max(0, hold_from - 6), min(n, frame + 8)
        for t in range(lo, hi):
            if hold_from <= t <= frame:
                continue
            edge = hold_from if t < hold_from else frame
            weight = float(_smoothstep(1.0 - abs(t - edge) / 6.0))
            sample = frame if t > frame else hold_from
            sample_bias = float(toe_series[sample] - mid_z[sample])
            extra[t] = (1.0 - weight) * extra[t] + weight * sample_bias
    return desired + mid_z + extra


def _apply_player_plan(
    joints: np.ndarray,
    player: str,
    events: list[dict],
    gap_lanes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep the video's hop and ready stance. Only ease the body onto the receiving line.

    The striking foot is on that line about 0.4 s before the ball arrives, so the
    player can bounce in place instead of sliding into the pass.
    """
    n = len(joints)
    out = joints.copy()
    shift = _ready_shift(n, player, events, joints, gap_lanes)
    out[:, :, 2] = joints[:, :, 2] - shift[:, None]
    plant = np.zeros((n, 2), dtype=np.float64)
    for event in events:
        if event["player"] != player:
            continue
        frame = int(event["frame"])
        side = 0 if event["side"] == "left" else 1
        ankle_i = SMPLX_IDX["left_ankle" if side == 0 else "right_ankle"]
        toe_i = SMPLX_IDX["left_toe" if side == 0 else "right_toe"]
        lo, hi = max(0, frame - 5), min(n, frame + 3)
        for t in range(lo, hi):
            weight = float(_smoothstep(1.0 - abs(t - frame) / 5.0))
            out[t, ankle_i, 1] = (1.0 - weight) * joints[t, ankle_i, 1] + weight * 0.05
            out[t, toe_i, 1] = (1.0 - weight) * joints[t, toe_i, 1] + weight * 0.02
        support = 1 - side
        plant[max(0, frame - 2) : min(n, frame + 2), support] = 1.0
    return out, plant



def station_pair_on_ladder(
    joints_a: np.ndarray,
    joints_b: np.ndarray,
    contacts: list[dict],
    gap_lanes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sidestep each player onto the gap they are about to play.

    A shared whole-body slide reads as skating. One foot stays planted while
    the other swings, and the strike itself uses the source kick placed on the ball.
    """
    n = min(len(joints_a), len(joints_b))
    events = _contact_stations(contacts, gap_lanes)
    aligned_a, plant_a = _apply_player_plan(joints_a[:n].copy(), "a", events, gap_lanes)
    aligned_b, plant_b = _apply_player_plan(joints_b[:n].copy(), "b", events, gap_lanes)
    return aligned_a, aligned_b, plant_a, plant_b


def prepare_soccer_targets(source_joints: np.ndarray, pelvis_height: float):
    """Keep the source bounce: do not flatten every frame onto the lower ankle."""
    joints = points_core4d_to_robot(source_joints)
    pelvis = joints[:, SMPLX_IDX["pelvis"]]
    ankles = (joints[:, SMPLX_IDX["left_ankle"]] + joints[:, SMPLX_IDX["right_ankle"]]) / 2
    human_leg = float(np.median(np.linalg.norm(pelvis - ankles, axis=-1)))
    scale = (pelvis_height - 0.03) / human_leg
    root = pelvis.copy()
    root[:, 2] = pelvis_height + scale * (pelvis[:, 2] - np.median(pelvis[:, 2]))
    indices = [SMPLX_IDX[name] for name in KEYPOINT_NAMES]
    local = (joints[:, indices] - pelvis[:, None]) * scale
    targets = root[:, None] + local
    ankle_i = (KEYPOINT_NAMES.index("left_ankle"), KEYPOINT_NAMES.index("right_ankle"))
    lower = np.minimum(targets[:, ankle_i[0], 2], targets[:, ankle_i[1], 2])
    # Land the typical stance, keep hops/ready bounces above the floor.
    targets[..., 2] += SOLE_Z - float(np.percentile(lower, 8))
    root[:, 2] = targets[:, KEYPOINT_NAMES.index("pelvis"), 2]
    for name in ("left_ankle", "right_ankle", "left_toe", "right_toe"):
        idx = KEYPOINT_NAMES.index(name)
        targets[:, idx, 2] = np.maximum(targets[:, idx, 2], 0.015)
    yaw = _yaw_from_shoulders(joints)
    return targets, root, yaw, scale


def local_keypoints(transforms, dtype) -> torch.Tensor:
    offset = TOE_OFFSET_LOCAL.to(dtype=dtype)
    parts = []
    for name in KEYPOINT_NAMES:
        matrix = transforms[SOCCER_LINKS[name]].get_matrix()
        origin = matrix[:, :3, 3]
        if name.endswith("_toe"):
            origin = origin + torch.einsum("nij,j->ni", matrix[:, :3, :3], offset)
        parts.append(origin)
    return torch.stack(parts, dim=1)


def contact_windows(
    contacts: list[dict],
    player: str,
    n: int,
    radius: int = 8,
    step_plant: np.ndarray | None = None,
):
    """Kick/receive foot is active on the approach; plant the support foot."""
    kick = np.zeros((n, 2), dtype=np.float64)
    plant = np.zeros((n, 2), dtype=np.float64) if step_plant is None else step_plant.copy()
    for event in contacts:
        if event["player"] != player:
            continue
        kick_side_name = event.get("kick_side", event["side"])
        kick_side = 0 if kick_side_name == "left" else 1
        plant_side = 1 - kick_side
        frame = int(event.get("kick_frame", event["frame"]))
        lo, hi = max(0, frame - radius), min(n, frame + 3)
        span = max(hi - lo, 1)
        ramp = np.linspace(0.55, 1.25, span)
        kick[lo:hi, kick_side] = np.maximum(kick[lo:hi, kick_side], ramp)
        kick[frame, kick_side] = 1.7
        plo, phi = max(0, frame - 3), min(n, frame + 2)
        plant[plo:phi, plant_side] = 1.0
    return kick, plant


def retarget_soccer(
    source_joints: np.ndarray,
    urdf_path: Path,
    config: IndependentRetargetConfig,
    kick: np.ndarray,
    plant: np.ndarray,
    loss_weights: SoccerRetargetWeights | None = None,
    precomputed_targets: tuple[np.ndarray, np.ndarray, np.ndarray, float] | None = None,
    ball_contact: dict | None = None,
) -> IndependentRetargetResult:
    """Optimise G1 joints/root toward soccer keypoint targets.

    ``precomputed_targets`` = (targets, base_root, initial_yaw, scale) bypasses
    ``prepare_soccer_targets`` so controlled experiments can share one frozen
    reference. The optimiser still starts from the same deterministic
    initialisation; no previous solution is loaded.

    ``ball_contact`` (experimental, default None = accepted behaviour) adds an
    explicit per-robot object term: ``{"ball": (n,3) robot-frame centres,
    "events": [(frame, keypoint_index), ...], "weight": w, "distance": d}``
    penalising ``(||toe - ball|| - d)^2`` at the listed frames. Used only by
    the independent object-aware baseline probe.
    """
    loss_weights = loss_weights or SoccerRetargetWeights()
    dtype = torch.float64
    chain = pk.build_chain_from_urdf(urdf_path.read_text()).to(dtype=dtype)
    joint_names = tuple(chain.get_joint_parameter_names())
    if precomputed_targets is None:
        targets_np, base_root_np, initial_yaw_np, scale = prepare_soccer_targets(
            source_joints, config.nominal_pelvis_height
        )
    else:
        targets_np, base_root_np, initial_yaw_np, scale = precomputed_targets
        targets_np = np.asarray(targets_np, dtype=np.float64)
        base_root_np = np.asarray(base_root_np, dtype=np.float64)
        initial_yaw_np = np.asarray(initial_yaw_np, dtype=np.float64)
        scale = float(scale)
    frame_count = len(source_joints)
    joint_map = {joint.name: joint for joint in load_actuated_joint_limits(urdf_path)}
    lower = torch.tensor([joint_map[name].lower for name in joint_names], dtype=dtype)
    upper = torch.tensor([joint_map[name].upper for name in joint_names], dtype=dtype)
    midpoint = (lower + upper) / 2
    half_range = (upper - lower) / 2
    raw_initial = torch.atanh(
        torch.clamp(-midpoint / half_range, -0.999, 0.999)
    ).repeat(frame_count, 1)

    raw_joints = torch.nn.Parameter(raw_initial)
    root_delta = torch.nn.Parameter(torch.zeros(frame_count, 3, dtype=dtype))
    root_yaw = torch.nn.Parameter(torch.tensor(initial_yaw_np, dtype=dtype))
    yaw_prior = torch.tensor(initial_yaw_np, dtype=dtype)
    base_root = torch.tensor(base_root_np, dtype=dtype)
    targets = torch.tensor(targets_np, dtype=dtype)
    target_velocity = targets[1:] - targets[:-1]
    kick_t = torch.tensor(kick, dtype=dtype)
    plant_t = torch.tensor(plant, dtype=dtype)
    optimizer = torch.optim.Adam(
        [raw_joints, root_delta, root_yaw], lr=config.learning_rate
    )

    base_w = torch.tensor([BASE_WEIGHTS[name] for name in KEYPOINT_NAMES], dtype=dtype)
    left_idx = [KEYPOINT_NAMES.index(n) for n in ("left_hip", "left_knee", "left_ankle", "left_toe")]
    right_idx = [KEYPOINT_NAMES.index(n) for n in ("right_hip", "right_knee", "right_ankle", "right_toe")]
    lank = KEYPOINT_NAMES.index("left_ankle")
    rank = KEYPOINT_NAMES.index("right_ankle")
    pelvis_i = KEYPOINT_NAMES.index("pelvis")
    hip_yaw_idx = torch.tensor(
        [i for i, name in enumerate(joint_names) if "hip_yaw" in name], dtype=torch.long
    )

    ball_t = None
    if ball_contact is not None:
        ball_t = torch.tensor(np.asarray(ball_contact["ball"], dtype=np.float64), dtype=dtype)
        ball_frames = torch.tensor([int(f) for f, _ in ball_contact["events"]], dtype=torch.long)
        ball_kps = torch.tensor([int(k) for _, k in ball_contact["events"]], dtype=torch.long)
        ball_weight = float(ball_contact.get("weight", 30.0))
        ball_distance = float(ball_contact.get("distance", 0.11))

    history: list[float] = []
    best = None
    for _ in range(config.iterations):
        joint_positions = midpoint + half_range * torch.tanh(raw_joints)
        transforms = chain.forward_kinematics(joint_positions)
        local = local_keypoints(transforms, dtype)
        root = base_root + root_delta
        pred = _apply_root_transform(local, root, root_yaw)
        keypoint_weights = base_w[None].repeat(frame_count, 1)
        event_boost = 1.0 if loss_weights.event_keypoint_boost else 0.0
        boost_l = 1.0 + event_boost * 2.4 * kick_t[:, 0:1]
        boost_r = 1.0 + event_boost * 2.4 * kick_t[:, 1:2]
        for idx in left_idx:
            keypoint_weights[:, idx] = keypoint_weights[:, idx] * boost_l[:, 0]
        for idx in right_idx:
            keypoint_weights[:, idx] = keypoint_weights[:, idx] * boost_r[:, 0]
        for idx in (lank, KEYPOINT_NAMES.index("left_toe")):
            keypoint_weights[:, idx] = keypoint_weights[:, idx] * (
                1.0 + event_boost * 1.6 * kick_t[:, 0]
            )
        for idx in (rank, KEYPOINT_NAMES.index("right_toe")):
            keypoint_weights[:, idx] = keypoint_weights[:, idx] * (
                1.0 + event_boost * 1.6 * kick_t[:, 1]
            )
        keypoint_loss = torch.mean(
            torch.sum((pred - targets) ** 2, dim=-1) * keypoint_weights
        )
        root_loss = torch.mean(root_delta**2)
        yaw_loss = torch.mean((root_yaw - yaw_prior) ** 2)
        hip_yaw_loss = torch.mean(joint_positions[:, hip_yaw_idx] ** 2)
        vel = joint_positions[1:] - joint_positions[:-1]
        acc = joint_positions[2:] - 2 * joint_positions[1:-1] + joint_positions[:-2]
        kick_phase = torch.maximum(kick_t[:, 0], kick_t[:, 1])
        swing = torch.clamp(torch.maximum(kick_phase[1:], kick_phase[:-1]), 0.0, 1.4) / 1.4
        smooth_relaxation = 0.75 if loss_weights.event_smooth_relaxation else 0.0
        smooth_w = (1.0 - smooth_relaxation * swing)[:, None]
        pred_velocity = pred[1:] - pred[:-1]
        # Preserve the source's preparation, impact, and follow-through velocity.
        left_phase = torch.maximum(kick_t[1:, 0], kick_t[:-1, 0])
        right_phase = torch.maximum(kick_t[1:, 1], kick_t[:-1, 1])
        kick_velocity_loss = (
            torch.sum(
                left_phase[:, None]
                * (
                    (pred_velocity[:, lank] - target_velocity[:, lank]) ** 2
                    + (
                        pred_velocity[:, KEYPOINT_NAMES.index("left_toe")]
                        - target_velocity[:, KEYPOINT_NAMES.index("left_toe")]
                    )
                    ** 2
                )
            )
            + torch.sum(
                right_phase[:, None]
                * (
                    (pred_velocity[:, rank] - target_velocity[:, rank]) ** 2
                    + (
                        pred_velocity[:, KEYPOINT_NAMES.index("right_toe")]
                        - target_velocity[:, KEYPOINT_NAMES.index("right_toe")]
                    )
                    ** 2
                )
            )
        ) / torch.clamp(
            2.0 * (left_phase.sum() + right_phase.sum()), min=1.0
        )
        upper_idx = [
            KEYPOINT_NAMES.index(name)
            for name in ("neck", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow")
        ]
        upper_velocity_loss = torch.mean(
            (pred_velocity[:, upper_idx] - target_velocity[:, upper_idx]) ** 2
        )
        neck_i = KEYPOINT_NAMES.index("neck")
        left_sh = KEYPOINT_NAMES.index("left_shoulder")
        right_sh = KEYPOINT_NAMES.index("right_shoulder")

        def _unit(vec: torch.Tensor) -> torch.Tensor:
            return vec / torch.clamp(torch.linalg.norm(vec, dim=-1, keepdim=True), min=1e-6)

        # torso_link origin sits on the waist joint, so pitch does not move it.
        # Sample a point up the chest and match the source spine lean.
        torso = transforms["torso_link"].get_matrix()
        chest_offset = torch.tensor([0.0, 0.0, 0.28], dtype=dtype)
        chest_local = torso[:, :3, 3] + torch.einsum("nij,j->ni", torso[:, :3, :3], chest_offset)
        chest = _apply_root_transform(chest_local[:, None], root, root_yaw)[:, 0]
        up_loss = torch.mean(
            (_unit(chest - pred[:, pelvis_i]) - _unit(targets[:, neck_i] - targets[:, pelvis_i])) ** 2
        )
        shoulder_line_loss = torch.mean(
            (_unit(pred[:, left_sh] - pred[:, right_sh]) - _unit(targets[:, left_sh] - targets[:, right_sh]))
            ** 2
        )
        # Allow the kicking leg to move quickly; keep the planted foot still.
        plant_slip = (
            torch.mean(plant_t[1:, 0] * torch.sum((pred[1:, lank] - pred[:-1, lank]) ** 2, dim=-1))
            + torch.mean(plant_t[1:, 1] * torch.sum((pred[1:, rank] - pred[:-1, rank]) ** 2, dim=-1))
        )
        sole = torch.mean(
            torch.relu(0.012 - pred[:, lank, 2]) ** 2 + torch.relu(0.012 - pred[:, rank, 2]) ** 2
        )
        bounce = torch.mean((pred[:, pelvis_i, 2] - targets[:, pelvis_i, 2]) ** 2)
        bounce_vel = torch.mean(
            (
                (pred[1:, pelvis_i, 2] - pred[:-1, pelvis_i, 2])
                - (targets[1:, pelvis_i, 2] - targets[:-1, pelvis_i, 2])
            )
            ** 2
        )
        loss = (
            config.keypoint_weight * keypoint_loss
            + config.root_regularization_weight * root_loss
            + 0.4 * yaw_loss
            + 0.06 * hip_yaw_loss
            + 0.02 * torch.mean(smooth_w * vel**2)
            + 0.04 * torch.mean(smooth_w[1:] * acc**2)
            + loss_weights.support_plant * plant_slip
            + loss_weights.support_floor * sole
            + 18.0 * bounce
            + 40.0 * bounce_vel
            + 8.0 * upper_velocity_loss
            + 6.0 * up_loss
            + 4.0 * shoulder_line_loss
            + loss_weights.kick_velocity * kick_velocity_loss
        )
        if ball_t is not None:
            toe_at_events = pred[ball_frames, ball_kps]
            gap = torch.linalg.norm(toe_at_events - ball_t[ball_frames], dim=-1) - ball_distance
            loss = loss + ball_weight * torch.mean(gap**2)
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
                pred.detach().clone(),
            )
        optimizer.step()

    _, best_joints, best_root, best_yaw, best_keypoints = best
    return IndependentRetargetResult(
        joint_names=joint_names,
        keypoint_names=KEYPOINT_NAMES,
        joint_positions=best_joints.cpu().numpy(),
        root_translation=best_root.cpu().numpy(),
        root_yaw=best_yaw.cpu().numpy(),
        robot_keypoints=best_keypoints.cpu().numpy(),
        target_keypoints=targets_np,
        scale=scale,
        loss_history=np.asarray(history),
        config_json=json.dumps(
            {
                "ik": asdict(config),
                "soccer": asdict(loss_weights),
            },
            sort_keys=True,
        ),
    )


def save_result(result: IndependentRetargetResult, path: Path) -> None:
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


def align_second_pass_stations(g1_dir: Path, gap_lanes: np.ndarray) -> None:
    """Keep robot B's natural joints, but align its right-stop/left-kick stance.

    A slow world-Y offset places the right foot on pass 1 at f15 and the left
    foot on pass 2 at f21. Joint angles and the source-derived action are unchanged.
    """
    def apply_offset(path: Path, anchors: list[tuple[int, float]]) -> None:
        payload = {key: np.array(value) for key, value in np.load(path).items()}
        offset = np.zeros(len(payload["robot_keypoints"]), dtype=np.float64)
        for (f0, y0), (f1, y1) in zip(anchors, anchors[1:]):
            weights = np.asarray(
                _smoothstep(np.linspace(0.0, 1.0, f1 - f0 + 1)),
                dtype=np.float64,
            )
            offset[f0 : f1 + 1] = y0 + weights * (y1 - y0)
        payload["root_translation"][:, 1] += offset
        payload["robot_keypoints"][:, :, 1] += offset[:, None]
        if "target_keypoints" in payload:
            payload["target_keypoints"][:, :, 1] += offset[:, None]
        np.savez_compressed(path, **payload)

    path_a = g1_dir / "robot_a.npz"
    payload_a = {key: np.array(value) for key, value in np.load(path_a).items()}
    names_a = [str(name) for name in payload_a["keypoint_names"]]
    first_offset = float(
        gap_lanes[3]
        - payload_a["robot_keypoints"][0, names_a.index("right_toe"), 1]
    )
    # Hold through contact, then move during the visible two-foot hop.
    apply_offset(
        path_a,
        [(0, first_offset), (5, first_offset), (14, 0.0), (15, 0.0)],
    )

    path = g1_dir / "robot_b.npz"
    payload = {key: np.array(value) for key, value in np.load(path).items()}
    names = [str(name) for name in payload["keypoint_names"]]
    keypoints = payload["robot_keypoints"]
    right_offset = float(
        gap_lanes[3] - keypoints[15, names.index("right_toe"), 1]
    )
    left_offset = float(
        gap_lanes[4] - keypoints[21, names.index("left_toe"), 1]
    )
    shared_offset = 0.5 * (right_offset + left_offset)
    anchors = [
        (0, 0.0),
        (6, 0.0),
        (12, shared_offset),
        (21, shared_offset),
        (27, shared_offset),
        (34, 0.0),
        (35, 0.0),
    ]
    apply_offset(path, anchors)
    print(
        "aligned opening stations",
        f"a-f0={first_offset:+.3f}m",
        f"b-hop={shared_offset:+.3f}m",
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--joints",
        type=Path,
        default=Path("data/exports/touch_ladder_demo/smplx_joints.npz"),
    )
    parser.add_argument(
        "--contacts",
        type=Path,
        default=Path("data/exports/touch_ladder_demo/ball_cones.json"),
    )
    parser.add_argument(
        "--urdf",
        type=Path,
        default=g1_urdf(),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/exports/touch_ladder_demo/g1"),
    )
    parser.add_argument("--iterations", type=int, default=260)
    args = parser.parse_args()
    guard_release_writes(args, args.contacts.with_suffix('.npz'))

    packed = np.load(args.joints)
    contacts = json.loads(args.contacts.read_text())["contacts"]
    _, gap_lanes = ladder_geometry()
    joints_a, joints_b, plant_a, plant_b = station_pair_on_ladder(
        packed["player_a_joints"],
        packed["player_b_joints"],
        contacts,
        gap_lanes,
    )
    step_plant = {"a": plant_a, "b": plant_b}
    pass_lanes = near_return_lanes(max(len(contacts) - 1, 1))
    packed_joints = {"player_a_joints": joints_a, "player_b_joints": joints_b}
    print("source pass lanes", list(pass_lanes), "gap_y", np.round(gap_lanes, 3))
    config = IndependentRetargetConfig(
        iterations=args.iterations,
        learning_rate=0.04,
        keypoint_weight=1.0,
        root_regularization_weight=0.35,
        velocity_weight=0.035,
        acceleration_weight=0.08,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("robot_a.npz", "robot_b.npz"):
        archive_export(args.out_dir / name, tag="pre_ik")
    scales = {}
    for name, key, player in (
        ("robot_a", "player_a_joints", "a"),
        ("robot_b", "player_b_joints", "b"),
    ):
        joints = packed_joints[key]
        n = len(joints)
        kick, plant = contact_windows(contacts, player, n, step_plant=step_plant[player])
        print(
            f"retarget {name} {joints.shape} kick_frames={int((kick > 0).any(1).sum())} ..."
        )
        result = retarget_soccer(joints, args.urdf, config, kick, plant)
        save_result(result, args.out_dir / f"{name}.npz")
        scales[name] = float(result.scale)
        names = list(result.joint_names)
        lp = names.index("left_ankle_pitch_joint")
        rp = names.index("right_ankle_pitch_joint")
        print(
            f"  scale={result.scale:.3f} loss={result.loss_history[-1]:.4f} "
            f"ankle_pitch_max={max(abs(result.joint_positions[:, lp]).max(), abs(result.joint_positions[:, rp]).max()):.3f} "
            f"pelvis_z_ptp={np.ptp(result.robot_keypoints[:, 0, 2]):.3f}"
        )
    (args.out_dir / "scales.json").write_text(json.dumps(scales, indent=2) + "\n")
    align_second_pass_stations(args.out_dir, gap_lanes)
    snap_ball_to_g1_feet(args.out_dir, contacts, args.contacts.with_suffix(".npz"))


def _push_off_cones(ball: np.ndarray, cones: np.ndarray, clearance: float) -> np.ndarray:
    """Keep the pass in the gaps instead of tunneling through a cone."""
    out = ball.copy()
    for t in range(len(out)):
        for cone in cones:
            delta = out[t, :2] - cone[:2]
            dist = float(np.linalg.norm(delta))
            if dist < clearance:
                if dist < 1e-8:
                    delta = np.array([1.0, 0.0])
                    dist = 1.0
                out[t, :2] = cone[:2] + delta / dist * clearance
    return out


def _robot_to_yup(points: np.ndarray) -> np.ndarray:
    return np.stack([points[..., 0], points[..., 2], -points[..., 1]], axis=-1)


def _hermite(p0: np.ndarray, p1: np.ndarray, v0: np.ndarray, v1: np.ndarray, n: int) -> np.ndarray:
    n = int(n)
    out = np.zeros((max(n, 1), 3), dtype=np.float64)
    if n <= 1:
        out[0] = p1
        return out[:1]
    u = np.linspace(0.0, 1.0, n, dtype=np.float64)[:, None]
    h00 = 2 * u**3 - 3 * u**2 + 1
    h10 = u**3 - 2 * u**2 + u
    h01 = -2 * u**3 + 3 * u**2
    h11 = u**3 - u**2
    out[:] = h00 * p0 + h10 * v0 + h01 * p1 + h11 * v1
    return out


def _source_progress(source_uv: np.ndarray) -> np.ndarray:
    """Monotone travel progress measured directly from the detected source ball."""
    if len(source_uv) <= 1:
        return np.zeros(len(source_uv), dtype=np.float64)
    direction = source_uv[-1] - source_uv[0]
    denominator = max(float(np.dot(direction, direction)), 1e-8)
    progress = ((source_uv - source_uv[0]) @ direction) / denominator
    progress = np.clip(progress, 0.0, 1.0)
    if len(progress) >= 5:
        progress = np.convolve(
            np.pad(progress, (2, 2), mode="edge"),
            np.ones(5) / 5.0,
            mode="valid",
        )
    progress = np.maximum.accumulate(progress)
    progress -= progress[0]
    progress /= max(float(progress[-1]), 1e-8)
    return progress


def _fit_homography(
    source: np.ndarray, target: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """Weighted normalized DLT; avoids an OpenCV dependency in the IK venv."""

    def normalize(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        center = points.mean(axis=0)
        centered = points - center
        scale = np.sqrt(2.0) / max(
            float(np.mean(np.linalg.norm(centered, axis=1))), 1e-8
        )
        transform = np.array(
            [
                [scale, 0.0, -scale * center[0]],
                [0.0, scale, -scale * center[1]],
                [0.0, 0.0, 1.0],
            ]
        )
        homogeneous = np.column_stack([points, np.ones(len(points))])
        normalized = homogeneous @ transform.T
        return normalized[:, :2], transform

    src, source_transform = normalize(np.asarray(source, dtype=np.float64))
    dst, target_transform = normalize(np.asarray(target, dtype=np.float64))
    rows = []
    for (u, v), (x, y), weight in zip(src, dst, weights, strict=True):
        root_weight = np.sqrt(float(weight))
        rows.append(
            root_weight
            * np.array([-u, -v, -1.0, 0.0, 0.0, 0.0, x * u, x * v, x])
        )
        rows.append(
            root_weight
            * np.array([0.0, 0.0, 0.0, -u, -v, -1.0, y * u, y * v, y])
        )
    _, _, vh = np.linalg.svd(np.stack(rows), full_matrices=False)
    normalized_h = vh[-1].reshape(3, 3)
    homography = (
        np.linalg.inv(target_transform) @ normalized_h @ source_transform
    )
    return homography / homography[2, 2]


def _project_homography(points: np.ndarray, homography: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack([points, np.ones(len(points))])
    mapped = homogeneous @ homography.T
    return mapped[:, :2] / mapped[:, 2:3]


def _smooth_segment(values: np.ndarray) -> np.ndarray:
    if len(values) < 7:
        return values.copy()
    kernel = np.array([1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0]) / 16.0
    result = values.copy()
    for axis in range(values.shape[1]):
        result[:, axis] = np.convolve(
            np.pad(values[:, axis], (3, 3), mode="edge"), kernel, mode="valid"
        )
    return result


def _eased_progress(frames: int, drag: float = 0.32) -> np.ndarray:
    tau = np.linspace(0.0, 1.0, frames)
    progress = tau + drag * tau * (1.0 - tau)
    progress /= max(float(progress[-1]), 1e-8)
    return progress


def _decel_polyline(points: list[np.ndarray], frames: int, drag: float = 0.22) -> np.ndarray:
    """One decelerating roll along straight pieces. Mild drag so it does not crawl-stop early."""
    pts = [np.asarray(point, dtype=np.float64) for point in points]
    seg = np.array([np.linalg.norm(b[:2] - a[:2]) for a, b in zip(pts, pts[1:])])
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(cum[-1])
    result = np.zeros((frames, 3), dtype=np.float64)
    if total < 1e-6:
        result[:] = pts[-1]
        result[:, 2] = 0.11
        return result
    distance = _eased_progress(frames, drag=drag) * total
    for index, traveled in enumerate(distance):
        piece = int(np.searchsorted(cum, traveled, side="right") - 1)
        piece = min(max(piece, 0), len(seg) - 1)
        span = seg[piece]
        blend = 0.0 if span < 1e-9 else (traveled - cum[piece]) / span
        blend = float(np.clip(blend, 0.0, 1.0))
        result[index, :2] = pts[piece][:2] + blend * (pts[piece + 1][:2] - pts[piece][:2])
    result[0, :2] = pts[0][:2]
    result[-1, :2] = pts[-1][:2]
    result[:, 2] = 0.11
    return result


def archive_export(path: Path, tag: str = "") -> Path | None:
    """Copy an existing export aside before rewriting it."""
    if not path.exists():
        return None
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir = path.parent / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{tag}" if tag else ""
    dest = archive_dir / f"{path.stem}_{stamp}{suffix}{path.suffix}"
    shutil.copy2(path, dest)
    print("archived", path, "->", dest)
    return dest


def snap_ball_to_g1_feet(g1_dir: Path, contacts: list[dict], ball_npz: Path) -> None:
    """Roll from the kicking foot through the source gap at that station."""
    archive_export(ball_npz, tag="pre_snap")
    robots = {
        "a": {
            key: np.array(value)
            for key, value in np.load(g1_dir / "robot_a.npz").items()
        },
        "b": {
            key: np.array(value)
            for key, value in np.load(g1_dir / "robot_b.npz").items()
        },
    }
    names = [str(name) for name in robots["a"]["keypoint_names"]]
    ankle_idx = {
        "left": names.index("left_ankle"),
        "right": names.index("right_ankle"),
    }
    toe_idx = {"left": names.index("left_toe"), "right": names.index("right_toe")}
    pelvis_idx = names.index("pelvis")
    n = min(len(robots["a"]["robot_keypoints"]), len(robots["b"]["robot_keypoints"]))

    def contact_pose(
        event: dict, *, kick: bool
    ) -> tuple[int, str, np.ndarray, np.ndarray]:
        kicker = robots[event["player"]]["robot_keypoints"]
        other = robots["b" if event["player"] == "a" else "a"]["robot_keypoints"]
        frame = int(
            event.get("kick_frame", event["frame"]) if kick else event["frame"]
        )
        side = event.get("kick_side", event["side"]) if kick else event["side"]
        alpha = float(event.get("foot_alpha", 1.0))
        touch = (1.0 - alpha) * kicker[frame, ankle_idx[side]] + alpha * kicker[
            frame, toe_idx[side]
        ]
        toward = other[frame, pelvis_idx] - kicker[frame, pelvis_idx]
        toward[2] = 0.0
        toward = toward / (np.linalg.norm(toward) + 1e-8)
        prev_frame = max(0, frame - 2)
        next_frame = min(n - 1, frame + 2)
        kick_vel = kicker[next_frame, toe_idx[side]] - kicker[prev_frame, toe_idx[side]]
        kick_vel[2] = 0.0
        if np.dot(kick_vel, toward) > 0.01:
            blended = 0.6 * toward + 0.4 * kick_vel / (np.linalg.norm(kick_vel) + 1e-8)
            blended[2] = 0.0
            toward = blended / (np.linalg.norm(blended) + 1e-8)
        # A receive must visibly reach the shoe mesh; a kick starts with a
        # slightly larger forward offset so the ball leaves cleanly.
        contact_offset = 0.095 if kick else 0.075
        pos = touch + contact_offset * toward
        pos[2] = 0.11
        return frame, side, pos, toward

    existing = dict(np.load(ball_npz))
    cones_robot = points_core4d_to_robot(
        np.asarray(existing["cone_xyz_yup"], dtype=np.float64)
    )
    cone_y, gap_lanes = ladder_geometry(len(cones_robot))
    cone_x = 0.5 * float(
        robots["a"]["root_translation"][:, 0].mean()
        + robots["b"]["root_translation"][:, 0].mean()
    )
    cone_order = np.argsort(cones_robot[:, 1])[::-1]
    for rank, cone_index in enumerate(cone_order):
        cones_robot[cone_index, 0] = cone_x
        cones_robot[cone_index, 1] = cone_y[rank]
    existing["cone_xyz_yup"] = _robot_to_yup(cones_robot)

    events = []
    for event in contacts:
        receive_frame, receive_side, receive_pos, _ = contact_pose(
            event, kick=False
        )
        kick_frame, kick_side, kick_pos, toward = contact_pose(event, kick=True)
        if kick_frame == receive_frame and kick_side == receive_side:
            kick_pos = receive_pos.copy()
        events.append(
            {
                "receive_frame": receive_frame,
                "receive_side": receive_side,
                "receive_pos": receive_pos,
                "kick_frame": kick_frame,
                "kick_side": kick_side,
                "kick_pos": kick_pos,
                "toward": toward,
                "label": f"{event['player']}-{receive_side}->{kick_side}",
            }
        )
        print(
            f"  ball contact {event['player']} receive "
            f"f{receive_frame}-{receive_side} kick f{kick_frame}-{kick_side}"
        )
    pass_lanes = near_return_lanes(max(len(contacts) - 1, 1))
    for event in events:
        for key in ("receive_pos", "kick_pos"):
            event[key] = np.asarray(event[key], dtype=np.float64).copy()
            event[key][2] = 0.11
    print("  passes", list(pass_lanes), "gap_y", np.round(gap_lanes, 3))
    ball_robot = np.zeros((n, 3), dtype=np.float64)
    first_end = int(events[0]["kick_frame"])
    first_pos = np.asarray(events[0]["kick_pos"], dtype=np.float64)
    first_start = first_pos + 0.45 * np.asarray(events[0]["toward"])
    first_progress = _eased_progress(first_end + 1, drag=0.18)
    ball_robot[: first_end + 1] = (
        first_start[None]
        + first_progress[:, None] * (first_pos - first_start)[None]
    )
    ball_robot[: first_end + 1, 2] = 0.11
    gap_order = []
    for i, event in enumerate(events):
        receive_frame = int(event["receive_frame"])
        start = int(event["kick_frame"])
        p0 = np.asarray(event["kick_pos"], dtype=np.float64)
        if start > receive_frame:
            receive = np.asarray(event["receive_pos"], dtype=np.float64)
            control_n = start - receive_frame + 1
            weights = np.asarray(
                _smoothstep(np.linspace(0.0, 1.0, control_n)), dtype=np.float64
            )
            ball_robot[receive_frame : start + 1] = (
                receive[None] + weights[:, None] * (p0 - receive)[None]
            )
        if i + 1 >= len(events):
            final_end = p0 + 0.30 * np.asarray(event["toward"])
            final_progress = _eased_progress(n - start, drag=0.18)
            ball_robot[start:] = p0[None] + final_progress[:, None] * (
                final_end - p0
            )[None]
            ball_robot[start:, 2] = 0.11
            break
        end = int(events[i + 1]["receive_frame"])
        p1 = np.asarray(events[i + 1]["receive_pos"], dtype=np.float64)
        lane_index = int(pass_lanes[i])
        gap_y = float(gap_lanes[lane_index])
        next_y = (
            float(gap_lanes[int(pass_lanes[i + 1])])
            if i + 1 < len(pass_lanes)
            else gap_y
        )
        total = end - start + 1
        if i in (0, 2):
            # First and third passes are single-impulse straight rolls.
            # Their two foot contacts are already aligned to the accepted gap.
            segment = _decel_polyline([p0, p1], total, drag=0.18)
            segment[0] = p0
            segment[-1] = p1
            gap_order.append(lane_index)
            ball_robot[start : end + 1] = segment
            print(
                f"  pass {start:03d}->{end:03d} gap={lane_index} "
                "direct_foot_to_foot"
            )
            continue
        receive_step = 0
        if abs(next_y - gap_y) >= 0.08 and i + 1 < len(pass_lanes):
            need = int(np.ceil(1.5 * abs(next_y - gap_y) * 24.0 / 2.0))
            receive_step = int(min(max(need, 1), max(total - 8, 1)))
        launch_step = 0
        if abs(float(p0[1]) - gap_y) >= 0.08:
            launch_step = min(3, max(total - receive_step - 8, 0))
        flight_n = total - launch_step - receive_step
        flight = _decel_polyline(
            [np.array([p0[0], gap_y, 0.11]), np.array([p1[0], gap_y, 0.11])],
            flight_n,
            drag=0.18,
        )
        parts = []
        if launch_step:
            weights = np.asarray(
                _smoothstep(np.linspace(0.0, 1.0, launch_step + 1)[:-1]),
                dtype=np.float64,
            )
            launch = np.column_stack(
                [
                    np.full(launch_step, p0[0]),
                    p0[1] + weights * (gap_y - p0[1]),
                    np.full(launch_step, 0.11),
                ]
            )
            parts.append(launch)
        parts.append(flight)
        if receive_step:
            weights = np.asarray(
                _smoothstep(np.linspace(0.0, 1.0, receive_step + 1)[1:]),
                dtype=np.float64,
            )
            settle = np.column_stack(
                [
                    np.full(receive_step, p1[0]),
                    gap_y + weights * (p1[1] - gap_y),
                    np.full(receive_step, 0.11),
                ]
            )
            parts.append(settle)
        segment = np.vstack(parts)
        segment[0] = p0
        segment[-1] = p1
        gap_order.append(lane_index)
        ball_robot[start : end + 1] = segment
        print(
            f"  pass {start:03d}->{end:03d} gap={lane_index} "
            f"straight_y={gap_y:+.3f} launch={launch_step} receive={receive_step}"
        )
    existing["ball_xyz_yup"] = _robot_to_yup(ball_robot)
    existing["gap_lanes_robot_y"] = gap_lanes
    existing["source_gap_order"] = np.asarray(gap_order, dtype=np.int32)
    existing["method"] = np.asarray("straight_gap_pass")
    np.savez_compressed(ball_npz, **existing)
    print("updated", ball_npz)


if __name__ == "__main__":
    main()

"""Post-hoc cross-agent coordination metrics for the CORE4D Join evaluation.

Everything here is computed from already-saved robot trajectories, the official
contact-analysis masks, and the official rubber-hand sample geometry.  No
optimization is run and no thresholds differ from the paper metric contract:

* source expected contact : SMPL-X fingertip min distance, hysteresis 0.05/0.07 m
* robot predicted contact : rubber-hand sample min distance, hysteresis 0.04/0.055 m
* stable onset / end      : first run of >= 5 frames (``first_stable_contact``)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import ConvexHull, cKDTree

from .audit import load_npz_mapping
from .contact import (
    PHASE_NAMES,
    fingertips_in_object_space,
    first_stable_contact,
    hysteresis_contact,
    resolve_object_model,
    stable_contact_end,
)
from .coordinates import points_robot_to_core4d
from .metrics import binary_f1, rubber_hand_world_volume_points
from .retarget.balance import LEFT_HAND_VOLUME_OFFSETS_M, points_in_object_frame

SOURCE_CONTACT_ENTER_M = 0.05
SOURCE_CONTACT_EXIT_M = 0.07
ROBOT_CONTACT_ENTER_M = 0.04
ROBOT_CONTACT_EXIT_M = 0.055
STABLE_MIN_FRAMES = 5
ONSET_SENSITIVITY_MIN_FRAMES = (1, 3, 5, 8)
PENETRATION_THRESHOLDS_CM = (0.0, 0.5, 1.0)
HAND_NAMES = ("left", "right")
DEFAULT_URDF = Path(
    "third_party/unitree_ros/robots/g1_description/g1_29dof_rev_1_0.urdf"
)


@dataclass
class SequenceSignals:
    """Per-frame signals for one evaluated dual-robot result."""

    sequence_id: str
    label: str
    fps: float
    phases: np.ndarray
    expected: dict[int, np.ndarray]
    expected_hand: dict[int, np.ndarray]
    predicted: dict[int, np.ndarray]
    predicted_hand_or: dict[int, np.ndarray]
    hand_distance_m: dict[int, np.ndarray]
    hand_penetration_cm: dict[int, np.ndarray]
    hand_penetration_per_hand_cm: dict[int, np.ndarray]
    robot_hand_local: dict[int, np.ndarray]
    source_anchor_local: dict[int, np.ndarray]
    extras: dict[str, object] = field(default_factory=dict)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return [start, end) index pairs of maximal True runs."""
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return []
    padded = np.concatenate([[False], mask, [False]])
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(edges[i]), int(edges[i + 1])) for i in range(0, len(edges), 2)]


def _longest_run(mask: np.ndarray) -> int:
    runs = _runs(mask)
    return max((end - start for start, end in runs), default=0)


def _safe_onset(mask: np.ndarray, min_frames: int) -> int | None:
    try:
        return int(first_stable_contact(mask, min_frames=min_frames))
    except ValueError:
        return None


def _safe_end(mask: np.ndarray, onset: int | None, min_frames: int) -> int | None:
    if onset is None:
        return None
    return int(stable_contact_end(mask, onset, min_frames=min_frames))


def _resolve_result(result_root: Path, sequence_id: str, person: int) -> Path:
    stem = sequence_id.replace("/", "_")
    nested = result_root / stem / f"robot{person}.npz"
    return nested if nested.exists() else result_root / f"robot{person}.npz"


def _source_surface_anchors(
    person_vertices: np.ndarray,
    object_poses: np.ndarray,
    object_vertices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Object-frame surface point nearest the closest fingertip of each hand.

    This is the source-defined contact anchor used by the optimizer, with zero
    palm offset, no smoothing, and no grasp-height clamp so that it is neutral
    with respect to every method configuration.
    """
    local = fingertips_in_object_space(person_vertices, object_poses)
    tree = cKDTree(object_vertices)
    distances, indices = tree.query(local.reshape(-1, 3), k=1)
    distances = distances.reshape(len(person_vertices), 10)
    indices = indices.reshape(len(person_vertices), 10)
    anchors = np.zeros((len(person_vertices), 2, 3), dtype=np.float64)
    per_hand_min = np.zeros((len(person_vertices), 2), dtype=np.float64)
    rows = np.arange(len(person_vertices))
    for hand in range(2):
        sl = slice(hand * 5, (hand + 1) * 5)
        best = np.argmin(distances[:, sl], axis=1)
        anchors[:, hand] = object_vertices[indices[:, sl][rows, best]]
        per_hand_min[:, hand] = distances[:, sl][rows, best]
    return anchors, per_hand_min


def load_sequence_signals(
    sequence_root: Path,
    sequence_id: str,
    object_models_root: Path,
    contact_analysis_path: Path,
    result_root: Path,
    *,
    label: str,
    fps: float = 15.0,
    urdf_path: Path = DEFAULT_URDF,
) -> SequenceSignals:
    """Load every per-frame signal needed for the coordination metrics."""
    sequence_dir = sequence_root.joinpath(*sequence_id.split("/"))
    object_pose_path = sequence_dir / "smooth_objposes_so3.npy"
    if not object_pose_path.exists():
        object_pose_path = sequence_dir / "smooth_objposes.npy"
    object_poses = np.load(object_pose_path)
    mesh = trimesh.load_mesh(
        resolve_object_model(sequence_dir, object_models_root), process=False
    )
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    hull = ConvexHull(vertices)
    tree = cKDTree(vertices)

    with np.load(contact_analysis_path) as contact:
        phases = np.asarray(contact["phase"])
        expected = {
            1: np.asarray(contact["person1_contact"], dtype=bool),
            2: np.asarray(contact["person2_contact"], dtype=bool),
        }
        fingertip = {
            1: np.asarray(contact["person1_fingertip_distances"]),
            2: np.asarray(contact["person2_fingertip_distances"]),
        }
    expected_hand = {}
    for person in (1, 2):
        expected_hand[person] = np.stack(
            [
                hysteresis_contact(
                    fingertip[person][:, hand * 5 : (hand + 1) * 5].min(axis=1),
                    enter_threshold=SOURCE_CONTACT_ENTER_M,
                    exit_threshold=SOURCE_CONTACT_EXIT_M,
                )
                for hand in range(2)
            ],
            axis=1,
        )

    source_anchor_local = {}
    for person in (1, 2):
        source = load_npz_mapping(sequence_dir / f"person{person}_poses.npz")
        anchors, _ = _source_surface_anchors(
            np.asarray(source["vertices"]), object_poses, vertices
        )
        source_anchor_local[person] = anchors

    n_left = len(LEFT_HAND_VOLUME_OFFSETS_M)
    predicted: dict[int, np.ndarray] = {}
    predicted_hand_or: dict[int, np.ndarray] = {}
    hand_distance_m: dict[int, np.ndarray] = {}
    hand_penetration_cm: dict[int, np.ndarray] = {}
    hand_penetration_per_hand_cm: dict[int, np.ndarray] = {}
    robot_hand_local: dict[int, np.ndarray] = {}
    frame_counts = {}
    robot_keypoints: dict[int, np.ndarray] = {}
    for person in (1, 2):
        result = load_npz_mapping(_resolve_result(result_root, sequence_id, person))
        joint_positions = np.asarray(result["joint_positions"])
        frame_counts[person] = len(joint_positions)
        if "robot_keypoints" in result:
            robot_keypoints[person] = np.asarray(result["robot_keypoints"], dtype=np.float64)
        volume_world = rubber_hand_world_volume_points(
            urdf_path,
            joint_positions,
            np.asarray(result["root_translation"]),
            np.asarray(result["root_yaw"]),
        )
        local = points_in_object_frame(points_robot_to_core4d(volume_world), object_poses)
        count = local.shape[0]
        flat = local.reshape(-1, 3)
        distances, _ = tree.query(flat, k=1)
        distances = distances.reshape(count, -1)
        per_hand = np.stack(
            [distances[:, :n_left].min(axis=1), distances[:, n_left:].min(axis=1)],
            axis=1,
        )
        hand_distance_m[person] = per_hand
        # Official robot-level contact state: hysteresis on the min over hands.
        predicted[person] = hysteresis_contact(
            per_hand.min(axis=1),
            enter_threshold=ROBOT_CONTACT_ENTER_M,
            exit_threshold=ROBOT_CONTACT_EXIT_M,
        )
        predicted_hand_or[person] = np.logical_or.reduce(
            [
                hysteresis_contact(
                    per_hand[:, hand],
                    enter_threshold=ROBOT_CONTACT_ENTER_M,
                    exit_threshold=ROBOT_CONTACT_EXIT_M,
                )
                for hand in range(2)
            ]
        )
        # Per-frame convex-hull penetration depth (deepest sample), cm.
        values = flat @ hull.equations[:, :3].T + hull.equations[:, 3]
        depth = np.maximum(-values.max(axis=1), 0.0).reshape(count, -1) * 100.0
        hand_penetration_per_hand_cm[person] = np.stack(
            [depth[:, :n_left].max(axis=1), depth[:, n_left:].max(axis=1)], axis=1
        )
        hand_penetration_cm[person] = depth.max(axis=1)
        rows = np.arange(count)
        nearest = np.stack(
            [
                local[rows, np.argmin(distances[:, :n_left], axis=1)],
                local[rows, n_left + np.argmin(distances[:, n_left:], axis=1)],
            ],
            axis=1,
        )
        robot_hand_local[person] = nearest

    frames = len(phases)
    for person in (1, 2):
        if frame_counts[person] != frames:
            raise ValueError(
                f"{label} {sequence_id}: robot{person} has {frame_counts[person]} "
                f"frames but contact analysis has {frames}"
            )
    extras: dict[str, object] = {"result_root": str(result_root)}
    if len(robot_keypoints) == 2:
        # Per-frame minimum distance between any robot-1 and robot-2 keypoint (m).
        diff = robot_keypoints[1][:, :, None, :] - robot_keypoints[2][:, None, :, :]
        extras["inter_robot_min_distance_m"] = np.linalg.norm(diff, axis=-1).min(axis=(1, 2))
    return SequenceSignals(
        sequence_id=sequence_id,
        label=label,
        fps=fps,
        phases=phases,
        expected=expected,
        expected_hand=expected_hand,
        predicted=predicted,
        predicted_hand_or=predicted_hand_or,
        hand_distance_m=hand_distance_m,
        hand_penetration_cm=hand_penetration_cm,
        hand_penetration_per_hand_cm=hand_penetration_per_hand_cm,
        robot_hand_local=robot_hand_local,
        source_anchor_local=source_anchor_local,
        extras=extras,
    )


def _overlap_stats(expected: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    expected = np.asarray(expected, dtype=bool)
    predicted = np.asarray(predicted, dtype=bool)
    tp = int((expected & predicted).sum())
    fp = int((~expected & predicted).sum())
    fn = int((expected & ~predicted).sum())
    precision = tp / (tp + fp) if tp + fp else (1.0 if fn == 0 else 0.0)
    recall = tp / (tp + fn) if tp + fn else 1.0
    union = tp + fp + fn
    return {
        "true_positive_frames": tp,
        "false_positive_frames": fp,
        "false_negative_frames": fn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(binary_f1(expected, predicted)),
        "iou": float(tp / union) if union else 1.0,
    }


def _dropout_events(
    expected_joint: np.ndarray,
    predicted_joint: np.ndarray,
    window: tuple[int, int] | None,
) -> dict[str, object]:
    if window is None:
        return {
            "window": None,
            "dropout_events": None,
            "missed_runs_total": None,
            "maintained_fraction_in_window": None,
        }
    start, end = window
    exp = expected_joint[start:end]
    pred = predicted_joint[start:end]
    missed = exp & ~pred
    runs = _runs(missed)
    missed_total = len(runs)
    # A leading miss before any simultaneous contact is onset delay, not a dropout.
    dropouts = [
        run for run in runs if pred[: run[0]].any()
    ]
    maintained = float((exp & pred).sum() / exp.sum()) if exp.sum() else None
    return {
        "window": [int(start), int(end)],
        "window_expected_joint_frames": int(exp.sum()),
        "dropout_events": len(dropouts),
        "dropout_frames_total": int(sum(end_ - start_ for start_, end_ in dropouts)),
        "missed_runs_total": int(missed_total),
        "maintained_fraction_in_window": maintained,
    }


def coordination_metrics(signals: SequenceSignals) -> dict[str, object]:
    """Compute joint-contact, timing, bilateral error, and geometry-valid proxy."""
    fps = signals.fps
    expected_joint = signals.expected[1] & signals.expected[2]
    predicted_joint = signals.predicted[1] & signals.predicted[2]
    frames = len(expected_joint)

    # ---- A. joint-contact overlap -------------------------------------------------
    overlap = _overlap_stats(expected_joint, predicted_joint)
    per_robot = {
        str(person): {
            **_overlap_stats(signals.expected[person], signals.predicted[person]),
            "expected_frames": int(signals.expected[person].sum()),
            "predicted_frames": int(signals.predicted[person].sum()),
        }
        for person in (1, 2)
    }

    # ---- B. timing / continuity ---------------------------------------------------
    onset_expected = _safe_onset(expected_joint, STABLE_MIN_FRAMES)
    onset_predicted = _safe_onset(predicted_joint, STABLE_MIN_FRAMES)
    end_expected = _safe_end(expected_joint, onset_expected, STABLE_MIN_FRAMES)
    end_predicted = _safe_end(predicted_joint, onset_predicted, STABLE_MIN_FRAMES)
    window = (
        (onset_expected, end_expected)
        if onset_expected is not None and end_expected is not None
        else None
    )
    continuity = _dropout_events(expected_joint, predicted_joint, window)
    longest_predicted = _longest_run(predicted_joint)
    longest_true = _longest_run(predicted_joint & expected_joint)
    longest_expected = _longest_run(expected_joint)
    sensitivity = {}
    for min_frames in ONSET_SENSITIVITY_MIN_FRAMES:
        exp_on = _safe_onset(expected_joint, min_frames)
        pred_on = _safe_onset(predicted_joint, min_frames)
        sensitivity[f"min_frames_{min_frames}"] = {
            "expected_onset": exp_on,
            "predicted_onset": pred_on,
            "onset_error_frames": (
                int(pred_on - exp_on) if exp_on is not None and pred_on is not None else None
            ),
        }
    alt_joint = signals.predicted_hand_or[1] & signals.predicted_hand_or[2]
    alt_onset = _safe_onset(alt_joint, STABLE_MIN_FRAMES)
    sensitivity["per_hand_hysteresis_or"] = {
        "f1": float(binary_f1(expected_joint, alt_joint)),
        "predicted_onset": alt_onset,
        "frames_differing_from_official": int((alt_joint != predicted_joint).sum()),
    }

    timing = {
        "expected_joint_onset_frame": onset_expected,
        "predicted_joint_onset_frame": onset_predicted,
        "joint_onset_error_frames": (
            int(onset_predicted - onset_expected)
            if onset_expected is not None and onset_predicted is not None
            else None
        ),
        "expected_joint_end_frame": end_expected,
        "predicted_joint_end_frame": end_predicted,
        "joint_end_error_frames": (
            int(end_predicted - end_expected)
            if end_expected is not None and end_predicted is not None
            else None
        ),
        "longest_predicted_joint_run_frames": int(longest_predicted),
        "longest_predicted_joint_run_s": float(longest_predicted / fps),
        "longest_true_joint_run_frames": int(longest_true),
        "longest_expected_joint_run_frames": int(longest_expected),
        "longest_run_ratio": (
            float(longest_true / longest_expected) if longest_expected else None
        ),
        **continuity,
        "stable_rule": f"first run >= {STABLE_MIN_FRAMES} frames",
        "onset_sensitivity": sensitivity,
    }
    # Role-transition timing: per-robot stable grasp onset / release, and
    # whether the ORDER of events (who grasps first, who releases first) is
    # preserved relative to the source.
    role: dict[str, object] = {}
    onsets_exp: dict[int, int | None] = {}
    onsets_pred: dict[int, int | None] = {}
    release_exp: dict[int, int | None] = {}
    release_pred: dict[int, int | None] = {}
    for person in (1, 2):
        onsets_exp[person] = _safe_onset(signals.expected[person], STABLE_MIN_FRAMES)
        onsets_pred[person] = _safe_onset(signals.predicted[person], STABLE_MIN_FRAMES)
        release_exp[person] = _safe_end(
            signals.expected[person], onsets_exp[person], STABLE_MIN_FRAMES
        )
        release_pred[person] = _safe_end(
            signals.predicted[person], onsets_pred[person], STABLE_MIN_FRAMES
        )
        role[f"robot{person}_onset_expected"] = onsets_exp[person]
        role[f"robot{person}_onset_predicted"] = onsets_pred[person]
        role[f"robot{person}_onset_error_frames"] = (
            int(onsets_pred[person] - onsets_exp[person])
            if onsets_exp[person] is not None and onsets_pred[person] is not None
            else None
        )
        role[f"robot{person}_release_expected"] = release_exp[person]
        role[f"robot{person}_release_predicted"] = release_pred[person]
        role[f"robot{person}_release_error_frames"] = (
            int(release_pred[person] - release_exp[person])
            if release_exp[person] is not None and release_pred[person] is not None
            else None
        )

    def _order(a: int | None, b: int | None) -> int | None:
        if a is None or b is None:
            return None
        return int(np.sign(a - b))

    role["grasp_order_expected"] = _order(onsets_exp[1], onsets_exp[2])
    role["grasp_order_predicted"] = _order(onsets_pred[1], onsets_pred[2])
    role["release_order_expected"] = _order(release_exp[1], release_exp[2])
    role["release_order_predicted"] = _order(release_pred[1], release_pred[2])
    role["grasp_order_preserved"] = (
        role["grasp_order_expected"] == role["grasp_order_predicted"]
        if role["grasp_order_expected"] is not None
        and role["grasp_order_predicted"] is not None
        else None
    )
    role["release_order_preserved"] = (
        role["release_order_expected"] == role["release_order_predicted"]
        if role["release_order_expected"] is not None
        and role["release_order_predicted"] is not None
        else None
    )
    # Handover lag: time between the two robots' grasp onsets (source vs robot).
    if None not in (onsets_exp[1], onsets_exp[2], onsets_pred[1], onsets_pred[2]):
        lag_exp = abs(onsets_exp[1] - onsets_exp[2])
        lag_pred = abs(onsets_pred[1] - onsets_pred[2])
        role["onset_lag_expected_frames"] = int(lag_exp)
        role["onset_lag_predicted_frames"] = int(lag_pred)
        role["onset_lag_error_frames"] = int(lag_pred - lag_exp)
    else:
        role["onset_lag_error_frames"] = None
    timing["role_transition"] = role

    # ---- C. bilateral object-relative anchor error -------------------------------
    eval_frames = np.flatnonzero(expected_joint)
    robot_error_cm = {1: [], 2: []}
    for t in eval_frames:
        for person in (1, 2):
            active = signals.expected_hand[person][t]
            if not active.any():
                robot_error_cm[person].append(np.nan)
                continue
            errors = np.linalg.norm(
                signals.robot_hand_local[person][t, active]
                - signals.source_anchor_local[person][t, active],
                axis=-1,
            )
            robot_error_cm[person].append(float(errors.mean() * 100.0))
    e1 = np.asarray(robot_error_cm[1])
    e2 = np.asarray(robot_error_cm[2])
    valid = ~np.isnan(e1) & ~np.isnan(e2)
    if valid.any():
        worse = np.maximum(e1[valid], e2[valid])
        better = np.minimum(e1[valid], e2[valid])
        bilateral = {
            "evaluated_frames": int(valid.sum()),
            "bilateral_max_error_cm": float(worse.mean()),
            "bilateral_min_error_cm": float(better.mean()),
            "robot1_error_cm": float(np.nanmean(e1[valid])),
            "robot2_error_cm": float(np.nanmean(e2[valid])),
            "asymmetry_cm": float((worse - better).mean()),
            "robot1_is_worse_fraction": float((e1[valid] >= e2[valid]).mean()),
            "bilateral_max_error_p95_cm": float(np.percentile(worse, 95)),
        }
    else:
        bilateral = {"evaluated_frames": 0}

    # ---- D. geometry-valid joint contact (geometric proxy) -------------------------
    geometry = {}
    for tau in PENETRATION_THRESHOLDS_CM:
        ok1 = signals.hand_penetration_cm[1] <= tau + 1e-9
        ok2 = signals.hand_penetration_cm[2] <= tau + 1e-9
        valid_joint = predicted_joint & ok1 & ok2
        stats = _overlap_stats(expected_joint, valid_joint)
        key = f"tau_{tau:g}cm"
        geometry[key] = {
            "threshold_cm": tau,
            "valid_joint_frames": int(valid_joint.sum()),
            "coverage_of_expected_joint": stats["recall"],
            "f1": stats["f1"],
            "iou": stats["iou"],
            "precision": stats["precision"],
            "predicted_joint_frames_invalidated": int((predicted_joint & ~valid_joint).sum()),
            "robot1_valid_fraction_during_predicted_joint": (
                float(ok1[predicted_joint].mean()) if predicted_joint.any() else None
            ),
            "robot2_valid_fraction_during_predicted_joint": (
                float(ok2[predicted_joint].mean()) if predicted_joint.any() else None
            ),
            # Longest uninterrupted geometry-valid simultaneous-contact span.
            "longest_valid_joint_run_frames": int(_longest_run(valid_joint)),
            "longest_valid_true_joint_run_frames": int(
                _longest_run(valid_joint & expected_joint)
            ),
            "longest_valid_true_run_ratio": (
                float(_longest_run(valid_joint & expected_joint) / longest_expected)
                if longest_expected
                else None
            ),
        }
    inter_robot: dict[str, object] = {}
    inter_distance = signals.extras.get("inter_robot_min_distance_m")
    if isinstance(inter_distance, np.ndarray):
        inter_robot = {
            "min_keypoint_distance_m": float(inter_distance.min()),
            "mean_keypoint_distance_m": float(inter_distance.mean()),
            "frames_below_0p12m": int((inter_distance < 0.12).sum()),
            "frames_below_0p05m": int((inter_distance < 0.05).sum()),
            "min_during_expected_joint_m": (
                float(inter_distance[expected_joint].min()) if expected_joint.any() else None
            ),
        }
    penetration_summary = {
        str(person): {
            "max_cm": float(signals.hand_penetration_cm[person].max()),
            "mean_during_predicted_joint_cm": (
                float(signals.hand_penetration_cm[person][predicted_joint].mean())
                if predicted_joint.any()
                else None
            ),
        }
        for person in (1, 2)
    }

    phase_counts = {
        name: int((signals.phases == index).sum()) for index, name in enumerate(PHASE_NAMES)
    }
    return {
        "sequence_id": signals.sequence_id,
        "label": signals.label,
        "fps": fps,
        "frames": frames,
        "contract": {
            "source_contact_hysteresis_m": [SOURCE_CONTACT_ENTER_M, SOURCE_CONTACT_EXIT_M],
            "robot_contact_hysteresis_m": [ROBOT_CONTACT_ENTER_M, ROBOT_CONTACT_EXIT_M],
            "robot_contact_state": "hysteresis on min over both rubber-hand sample sets",
            "stable_min_frames": STABLE_MIN_FRAMES,
        },
        "expected_joint_frames": int(expected_joint.sum()),
        "predicted_joint_frames": int(predicted_joint.sum()),
        "expected_joint_frames_in_phase4": int((expected_joint & (signals.phases == 4)).sum()),
        "phase_counts": phase_counts,
        "joint_contact": overlap,
        "per_robot_contact": per_robot,
        "timing": timing,
        "bilateral": bilateral,
        "geometry_valid": geometry,
        "hand_penetration": penetration_summary,
        "inter_robot": inter_robot,
        "result_root": signals.extras.get("result_root"),
    }


def flatten_metrics(metrics: dict[str, object]) -> dict[str, object]:
    """Flatten the nested metrics into one CSV row."""
    jc = metrics["joint_contact"]
    tm = metrics["timing"]
    bl = metrics["bilateral"]
    gv = metrics["geometry_valid"]
    pr = metrics["per_robot_contact"]
    row: dict[str, object] = {
        "label": metrics["label"],
        "sequence_id": metrics["sequence_id"],
        "frames": metrics["frames"],
        "expected_joint_frames": metrics["expected_joint_frames"],
        "predicted_joint_frames": metrics["predicted_joint_frames"],
        "joint_precision": jc["precision"],
        "joint_recall": jc["recall"],
        "joint_f1": jc["f1"],
        "joint_iou": jc["iou"],
        "joint_coverage": jc["recall"],
        "robot1_f1": pr["1"]["f1"],
        "robot2_f1": pr["2"]["f1"],
        "joint_onset_expected": tm["expected_joint_onset_frame"],
        "joint_onset_predicted": tm["predicted_joint_onset_frame"],
        "joint_onset_error_frames": tm["joint_onset_error_frames"],
        "joint_end_expected": tm["expected_joint_end_frame"],
        "joint_end_predicted": tm["predicted_joint_end_frame"],
        "joint_end_error_frames": tm["joint_end_error_frames"],
        "longest_predicted_joint_run_frames": tm["longest_predicted_joint_run_frames"],
        "longest_true_joint_run_frames": tm["longest_true_joint_run_frames"],
        "longest_expected_joint_run_frames": tm["longest_expected_joint_run_frames"],
        "longest_run_ratio": tm["longest_run_ratio"],
        "dropout_events": tm["dropout_events"],
        "dropout_frames_total": tm.get("dropout_frames_total"),
        "maintained_fraction_in_window": tm["maintained_fraction_in_window"],
        "bilateral_max_error_cm": bl.get("bilateral_max_error_cm"),
        "bilateral_min_error_cm": bl.get("bilateral_min_error_cm"),
        "robot1_anchor_error_cm": bl.get("robot1_error_cm"),
        "robot2_anchor_error_cm": bl.get("robot2_error_cm"),
        "anchor_asymmetry_cm": bl.get("asymmetry_cm"),
        "robot1_is_worse_fraction": bl.get("robot1_is_worse_fraction"),
        "robot1_hand_pen_max_cm": metrics["hand_penetration"]["1"]["max_cm"],
        "robot2_hand_pen_max_cm": metrics["hand_penetration"]["2"]["max_cm"],
    }
    for key, values in gv.items():
        row[f"valid_joint_coverage_{key}"] = values["coverage_of_expected_joint"]
        row[f"valid_joint_f1_{key}"] = values["f1"]
        row[f"valid_joint_iou_{key}"] = values["iou"]
        row[f"longest_valid_joint_run_frames_{key}"] = values.get("longest_valid_joint_run_frames")
        row[f"longest_valid_true_joint_run_frames_{key}"] = values.get(
            "longest_valid_true_joint_run_frames"
        )
    role = tm.get("role_transition", {})
    for key in (
        "robot1_onset_error_frames",
        "robot2_onset_error_frames",
        "robot1_release_error_frames",
        "robot2_release_error_frames",
        "grasp_order_preserved",
        "release_order_preserved",
        "onset_lag_error_frames",
    ):
        row[key] = role.get(key)
    inter = metrics.get("inter_robot", {})
    row["inter_robot_min_keypoint_distance_m"] = inter.get("min_keypoint_distance_m")
    row["inter_robot_frames_below_0p12m"] = inter.get("frames_below_0p12m")
    return row


def evaluate_result(
    sequence_root: Path,
    sequence_id: str,
    object_models_root: Path,
    contact_analysis_root: Path,
    result_root: Path,
    *,
    label: str,
    fps: float = 15.0,
    urdf_path: Path = DEFAULT_URDF,
) -> tuple[dict[str, object], SequenceSignals]:
    stem = sequence_id.replace("/", "_")
    signals = load_sequence_signals(
        sequence_root,
        sequence_id,
        object_models_root,
        contact_analysis_root / f"{stem}.npz",
        result_root,
        label=label,
        fps=fps,
        urdf_path=urdf_path,
    )
    return coordination_metrics(signals), signals


def write_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n", "utf-8")


def _json_default(value: object) -> object:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Unserializable: {type(value)!r}")

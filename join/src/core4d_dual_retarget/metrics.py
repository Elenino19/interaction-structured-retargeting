"""Kinematic and contact metrics for independent retargeting baselines."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from .audit import load_npz_mapping
from .contact import first_stable_contact, hysteresis_contact, resolve_object_model
from .coordinates import points_robot_to_core4d
from .retarget.balance import (
    COLLISION_KEYPOINT_NAMES,
    HAND_KEYPOINT_NAMES,
    LEFT_FOOT_VOLUME_OFFSETS_M,
    LEFT_HAND_VOLUME_OFFSETS_M,
    RIGHT_FOOT_VOLUME_OFFSETS_M,
    RIGHT_HAND_VOLUME_OFFSETS_M,
    aabb_from_vertices,
    aabb_inside_depth,
    com_mass_vector,
    human_foot_contact,
    numpy_apply_root_yaw,
    numpy_transform_offsets,
    points_in_object_frame,
    support_excess_xy,
    weighted_com,
)

PHUMA_NON_SKATING_CM_S = 10.0
NEAR_GROUND_HEIGHT_M = 0.02
SOLE_FLOAT_TOL_M = 0.01
HAND_CONTACT_FLUSH_CM = 1.0
PENETRATION_PUBLISH_CM = 1.0
GEOMETRY_KEYS = (
    "planted_foot_speed_mean_cm_s",
    "planted_foot_speed_p50_cm_s",
    "planted_foot_speed_p95_cm_s",
    "non_skating_frame_frac",
    "sole_below_ground_max_cm",
    "sole_below_ground_mean_cm",
    "sole_zmin_cm",
    "sole_float_mean_cm",
    "hand_in_box_max_cm",
    "foot_in_box_max_cm",
    "hand_in_hull_max_cm",
    "foot_in_hull_max_cm",
    "mean_palm_distance_during_expected_contact_cm",
)


def binary_f1(expected: np.ndarray, predicted: np.ndarray) -> float:
    """Compute binary F1 with a well-defined perfect empty/empty case."""
    expected = np.asarray(expected, dtype=bool)
    predicted = np.asarray(predicted, dtype=bool)
    true_positive = int(np.logical_and(expected, predicted).sum())
    false_positive = int(np.logical_and(~expected, predicted).sum())
    false_negative = int(np.logical_and(expected, ~predicted).sum())
    denominator = 2 * true_positive + false_positive + false_negative
    return 1.0 if denominator == 0 else 2 * true_positive / denominator


def derivative_rms(values: np.ndarray, *, order: int, fps: float) -> float:
    """RMS finite-difference derivative in source units per second**order."""
    derivative = np.diff(np.asarray(values, dtype=np.float64), n=order, axis=0)
    derivative *= fps**order
    return float(np.sqrt(np.mean(derivative**2)))


def shin_xy_intersection_frames(
    robot_keypoints: np.ndarray,
    keypoint_names: list[str],
) -> int:
    """Count frames where left/right knee-to-ankle segments intersect in XY."""
    required = ("left_knee", "left_ankle", "right_knee", "right_ankle")
    if any(name not in keypoint_names for name in required):
        return 0
    lk, la, rk, ra = (keypoint_names.index(name) for name in required)
    a = np.asarray(robot_keypoints[:, lk, :2], dtype=np.float64)
    b = np.asarray(robot_keypoints[:, la, :2], dtype=np.float64)
    c = np.asarray(robot_keypoints[:, rk, :2], dtype=np.float64)
    d = np.asarray(robot_keypoints[:, ra, :2], dtype=np.float64)

    def orientation(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> np.ndarray:
        return (q[:, 0] - p[:, 0]) * (r[:, 1] - p[:, 1]) - (
            q[:, 1] - p[:, 1]
        ) * (r[:, 0] - p[:, 0])

    o1 = orientation(a, b, c)
    o2 = orientation(a, b, d)
    o3 = orientation(c, d, a)
    o4 = orientation(c, d, b)
    eps = 1e-9
    intersects = (o1 * o2 < -eps) & (o3 * o4 < -eps)
    return int(intersects.sum())


def robot_palm_distances_to_object(
    robot_keypoints: np.ndarray,
    keypoint_names: np.ndarray,
    object_poses: np.ndarray,
    object_vertices: np.ndarray,
) -> np.ndarray:
    """Return left/right G1 palm-origin distances to the CORE4D object mesh."""
    names = [str(name) for name in keypoint_names]
    indices = [names.index("left_wrist"), names.index("right_wrist")]
    palms_core4d = points_robot_to_core4d(robot_keypoints[:, indices])
    rotations = object_poses[:, :3, :3]
    translations = object_poses[:, :3, 3]
    local_palms = np.einsum(
        "tki,tij->tkj",
        palms_core4d - translations[:, None],
        rotations,
    )
    tree = cKDTree(object_vertices)
    distances, _ = tree.query(local_palms.reshape(-1, 3), k=1)
    return distances.reshape(len(robot_keypoints), 2)


def _result_config(result: np.lib.npyio.NpzFile) -> dict[str, object]:
    if "config_json" not in result.files:
        return {}
    raw = result["config_json"]
    text = raw.item() if hasattr(raw, "item") else str(raw)
    try:
        parsed = json.loads(str(text))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def rubber_hand_world_volume_points(
    urdf_path: Path,
    joint_positions: np.ndarray,
    root_translation: np.ndarray,
    root_yaw: np.ndarray,
) -> np.ndarray:
    """Return (T, 2N, 3) rubber-hand mesh samples in the robot world frame."""
    return robot_link_volume_world_points(
        urdf_path,
        joint_positions,
        root_translation,
        root_yaw,
        (
            ("left_rubber_hand", np.asarray(LEFT_HAND_VOLUME_OFFSETS_M)),
            ("right_rubber_hand", np.asarray(RIGHT_HAND_VOLUME_OFFSETS_M)),
        ),
    )


def rubber_foot_world_volume_points(
    urdf_path: Path,
    joint_positions: np.ndarray,
    root_translation: np.ndarray,
    root_yaw: np.ndarray,
) -> np.ndarray:
    """Return (T, 2N, 3) ankle-roll mesh samples in the robot world frame."""
    return robot_link_volume_world_points(
        urdf_path,
        joint_positions,
        root_translation,
        root_yaw,
        (
            ("left_ankle_roll_link", np.asarray(LEFT_FOOT_VOLUME_OFFSETS_M)),
            ("right_ankle_roll_link", np.asarray(RIGHT_FOOT_VOLUME_OFFSETS_M)),
        ),
    )


def robot_hand_volume_distances_to_object(
    volume_world: np.ndarray,
    object_poses: np.ndarray,
    object_vertices: np.ndarray,
) -> np.ndarray:
    """Min unsigned distance from each hand's mesh samples to the object."""
    n_left = len(LEFT_HAND_VOLUME_OFFSETS_M)
    core4d = points_robot_to_core4d(volume_world)
    rotations = object_poses[:, :3, :3]
    translations = object_poses[:, :3, 3]
    local = np.einsum("tki,tij->tkj", core4d - translations[:, None], rotations)
    tree = cKDTree(object_vertices)
    distances, _ = tree.query(local.reshape(-1, 3), k=1)
    distances = distances.reshape(len(volume_world), -1)
    left = distances[:, :n_left].min(axis=1)
    right = distances[:, n_left:].min(axis=1)
    return np.stack([left, right], axis=1)


def robot_link_volume_world_points(
    urdf_path: Path,
    joint_positions: np.ndarray,
    root_translation: np.ndarray,
    root_yaw: np.ndarray,
    links_and_offsets: tuple[tuple[str, np.ndarray], ...],
) -> np.ndarray:
    """Return (T, sum N_i, 3) samples for named links in the robot world frame."""
    import pytorch_kinematics as pk
    import torch

    chain = pk.build_chain_from_urdf(urdf_path.read_text()).to(dtype=torch.float64)
    transforms = chain.forward_kinematics(torch.tensor(joint_positions, dtype=torch.float64))
    chunks = [
        numpy_transform_offsets(
            transforms[link].get_matrix().detach().cpu().numpy(),
            np.asarray(offsets, dtype=np.float64),
        )
        for link, offsets in links_and_offsets
    ]
    return numpy_apply_root_yaw(
        np.concatenate(chunks, axis=1),
        np.asarray(root_translation),
        np.asarray(root_yaw),
    )


def planted_slip_cm_s(
    robot_keypoints: np.ndarray,
    keypoint_names: list[str],
    root_translation: np.ndarray,
    source_joints: np.ndarray,
    *,
    fps: float,
) -> tuple[float | None, float | None]:
    """Mean planted ankle / dual-stance root XY speed in cm/s."""
    contact = human_foot_contact(np.asarray(source_joints))
    left = keypoint_names.index("left_ankle")
    right = keypoint_names.index("right_ankle")
    ankle_speed = (
        np.linalg.norm(np.diff(robot_keypoints[:, [left, right], :2], axis=0), axis=-1)
        * fps
        * 100
    )
    planted = contact[1:]
    ankle = float(ankle_speed[planted].mean()) if planted.any() else None
    both = contact[1:, 0] & contact[1:, 1]
    root_speed = np.linalg.norm(np.diff(root_translation[:, :2], axis=0), axis=-1) * fps * 100
    root = float(root_speed[both].mean()) if both.any() else None
    return ankle, root


def planted_foot_geometry(
    foot_world: np.ndarray,
    *,
    fps: float,
    near_m: float = NEAR_GROUND_HEIGHT_M,
    skate_cm_s: float = PHUMA_NON_SKATING_CM_S,
) -> dict[str, float | None]:
    """PHUMA-style sole height and contact-foot horizontal speed from foot grids."""
    n_left = len(LEFT_FOOT_VOLUME_OFFSETS_M)
    left = foot_world[:, :n_left]
    right = foot_world[:, n_left:]
    zmin = np.stack([left[..., 2].min(axis=1), right[..., 2].min(axis=1)], axis=1)
    centroids = np.stack([left.mean(axis=1), right.mean(axis=1)], axis=1)
    speed = np.linalg.norm(np.diff(centroids[:, :, :2], axis=0), axis=-1) * fps * 100
    planted = zmin[1:] < near_m
    if planted.any():
        speeds = speed[planted]
        mean = float(speeds.mean())
        p50 = float(np.median(speeds))
        p95 = float(np.percentile(speeds, 95))
        frac = float((speeds < skate_cm_s).mean())
    else:
        mean = p50 = p95 = frac = None
    below = np.maximum(-zmin, 0.0) * 100
    support_z = zmin.min(axis=1)
    float_cm = np.maximum(support_z - SOLE_FLOAT_TOL_M, 0.0) * 100
    return {
        "planted_foot_speed_mean_cm_s": mean,
        "planted_foot_speed_p50_cm_s": p50,
        "planted_foot_speed_p95_cm_s": p95,
        "non_skating_frame_frac": frac,
        "sole_below_ground_max_cm": float(below.max()),
        "sole_below_ground_mean_cm": (
            float(below[below > 0].mean()) if np.any(below > 0) else 0.0
        ),
        "sole_zmin_cm": float(zmin.min() * 100),
        "sole_float_mean_cm": float(float_cm.mean()),
    }


def aabb_inside_max_mean_cm(
    world_points: np.ndarray,
    object_poses: np.ndarray,
    vertices: np.ndarray,
) -> tuple[float, float]:
    depth = aabb_inside_depth(
        points_in_object_frame(points_robot_to_core4d(world_points), object_poses),
        *aabb_from_vertices(np.asarray(vertices)),
        0.0,
    )
    return float(depth.max() * 100), float(depth.mean() * 100)


def hull_inside_max_cm(
    world_points: np.ndarray,
    object_poses: np.ndarray,
    vertices: np.ndarray,
) -> float:
    from scipy.spatial import ConvexHull

    local = points_in_object_frame(points_robot_to_core4d(world_points), object_poses)
    pts = local.reshape(-1, 3)
    hull = ConvexHull(np.asarray(vertices, dtype=np.float64))
    values = pts @ hull.equations[:, :3].T + hull.equations[:, 3]
    # scipy: inside iff all half-spaces are <= 0; -max(values) is plane depth.
    inside_depth = np.maximum(-values.max(axis=1), 0.0)
    return float(inside_depth.max() * 100)


def apply_sole_lift_to_root(
    root_translation: np.ndarray,
    foot_zmin: np.ndarray,
    *,
    min_z: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Raise root z per frame so the lowest sole sample is at least ``min_z``."""
    lift = np.maximum(min_z - np.asarray(foot_zmin), 0.0)
    out = np.array(root_translation, copy=True, dtype=np.float64)
    out[:, 2] += lift
    return out, lift


def write_geometry_report(metrics_root: Path, output_path: Path) -> dict[str, object]:
    """Flatten per-robot literature geometry metrics into one JSON table."""
    rows: list[dict[str, object]] = []
    for path in sorted(metrics_root.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for person, robot in payload["robots"].items():
            row = {
                "sequence_id": payload["sequence_id"],
                "robot": person,
                "contact_f1": robot.get("contact_f1"),
                "ankle_height_mae_m": robot.get("ankle_height_mae_m"),
            }
            for key in GEOMETRY_KEYS:
                row[key] = robot.get(key)
            rows.append(row)
    report = {
        "metrics_root": str(metrics_root),
        "publish_bar": {
            "planted_foot_speed_mean_cm_s": PHUMA_NON_SKATING_CM_S,
            "sole_below_ground_max_cm": PENETRATION_PUBLISH_CM,
            "hand_in_box_max_cm": PENETRATION_PUBLISH_CM,
            "foot_in_box_max_cm": PENETRATION_PUBLISH_CM,
            "mean_palm_distance_during_expected_contact_cm": HAND_CONTACT_FLUSH_CM,
        },
        "rows": rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def evaluate_independent_sequence(
    sequence_root: Path,
    sequence_id: str,
    object_models_root: Path,
    contact_analysis_path: Path,
    result_root: Path,
    *,
    fps: float = 15.0,
    robot_contact_enter: float = 0.08,
    robot_contact_exit: float = 0.10,
    paper_metric_mode: bool = False,
    reference_root: Path | None = None,
    urdf_path: Path = Path(
        "third_party/unitree_ros/robots/g1_description/g1_29dof_rev_1_0.urdf"
    ),
) -> dict[str, object]:
    """Evaluate both independently retargeted robots against source contacts."""
    sequence_dir = sequence_root.joinpath(*sequence_id.split("/"))
    object_pose_path = sequence_dir / "smooth_objposes_so3.npy"
    if not object_pose_path.exists():
        object_pose_path = sequence_dir / "smooth_objposes.npy"
    object_poses = np.load(object_pose_path)
    object_model = resolve_object_model(sequence_dir, object_models_root)
    mesh = trimesh.load_mesh(object_model, process=False)
    vertices = np.asarray(mesh.vertices)
    with np.load(contact_analysis_path) as contact_data:
        expected_contacts = {
            1: contact_data["person1_contact"].astype(bool),
            2: contact_data["person2_contact"].astype(bool),
        }

    payload: dict[str, object] = {
        "sequence_id": sequence_id,
        "fps": fps,
        "robot_contact_thresholds_m": {
            "enter": robot_contact_enter,
            "exit": robot_contact_exit,
        },
        "paper_metric_mode": paper_metric_mode,
        "robots": {},
    }
    robots: dict[str, object] = {}
    for person in (1, 2):
        result_path = result_root / sequence_id.replace("/", "_") / f"robot{person}.npz"
        with np.load(result_path) as result:
            robot_keypoints = result["robot_keypoints"]
            target_keypoints = result["target_keypoints"]
            joint_positions = result["joint_positions"]
            keypoint_names = [str(name) for name in result["keypoint_names"]]
            root_yaw = result["root_yaw"]
            root_translation = result["root_translation"]
            cfg = _result_config(result)
            use_hand_volume = paper_metric_mode or (
                float(cfg.get("hand_volume_clearance_weight", 0.0) or 0.0) > 0
            )
            enter = robot_contact_enter
            exit_ = robot_contact_exit
            volume_world = rubber_hand_world_volume_points(
                urdf_path,
                joint_positions,
                root_translation,
                root_yaw,
            )
            foot_world = rubber_foot_world_volume_points(
                urdf_path,
                joint_positions,
                root_translation,
                root_yaw,
            )
            if use_hand_volume:
                palm_distances = robot_hand_volume_distances_to_object(
                    volume_world,
                    object_poses,
                    vertices,
                )
                hand_penetration = float(
                    aabb_inside_depth(
                        points_in_object_frame(
                            points_robot_to_core4d(volume_world),
                            object_poses,
                        ),
                        *aabb_from_vertices(vertices),
                        0.0,
                    ).mean()
                    * 100
                )
                if not paper_metric_mode and enter >= 0.08 and exit_ >= 0.10:
                    enter = 0.04
                    exit_ = 0.055
            else:
                palm_distances = robot_palm_distances_to_object(
                    robot_keypoints,
                    np.asarray(keypoint_names),
                    object_poses,
                    vertices,
                )
                hand_penetration = None
            geometry = planted_foot_geometry(foot_world, fps=fps)
            hand_in_max, _ = aabb_inside_max_mean_cm(volume_world, object_poses, vertices)
            foot_in_max, _ = aabb_inside_max_mean_cm(foot_world, object_poses, vertices)
            geometry["hand_in_box_max_cm"] = hand_in_max
            geometry["foot_in_box_max_cm"] = foot_in_max
            geometry["hand_in_hull_max_cm"] = hull_inside_max_cm(
                volume_world, object_poses, vertices
            )
            geometry["foot_in_hull_max_cm"] = hull_inside_max_cm(
                foot_world, object_poses, vertices
            )
        minimum_distance = palm_distances.min(axis=1)
        predicted_contact = hysteresis_contact(
            minimum_distance,
            enter_threshold=enter,
            exit_threshold=exit_,
        )
        expected = expected_contacts[person]
        try:
            expected_onset = first_stable_contact(expected)
        except ValueError:
            expected_onset = None
        try:
            predicted_onset = first_stable_contact(predicted_contact)
        except ValueError:
            predicted_onset = None
        if expected_onset is None or predicted_onset is None:
            onset_error_frames = None
        else:
            onset_error_frames = int(predicted_onset - expected_onset)

        expected_distances = minimum_distance[expected]
        left_ankle = keypoint_names.index("left_ankle")
        right_ankle = keypoint_names.index("right_ankle")
        contact_all = np.ones(len(robot_keypoints), dtype=bool)
        source_path = sequence_dir / f"person{person}_poses.npz"
        if source_path.exists():
            planted_ankle, planted_root = planted_slip_cm_s(
                robot_keypoints,
                keypoint_names,
                root_translation,
                np.asarray(load_npz_mapping(source_path)["joints"]),
                fps=fps,
            )
        else:
            planted_ankle, planted_root = None, None
        root_error_cm = None
        if reference_root is not None:
            reference_path = (
                reference_root / sequence_id.replace("/", "_") / f"robot{person}.npz"
            )
            if reference_path.exists():
                with np.load(reference_path) as reference:
                    reference_translation = np.asarray(
                        reference["root_translation"], dtype=np.float64
                    )
                count = min(len(root_translation), len(reference_translation))
                root_error_cm = float(
                    np.linalg.norm(
                        root_translation[:count] - reference_translation[:count],
                        axis=-1,
                    ).mean()
                    * 100
                )
        unwrapped_yaw = np.unwrap(np.asarray(root_yaw, dtype=np.float64))
        robots[str(person)] = {
            "semantic_keypoint_error_cm": float(
                np.linalg.norm(robot_keypoints - target_keypoints, axis=-1).mean() * 100
            ),
            "contact_f1": binary_f1(expected, predicted_contact),
            "expected_contact_onset_frame": expected_onset,
            "predicted_contact_onset_frame": predicted_onset,
            "contact_onset_error_frames": onset_error_frames,
            "mean_palm_distance_during_expected_contact_cm": (
                float(expected_distances.mean() * 100) if len(expected_distances) else None
            ),
            "minimum_palm_distance_cm": float(minimum_distance.min() * 100),
            "joint_velocity_rms_rad_s": derivative_rms(
                joint_positions, order=1, fps=fps
            ),
            "joint_acceleration_rms_rad_s2": derivative_rms(
                joint_positions, order=2, fps=fps
            ),
            "joint_jerk_rms_rad_s3": derivative_rms(
                joint_positions, order=3, fps=fps
            ),
            "root_velocity_rms_m_s": derivative_rms(
                root_translation, order=1, fps=fps
            ),
            "root_acceleration_rms_m_s2": derivative_rms(
                root_translation, order=2, fps=fps
            ),
            "root_jerk_rms_m_s3": derivative_rms(
                root_translation, order=3, fps=fps
            ),
            "root_yaw_jerk_rms_rad_s3": derivative_rms(
                unwrapped_yaw, order=3, fps=fps
            ),
            "root_trajectory_error_cm": root_error_cm,
            "com_support_excess_mean_m": float(
                support_excess_xy(
                    weighted_com(robot_keypoints, com_mass_vector(keypoint_names))[:, :2],
                    robot_keypoints[:, left_ankle, :2],
                    robot_keypoints[:, right_ankle, :2],
                    contact_all,
                    contact_all,
                    root_yaw,
                ).mean()
            ),
            "ankle_height_mae_m": float(
                np.abs(robot_keypoints[:, [left_ankle, right_ankle], 2] - 0.055).mean()
            ),
            "planted_ankle_slip_mean_cm_s": planted_ankle,
            "planted_root_slip_mean_cm_s": planted_root,
            "body_object_penetration_mean_cm": float(
                aabb_inside_depth(
                    points_in_object_frame(
                        points_robot_to_core4d(
                            robot_keypoints[
                                :,
                                [
                                    keypoint_names.index(name)
                                    for name in COLLISION_KEYPOINT_NAMES
                                    if name in keypoint_names
                                ],
                            ]
                        ),
                        object_poses,
                    ),
                    *aabb_from_vertices(vertices),
                    0.12,
                ).mean()
                * 100
            ),
            "hand_object_penetration_mean_cm": (
                hand_penetration
                if hand_penetration is not None
                else float(
                    aabb_inside_depth(
                        points_in_object_frame(
                            points_robot_to_core4d(
                                robot_keypoints[
                                    :,
                                    [
                                        keypoint_names.index(name)
                                        for name in HAND_KEYPOINT_NAMES
                                        if name in keypoint_names
                                    ],
                                ]
                            ),
                            object_poses,
                        ),
                        *aabb_from_vertices(vertices),
                        0.06,
                    ).mean()
                    * 100
                )
            ),
            "hand_object_penetration_max_cm": geometry["hand_in_hull_max_cm"],
            "foot_object_penetration_max_cm": geometry["foot_in_hull_max_cm"],
            "leg_xy_intersection_frames": shin_xy_intersection_frames(
                robot_keypoints, keypoint_names
            ),
            **geometry,
        }
    payload["robots"] = robots
    return payload


def write_metrics(metrics: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

"""Export kinematic dual-G1 trajectories as residual-control observations."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .coordinates import poses_core4d_to_robot
from .dataset import BOX_SEQUENCES, GENERALIZATION_SEQUENCES, SPLIT


def trajectory_to_observations(
    robot: dict[str, np.ndarray],
    *,
    source_fps: float = 15.0,
) -> dict[str, np.ndarray]:
    joints = np.asarray(robot["joint_positions"], dtype=np.float64)
    roots = np.asarray(robot["root_translation"], dtype=np.float64)
    yaws = np.asarray(robot["root_yaw"], dtype=np.float64)
    dt = 1.0 / source_fps
    return {
        "joint_positions": joints,
        "joint_velocities": np.gradient(joints, dt, axis=0),
        "root_translation": roots,
        "root_velocity": np.gradient(roots, dt, axis=0),
        "root_yaw": yaws,
        "yaw_velocity": np.gradient(yaws, dt),
        "contact_mask": np.asarray(robot["contact_mask"]),
        "contact_anchors": np.asarray(robot["contact_anchors"]),
        "robot_keypoints": np.asarray(robot["robot_keypoints"]),
        "target_keypoints": np.asarray(robot["target_keypoints"]),
        "joint_names": np.asarray(robot["joint_names"]),
        "keypoint_names": np.asarray(robot["keypoint_names"]),
    }


def export_residual_clip(
    sequence_id: str,
    motion_root: Path,
    result_root: Path,
    output_dir: Path,
    *,
    source_fps: float = 15.0,
) -> dict[str, object]:
    stem = sequence_id.replace("/", "_")
    sequence_dir = motion_root.joinpath(*sequence_id.split("/"))
    object_pose_path = sequence_dir / "smooth_objposes_so3.npy"
    if not object_pose_path.exists():
        object_pose_path = sequence_dir / "smooth_objposes.npy"
    object_poses = poses_core4d_to_robot(np.load(object_pose_path))
    payload: dict[str, np.ndarray] = {
        "sequence_id": np.asarray(sequence_id),
        "source_fps": np.asarray(source_fps),
        "object_poses": object_poses,
    }
    frames = []
    for person in (1, 2):
        with np.load(result_root / stem / f"robot{person}.npz") as raw:
            robot = {name: raw[name] for name in raw.files}
        obs = trajectory_to_observations(robot, source_fps=source_fps)
        frames.append(len(obs["joint_positions"]))
        for key, value in obs.items():
            payload[f"robot{person}_{key}"] = value
    n = min(frames + [len(object_poses)])
    payload["object_poses"] = object_poses[:n]
    for person in (1, 2):
        for key in (
            "joint_positions",
            "joint_velocities",
            "root_translation",
            "root_velocity",
            "root_yaw",
            "yaw_velocity",
            "contact_mask",
            "contact_anchors",
            "robot_keypoints",
            "target_keypoints",
        ):
            field = f"robot{person}_{key}"
            payload[field] = payload[field][:n]
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / f"{stem}.npz"
    np.savez_compressed(npz_path, **payload)
    return {
        "sequence_id": sequence_id,
        "split": SPLIT.get(sequence_id),
        "frames": n,
        "path": str(npz_path),
    }


def export_residual_dataset(
    motion_root: Path,
    result_root: Path,
    output_root: Path,
    *,
    include_holdout: bool = True,
) -> dict[str, object]:
    sequences = list(BOX_SEQUENCES)
    if include_holdout:
        sequences.extend(GENERALIZATION_SEQUENCES)
    clips = [
        export_residual_clip(sequence_id, motion_root, result_root, output_root)
        for sequence_id in sequences
    ]
    manifest = {
        "method": "balanced_smooth_residual_obs",
        "note": (
            "Kinematic observations only. Not a trained controller. "
            "Intended for residual RL / WBC tracking research."
        ),
        "clips": clips,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def load_residual_gains(path: Path) -> tuple[float, float, float, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    gains = payload["gains"]
    return (float(gains[0]), float(gains[1]), float(gains[2]), float(gains[3]))


def residual_hold_cost(metrics: dict[str, object], duration: float) -> float:
    robots = metrics["robots"]
    falls = []
    heights = []
    for person in ("1", "2"):
        fall = robots[person]["time_to_fall_s"]
        falls.append(duration if fall is None else float(fall))
        heights.append(float(robots[person]["minimum_pelvis_height_m"]))
    worst_fall = min(falls)
    if worst_fall < duration - 1e-6:
        return float(20.0 - worst_fall)
    return float(1.2 - min(heights) + 0.4 * max(0.45 - min(heights), 0.0))


def train_linear_stance_residual(
    scene_xml: Path,
    object_poses: np.ndarray,
    robot1_path: Path,
    robot2_path: Path,
    *,
    duration: float = 6.0,
    samples: int = 20,
    seed: int = 0,
    residual_joints: str = "ankle",
) -> dict[str, object]:
    """Random-search then Nelder-Mead a 4-gain IMU residual on a free-base hold."""
    from scipy.optimize import minimize

    from .pd_tracking import PDTrackingConfig, simulate_pd_tracking

    rng = np.random.default_rng(seed)
    evaluations: list[dict[str, object]] = []

    def evaluate(vector: np.ndarray) -> float:
        gains = tuple(float(value) for value in np.asarray(vector, dtype=np.float64)[:4])
        _, metrics = simulate_pd_tracking(
            scene_xml,
            object_poses,
            robot1_path,
            robot2_path,
            PDTrackingConfig(
                hold_first_frame_seconds=duration,
                control_mode="position",
                lift_feet=True,
                project_com=False,
                online_com=False,
                residual_gains=gains,
                residual_joints=residual_joints,
                leg_kp=200.0,
                arm_kp=40.0,
                waist_kp=60.0,
                com_kp=2.5,
                com_kd=0.25,
                com_mode="heuristic",
            ),
        )
        cost = residual_hold_cost(metrics, duration)
        evaluations.append(
            {
                "gains": list(gains),
                "cost": cost,
                "fall_s": [
                    metrics["robots"]["1"]["time_to_fall_s"],
                    metrics["robots"]["2"]["time_to_fall_s"],
                ],
                "min_z_m": [
                    metrics["robots"]["1"]["minimum_pelvis_height_m"],
                    metrics["robots"]["2"]["minimum_pelvis_height_m"],
                ],
            }
        )
        return cost

    best_vector = np.zeros(4, dtype=np.float64)
    best_cost = evaluate(best_vector)
    seeds = [
        np.array([0.0, 0.08, 0.0, 0.08]),
        np.array([0.0, -0.08, 0.0, -0.08]),
        np.array([0.15, 0.05, 0.15, 0.05]),
        np.array([-0.15, -0.05, -0.15, -0.05]),
        np.array([0.0, 0.12, 0.0, 0.0]),
        np.array([0.0, -0.12, 0.0, 0.0]),
    ]
    for seed_vector in seeds:
        cost = evaluate(seed_vector)
        if cost < best_cost - 1e-4:
            best_cost = cost
            best_vector = np.array(seed_vector, dtype=np.float64, copy=True)
    for scale in (0.05, 0.12, 0.25):
        for _ in range(max(samples // 3, 4)):
            candidate = rng.normal(0.0, scale, size=4)
            cost = evaluate(candidate)
            if cost < best_cost - 1e-4:
                best_cost = cost
                best_vector = np.array(candidate, copy=True)
    result = minimize(
        evaluate,
        best_vector,
        method="Nelder-Mead",
        options={"maxiter": 10, "maxfev": 16, "xatol": 0.02, "fatol": 0.02},
    )
    final = np.asarray(result.x, dtype=np.float64)
    final_cost = evaluate(final)
    if final_cost > best_cost:
        final = best_vector
        final_cost = best_cost
    ranked = sorted(evaluations, key=lambda row: float(row["cost"]))
    return {
        "gains": [float(value) for value in final],
        "cost": float(final_cost),
        "duration_s": duration,
        "nfev": len(evaluations),
        "best_trials": ranked[:8],
        "nm_success": bool(result.success),
        "nm_message": str(result.message),
        "residual_joints": residual_joints,
    }

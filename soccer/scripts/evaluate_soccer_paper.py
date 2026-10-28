#!/usr/bin/env python3
"""Create frozen paper-stage metrics from the existing soccer ablations."""

from __future__ import annotations

import sys as _fg_sys
from pathlib import Path as _FgPath
_fg_sys.path.insert(0, str(_FgPath(__file__).resolve().parent))
from frozen_guard import guard_release_writes


import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_ablation_experiments import (
    BALL_RADIUS_M,
    FLOOR_PROXY_Z_M,
    VARIANT_ORDER,
    evaluate_variant,
    load_robot,
    yup_to_robot,
)

FPS = 24.0
TIMING_WINDOW_RADIUS_FRAMES = 4
SUPPORT_BEFORE_FRAMES = 8
SUPPORT_AFTER_FRAMES = 2
PHASES = ("receive", "kick", "overall")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def event_phases(contacts: list[dict]) -> list[dict]:
    events = []
    for index, event in enumerate(contacts):
        if index > 0:
            events.append(
                {
                    "phase": "receive",
                    "player": event["player"],
                    "frame": int(event["frame"]),
                    "foot": event["side"],
                }
            )
        events.append(
            {
                "phase": "kick",
                "player": event["player"],
                "frame": int(event.get("kick_frame", event["frame"])),
                "foot": event.get("kick_side", event["side"]),
            }
        )
    return events


def _summary(samples: list[dict]) -> dict[str, float | int]:
    if not samples:
        return {"count": 0}
    return {
        "count": len(samples),
        "target_foot_distance_mean_m": float(
            np.mean([sample["target_foot_distance_m"] for sample in samples])
        ),
        "wrong_foot_distance_mean_m": float(
            np.mean([sample["wrong_foot_distance_m"] for sample in samples])
        ),
        "contact_surface_error_mean_m": float(
            np.mean([sample["contact_surface_error_m"] for sample in samples])
        ),
        "correct_foot_association_rate": float(
            np.mean([sample["correct_foot_association"] for sample in samples])
        ),
        "timing_error_mean_frames": float(
            np.mean([sample["timing_error_frames"] for sample in samples])
        ),
        "timing_error_mean_seconds": float(
            np.mean([sample["timing_error_seconds"] for sample in samples])
        ),
        "timing_error_p95_frames": float(
            np.percentile([sample["timing_error_frames"] for sample in samples], 95)
        ),
    }


def phase_contact_and_timing(
    robots: dict[str, dict[str, np.ndarray]],
    ball: np.ndarray,
    contacts: list[dict],
) -> dict:
    samples = []
    n = len(ball)
    for event in event_phases(contacts):
        robot = robots[event["player"]]
        names = list(robot["keypoint_names"])
        target_index = names.index(f"{event['foot']}_toe")
        wrong_foot = "right" if event["foot"] == "left" else "left"
        wrong_index = names.index(f"{wrong_foot}_toe")
        frame = event["frame"]
        target = robot["robot_keypoints"][frame, target_index]
        wrong = robot["robot_keypoints"][frame, wrong_index]
        target_distance = float(np.linalg.norm(ball[frame] - target))
        wrong_distance = float(np.linalg.norm(ball[frame] - wrong))

        lo = max(0, frame - TIMING_WINDOW_RADIUS_FRAMES)
        hi = min(n, frame + TIMING_WINDOW_RADIUS_FRAMES + 1)
        local_target = robot["robot_keypoints"][lo:hi, target_index]
        local_distance = np.linalg.norm(ball[lo:hi] - local_target, axis=1)
        predicted = lo + int(np.argmin(local_distance))
        timing_frames = abs(predicted - frame)
        samples.append(
            {
                **event,
                "predicted_interaction_frame": predicted,
                "timing_window_start": lo,
                "timing_window_end_inclusive": hi - 1,
                "timing_error_frames": timing_frames,
                "timing_error_seconds": timing_frames / FPS,
                "target_foot_distance_m": target_distance,
                "wrong_foot_distance_m": wrong_distance,
                "contact_surface_error_m": abs(target_distance - BALL_RADIUS_M),
                "correct_foot_association": bool(target_distance < wrong_distance),
            }
        )
    receive = [sample for sample in samples if sample["phase"] == "receive"]
    kick = [sample for sample in samples if sample["phase"] == "kick"]
    return {
        "definition": {
            "predicted_interaction": (
                "Earliest frame with minimum ball-center to annotated target-toe "
                "distance in the fixed annotated_frame +/- 4 frame window."
            ),
            "correct_foot_association": (
                "Target foot is strictly closer to the ball than the opposite foot "
                "at the annotated frame."
            ),
        },
        "receive": _summary(receive),
        "kick": _summary(kick),
        "overall": _summary(samples),
        "samples": samples,
    }


def support_and_floor_metrics(
    robots: dict[str, dict[str, np.ndarray]],
    contacts: list[dict],
) -> dict:
    event_displacement = []
    speeds = []
    floor_depth = []
    samples = []
    for player, robot in robots.items():
        names = list(robot["keypoint_names"])
        keypoints = robot["robot_keypoints"]
        for name in ("left_ankle", "left_toe", "right_ankle", "right_toe"):
            z = keypoints[:, names.index(name), 2]
            floor_depth.extend(np.maximum(FLOOR_PROXY_Z_M - z, 0.0).tolist())
        for event in contacts:
            if event["player"] != player:
                continue
            frame = int(event.get("kick_frame", event["frame"]))
            kick_foot = event.get("kick_side", event["side"])
            support_foot = "right" if kick_foot == "left" else "left"
            toe = keypoints[:, names.index(f"{support_foot}_toe"), :2]
            lo = max(0, frame - SUPPORT_BEFORE_FRAMES)
            hi = min(len(toe), frame + SUPPORT_AFTER_FRAMES + 1)
            step = np.linalg.norm(np.diff(toe[lo:hi], axis=0), axis=1)
            displacement = float(step.sum())
            event_displacement.append(displacement)
            speeds.extend((step * FPS).tolist())
            samples.append(
                {
                    "player": player,
                    "kick_frame": frame,
                    "support_foot": support_foot,
                    "interval_start": lo,
                    "interval_end_inclusive": hi - 1,
                    "horizontal_displacement_m": displacement,
                    "speed_mean_m_per_s": float(np.mean(step * FPS)),
                    "speed_max_m_per_s": float(np.max(step * FPS)),
                }
            )
    floor = np.asarray(floor_depth)
    speed = np.asarray(speeds)
    return {
        "definition": {
            "support_interval": "Annotated kick frame - 8 through kick frame + 2.",
            "support_foot": "Foot opposite the annotated kick foot.",
            "floor_proxy": (
                "Depth of ankle/toe keypoint below the fixed 0.012 m threshold."
            ),
        },
        "stance_foot_horizontal_displacement_mean_m": float(
            np.mean(event_displacement)
        ),
        "stance_foot_horizontal_displacement_p95_m": float(
            np.percentile(event_displacement, 95)
        ),
        "stance_foot_speed_mean_m_per_s": float(np.mean(speed)),
        "stance_foot_speed_p50_m_per_s": float(np.percentile(speed, 50)),
        "stance_foot_speed_p95_m_per_s": float(np.percentile(speed, 95)),
        "stance_foot_speed_max_m_per_s": float(np.max(speed)),
        "floor_penetration_keypoint_mean_m": float(np.mean(floor)),
        "floor_penetration_keypoint_max_m": float(np.max(floor)),
        "floor_penetration_keypoint_fraction": float(np.mean(floor > 0.0)),
        "samples": samples,
    }


def _vector_rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum(values**2, axis=-1))))


def _scalar_rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(values**2)))


def smoothness_metrics(
    robots: dict[str, dict[str, np.ndarray]],
) -> dict:
    root_velocity = []
    root_acceleration = []
    root_jerk = []
    joint_velocity = []
    joint_acceleration = []
    joint_jerk = []
    for robot in robots.values():
        root = robot["root_translation"]
        joints = robot["joint_positions"]
        root_velocity.append(np.diff(root, n=1, axis=0) * FPS)
        root_acceleration.append(np.diff(root, n=2, axis=0) * FPS**2)
        root_jerk.append(np.diff(root, n=3, axis=0) * FPS**3)
        joint_velocity.append(np.diff(joints, n=1, axis=0) * FPS)
        joint_acceleration.append(np.diff(joints, n=2, axis=0) * FPS**2)
        joint_jerk.append(np.diff(joints, n=3, axis=0) * FPS**3)
    return {
        "definition": (
            "Forward finite difference of order n multiplied by fps^n; root RMS "
            "uses Euclidean XYZ magnitude, joint RMS pools all frames and 29 DoF."
        ),
        "root_velocity_rms_m_per_s": _vector_rms(np.concatenate(root_velocity)),
        "root_acceleration_rms_m_per_s2": _vector_rms(
            np.concatenate(root_acceleration)
        ),
        "root_jerk_rms_m_per_s3": _vector_rms(np.concatenate(root_jerk)),
        "joint_velocity_rms_rad_per_s": _scalar_rms(
            np.concatenate(joint_velocity)
        ),
        "joint_acceleration_rms_rad_per_s2": _scalar_rms(
            np.concatenate(joint_acceleration)
        ),
        "joint_jerk_rms_rad_per_s3": _scalar_rms(np.concatenate(joint_jerk)),
    }


def mesh_metrics(
    variant_dir: Path,
    robots: dict[str, dict[str, np.ndarray]],
    ball: np.ndarray,
    contacts: list[dict],
) -> dict:
    try:
        import mujoco

        from isr_assets import ensure_join_on_path
        ensure_join_on_path()
        from core4d_dual_retarget.mujoco_scene import _set_robot_qpos
    except ImportError as exc:
        return {"available": False, "reason": f"MuJoCo import failed: {exc}"}

    scene = variant_dir / "soccer_scene.xml"
    if not scene.exists():
        return {"available": False, "reason": f"Missing scene: {scene}"}
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    foot_geoms: dict[tuple[str, str], int] = {}
    for player, prefix in (("a", "r1_"), ("b", "r2_")):
        for side in ("left", "right"):
            body_name = f"{prefix}{side}_ankle_roll_link"
            body_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, body_name
            )
            matches = [
                geom
                for geom in range(model.ngeom)
                if model.geom_bodyid[geom] == body_id
                and model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH
            ]
            if not matches:
                return {"available": False, "reason": f"No mesh geom for {body_name}"}
            foot_geoms[(player, side)] = matches[0]

    def world_vertices(geom: int) -> np.ndarray:
        mesh = int(model.geom_dataid[geom])
        start = int(model.mesh_vertadr[mesh])
        count = int(model.mesh_vertnum[mesh])
        local = model.mesh_vert[start : start + count]
        rotation = data.geom_xmat[geom].reshape(3, 3)
        return local @ rotation.T + data.geom_xpos[geom]

    event_lookup: dict[int, list[dict]] = {}
    for event in event_phases(contacts):
        event_lookup.setdefault(event["frame"], []).append(event)
    ground_depth = []
    contact_gap = []
    contact_samples = []
    n = min(len(robot["joint_positions"]) for robot in robots.values())
    for frame in range(n):
        for player, prefix in (("a", "r1_"), ("b", "r2_")):
            robot = robots[player]
            _set_robot_qpos(
                model,
                data,
                prefix,
                robot["joint_names"],
                robot["joint_positions"][frame],
                robot["root_translation"][frame],
                float(robot["root_yaw"][frame]),
            )
        mujoco.mj_forward(model, data)
        vertices = {
            key: world_vertices(geom) for key, geom in foot_geoms.items()
        }
        for value in vertices.values():
            ground_depth.append(max(0.0, -float(np.min(value[:, 2]))))
        for event in event_lookup.get(frame, []):
            value = vertices[(event["player"], event["foot"])]
            nearest_vertex = float(
                np.min(np.linalg.norm(value - ball[frame], axis=1))
            )
            gap = nearest_vertex - BALL_RADIUS_M
            contact_gap.append(gap)
            contact_samples.append(
                {
                    **event,
                    "nearest_mesh_vertex_to_ball_surface_gap_m": gap,
                }
            )
    ground = np.asarray(ground_depth)
    return {
        "available": True,
        "definition": {
            "ground": (
                "Maximum depth of transformed G1 foot-mesh vertices below Z=0."
            ),
            "ball_foot": (
                "Nearest transformed target-foot mesh vertex to ball center minus "
                "0.11 m radius; vertex-sampled proxy, not exact triangle distance."
            ),
        },
        "foot_ground_penetration_mesh_mean_m": float(np.mean(ground)),
        "foot_ground_penetration_mesh_max_m": float(np.max(ground)),
        "foot_ground_penetration_mesh_fraction": float(np.mean(ground > 0.0)),
        "ball_target_foot_mesh_vertex_gap_mean_m": float(np.mean(contact_gap)),
        "ball_target_foot_mesh_vertex_gap_abs_mean_m": float(
            np.mean(np.abs(contact_gap))
        ),
        "self_collision": {
            "available": False,
            "reason": (
                "The rendering scene disables/filters relevant collision pairs; "
                "no robust non-adjacent mesh self-distance implementation is present."
            ),
        },
        "samples": contact_samples,
    }


def flatten_row(
    sequence_id: str,
    method: str,
    num_frames: int,
    metrics: dict,
) -> dict[str, Any]:
    phase = metrics["interaction"]
    support = metrics["support"]
    smooth = metrics["smoothness"]
    passes = metrics["passes"]
    return {
        "sequence_id": sequence_id,
        "method": method,
        "num_frames": num_frames,
        "num_receive_events": phase["receive"]["count"],
        "num_kick_events": phase["kick"]["count"],
        "target_foot_distance_mean_m": phase["overall"][
            "target_foot_distance_mean_m"
        ],
        "contact_surface_error_mean_m": phase["overall"][
            "contact_surface_error_mean_m"
        ],
        "correct_foot_association_rate": phase["overall"][
            "correct_foot_association_rate"
        ],
        "receive_timing_error_mean_frames": phase["receive"][
            "timing_error_mean_frames"
        ],
        "kick_timing_error_mean_frames": phase["kick"][
            "timing_error_mean_frames"
        ],
        "support_foot_displacement_mean_m": support[
            "stance_foot_horizontal_displacement_mean_m"
        ],
        "support_foot_speed_p95_m_s": support[
            "stance_foot_speed_p95_m_per_s"
        ],
        "floor_penetration_max_m": support[
            "floor_penetration_keypoint_max_m"
        ],
        "root_jerk_rms_m_s3": smooth["root_jerk_rms_m_per_s3"],
        "joint_jerk_rms_rad_s3": smooth["joint_jerk_rms_rad_per_s3"],
        "lane_crossing_error_mean_m": passes["lane_crossing_error_mean_m"],
        "cone_surface_clearance_min_m": passes["cone_surface_clearance_min_m"],
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ablation-root",
        type=Path,
        default=Path("data/exports/touch_ladder_demo/experiments/ablation"),
    )
    parser.add_argument(
        "--contacts",
        type=Path,
        default=Path("data/exports/touch_ladder_demo/ball_cones.json"),
    )
    parser.add_argument(
        "--prep",
        type=Path,
        default=Path("data/clips/touch_ladder_demo_prep.json"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/exports/touch_ladder_demo/paper_eval"),
    )
    parser.add_argument("--skip-mesh", action="store_true")
    args = parser.parse_args()
    guard_release_writes(args)
    args.out.mkdir(parents=True, exist_ok=True)
    sequence_dir = args.out / "sequences/touch_ladder_demo"
    sequence_dir.mkdir(parents=True, exist_ok=True)
    contacts = json.loads(args.contacts.read_text())["contacts"]
    prep = json.loads(args.prep.read_text())
    (sequence_dir / "events.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "source": str(args.contacts),
                "contacts": contacts,
            },
            indent=2,
        )
        + "\n"
    )
    sequence_metadata = {
        "schema_version": "1.0.0",
        "sequence_id": "touch_ladder_demo",
        "source_metadata": prep,
        "source_video_packaged": False,
        "source_video_packaging_reason": "Redistribution permission not granted.",
        "preprocessing_metadata": str(args.prep),
        "event_annotations": str(sequence_dir / "events.json"),
        "variants": {
            variant: {
                "robot_a": str(args.ablation_root / variant / "robot_a.npz"),
                "robot_b": str(args.ablation_root / variant / "robot_b.npz"),
                "ball_cones": str(args.ablation_root / variant / "ball_cones.npz"),
                "preview": str(args.ablation_root / variant / "preview.mp4"),
            }
            for variant in VARIANT_ORDER
        },
    }
    (sequence_dir / "metadata.json").write_text(
        json.dumps(sequence_metadata, indent=2) + "\n"
    )

    all_metrics = {}
    summary_rows = []
    phase_rows = []
    for variant in VARIANT_ORDER:
        variant_dir = args.ablation_root / variant
        robots = {
            "a": load_robot(variant_dir / "robot_a.npz"),
            "b": load_robot(variant_dir / "robot_b.npz"),
        }
        environment = np.load(variant_dir / "ball_cones.npz")
        ball = yup_to_robot(environment["ball_xyz_yup"])
        legacy = evaluate_variant(variant_dir, contacts)
        metrics = {
            "interaction": phase_contact_and_timing(robots, ball, contacts),
            "support": support_and_floor_metrics(robots, contacts),
            "smoothness": smoothness_metrics(robots),
            "passes": legacy["passes"],
            "legacy_compatibility": legacy,
            "mesh_geometry": (
                {"available": False, "reason": "Skipped by command line."}
                if args.skip_mesh
                else mesh_metrics(variant_dir, robots, ball, contacts)
            ),
        }
        all_metrics[variant] = metrics
        summary_rows.append(
            flatten_row("touch_ladder_demo", variant, len(ball), metrics)
        )
        for phase in PHASES:
            values = metrics["interaction"][phase]
            phase_rows.append(
                {
                    "sequence_id": "touch_ladder_demo",
                    "method": variant,
                    "phase": phase,
                    **values,
                }
            )

    output = {
        "schema_version": "1.0.0",
        "sequence_id": "touch_ladder_demo",
        "fps": FPS,
        "timing_window_radius_frames": TIMING_WINDOW_RADIUS_FRAMES,
        "support_interval": {
            "frames_before_kick": SUPPORT_BEFORE_FRAMES,
            "frames_after_kick": SUPPORT_AFTER_FRAMES,
        },
        "claim_boundary": (
            "Single-sequence internal kinematic/geometric evaluation from "
            "monocular pseudo-labels and prescribed ball paths."
        ),
        "variants": all_metrics,
    }
    metrics_path = sequence_dir / "metrics.json"
    metrics_path.write_text(json.dumps(output, indent=2) + "\n")
    write_csv(args.out / "summary.csv", summary_rows)
    write_csv(args.out / "receive_kick_summary.csv", phase_rows)

    aggregate = {
        "schema_version": "1.0.0",
        "num_sequences": 1,
        "standard_deviation": None,
        "standard_deviation_note": (
            "Unavailable: only one qualifying sequence. Repeated deterministic "
            "runs are not used as independent samples."
        ),
        "methods": {
            row["method"]: {
                "mean": {key: value for key, value in row.items() if key not in ("sequence_id", "method")},
                "median": {key: value for key, value in row.items() if key not in ("sequence_id", "method")},
            }
            for row in summary_rows
        },
    }
    (args.out / "aggregate.json").write_text(json.dumps(aggregate, indent=2) + "\n")
    aggregate_rows = []
    excluded = {"sequence_id", "method"}
    for row in summary_rows:
        for metric, value in row.items():
            if metric in excluded:
                continue
            aggregate_rows.append(
                {
                    "method": row["method"],
                    "metric": metric,
                    "n_sequences": 1,
                    "mean": value,
                    "median": value,
                    "standard_deviation": "",
                }
            )
    write_csv(args.out / "aggregate.csv", aggregate_rows)

    motion_rows = []
    for method, values in all_metrics.items():
        support = values["support"]
        smooth = values["smoothness"]
        mesh = values["mesh_geometry"]
        motion_rows.append(
            {
                "sequence_id": "touch_ladder_demo",
                "method": method,
                "stance_foot_displacement_mean_m": support[
                    "stance_foot_horizontal_displacement_mean_m"
                ],
                "stance_foot_speed_mean_m_s": support[
                    "stance_foot_speed_mean_m_per_s"
                ],
                "stance_foot_speed_p50_m_s": support[
                    "stance_foot_speed_p50_m_per_s"
                ],
                "stance_foot_speed_p95_m_s": support[
                    "stance_foot_speed_p95_m_per_s"
                ],
                "stance_foot_speed_max_m_s": support[
                    "stance_foot_speed_max_m_per_s"
                ],
                "keypoint_floor_penetration_mean_m": support[
                    "floor_penetration_keypoint_mean_m"
                ],
                "keypoint_floor_penetration_max_m": support[
                    "floor_penetration_keypoint_max_m"
                ],
                "keypoint_floor_penetration_fraction": support[
                    "floor_penetration_keypoint_fraction"
                ],
                "root_velocity_rms_m_s": smooth["root_velocity_rms_m_per_s"],
                "root_acceleration_rms_m_s2": smooth[
                    "root_acceleration_rms_m_per_s2"
                ],
                "root_jerk_rms_m_s3": smooth["root_jerk_rms_m_per_s3"],
                "joint_velocity_rms_rad_s": smooth[
                    "joint_velocity_rms_rad_per_s"
                ],
                "joint_acceleration_rms_rad_s2": smooth[
                    "joint_acceleration_rms_rad_per_s2"
                ],
                "joint_jerk_rms_rad_s3": smooth[
                    "joint_jerk_rms_rad_per_s3"
                ],
                "mesh_floor_penetration_mean_m": (
                    mesh.get("foot_ground_penetration_mesh_mean_m")
                ),
                "mesh_floor_penetration_max_m": (
                    mesh.get("foot_ground_penetration_mesh_max_m")
                ),
                "mesh_floor_penetration_fraction": (
                    mesh.get("foot_ground_penetration_mesh_fraction")
                ),
                "mesh_vertex_ball_gap_abs_mean_m": (
                    mesh.get("ball_target_foot_mesh_vertex_gap_abs_mean_m")
                ),
            }
        )
    write_csv(args.out / "motion_geometry_summary.csv", motion_rows)

    cross_task = {
        "schema_version": "1.0.0",
        "units": {
            "interaction_distance": "m",
            "interaction_timing": "frames and s",
            "geometry": "m",
            "smoothness_root": "m/s^3",
            "smoothness_joint": "rad/s^3",
        },
        "categories": {
            method: {
                "interaction_distance": values["interaction"]["overall"][
                    "target_foot_distance_mean_m"
                ],
                "interaction_timing_frames": values["interaction"]["overall"][
                    "timing_error_mean_frames"
                ],
                "geometry_floor_penetration_max_m": values["support"][
                    "floor_penetration_keypoint_max_m"
                ],
                "geometry_cone_surface_clearance_min_m": values["passes"][
                    "cone_surface_clearance_min_m"
                ],
                "smoothness_root_jerk_rms_m_per_s3": values["smoothness"][
                    "root_jerk_rms_m_per_s3"
                ],
                "smoothness_joint_jerk_rms_rad_per_s3": values["smoothness"][
                    "joint_jerk_rms_rad_per_s3"
                ],
            }
            for method, values in all_metrics.items()
        },
    }
    (args.out / "cross_task_metrics.json").write_text(
        json.dumps(cross_task, indent=2) + "\n"
    )

    frozen = {
        "accepted_preview": {
            "path": "data/exports/touch_ladder_demo/comparison/current_hop_timed_lateral.mp4",
            "sha256": sha256(
                Path(
                    "data/exports/touch_ladder_demo/comparison/"
                    "current_hop_timed_lateral.mp4"
                )
            ),
        },
        "accepted_robot_a": {
            "path": "data/exports/touch_ladder_demo/g1/robot_a.npz",
            "sha256": sha256(Path("data/exports/touch_ladder_demo/g1/robot_a.npz")),
        },
        "accepted_robot_b": {
            "path": "data/exports/touch_ladder_demo/g1/robot_b.npz",
            "sha256": sha256(Path("data/exports/touch_ladder_demo/g1/robot_b.npz")),
        },
        "accepted_ball": {
            "path": "data/exports/touch_ladder_demo/ball_cones.npz",
            "sha256": sha256(Path("data/exports/touch_ladder_demo/ball_cones.npz")),
        },
        "paper_metrics": str(metrics_path),
    }
    (args.out / "frozen_inputs.json").write_text(json.dumps(frozen, indent=2) + "\n")
    print(json.dumps({"summary": summary_rows, "metrics": str(metrics_path)}, indent=2))


if __name__ == "__main__":
    main()

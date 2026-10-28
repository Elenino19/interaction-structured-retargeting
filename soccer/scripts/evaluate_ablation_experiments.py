#!/usr/bin/env python3
"""Evaluate all Touch Ladder ablations with fixed geometric proxy metrics."""

from __future__ import annotations

import sys as _fg_sys
from pathlib import Path as _FgPath
_fg_sys.path.insert(0, str(_FgPath(__file__).resolve().parent))
from frozen_guard import guard_release_writes


import argparse
import csv
import json
from pathlib import Path

import numpy as np

BALL_RADIUS_M = 0.11
CONE_RADIUS_M = 0.06
FLOOR_PROXY_Z_M = 0.012
FPS = 24.0
PASS_LANES = (3, 4, 5, 4, 3, 2)
VARIANT_ORDER = ("basic_ik", "support", "event_semantics", "full")


def yup_to_robot(points: np.ndarray) -> np.ndarray:
    return np.stack([points[..., 0], -points[..., 2], points[..., 1]], axis=-1)


def load_robot(path: Path) -> dict[str, np.ndarray]:
    return {key: np.array(value) for key, value in np.load(path).items()}


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def _p95(values: list[float]) -> float:
    return float(np.percentile(values, 95)) if values else float("nan")


def contact_metrics(
    robots: dict[str, dict[str, np.ndarray]],
    ball: np.ndarray,
    contacts: list[dict],
) -> dict:
    target_distance: list[float] = []
    wrong_distance: list[float] = []
    surface_error: list[float] = []
    role_margin: list[float] = []
    samples = []
    for index, event in enumerate(contacts):
        phases = []
        if index > 0:
            phases.append(("receive", int(event["frame"]), event["side"]))
        phases.append(
            (
                "kick",
                int(event.get("kick_frame", event["frame"])),
                event.get("kick_side", event["side"]),
            )
        )
        robot = robots[event["player"]]
        names = list(robot["keypoint_names"])
        for phase, frame, side in phases:
            target = robot["robot_keypoints"][frame, names.index(f"{side}_toe")]
            wrong_side = "right" if side == "left" else "left"
            wrong = robot["robot_keypoints"][
                frame, names.index(f"{wrong_side}_toe")
            ]
            target_d = float(np.linalg.norm(ball[frame] - target))
            wrong_d = float(np.linalg.norm(ball[frame] - wrong))
            target_distance.append(target_d)
            wrong_distance.append(wrong_d)
            surface_error.append(abs(target_d - BALL_RADIUS_M))
            role_margin.append(wrong_d - target_d)
            samples.append(
                {
                    "player": event["player"],
                    "phase": phase,
                    "frame": frame,
                    "foot": side,
                    "target_distance_m": target_d,
                    "wrong_foot_distance_m": wrong_d,
                    "surface_error_m": abs(target_d - BALL_RADIUS_M),
                    "wrong_minus_target_margin_m": wrong_d - target_d,
                }
            )
    return {
        "target_foot_distance_mean_m": _mean(target_distance),
        "target_foot_distance_p95_m": _p95(target_distance),
        "wrong_foot_distance_mean_m": _mean(wrong_distance),
        "contact_surface_error_mean_m": _mean(surface_error),
        "wrong_minus_target_margin_mean_m": _mean(role_margin),
        "wrong_foot_farther_fraction": float(
            np.mean(np.asarray(role_margin) > 0.0)
        ),
        "samples": samples,
    }


def robot_motion_metrics(
    robots: dict[str, dict[str, np.ndarray]],
    contacts: list[dict],
) -> dict:
    support_sliding = []
    floor_depth = []
    pelvis_acceleration = []
    for player, robot in robots.items():
        names = list(robot["keypoint_names"])
        keypoints = robot["robot_keypoints"]
        for name in ("left_ankle", "left_toe", "right_ankle", "right_toe"):
            z = keypoints[:, names.index(name), 2]
            floor_depth.extend(np.maximum(FLOOR_PROXY_Z_M - z, 0.0).tolist())
        pelvis_z = keypoints[:, names.index("pelvis"), 2]
        pelvis_acceleration.extend((np.diff(pelvis_z, n=2) * FPS**2).tolist())
        for event in contacts:
            if event["player"] != player:
                continue
            frame = int(event.get("kick_frame", event["frame"]))
            kick_side = event.get("kick_side", event["side"])
            support_side = "right" if kick_side == "left" else "left"
            toe = keypoints[:, names.index(f"{support_side}_toe"), :2]
            lo, hi = max(0, frame - 3), min(len(toe), frame + 2)
            support_sliding.append(
                float(np.linalg.norm(np.diff(toe[lo:hi], axis=0), axis=1).sum())
            )
    floor = np.asarray(floor_depth)
    acceleration = np.asarray(pelvis_acceleration)
    return {
        "support_foot_sliding_mean_m_per_event": _mean(support_sliding),
        "support_foot_sliding_p95_m_per_event": _p95(support_sliding),
        "floor_penetration_proxy_mean_m": float(np.mean(floor)),
        "floor_penetration_proxy_max_m": float(np.max(floor)),
        "floor_penetration_proxy_fraction": float(np.mean(floor > 0.0)),
        "pelvis_vertical_acceleration_rms_m_per_s2": float(
            np.sqrt(np.mean(acceleration**2))
        ),
    }


def pass_metrics(
    robots: dict[str, dict[str, np.ndarray]],
    ball: np.ndarray,
    cones: np.ndarray,
    contacts: list[dict],
    gap_lanes: np.ndarray,
) -> dict:
    cone_x = float(np.mean(cones[:, 0]))
    crossing_error = []
    cone_center_clearance = []
    cone_surface_clearance = []
    path_excess = []
    heading_change = []
    foot_lane_error = []
    samples = []
    for index, lane_index in enumerate(PASS_LANES):
        start = int(contacts[index].get("kick_frame", contacts[index]["frame"]))
        end = int(contacts[index + 1]["frame"])
        segment = ball[start : end + 1]
        lane_y = float(gap_lanes[lane_index])
        crossing_offset = int(np.argmin(np.abs(segment[:, 0] - cone_x)))
        crossing_y = float(segment[crossing_offset, 1])
        crossing_error.append(abs(crossing_y - lane_y))
        center_clearance = float(
            np.linalg.norm(
                segment[:, None, :2] - cones[None, :, :2],
                axis=-1,
            ).min()
        )
        cone_center_clearance.append(center_clearance)
        cone_surface_clearance.append(
            center_clearance - BALL_RADIUS_M - CONE_RADIUS_M
        )

        central = segment[np.abs(segment[:, 0] - cone_x) <= 0.55, :2]
        if len(central) >= 2:
            deltas = np.diff(central, axis=0)
            lengths = np.linalg.norm(deltas, axis=1)
            chord = float(np.linalg.norm(central[-1] - central[0]))
            path_excess.append(float(lengths.sum() / max(chord, 1e-8) - 1.0))
            moving = lengths > 1e-8
            headings = np.unwrap(
                np.arctan2(deltas[moving, 1], deltas[moving, 0])
            )
            heading_change.append(
                float(np.degrees(np.abs(np.diff(headings)).sum()))
                if len(headings) >= 2
                else 0.0
            )

        for event_index, phase in ((index, "kick"), (index + 1, "receive")):
            event = contacts[event_index]
            robot = robots[event["player"]]
            names = list(robot["keypoint_names"])
            if phase == "kick":
                frame = int(event.get("kick_frame", event["frame"]))
                side = event.get("kick_side", event["side"])
            else:
                frame = int(event["frame"])
                side = event["side"]
            toe_y = float(
                robot["robot_keypoints"][
                    frame, names.index(f"{side}_toe"), 1
                ]
            )
            foot_lane_error.append(abs(toe_y - lane_y))
        samples.append(
            {
                "pass": index + 1,
                "start_frame": start,
                "end_frame": end,
                "lane_index": lane_index,
                "lane_y_m": lane_y,
                "crossing_y_m": crossing_y,
                "crossing_error_m": abs(crossing_y - lane_y),
                "cone_center_clearance_m": center_clearance,
                "cone_surface_clearance_m": (
                    center_clearance - BALL_RADIUS_M - CONE_RADIUS_M
                ),
            }
        )
    return {
        "pass_curvature_path_excess_mean": _mean(path_excess),
        "pass_heading_change_mean_deg": _mean(heading_change),
        "cone_center_clearance_min_m": float(np.min(cone_center_clearance)),
        "cone_surface_clearance_min_m": float(np.min(cone_surface_clearance)),
        "lane_crossing_error_mean_m": _mean(crossing_error),
        "contact_foot_lane_error_mean_m": _mean(foot_lane_error),
        "samples": samples,
    }


def evaluate_variant(
    variant_dir: Path,
    contacts: list[dict],
) -> dict:
    robots = {
        "a": load_robot(variant_dir / "robot_a.npz"),
        "b": load_robot(variant_dir / "robot_b.npz"),
    }
    environment = np.load(variant_dir / "ball_cones.npz")
    ball = yup_to_robot(environment["ball_xyz_yup"])
    cones = yup_to_robot(environment["cone_xyz_yup"])
    gap_lanes = np.asarray(
        environment.get(
            "gap_lanes_robot_y",
            np.asarray([1.25, 0.75, 0.25, -0.25, -0.75, -1.25]),
        )
    )
    return {
        "contact": contact_metrics(robots, ball, contacts),
        "robot_motion": robot_motion_metrics(robots, contacts),
        "passes": pass_metrics(robots, ball, cones, contacts, gap_lanes),
    }


def flatten_summary(metrics: dict) -> dict[str, float | str]:
    return {
        "target_foot_distance_mean_m": metrics["contact"][
            "target_foot_distance_mean_m"
        ],
        "wrong_foot_distance_mean_m": metrics["contact"][
            "wrong_foot_distance_mean_m"
        ],
        "contact_surface_error_mean_m": metrics["contact"][
            "contact_surface_error_mean_m"
        ],
        "wrong_minus_target_margin_mean_m": metrics["contact"][
            "wrong_minus_target_margin_mean_m"
        ],
        "support_foot_sliding_mean_m_per_event": metrics["robot_motion"][
            "support_foot_sliding_mean_m_per_event"
        ],
        "floor_penetration_proxy_max_m": metrics["robot_motion"][
            "floor_penetration_proxy_max_m"
        ],
        "pelvis_vertical_acceleration_rms_m_per_s2": metrics["robot_motion"][
            "pelvis_vertical_acceleration_rms_m_per_s2"
        ],
        "pass_curvature_path_excess_mean": metrics["passes"][
            "pass_curvature_path_excess_mean"
        ],
        "pass_heading_change_mean_deg": metrics["passes"][
            "pass_heading_change_mean_deg"
        ],
        "cone_surface_clearance_min_m": metrics["passes"][
            "cone_surface_clearance_min_m"
        ],
        "lane_crossing_error_mean_m": metrics["passes"][
            "lane_crossing_error_mean_m"
        ],
        "contact_foot_lane_error_mean_m": metrics["passes"][
            "contact_foot_lane_error_mean_m"
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/exports/touch_ladder_demo/experiments/ablation"),
    )
    parser.add_argument(
        "--contacts",
        type=Path,
        default=Path("data/exports/touch_ladder_demo/ball_cones.json"),
    )
    args = parser.parse_args()
    guard_release_writes(args, args.root / 'metrics.json', args.root / 'metrics.csv')
    contacts = json.loads(args.contacts.read_text())["contacts"]
    variants = {
        name: evaluate_variant(args.root / name, contacts)
        for name in VARIANT_ORDER
    }
    output = {
        "schema_version": "1.0.0",
        "experiment": "touch_ladder_ablation",
        "fps": FPS,
        "metric_scope": (
            "Single accepted sequence; internal geometric proxies computed "
            "from prescribed ball trajectories and retargeted keypoints."
        ),
        "definitions": {
            "contact": (
                "Ball-center distances at fixed receive/kick events; surface "
                "error is abs(center distance - 0.11 m ball radius)."
            ),
            "support_sliding": (
                "Cumulative horizontal support-toe displacement from kick-3 "
                "through kick+1, using the opposite of the annotated kick foot."
            ),
            "floor_penetration_proxy": (
                "Positive depth below the 0.012 m ankle/toe keypoint threshold; "
                "not a mesh collision or force measurement."
            ),
            "pelvis_jitter": "RMS second finite difference of pelvis Z times fps^2.",
            "pass_curvature": (
                "Path-length excess and accumulated heading change within "
                "abs(ball_x - cone_x) <= 0.55 m."
            ),
            "cone_clearance": (
                "Minimum center distance minus 0.11 m ball and 0.06 m cone radii."
            ),
        },
        "variants": variants,
    }
    (args.root / "metrics.json").write_text(json.dumps(output, indent=2) + "\n")
    rows = []
    for name in VARIANT_ORDER:
        rows.append({"variant": name, **flatten_summary(variants[name])})
    with (args.root / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({row["variant"]: row for row in rows}, indent=2))


if __name__ == "__main__":
    main()

"""Paper-stage exports for the controlled CORE4D Join evaluation."""

from __future__ import annotations

import csv
import json
import shutil
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from .dataset import SPLIT
from .retarget.paper_presets import (
    PAPER_VARIANT_LABELS,
    PAPER_VARIANTS,
    paper_join_config_dict,
)

PAPER_COLUMNS = (
    "sequence_id",
    "box_id",
    "split",
    "row_type",
    "robot_id",
    "contact_f1",
    "contact_onset_error_frames",
    "contact_onset_abs_error_frames",
    "mean_palm_distance_cm",
    "minimum_palm_distance_cm",
    "semantic_keypoint_error_cm",
    "root_trajectory_error_cm",
    "joint_velocity_rms_rad_s",
    "joint_acceleration_rms_rad_s2",
    "joint_jerk_rms_rad_s3",
    "root_velocity_rms_m_s",
    "root_acceleration_rms_m_s2",
    "root_jerk_rms_m_s3",
    "body_object_penetration_mean_cm",
    "hand_object_penetration_max_cm",
    "foot_object_penetration_max_cm",
    "leg_xy_intersection_frames",
    "planted_foot_speed_mean_cm_s",
    "planted_foot_speed_p50_cm_s",
    "planted_foot_speed_p95_cm_s",
    "non_skating_frame_frac",
    "sole_below_ground_max_cm",
    "sole_float_mean_cm",
    "ankle_height_mae_m",
    "interaction_distance_cm",
    "timing_error_frames",
    "geometry_violation_cm",
    "smoothness_joint_jerk_rad_s3",
)

ABLATION_KEYS = (
    ("contact_f1", "Contact F1 ↑"),
    ("contact_onset_abs_error_frames", "Onset Error [frames] ↓"),
    ("mean_palm_distance_cm", "Palm Distance [cm] ↓"),
    ("semantic_keypoint_error_cm", "Semantic JPE [cm] ↓"),
    ("root_trajectory_error_cm", "Root Error [cm] ↓"),
    ("joint_jerk_rms_rad_s3", "Joint Jerk [rad/s³] ↓"),
    ("root_jerk_rms_m_s3", "Root Jerk [m/s³] ↓"),
    ("hand_object_penetration_max_cm", "Hand Pen. [cm] ↓"),
    ("foot_object_penetration_max_cm", "Foot Pen. [cm] ↓"),
    ("planted_foot_speed_p95_cm_s", "Foot Slide p95 [cm/s] ↓"),
)


def _box_id(sequence_id: str) -> str:
    return "Box024" if sequence_id.rsplit("/", 1)[-1] in {"018", "019", "020"} else "Box001"


def _paper_robot_row(
    sequence_id: str,
    robot_id: str,
    robot: dict[str, object],
) -> dict[str, object]:
    onset = robot.get("contact_onset_error_frames")
    hand = robot.get("hand_object_penetration_max_cm")
    foot = robot.get("foot_object_penetration_max_cm")
    ground = robot.get("sole_below_ground_max_cm")
    geometry_values = [
        float(value)
        for value in (hand, foot, ground)
        if value is not None
    ]
    return {
        "sequence_id": sequence_id,
        "box_id": _box_id(sequence_id),
        "split": SPLIT[sequence_id],
        "row_type": "robot",
        "robot_id": robot_id,
        "contact_f1": robot.get("contact_f1"),
        "contact_onset_error_frames": onset,
        "contact_onset_abs_error_frames": abs(float(onset)) if onset is not None else None,
        "mean_palm_distance_cm": robot.get(
            "mean_palm_distance_during_expected_contact_cm"
        ),
        "minimum_palm_distance_cm": robot.get("minimum_palm_distance_cm"),
        "semantic_keypoint_error_cm": robot.get("semantic_keypoint_error_cm"),
        "root_trajectory_error_cm": robot.get("root_trajectory_error_cm"),
        "joint_velocity_rms_rad_s": robot.get("joint_velocity_rms_rad_s"),
        "joint_acceleration_rms_rad_s2": robot.get(
            "joint_acceleration_rms_rad_s2"
        ),
        "joint_jerk_rms_rad_s3": robot.get("joint_jerk_rms_rad_s3"),
        "root_velocity_rms_m_s": robot.get("root_velocity_rms_m_s"),
        "root_acceleration_rms_m_s2": robot.get("root_acceleration_rms_m_s2"),
        "root_jerk_rms_m_s3": robot.get("root_jerk_rms_m_s3"),
        "body_object_penetration_mean_cm": robot.get(
            "body_object_penetration_mean_cm"
        ),
        "hand_object_penetration_max_cm": hand,
        "foot_object_penetration_max_cm": foot,
        "leg_xy_intersection_frames": robot.get("leg_xy_intersection_frames"),
        "planted_foot_speed_mean_cm_s": robot.get(
            "planted_foot_speed_mean_cm_s"
        ),
        "planted_foot_speed_p50_cm_s": robot.get("planted_foot_speed_p50_cm_s"),
        "planted_foot_speed_p95_cm_s": robot.get("planted_foot_speed_p95_cm_s"),
        "non_skating_frame_frac": robot.get("non_skating_frame_frac"),
        "sole_below_ground_max_cm": ground,
        "sole_float_mean_cm": robot.get("sole_float_mean_cm"),
        "ankle_height_mae_m": robot.get("ankle_height_mae_m"),
        "interaction_distance_cm": robot.get(
            "mean_palm_distance_during_expected_contact_cm"
        ),
        "timing_error_frames": abs(float(onset)) if onset is not None else None,
        "geometry_violation_cm": max(geometry_values) if geometry_values else None,
        "smoothness_joint_jerk_rad_s3": robot.get("joint_jerk_rms_rad_s3"),
    }


def _average_rows(rows: list[dict[str, object]]) -> dict[str, object]:
    average = dict(rows[0])
    average["row_type"] = "sequence_average"
    average["robot_id"] = "avg"
    for key in PAPER_COLUMNS:
        values = [
            float(row[key])
            for row in rows
            if row.get(key) is not None and isinstance(row.get(key), (int, float))
        ]
        if values:
            average[key] = float(np.mean(values))
    return average


def load_paper_rows(metrics_paths: Iterable[Path]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(metrics_paths):
        payload = json.loads(path.read_text(encoding="utf-8"))
        sequence_id = str(payload["sequence_id"])
        robot_rows = [
            _paper_robot_row(sequence_id, robot_id, robot)
            for robot_id, robot in sorted(payload["robots"].items())
        ]
        rows.extend(robot_rows)
        rows.append(_average_rows(robot_rows))
    return rows


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PAPER_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summarize_sequence_rows(rows: list[dict[str, object]]) -> dict[str, object]:
    sequence_rows = [row for row in rows if row["row_type"] == "sequence_average"]
    summary: dict[str, object] = {"aggregation": "six sequence averages", "metrics": {}}
    for key in PAPER_COLUMNS:
        values = [
            float(row[key])
            for row in sequence_rows
            if row.get(key) is not None and isinstance(row.get(key), (int, float))
        ]
        if not values:
            continue
        array = np.asarray(values, dtype=np.float64)
        summary["metrics"][key] = {
            "count": len(values),
            "mean": float(array.mean()),
            "std": float(array.std()),
            "median": float(np.median(array)),
            "min": float(array.min()),
            "max": float(array.max()),
        }
    return summary


def _format_value(value: object) -> str:
    if value is None:
        return "—"
    if isinstance(value, (int, float)):
        return f"{float(value):.3f}"
    return str(value)


def write_aggregate_markdown(summary: dict[str, object], path: Path) -> None:
    lines = [
        "# Join final-six aggregate",
        "",
        "Aggregation unit: six sequence averages (two robots averaged per sequence).",
        "",
        "| Metric | Mean | Std | Median | Min | Max |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key, values in summary["metrics"].items():
        lines.append(
            f"| {key} | {_format_value(values['mean'])} | "
            f"{_format_value(values['std'])} | {_format_value(values['median'])} | "
            f"{_format_value(values['min'])} | {_format_value(values['max'])} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def ablation_rows(metrics_paths: dict[str, Path]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for variant in PAPER_VARIANTS:
        payload = json.loads(metrics_paths[variant].read_text(encoding="utf-8"))
        robot_rows = [
            _paper_robot_row(str(payload["sequence_id"]), robot_id, robot)
            for robot_id, robot in sorted(payload["robots"].items())
        ]
        average = _average_rows(robot_rows)
        average["variant"] = variant
        average["method"] = PAPER_VARIANT_LABELS[variant]
        rows.append(average)
    return rows


def write_ablation_tables(rows: list[dict[str, object]], root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    csv_path = root / "join_ablation_018.csv"
    fields = ["variant", "method", *(key for key, _ in ABLATION_KEYS)]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    headers = ["Variant", "Method", *(label for _, label in ABLATION_KEYS)]
    md = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---", "---", *(["---:"] * len(ABLATION_KEYS))]) + "|",
    ]
    for row in rows:
        md.append(
            "| "
            + " | ".join(
                [
                    str(row["variant"]),
                    str(row["method"]),
                    *(_format_value(row.get(key)) for key, _ in ABLATION_KEYS),
                ]
            )
            + " |"
        )
    (root / "join_ablation_018.md").write_text(
        "# Controlled 018 ablation\n\n"
        "All rows use the same independent initialization and paper metric mode.\n\n"
        + "\n".join(md)
        + "\n",
        encoding="utf-8",
    )

    latex_rows = []
    for row in rows:
        cells = [
            str(row["variant"]),
            str(row["method"]).replace("_", r"\_"),
            *(_format_value(row.get(key)) for key, _ in ABLATION_KEYS),
        ]
        latex_rows.append(" & ".join(cells) + r" \\")
    latex = (
        "\\begin{tabular}{ll"
        + "r" * len(ABLATION_KEYS)
        + "}\n\\toprule\n"
        + " & ".join(headers)
        + r" \\"
        + "\n\\midrule\n"
        + "\n".join(latex_rows)
        + "\n\\bottomrule\n\\end{tabular}\n"
    )
    (root / "join_ablation_018.tex").write_text(latex, encoding="utf-8")


def write_ablation_component_matrix(root: Path) -> None:
    """Persist the exact cumulative component contract and resolved configs."""
    components = {
        "J0": (False, False, False, False, False, False),
        "J1": (True, False, False, False, False, False),
        "J2": (True, True, True, False, False, False),
        "J3": (True, True, True, True, True, False),
        "J4": (True, True, True, True, True, True),
    }
    labels = (
        "Object anchors",
        "Phase weighting",
        "Partner clearance",
        "Geometry / gait",
        "Put-down support",
        "Source approach + idle dual support",
    )
    lines = [
        "# Controlled 018 ablation components",
        "",
        "| Variant | Method | " + " | ".join(labels) + " |",
        "|---|---|" + "|".join(["---:"] * len(labels)) + "|",
    ]
    resolved = {}
    for variant in PAPER_VARIANTS:
        values = components[variant]
        lines.append(
            f"| {variant} | {PAPER_VARIANT_LABELS[variant]} | "
            + " | ".join("on" if value else "off" for value in values)
            + " |"
        )
        resolved[variant] = paper_join_config_dict(variant)
    (root / "join_ablation_components.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    (root / "join_ablation_configs.json").write_text(
        json.dumps(resolved, indent=2) + "\n", encoding="utf-8"
    )


KNOWN_VISUAL_NOTES = {
    "018": "Accepted demo; shallow person-2 foot/box overlap is metric-visible.",
    "019": "Shallow hand/foot penetrations are subtle in the preview.",
    "020": "Person-2 joining delay and person-1 foot/box penetration are visible.",
    "108": "Person-1 joining delay is visually obvious.",
    "109": "Person-2 hand and person-1 foot penetration are visually apparent.",
    "110": "Foot/box penetration and skating are visually obvious.",
}


def write_failure_cases(rows: list[dict[str, object]], path: Path) -> None:
    by_sequence: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        if row["row_type"] == "robot":
            by_sequence.setdefault(str(row["sequence_id"]), []).append(row)
    lines = [
        "# Final-six failure cases",
        "",
        (
            "Thresholds used for discussion: F1 < 0.8, |onset| > 5 frames, "
            "penetration/ground > 1 cm, planted-foot p95 > 10 cm/s."
        ),
        "",
    ]
    for sequence_id, sequence_rows in sorted(by_sequence.items()):
        short = sequence_id.rsplit("/", 1)[-1]
        lines.extend([f"## {sequence_id}", "", KNOWN_VISUAL_NOTES[short], ""])
        for row in sequence_rows:
            failures: list[str] = []
            if float(row["contact_f1"]) < 0.8:
                failures.append(f"weak contact F1={float(row['contact_f1']):.3f}")
            if float(row["contact_onset_abs_error_frames"]) > 5:
                failures.append(
                    f"onset error={float(row['contact_onset_abs_error_frames']):.0f} frames"
                )
            if float(row["geometry_violation_cm"]) > 1:
                failures.append(
                    f"geometry violation={float(row['geometry_violation_cm']):.2f} cm"
                )
            if float(row["planted_foot_speed_p95_cm_s"]) > 10:
                failures.append(
                    f"foot-slide p95={float(row['planted_foot_speed_p95_cm_s']):.1f} cm/s"
                )
            text = "; ".join(failures) if failures else "no threshold failure"
            lines.append(f"- Robot {row['robot_id']}: {text}.")
        lines.extend(
            [
                "",
                (
                    "Interpretation: these are kinematic-retargeting limitations; "
                    "they are not evidence of successful dynamic tracking."
                ),
                "",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def copy_frozen_sequence(
    sequence_id: str,
    frozen_root: Path,
    frozen_metrics_root: Path,
    frozen_video_root: Path,
    output_root: Path,
) -> Path:
    """Copy one frozen sequence into a paper folder, refusing overwrite."""
    short = sequence_id.rsplit("/", 1)[-1]
    stem = sequence_id.replace("/", "_")
    destination = output_root / short
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    destination.mkdir(parents=True)
    source_dir = frozen_root / stem
    for person in (1, 2):
        shutil.copy2(source_dir / f"robot{person}.npz", destination / f"robot{person}.npz")
    with np.load(source_dir / "robot1.npz") as payload:
        raw_config = payload["config_json"] if "config_json" in payload.files else None
        config_text = str(raw_config.item()) if raw_config is not None else "{}"
    config = json.loads(config_text)
    provenance = {
        "sequence_id": sequence_id,
        "method": "frozen_v41",
        "source_root": str(frozen_root),
        "config": config,
    }
    (destination / "config.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    metrics_path = frozen_metrics_root / f"{stem}.json"
    if metrics_path.exists():
        shutil.copy2(metrics_path, destination / "legacy_metrics.json")
    video_path = frozen_video_root / f"{short}_clearance_v41.mp4"
    if video_path.exists():
        shutil.copy2(video_path, destination / "frozen_preview.mp4")
    return destination

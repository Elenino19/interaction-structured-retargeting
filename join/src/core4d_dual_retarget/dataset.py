"""Quality labels and a sequence-level split for the dual-G1 dataset."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

BOX_SEQUENCES = (
    "20231108/018",
    "20231108/019",
    "20231108/020",
    "20231108/108",
    "20231108/109",
    "20231108/110",
)

GENERALIZATION_SEQUENCES = (
    "20231108/000",
    "20231108/001",
    "20231108/036",
    "20231108/037",
    "20231108/054",
    "20231108/055",
)

SEQUENCES = BOX_SEQUENCES

# Sequence-level split: no frame from the same clip appears in two splits.
SPLIT = {
    "20231108/018": "train",
    "20231108/019": "train",
    "20231108/108": "train",
    "20231108/020": "val",
    "20231108/109": "test",
    "20231108/110": "test",
    "20231108/000": "holdout_barrel",
    "20231108/001": "holdout_barrel",
    "20231108/036": "holdout_board",
    "20231108/037": "holdout_board",
    "20231108/054": "holdout_desk",
    "20231108/055": "holdout_desk",
}


from .metrics import (
    HAND_CONTACT_FLUSH_CM,
    PENETRATION_PUBLISH_CM,
    PHUMA_NON_SKATING_CM_S,
)


def _robot_pass(metrics: dict[str, object]) -> tuple[bool, list[str]]:
    failures: list[str] = []
    contact_f1 = metrics.get("contact_f1")
    if contact_f1 is None or float(contact_f1) < 0.90:
        failures.append("contact_f1<0.90")
    palm = metrics.get("mean_palm_distance_during_expected_contact_cm")
    if palm is None or float(palm) > 5.0:
        failures.append("palm_distance>5cm")
    excess = metrics.get("com_support_excess_mean_m")
    if excess is not None and float(excess) > 0.02:
        failures.append("com_support_excess>2cm")
    ankle = metrics.get("ankle_height_mae_m")
    if ankle is not None and float(ankle) > 0.03:
        failures.append("ankle_height_mae>3cm")
    jerk = metrics.get("joint_jerk_rms_rad_s3")
    if jerk is not None and float(jerk) > 60.0:
        failures.append("jerk>60")
    return not failures, failures


def _geometry_pass(metrics: dict[str, object]) -> tuple[bool, list[str]]:
    """Literature-aligned gate: sole/hand grids, not ankle-height MAE."""
    failures: list[str] = []
    contact_f1 = metrics.get("contact_f1")
    if contact_f1 is None or float(contact_f1) < 0.90:
        failures.append("contact_f1<0.90")
    palm = metrics.get("mean_palm_distance_during_expected_contact_cm")
    if palm is None or float(palm) > HAND_CONTACT_FLUSH_CM:
        failures.append(f"palm_distance>{HAND_CONTACT_FLUSH_CM:g}cm")
    excess = metrics.get("com_support_excess_mean_m")
    if excess is not None and float(excess) > 0.02:
        failures.append("com_support_excess>2cm")
    jerk = metrics.get("joint_jerk_rms_rad_s3")
    if jerk is not None and float(jerk) > 60.0:
        failures.append("jerk>60")
    speed = metrics.get("planted_foot_speed_mean_cm_s")
    if speed is None or float(speed) > PHUMA_NON_SKATING_CM_S:
        failures.append(f"planted_foot_speed>{PHUMA_NON_SKATING_CM_S:g}cm/s")
    below = metrics.get("sole_below_ground_max_cm")
    if below is None or float(below) > PENETRATION_PUBLISH_CM:
        failures.append(f"sole_below_ground>{PENETRATION_PUBLISH_CM:g}cm")
    hand_in = metrics.get("hand_in_hull_max_cm", metrics.get("hand_in_box_max_cm"))
    if hand_in is None or float(hand_in) > PENETRATION_PUBLISH_CM:
        failures.append(f"hand_in_object>{PENETRATION_PUBLISH_CM:g}cm")
    foot_in = metrics.get("foot_in_hull_max_cm", metrics.get("foot_in_box_max_cm"))
    if foot_in is None or float(foot_in) > PENETRATION_PUBLISH_CM:
        failures.append(f"foot_in_object>{PENETRATION_PUBLISH_CM:g}cm")
    return not failures, failures


def _id_stem(sequence_id: str, acceleration: str = "none") -> str:
    return f"{sequence_id.replace('/', '_')}_{acceleration}"


def layer2_clip_row(
    sequence_id: str,
    baseline_id: dict[str, object],
    new_id: dict[str, object],
    baseline_metrics: dict[str, object],
    new_metrics: dict[str, object],
) -> dict[str, object]:
    robots = {}
    all_pass = True
    for person in ("1", "2"):
        geom_pass, failures = _robot_pass(new_metrics["robots"][person])
        all_pass = all_pass and geom_pass
        base_r = baseline_id["robots"][person]
        new_r = new_id["robots"][person]
        robots[person] = {
            "geometry_passed": geom_pass,
            "geometry_failures": failures,
            "tensile_baseline": base_r["negative_normal_ratio"],
            "tensile_new": new_r["negative_normal_ratio"],
            "unilateral_residual_baseline": base_r["unilateral_base_residual_rms"],
            "unilateral_residual_new": new_r["unilateral_base_residual_rms"],
            "contact_f1_new": new_metrics["robots"][person]["contact_f1"],
            "palm_cm_new": new_metrics["robots"][person][
                "mean_palm_distance_during_expected_contact_cm"
            ],
            "jerk_new": new_metrics["robots"][person]["joint_jerk_rms_rad_s3"],
        }
    return {
        "sequence_id": sequence_id,
        "split": SPLIT.get(sequence_id),
        "passed": all_pass,
        "robots": robots,
    }


def build_layer2_report(
    baseline_id_root: Path,
    new_id_root: Path,
    baseline_metrics_root: Path,
    new_metrics_root: Path,
    *,
    sequence_ids: tuple[str, ...] | None = None,
    acceleration: str = "none",
) -> dict[str, object]:
    ids = sequence_ids or BOX_SEQUENCES
    rows = []
    missing = []
    for sequence_id in ids:
        stem = sequence_id.replace("/", "_")
        paths = {
            "baseline_id": baseline_id_root / _id_stem(sequence_id, acceleration) / "metrics.json",
            "new_id": new_id_root / _id_stem(sequence_id, acceleration) / "metrics.json",
            "baseline_metrics": baseline_metrics_root / f"{stem}.json",
            "new_metrics": new_metrics_root / f"{stem}.json",
        }
        if not all(path.exists() for path in paths.values()):
            missing.append(
                {
                    "sequence_id": sequence_id,
                    "missing": [name for name, path in paths.items() if not path.exists()],
                }
            )
            continue
        rows.append(
            layer2_clip_row(
                sequence_id,
                json.loads(paths["baseline_id"].read_text(encoding="utf-8")),
                json.loads(paths["new_id"].read_text(encoding="utf-8")),
                json.loads(paths["baseline_metrics"].read_text(encoding="utf-8")),
                json.loads(paths["new_metrics"].read_text(encoding="utf-8")),
            )
        )
    tensile_new = [row["robots"][p]["tensile_new"] for row in rows for p in ("1", "2")]
    tensile_base = [row["robots"][p]["tensile_baseline"] for row in rows for p in ("1", "2")]
    residual_new = [
        row["robots"][p]["unilateral_residual_new"] for row in rows for p in ("1", "2")
    ]
    residual_base = [
        row["robots"][p]["unilateral_residual_baseline"] for row in rows for p in ("1", "2")
    ]
    return {
        "clips": len(rows),
        "geometry_all_passed": bool(rows) and all(row["passed"] for row in rows),
        "mean_tensile_baseline": float(np.mean(tensile_base)) if tensile_base else None,
        "mean_tensile_new": float(np.mean(tensile_new)) if tensile_new else None,
        "mean_unilateral_residual_baseline": float(np.mean(residual_base)) if residual_base else None,
        "mean_unilateral_residual_new": float(np.mean(residual_new)) if residual_new else None,
        "missing": missing,
        "sequences": rows,
    }


def label_sequence(
    sequence_id: str,
    metrics_path: Path,
    robot1_path: Path,
    robot2_path: Path,
    *,
    geometry_gate: bool = False,
) -> dict[str, object]:
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    robots = payload["robots"]
    robot_labels = {}
    all_pass = True
    gate = _geometry_pass if geometry_gate else _robot_pass
    for person in ("1", "2"):
        passed, failures = gate(robots[person])
        all_pass = all_pass and passed
        robot_labels[person] = {"passed": passed, "failures": failures}
    with np.load(robot1_path) as left, np.load(robot2_path) as right:
        frames = int(min(len(left["joint_positions"]), len(right["joint_positions"])))
        has_config = "config_json" in left.files and "config_json" in right.files
    return {
        "sequence_id": sequence_id,
        "split": SPLIT[sequence_id],
        "frames": frames,
        "passed": all_pass and has_config,
        "has_embedded_config": has_config,
        "robots": robot_labels,
        "metrics_path": str(metrics_path),
        "trajectories": {"1": str(robot1_path), "2": str(robot2_path)},
    }


def build_dataset_manifest(
    result_root: Path,
    metrics_root: Path,
    output_path: Path,
    *,
    method: str = "balanced",
    sequence_ids: tuple[str, ...] | None = None,
    geometry_gate: bool = False,
) -> dict[str, object]:
    records = []
    selected = sequence_ids or SEQUENCES
    for sequence_id in selected:
        stem = sequence_id.replace("/", "_")
        records.append(
            label_sequence(
                sequence_id,
                metrics_root / f"{stem}.json",
                result_root / stem / "robot1.npz",
                result_root / stem / "robot2.npz",
                geometry_gate=geometry_gate,
            )
        )
    splits: dict[str, list[str]] = {}
    for record in records:
        splits.setdefault(str(record["split"]), []).append(str(record["sequence_id"]))
    manifest = {
        "method": method,
        "sequence_count": len(records),
        "passed_count": sum(bool(record["passed"]) for record in records),
        "splits": splits,
        "sequences": records,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest

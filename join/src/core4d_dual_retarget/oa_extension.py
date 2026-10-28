"""Evaluation driver for the object-aware baseline extension.

Every variant is a result root ``<variant_root>/raw/<stem>/robot{1,2}.npz``
produced by ``retarget-paper-join``.  This module applies the *frozen* paper
post-processing (``smooth-trajectories --clearance-recipe``) and the frozen
paper evaluator (``evaluate-paper-join``) through the CLI, adds the post-hoc
coordination metrics, and compares raw trajectories between variants.  The
frozen v41 results are only read.
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from .coordination_eval import evaluate_result, flatten_metrics, write_json
from .paper_eval import _average_rows, _paper_robot_row

PYTHON = sys.executable
CLI = [PYTHON, "-m", "core4d_dual_retarget.cli"]
DEFAULT_SEQUENCE_ROOT = Path("data/core4d/motions")
DEFAULT_OBJECT_MODELS = Path("data/core4d/object_models")
DEFAULT_CONTACT_ROOT = Path("outputs/contact_analysis")

PRIMARY_COLUMNS = [
    "valid_joint_coverage_tau_0.5cm",
    "valid_joint_f1_tau_0.5cm",
    "valid_joint_iou_tau_0.5cm",
    "joint_f1",
    "joint_iou",
    "joint_coverage",
    "bilateral_max_error_cm",
    "longest_valid_true_joint_run_frames_tau_0.5cm",
    "longest_valid_joint_run_frames_tau_0.5cm",
    "joint_onset_error_frames",
    "onset_lag_error_frames",
    "grasp_order_preserved",
    "release_order_preserved",
]
SECONDARY_COLUMNS = [
    "paper_contact_f1",
    "paper_hand_object_penetration_max_cm",
    "paper_foot_object_penetration_max_cm",
    "paper_geometry_violation_cm",
    "paper_planted_foot_speed_mean_cm_s",
    "paper_non_skating_frame_frac",
    "paper_joint_jerk_rms_rad_s3",
    "paper_root_jerk_rms_m_s3",
    "paper_semantic_keypoint_error_cm",
    "paper_root_trajectory_error_cm",
    "inter_robot_min_keypoint_distance_m",
    "inter_robot_frames_below_0p12m",
]


def _run(cmd: list[str], log: Path | None = None) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if log is not None:
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a", encoding="utf-8") as handle:
            handle.write("$ " + " ".join(cmd) + "\n" + result.stdout + result.stderr + "\n")
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(cmd)}\n{result.stderr}"
        )


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def ensure_smooth(variant_root: Path, sequence_id: str, log: Path | None = None) -> Path:
    """Apply the frozen paper smoothing recipe once (idempotent)."""
    stem = sequence_id.replace("/", "_")
    smooth_dir = variant_root / "smooth" / stem
    if not (smooth_dir / "robot1.npz").exists() or not (smooth_dir / "robot2.npz").exists():
        _run(
            [
                *CLI,
                "smooth-trajectories",
                sequence_id,
                "--clearance-recipe",
                "--result-root",
                str(variant_root / "raw"),
                "--output-root",
                str(variant_root / "smooth"),
            ],
            log,
        )
    return smooth_dir


def ensure_paper_metrics(
    result_root: Path, sequence_id: str, output_path: Path, log: Path | None = None
) -> dict:
    """Run the frozen paper evaluator once (idempotent) and load it."""
    if not output_path.exists():
        _run(
            [
                *CLI,
                "evaluate-paper-join",
                sequence_id,
                "--result-root",
                str(result_root),
                "--output-path",
                str(output_path),
            ],
            log,
        )
    return json.loads(output_path.read_text(encoding="utf-8"))


def paper_row(payload: dict) -> dict[str, object]:
    sequence_id = str(payload["sequence_id"])
    robot_rows = [
        _paper_robot_row(sequence_id, robot_id, robot)
        for robot_id, robot in sorted(payload["robots"].items())
    ]
    average = _average_rows(robot_rows)
    row = {f"paper_{k}": v for k, v in average.items() if k not in ("row_type", "robot_id")}
    for robot_row in robot_rows:
        rid = robot_row["robot_id"]
        for key in (
            "contact_f1",
            "contact_onset_error_frames",
            "hand_object_penetration_max_cm",
            "foot_object_penetration_max_cm",
            "sole_below_ground_max_cm",
            "planted_foot_speed_mean_cm_s",
            "joint_jerk_rms_rad_s3",
            "semantic_keypoint_error_cm",
        ):
            row[f"paper_{rid}_{key}"] = robot_row.get(key)
    return row


def evaluate_variant(
    *,
    label: str,
    group: str,
    variant_root: Path | None,
    smooth_root: Path,
    sequence_id: str,
    output_dir: Path,
    log: Path | None = None,
    sequence_root: Path = DEFAULT_SEQUENCE_ROOT,
    object_models_root: Path = DEFAULT_OBJECT_MODELS,
    contact_root: Path = DEFAULT_CONTACT_ROOT,
) -> dict[str, object]:
    """Smooth (if needed), run the frozen evaluator, and add coordination metrics."""
    stem = sequence_id.replace("/", "_")
    if variant_root is not None:
        ensure_smooth(variant_root, sequence_id, log)
    metrics_dir = output_dir / "paper_metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    payload = ensure_paper_metrics(
        smooth_root, sequence_id, metrics_dir / f"{label}__{stem}.json", log
    )
    metrics, _ = evaluate_result(
        sequence_root,
        sequence_id,
        object_models_root,
        contact_root,
        smooth_root,
        label=label,
    )
    write_json(metrics, output_dir / "per_result" / f"{label}__{stem}.json")
    row: dict[str, object] = {"label": label, "group": group, "sequence_id": sequence_id}
    row.update(flatten_metrics(metrics))
    row.update(paper_row(payload))
    row["smooth_root"] = str(smooth_root)
    row["robot1_md5"] = md5(smooth_root / stem / "robot1.npz")
    row["robot2_md5"] = md5(smooth_root / stem / "robot2.npz")
    return row


def compare_raw(
    a_root: Path, b_root: Path, sequence_id: str, *, label_a: str, label_b: str
) -> dict[str, object]:
    """Trajectory/objective/contact-mask differences between two raw results."""
    stem = sequence_id.replace("/", "_")
    out: dict[str, object] = {"a": label_a, "b": label_b, "sequence_id": sequence_id}
    for person in (1, 2):
        a = np.load(a_root / stem / f"robot{person}.npz")
        b = np.load(b_root / stem / f"robot{person}.npz")
        dq = a["joint_positions"] - b["joint_positions"]
        droot = a["root_translation"] - b["root_translation"]
        dyaw = a["root_yaw"] - b["root_yaw"]
        dkp = a["robot_keypoints"] - b["robot_keypoints"]
        out[f"robot{person}_joint_max_abs_rad"] = float(np.abs(dq).max())
        out[f"robot{person}_joint_rms_rad"] = float(np.sqrt((dq**2).mean()))
        out[f"robot{person}_root_max_abs_cm"] = float(np.abs(droot).max() * 100)
        out[f"robot{person}_root_rms_cm"] = float(np.sqrt((droot**2).sum(-1).mean()) * 100)
        out[f"robot{person}_yaw_max_abs_deg"] = float(np.degrees(np.abs(dyaw).max()))
        out[f"robot{person}_keypoint_max_abs_cm"] = float(np.abs(dkp).max() * 100)
        out[f"robot{person}_keypoint_rms_cm"] = float(np.sqrt((dkp**2).sum(-1).mean()) * 100)
        out[f"robot{person}_contact_mask_identical"] = bool(
            np.array_equal(a["contact_mask"], b["contact_mask"])
        )
        out[f"robot{person}_bit_identical"] = bool(
            np.array_equal(a["joint_positions"], b["joint_positions"])
            and np.array_equal(a["root_translation"], b["root_translation"])
            and np.array_equal(a["root_yaw"], b["root_yaw"])
        )
        out[f"robot{person}_final_loss_a"] = float(a["loss_history"][-1])
        out[f"robot{person}_final_loss_b"] = float(b["loss_history"][-1])
        out[f"robot{person}_best_loss_a"] = float(a["loss_history"].min())
        out[f"robot{person}_best_loss_b"] = float(b["loss_history"].min())
    return out


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _cell(row.get(k)) for k in fields})


def _cell(value: object) -> object:
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (np.floating,)):
        return f"{float(value):.6g}"
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def aggregate(rows: list[dict[str, object]], keys: list[str]) -> dict[str, object]:
    """Mean/SD/n over rows for numeric keys."""
    out: dict[str, object] = {"n": len(rows)}
    for key in keys:
        values = [
            float(r[key])
            for r in rows
            if r.get(key) is not None
            and not isinstance(r.get(key), (bool, np.bool_))
            and isinstance(r.get(key), (int, float, np.integer, np.floating))
        ]
        if values:
            out[f"{key}_mean"] = float(np.mean(values))
            out[f"{key}_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        bools = [r[key] for r in rows if isinstance(r.get(key), (bool, np.bool_))]
        if bools:
            out[f"{key}_frac_true"] = float(np.mean([bool(b) for b in bools]))
    return out


def term_log_summary(path: Path) -> dict[str, object]:
    """Summarize a per-iteration term log (value / grad norm / activation)."""
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {"path": str(path), "iterations": 0}

    def col(name: str) -> np.ndarray:
        return np.array([float(r[name]) for r in rows if r.get(name) not in (None, "")])

    summary: dict[str, object] = {"path": str(path), "iterations": len(rows)}
    for name in ("inter_robot_raw", "joint_contact_raw"):
        values = col(name)
        active = col(name.replace("_raw", "_active_pairs" if "inter" in name else "_active_frames"))
        grads = col(f"{name}_grad_norm")
        summary[name] = {
            "max_value": float(values.max()) if values.size else None,
            "mean_value": float(values.mean()) if values.size else None,
            "iterations_nonzero": int((values > 0).sum()) if values.size else 0,
            "activation_frequency": float((active > 0).mean()) if active.size else None,
            "mean_active_count": float(active.mean()) if active.size else None,
            "grad_norm_max": float(grads.max()) if grads.size else None,
            "grad_norm_mean": float(grads.mean()) if grads.size else None,
            "grad_norm_samples": int(grads.size),
        }
    weighted = {
        k: float(np.mean(col(k)))
        for k in rows[0]
        if k not in ("iteration", "loss")
        and not k.endswith(("_raw", "_grad_norm", "_active_pairs", "_active_frames"))
    }
    summary["mean_weighted_terms"] = weighted
    summary["loss_first"] = float(rows[0]["loss"])
    summary["loss_last"] = float(rows[-1]["loss"])
    summary["loss_min"] = float(col("loss").min())
    return summary

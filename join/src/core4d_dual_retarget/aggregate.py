"""Aggregate per-sequence metrics into paper-ready summary statistics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

METRIC_NAMES = (
    "semantic_keypoint_error_cm",
    "contact_f1",
    "contact_onset_error_frames",
    "mean_palm_distance_during_expected_contact_cm",
    "joint_jerk_rms_rad_s3",
    "com_support_excess_mean_m",
    "ankle_height_mae_m",
    "planted_foot_speed_mean_cm_s",
    "non_skating_frame_frac",
    "sole_below_ground_max_cm",
    "sole_float_mean_cm",
    "hand_in_box_max_cm",
    "foot_in_box_max_cm",
    "hand_in_hull_max_cm",
    "foot_in_hull_max_cm",
)


def _load_method_metrics(directory: Path) -> dict[str, dict[str, object]]:
    payloads: dict[str, dict[str, object]] = {}
    for path in sorted(directory.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payloads[str(payload["sequence_id"])] = payload
    return payloads


def _summarize(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(array),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def aggregate_comparison(
    independent_dir: Path,
    coupled_dir: Path,
    *,
    baseline_name: str = "independent",
    method_name: str = "coupled",
) -> dict[str, object]:
    """Aggregate paired baseline/method metrics over available sequences."""
    methods = {
        baseline_name: _load_method_metrics(independent_dir),
        method_name: _load_method_metrics(coupled_dir),
    }
    common_sequences = sorted(set(methods[baseline_name]) & set(methods[method_name]))
    if not common_sequences:
        raise ValueError("No paired metric files found")

    result: dict[str, object] = {
        "paired_sequence_count": len(common_sequences),
        "sequence_ids": common_sequences,
        "baseline_name": baseline_name,
        "method_name": method_name,
        "methods": {},
        f"{method_name}_minus_{baseline_name}": {},
    }
    method_summaries: dict[str, dict[str, object]] = {}
    for method, sequences in methods.items():
        summary: dict[str, object] = {}
        for metric in METRIC_NAMES:
            values: list[float] = []
            for sequence_id in common_sequences:
                for robot in ("1", "2"):
                    value = sequences[sequence_id]["robots"][robot].get(metric)
                    if value is None:
                        continue
                    if metric == "contact_onset_error_frames":
                        value = abs(value)
                    values.append(float(value))
            summary[metric] = _summarize(values)
        method_summaries[method] = summary
    result["methods"] = method_summaries
    delta = {
        metric: (
            None
            if method_summaries[method_name][metric]["mean"] is None
            or method_summaries[baseline_name][metric]["mean"] is None
            else (
                method_summaries[method_name][metric]["mean"]
                - method_summaries[baseline_name][metric]["mean"]
            )
        )
        for metric in METRIC_NAMES
    }
    result[f"{method_name}_minus_{baseline_name}"] = delta
    if baseline_name == "independent" and method_name == "coupled":
        result["coupled_minus_independent"] = delta
    return result


def write_aggregate(metrics: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

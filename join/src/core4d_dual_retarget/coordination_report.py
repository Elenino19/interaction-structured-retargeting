"""Orchestrate the post-hoc cross-agent coordination evaluation and exports."""

from __future__ import annotations

import csv
import hashlib
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

from .coordination_eval import (
    SequenceSignals,
    evaluate_result,
    flatten_metrics,
    write_json,
)
from .coordination_figures import (
    plot_coordination_comparison,
    plot_joint_contact_timeline,
    plot_joint_contact_timeline_variants,
)
from .dataset import BOX_SEQUENCES

CONTROLLED_VARIANTS = ("J0", "J1", "J2", "J3", "J4")
CONTROLLED_DISPLAY = {
    "J0": "J0 Independent IK",
    "J1": "J1 + Object contact",
    "J2": "J2 + Phase/partner",
    "J3": "J3 + Geometry/gait",
    "J4": "J4 Cold-start full",
}
COUPLING_TAGS = ("J4_rerun", "J4_noInter", "J4_noPhase")
COUPLING_DISPLAY = {
    "J4": "J4 (original run)",
    "J4_rerun": "J4 re-run (determinism control)",
    "J4_noInter": "J4 − inter-robot clearance",
    "J4_noPhase": "J4 − phase-3 contact weighting",
}
SUMMARY_KEYS = (
    "joint_precision",
    "joint_recall",
    "joint_f1",
    "joint_iou",
    "joint_onset_error_frames",
    "joint_end_error_frames",
    "longest_true_joint_run_frames",
    "longest_run_ratio",
    "dropout_events",
    "maintained_fraction_in_window",
    "bilateral_max_error_cm",
    "robot1_anchor_error_cm",
    "robot2_anchor_error_cm",
    "anchor_asymmetry_cm",
    "valid_joint_coverage_tau_0cm",
    "valid_joint_coverage_tau_0.5cm",
    "valid_joint_coverage_tau_1cm",
    "valid_joint_f1_tau_0.5cm",
    "robot1_f1",
    "robot2_f1",
)


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def _write_csv(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: object, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


def _markdown_table(rows: list[dict[str, object]], columns: list[tuple[str, str]]) -> str:
    head = "| " + " | ".join(name for _, name in columns) + " |"
    sep = "|" + "|".join("---:" if i else "---" for i, _ in enumerate(columns)) + "|"
    body = [
        "| " + " | ".join(_fmt(row.get(key)) for key, _ in columns) + " |" for row in rows
    ]
    return "\n".join([head, sep, *body])


def _aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out = []
    for stat in ("mean", "std", "median", "min", "max"):
        agg: dict[str, object] = {
            "group": "six_take_aggregate",
            "display": f"v41 staged ({stat} of 6)",
            "label": "v41_staged",
            "sequence_id": stat,
        }
        for key in SUMMARY_KEYS:
            values = np.asarray(
                [float(row[key]) for row in rows if row.get(key) is not None], dtype=float
            )
            if not len(values):
                agg[key] = None
                continue
            agg[key] = float(
                {
                    "mean": values.mean(),
                    "std": values.std(),
                    "median": np.median(values),
                    "min": values.min(),
                    "max": values.max(),
                }[stat]
            )
        out.append(agg)
    return out


def run_coordination_extension(
    *,
    output_root: Path,
    paper_root: Path = Path("outputs/paper_eval_join"),
    coupling_root: Path | None = None,
    frozen_root: Path = Path("outputs/balanced_clearance_v41_smooth"),
    motion_root: Path = Path("data/core4d/motions"),
    object_models_root: Path = Path("data/core4d/object_models"),
    contact_analysis_root: Path = Path("outputs/contact_analysis"),
    sequence_id: str = "20231108/018",
) -> dict[str, object]:
    """Evaluate every available result and write tables, figures, and provenance."""
    if coupling_root is None:
        coupling_root = output_root / "ablation_coupling_018"
    tables = output_root / "tables"
    figures = output_root / "figures"
    per_result = output_root / "per_result"
    for folder in (tables, figures, per_result):
        folder.mkdir(parents=True, exist_ok=True)

    nested: list[dict[str, object]] = []
    rows: list[dict[str, object]] = []
    signals_018: dict[str, tuple[SequenceSignals, dict[str, object]]] = {}

    def evaluate(label: str, group: str, display: str, seq: str, result_root: Path) -> dict:
        metrics, signals = evaluate_result(
            motion_root, seq, object_models_root, contact_analysis_root, result_root, label=label
        )
        metrics["group"] = group
        metrics["display"] = display
        nested.append(metrics)
        row = {"group": group, "display": display, **flatten_metrics(metrics)}
        rows.append(row)
        write_json(metrics, per_result / f"{group}__{label}__{seq.replace('/', '_')}.json")
        if seq == sequence_id:
            signals_018[label] = (signals, metrics)
        return row

    # 1. controlled cold-start J0--J4
    controlled_rows = []
    for variant in CONTROLLED_VARIANTS:
        root = paper_root / "ablation_018" / variant / "smooth"
        if not root.exists():
            raise FileNotFoundError(root)
        controlled_rows.append(
            evaluate(variant, "controlled_cold_start", CONTROLLED_DISPLAY[variant], sequence_id, root)
        )

    # 2. staged / warm-started frozen v41, all six takes
    six_rows = []
    hash_check = {}
    for seq in BOX_SEQUENCES:
        short = seq.rsplit("/", 1)[-1]
        root = paper_root / "final_six" / short
        stem = seq.replace("/", "_")
        hash_check[short] = {
            f"robot{p}": {
                "paper_copy_md5": _md5(root / f"robot{p}.npz"),
                "frozen_md5": _md5(frozen_root / stem / f"robot{p}.npz"),
            }
            for p in (1, 2)
        }
        for p in (1, 2):
            entry = hash_check[short][f"robot{p}"]
            entry["identical"] = entry["paper_copy_md5"] == entry["frozen_md5"]
        six_rows.append(
            evaluate("v41_staged", "staged_six_takes", f"v41 staged {short}", seq, root)
        )
    staged_018_row = next(row for row in six_rows if row["sequence_id"] == sequence_id)

    # 3. optional coupling ON/OFF controls
    coupling_rows = []
    j4_row = next(row for row in controlled_rows if row["label"] == "J4")
    if coupling_root.exists():
        coupling_rows.append({**j4_row, "group": "coupling_control", "display": COUPLING_DISPLAY["J4"]})
        for tag in COUPLING_TAGS:
            root = coupling_root / tag / "smooth"
            if root.exists():
                coupling_rows.append(
                    evaluate(tag, "coupling_control", COUPLING_DISPLAY[tag], sequence_id, root)
                )

    # ---- tables -------------------------------------------------------------------
    write_json(nested, output_root / "coordination_metrics.json")
    _write_csv(rows, output_root / "coordination_metrics.csv")
    _write_csv(controlled_rows, output_root / "controlled_comparison.csv")
    cold_vs_staged = [
        {**j4_row, "display": "J4 cold-start (controlled, independent init)"},
        {**staged_018_row, "display": "v41 staged (warm-started from v40)"},
    ]
    _write_csv(cold_vs_staged, output_root / "cold_vs_staged_018.csv")
    aggregate = _aggregate(six_rows)
    _write_csv([*six_rows, *aggregate], output_root / "six_sequence_comparison.csv")
    if coupling_rows:
        _write_csv(coupling_rows, output_root / "coupling_ablation_018.csv")

    columns = [
        ("display", "Variant"),
        ("joint_precision", "Joint P"),
        ("joint_recall", "Joint R / coverage"),
        ("joint_f1", "Joint F1"),
        ("joint_iou", "IoU"),
        ("joint_onset_error_frames", "Onset err [f]"),
        ("joint_end_error_frames", "End err [f]"),
        ("longest_true_joint_run_frames", "Longest run [f]"),
        ("dropout_events", "Dropouts"),
        ("bilateral_max_error_cm", "Bilateral err [cm]"),
        ("robot1_anchor_error_cm", "R1 err [cm]"),
        ("robot2_anchor_error_cm", "R2 err [cm]"),
        ("valid_joint_coverage_tau_0cm", "Valid cov τ=0"),
        ("valid_joint_coverage_tau_0.5cm", "Valid cov τ=0.5"),
        ("valid_joint_coverage_tau_1cm", "Valid cov τ=1"),
    ]
    six_columns = [("sequence_id", "Sequence"), *columns[1:]]
    md = [
        "# Coordination tables (auto-generated)",
        "",
        "## Controlled cold-start ablation, sequence 018 (same independent-IK initialization)",
        "",
        _markdown_table(controlled_rows, columns),
        "",
        "## Cold-start J4 vs staged v41, sequence 018 (different initialization — not a controlled row)",
        "",
        _markdown_table(cold_vs_staged, columns),
        "",
        "## Staged v41, six Join takes",
        "",
        _markdown_table([*six_rows, *aggregate[:2]], six_columns),
        "",
    ]
    if coupling_rows:
        md += [
            "## Single-term coupling controls, sequence 018 (J4 preset, identical init/budget/smoothing)",
            "",
            _markdown_table(coupling_rows, columns),
            "",
        ]
    (tables / "tables.md").write_text("\n".join(md), encoding="utf-8")

    # ---- figures ------------------------------------------------------------------
    staged_signals, staged_metrics = signals_018["v41_staged"]
    plot_joint_contact_timeline(
        staged_signals, staged_metrics, figures / "joint_contact_timeline.png"
    )
    variant_items = [signals_018[v] for v in ("J0", "J2", "J4")] + [signals_018["v41_staged"]]
    plot_joint_contact_timeline_variants(
        variant_items, figures / "joint_contact_timeline_variants.png"
    )
    comparison_rows = [
        {**controlled_rows[0], "display": "J0 indep."},
        {**controlled_rows[2], "display": "J2 phase/partner"},
        {**controlled_rows[4], "display": "J4 cold-start"},
        {**staged_018_row, "display": "v41 staged"},
    ]
    plot_coordination_comparison(comparison_rows, figures / "coordination_comparison.png")

    # ---- provenance ---------------------------------------------------------------
    git = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    provenance = {
        "command": " ".join(sys.argv),
        "python": platform.python_version(),
        "git_commit": git.stdout.strip() if git.returncode == 0 else "UNAVAILABLE",
        "paper_root": str(paper_root),
        "coupling_root": str(coupling_root) if coupling_root.exists() else None,
        "frozen_root": str(frozen_root),
        "frozen_copy_hash_check": hash_check,
        "all_frozen_copies_identical": all(
            entry["identical"] for seq in hash_check.values() for entry in seq.values()
        ),
        "n_results": len(nested),
    }
    write_json(provenance, output_root / "provenance.json")
    return provenance


def evaluate_single(
    *,
    result_root: Path,
    sequence_id: str,
    label: str,
    output_path: Path,
    motion_root: Path = Path("data/core4d/motions"),
    object_models_root: Path = Path("data/core4d/object_models"),
    contact_analysis_root: Path = Path("outputs/contact_analysis"),
) -> dict[str, object]:
    metrics, _ = evaluate_result(
        motion_root, sequence_id, object_models_root, contact_analysis_root, result_root, label=label
    )
    write_json(metrics, output_path)
    return metrics

"""Part E evaluation for the EXT-JC visual refinement.

Evaluates every reference method (frozen v41, OA-Independent, EXT-JC w10,
separable surrogate) and every refinement candidate found under
``runs_dev/<tag>`` / ``runs_test/<tag>`` with the FROZEN metric contract
(smoothing recipe -> ``evaluate-paper-join`` -> coordination metrics), adds the
frame-level visual diagnostics, and writes

* ``refinement_metrics.csv``        one row per (method, sequence),
* ``configuration_comparison.csv``  one row per configuration with dev/val
  aggregates, deltas vs. EXT-JC w10 and vs. the configuration's own
  separable-surrogate control.

Nothing here tunes anything: all thresholds live in the frozen evaluator.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ext_jc_visual import default_roots, summary_row

from core4d_dual_retarget.oa_extension import (
    PRIMARY_COLUMNS,
    SECONDARY_COLUMNS,
    evaluate_variant,
    write_csv,
)
from core4d_dual_retarget.visual_refinement import load_frame_diagnostics

ROOT = Path("outputs/paper_eval/ext_jc_visual_refinement")
SEQUENCES = ["20231108/018", "20231108/019", "20231108/020", "20231108/108", "20231108/109", "20231108/110"]
SPLIT = {
    "20231108/018": "dev",
    "20231108/019": "dev",
    "20231108/108": "dev",
    "20231108/020": "val",
    "20231108/109": "test",
    "20231108/110": "test",
}
REFERENCE_LABELS = {
    "v41": "Frozen v41",
    "oaind": "OA-Independent",
    "ext": "EXT-JC w10",
    "surrogate": "EXT w10 surrogate",
}
COMPARISON_METRICS = [
    "valid_joint_coverage_tau_0.5cm",
    "valid_joint_f1_tau_0.5cm",
    "joint_f1",
    "longest_valid_true_joint_run_frames_tau_0.5cm",
    "bilateral_max_error_cm",
    "joint_onset_error_frames",
    "paper_hand_object_penetration_max_cm",
    "diag_hand_pen_max_cm",
    "diag_frames_pen_gt_0p5",
    "diag_arm_jerk_max",
    "diag_waist_jerk_max",
    "paper_joint_jerk_rms_rad_s3",
    "paper_root_jerk_rms_m_s3",
    "paper_non_skating_frame_frac",
    "paper_planted_foot_speed_mean_cm_s",
    "paper_semantic_keypoint_error_cm",
    "paper_root_trajectory_error_cm",
    "paper_contact_f1",
]
HIGHER_IS_BETTER = {
    "valid_joint_coverage_tau_0.5cm",
    "valid_joint_f1_tau_0.5cm",
    "joint_f1",
    "longest_valid_true_joint_run_frames_tau_0.5cm",
    "paper_non_skating_frame_frac",
    "paper_contact_f1",
}


def diag_columns(seq: str, smooth_root: Path, label: str) -> dict[str, object]:
    diag = load_frame_diagnostics(seq, smooth_root, label=label)
    row = summary_row(diag)
    out: dict[str, object] = {}
    out["diag_hand_pen_max_cm"] = max(row["r1_hand_pen_max_cm"], row["r2_hand_pen_max_cm"])
    out["diag_frames_pen_gt_0p5"] = row["r1_frames_pen_gt_0p5"] + row["r2_frames_pen_gt_0p5"]
    out["diag_arm_jerk_max"] = max(row["r1_arm_jerk_max"], row["r2_arm_jerk_max"])
    out["diag_waist_jerk_max"] = max(row["r1_waist_jerk_max"], row["r2_waist_jerk_max"])
    out["diag_root_jerk_max"] = max(row["r1_root_jerk_max"], row["r2_root_jerk_max"])
    for key, value in row.items():
        if key.startswith("r") and key[1].isdigit():
            out[f"diag_{key}"] = value
    out["diag_joint_recall"] = row["joint_recall"]
    return out


def discover_candidates(root: Path) -> list[tuple[str, str, Path, str]]:
    found = []
    for runs_dir in (root / "runs_dev", root / "runs_test"):
        if not runs_dir.exists():
            continue
        for variant_root in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
            raw = variant_root / "raw"
            if not raw.exists():
                continue
            for stem_dir in sorted(raw.iterdir()):
                if (stem_dir / "robot1.npz").exists() and (stem_dir / "robot2.npz").exists():
                    seq = stem_dir.name.replace("_", "/", 1)
                    found.append((variant_root.name, runs_dir.name, variant_root, seq))
    return found


def config_description(variant_root: Path) -> str:
    cfg_path = variant_root / "config.json"
    if not cfg_path.exists():
        return ""
    payload = json.loads(cfg_path.read_text(encoding="utf-8"))
    overrides = payload.get("overrides", {})
    return "; ".join(f"{k}={v}" for k, v in sorted(overrides.items()))


def _num(value: object) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--sequences", default=",".join(SEQUENCES))
    parser.add_argument("--skip-references", action="store_true")
    args = parser.parse_args()
    root: Path = args.root
    sequences = [s if "/" in s else f"20231108/{s}" for s in args.sequences.split(",")]
    log = root / "evaluation_commands.log"
    rows: list[dict[str, object]] = []

    if not args.skip_references:
        for seq in sequences:
            roots = default_roots(seq)
            for key, label in REFERENCE_LABELS.items():
                smooth_root = roots[key]
                if not (smooth_root / seq.replace("/", "_") / "robot1.npz").exists():
                    print(f"skip reference {label} on {seq}: {smooth_root} missing")
                    continue
                row = evaluate_variant(
                    label=f"ref_{key}",
                    group="reference",
                    variant_root=None,
                    smooth_root=smooth_root,
                    sequence_id=seq,
                    output_dir=root,
                    log=log,
                )
                row.update(
                    {
                        "method": label,
                        "config_tag": key,
                        "coupled": key in ("v41", "ext"),
                        "surrogate_of": "ext" if key == "surrogate" else "",
                        "split": SPLIT[seq],
                        "config_overrides": "joint_contact_continuity_weight=10.0"
                        + ("; joint_contact_separable_surrogate=true" if key == "surrogate" else "")
                        if key in ("ext", "surrogate")
                        else "",
                    }
                )
                row.update(diag_columns(seq, smooth_root, label))
                rows.append(row)
                print("evaluated", label, seq)

    for tag, group, variant_root, seq in discover_candidates(root):
        if seq not in sequences:
            continue
        row = evaluate_variant(
            label=tag,
            group=group,
            variant_root=variant_root,
            smooth_root=variant_root / "smooth",
            sequence_id=seq,
            output_dir=root,
            log=log,
        )
        is_sur = tag.endswith("_surrogate")
        row.update(
            {
                "method": tag,
                "config_tag": tag,
                "coupled": not is_sur,
                "surrogate_of": tag.removesuffix("_surrogate") if is_sur else "",
                "split": SPLIT[seq],
                "config_overrides": config_description(variant_root),
            }
        )
        row.update(diag_columns(seq, variant_root / "smooth", tag))
        rows.append(row)
        print("evaluated", tag, seq)

    # Incremental runs must not drop rows from earlier runs.
    metrics_csv = root / "refinement_metrics.csv"
    fresh = {(str(r["method"]), str(r["sequence_id"])) for r in rows}
    if metrics_csv.exists():
        import csv as _csv

        with metrics_csv.open(encoding="utf-8") as handle:
            for old in _csv.DictReader(handle):
                if (old["method"], old["sequence_id"]) not in fresh:
                    rows.append(old)

    lead = ["method", "config_tag", "group", "sequence_id", "split", "coupled", "surrogate_of", "config_overrides"]
    diag_keys = sorted({k for r in rows for k in r if k.startswith("diag_")})
    ordered = []
    for r in rows:
        o = {k: r.get(k) for k in lead + PRIMARY_COLUMNS + SECONDARY_COLUMNS + diag_keys}
        o.update({k: v for k, v in r.items() if k not in o})
        ordered.append(o)
    write_csv(ordered, root / "refinement_metrics.csv")

    # ---- configuration comparison -------------------------------------------------
    by_tag: dict[str, dict[str, dict[str, object]]] = {}
    for r in rows:
        by_tag.setdefault(str(r["config_tag"]), {})[str(r["sequence_id"])] = r

    def agg(tag: str, split: str, metric: str) -> tuple[float, float, int]:
        vals = [
            _num(r.get(metric))
            for s, r in by_tag.get(tag, {}).items()
            if SPLIT[s] == split and not np.isnan(_num(r.get(metric)))
        ]
        if not vals:
            return float("nan"), float("nan"), 0
        return float(np.mean(vals)), float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0, len(vals)

    def paired_wins(tag_a: str, tag_b: str, metric: str, splits: tuple[str, ...]) -> str:
        wins = ties = losses = 0
        for s in SEQUENCES:
            if SPLIT[s] not in splits:
                continue
            a = by_tag.get(tag_a, {}).get(s)
            b = by_tag.get(tag_b, {}).get(s)
            if a is None or b is None:
                continue
            va, vb = _num(a.get(metric)), _num(b.get(metric))
            if np.isnan(va) or np.isnan(vb):
                continue
            better = va > vb if metric in HIGHER_IS_BETTER else va < vb
            if abs(va - vb) < 1e-9:
                ties += 1
            elif better:
                wins += 1
            else:
                losses += 1
        return f"{wins}/{ties}/{losses}"

    comparison: list[dict[str, object]] = []
    ext_adv: dict[tuple[str, str], float] = {}
    tags = [t for t in by_tag if not t.endswith("_surrogate") and t not in ("v41", "oaind", "surrogate")]
    tags = sorted(tags, key=lambda t: (t != "ext", t))
    for tag in tags:
        sur = "surrogate" if tag == "ext" else f"{tag}_surrogate"
        sample = next(iter(by_tag[tag].values()))
        comp: dict[str, object] = {
            "config": "EXT-JC w10 (frozen reference)" if tag == "ext" else tag,
            "config_tag": tag,
            "surrogate_control": sur if sur in by_tag else "",
            "config_overrides": sample.get("config_overrides", ""),
            "dev_sequences": ",".join(s[-3:] for s in sorted(by_tag[tag]) if SPLIT[s] == "dev"),
            "val_sequences": ",".join(s[-3:] for s in sorted(by_tag[tag]) if SPLIT[s] == "val"),
            "test_sequences": ",".join(s[-3:] for s in sorted(by_tag[tag]) if SPLIT[s] == "test"),
        }
        for metric in COMPARISON_METRICS:
            for split in ("dev", "val", "test"):
                m, sd, n = agg(tag, split, metric)
                if n == 0:
                    continue
                comp[f"{split}_{metric}_mean"] = round(m, 4)
                if split == "dev":
                    comp[f"{split}_{metric}_sd"] = round(sd, 4)
                em, _, en = agg("ext", split, metric)
                if en:
                    comp[f"{split}_{metric}_delta_vs_ext"] = round(m - em, 4)
                if sur in by_tag:
                    sm, _, sn = agg(sur, split, metric)
                    if sn:
                        comp[f"{split}_{metric}_delta_vs_surrogate"] = round(m - sm, 4)
                        if tag == "ext":
                            ext_adv[(split, metric)] = m - sm
                        elif (split, metric) in ext_adv and abs(ext_adv[(split, metric)]) > 1e-9:
                            comp[f"{split}_{metric}_advantage_retained_frac"] = round(
                                (m - sm) / ext_adv[(split, metric)], 3
                            )
        if sur in by_tag:
            for metric in ("valid_joint_coverage_tau_0.5cm", "joint_f1", "diag_hand_pen_max_cm"):
                comp[f"devval_wins_ties_losses_vs_surrogate_{metric}"] = paired_wins(
                    tag, sur, metric, ("dev", "val")
                )
        for metric in ("valid_joint_coverage_tau_0.5cm", "diag_hand_pen_max_cm", "paper_joint_jerk_rms_rad_s3"):
            comp[f"devval_wins_ties_losses_vs_ext_{metric}"] = paired_wins(tag, "ext", metric, ("dev", "val"))
        comparison.append(comp)
    write_csv(comparison, root / "configuration_comparison.csv")
    print(f"wrote {root / 'refinement_metrics.csv'} ({len(rows)} rows) and configuration_comparison.csv ({len(comparison)} configs)")


if __name__ == "__main__":
    main()

"""Build the Phase 7 tables and figure from ``all_variant_metrics.csv``.

Reads the evaluation produced by ``scripts/oa_run_evaluation.py`` and writes
``oa_independent_metrics.csv``, ``coupling_ablation_metrics.csv``,
``per_sequence_comparison.csv``, ``coordination_robustness.csv``,
``paper_ready_results_table.csv`` and ``paper_ready_comparison.{png,pdf}``.
No metric is recomputed here; the script only selects, aggregates and plots.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from core4d_dual_retarget.oa_extension import aggregate, write_csv

ROOT = Path("outputs/paper_eval/object_aware_baseline_extension")
SPLIT = {
    "20231108/018": "train",
    "20231108/019": "train",
    "20231108/108": "train",
    "20231108/020": "val",
    "20231108/109": "test",
    "20231108/110": "test",
}
PRIMARY = [
    ("valid_joint_coverage_tau_0.5cm", "geometry-valid joint coverage @0.5 cm", True),
    ("valid_joint_f1_tau_0.5cm", "geometry-valid joint F1 @0.5 cm", True),
    ("joint_f1", "joint-contact F1", True),
    ("joint_iou", "joint-contact IoU", True),
    ("bilateral_max_error_cm", "bilateral anchor error (cm)", False),
    (
        "longest_valid_true_joint_run_frames_tau_0.5cm",
        "longest valid simultaneous span (frames)",
        True,
    ),
    ("joint_onset_error_frames", "joint onset error (frames)", None),
    ("onset_lag_error_frames", "onset-lag error (frames)", None),
]
SECONDARY = [
    ("paper_hand_object_penetration_max_cm", "hand penetration max (cm)", False),
    ("paper_foot_object_penetration_max_cm", "foot penetration max (cm)", False),
    ("paper_geometry_violation_cm", "geometry violation (cm)", False),
    ("paper_planted_foot_speed_mean_cm_s", "planted-foot speed (cm/s)", False),
    ("paper_non_skating_frame_frac", "non-skating frame fraction", True),
    ("paper_joint_jerk_rms_rad_s3", "joint jerk RMS (rad/s^3)", False),
    ("paper_root_jerk_rms_m_s3", "root jerk RMS (m/s^3)", False),
    ("paper_semantic_keypoint_error_cm", "keypoint error (cm)", False),
    ("paper_root_trajectory_error_cm", "root trajectory error (cm)", False),
    ("paper_contact_f1", "per-robot contact F1 (paper)", True),
    ("inter_robot_min_keypoint_distance_m", "inter-robot min keypoint distance (m)", True),
]
KEYS = [k for k, _, _ in PRIMARY + SECONDARY]


def fnum(value: object) -> float | None:
    if value in (None, "", "None"):
        return None
    if isinstance(value, str) and value in ("True", "False"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_rows() -> list[dict[str, object]]:
    with (ROOT / "all_variant_metrics.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in KEYS + ["frames", "expected_joint_frames", "predicted_joint_frames"]:
            if key in row:
                value = fnum(row[key])
                row[key] = value if value is not None else row[key]
        for key in ("grasp_order_preserved", "release_order_preserved"):
            if row.get(key) in ("True", "False"):
                row[key] = row[key] == "True"
    return rows


METHOD_INFO = {
    "frozen/v41": ("Joint (frozen v41)", "staged"),
    "018/joint_J4_cold": ("Joint", "cold"),
    "018/joint_J4_cold_perRobotClip": ("Joint, per-robot clipping (diagnostic)", "cold"),
    "018/oaind_J4_cold": ("OA-Independent", "cold"),
    "018/joint_v41_staged": ("Joint (= frozen v41, reproduced)", "staged"),
    "018/oaind_v41_staged": ("OA-Independent", "staged"),
    "ext_018/ext_jc_w10_cold": ("EXT joint-contact w10 (coupled)", "cold"),
    "ext_018/ext_jc_w10_surrogate_cold": ("EXT w10 separable surrogate", "cold"),
    "ext_018/ext_jc_w40_cold": ("EXT joint-contact w40 (coupled)", "cold"),
    "ext_018/ext_jc_w40_surrogate_cold": ("EXT w40 separable surrogate", "cold"),
    "ext_018/ext_jc_w40_staged": ("EXT joint-contact w40 (coupled)", "staged"),
    "ext_018/ext_jc_w40_surrogate_staged": ("EXT w40 separable surrogate", "staged"),
    "multiseq/joint_v41_staged": ("Joint (= frozen v41, reproduced)", "staged"),
    "multiseq/oaind_v41_staged": ("OA-Independent", "staged"),
    "multiseq/ext_jc_w10_staged": ("EXT joint-contact w10 (coupled)", "staged"),
    "multiseq/ext_jc_w10_surrogate_staged": ("EXT w10 separable surrogate", "staged"),
}


def describe(label: str) -> tuple[str, str]:
    if label in METHOD_INFO:
        return METHOD_INFO[label]
    if label.startswith("stress_018/joint"):
        return ("Joint", "cold+perturbation")
    if label.startswith("stress_018/oaind"):
        return ("OA-Independent", "cold+perturbation")
    return (label, "?")


def select(rows, cols):
    out = []
    for row in rows:
        method, protocol = describe(str(row["label"]))
        rec = {
            "label": row["label"],
            "method": method,
            "protocol": protocol,
            "sequence_id": row["sequence_id"],
            "split": SPLIT.get(str(row["sequence_id"])),
        }
        for key in cols:
            rec[key] = row.get(key)
        rec["robot1_md5"] = row.get("robot1_md5")
        rec["robot2_md5"] = row.get("robot2_md5")
        out.append(rec)
    return out


def main() -> None:
    rows = load_rows()
    by_label = {}
    for row in rows:
        by_label.setdefault(str(row["label"]), []).append(row)

    # ---- OA-Independent vs Joint ------------------------------------------------
    oa_labels = [
        "018/joint_J4_cold",
        "018/oaind_J4_cold",
        "018/joint_J4_cold_perRobotClip",
        "018/joint_v41_staged",
        "018/oaind_v41_staged",
        "frozen/v41",
        "multiseq/joint_v41_staged",
        "multiseq/oaind_v41_staged",
    ]
    oa_rows = [r for r in rows if r["label"] in oa_labels]
    write_csv(select(oa_rows, KEYS), ROOT / "oa_independent_metrics.csv")

    # ---- coupling ablation (research extension ON / surrogate / OFF) -------------
    abl_labels = [
        "018/joint_J4_cold",
        "018/oaind_J4_cold",
        "ext_018/ext_jc_w10_cold",
        "ext_018/ext_jc_w10_surrogate_cold",
        "ext_018/ext_jc_w40_cold",
        "ext_018/ext_jc_w40_surrogate_cold",
        "frozen/v41",
        "018/oaind_v41_staged",
        "ext_018/ext_jc_w40_staged",
        "ext_018/ext_jc_w40_surrogate_staged",
        "multiseq/ext_jc_w10_staged",
        "multiseq/ext_jc_w10_surrogate_staged",
        "multiseq/oaind_v41_staged",
    ]
    abl_rows = [r for r in rows if r["label"] in abl_labels]
    abl = select(abl_rows, KEYS)
    for rec in abl:
        lab = str(rec["label"])
        rec["cross_agent_term"] = (
            "joint-contact product (coupled)"
            if "surrogate" not in lab and "ext_jc" in lab
            else "separable surrogate"
            if "surrogate" in lab
            else "none (inter-robot clearance inactive)"
        )
    write_csv(abl, ROOT / "coupling_ablation_metrics.csv")

    # ---- per-sequence comparison (staged protocol, six takes) ---------------------
    per_seq = []
    method_labels = {
        "Joint (frozen v41)": ["frozen/v41"],
        "OA-Independent": ["018/oaind_v41_staged", "multiseq/oaind_v41_staged"],
        "EXT joint-contact w10 (coupled)": ["multiseq/ext_jc_w10_staged"],
        "EXT w10 separable surrogate": ["multiseq/ext_jc_w10_surrogate_staged"],
    }
    summary_rows = []
    for method, labels in method_labels.items():
        recs = [r for r in rows if r["label"] in labels]
        for r in sorted(recs, key=lambda r: str(r["sequence_id"])):
            rec = {
                "method": method,
                "sequence_id": r["sequence_id"],
                "split": SPLIT.get(str(r["sequence_id"])),
                "label": r["label"],
            }
            for key in KEYS:
                rec[key] = r.get(key)
            per_seq.append(rec)
        for scope, flt in (
            ("all", lambda r: True),
            ("train", lambda r: SPLIT.get(str(r["sequence_id"])) == "train"),
            ("val", lambda r: SPLIT.get(str(r["sequence_id"])) == "val"),
            ("test", lambda r: SPLIT.get(str(r["sequence_id"])) == "test"),
            ("non-dev (019-110)", lambda r: str(r["sequence_id"]) != "20231108/018"),
        ):
            sub = [r for r in recs if flt(r)]
            if not sub:
                continue
            rec = {
                "method": method,
                "scope": scope,
                "sequences": ",".join(sorted(str(r["sequence_id"])[-3:] for r in sub)),
            }
            rec.update(aggregate(sub, KEYS))
            summary_rows.append(rec)
    write_csv(per_seq, ROOT / "per_sequence_comparison.csv")
    write_csv(summary_rows, ROOT / "per_sequence_summary_mean_sd.csv")

    # ---- coordination robustness ----------------------------------------------------
    rob = []
    base = {
        "Joint": by_label.get("018/joint_J4_cold", []),
        "OA-Independent": by_label.get("018/oaind_J4_cold", []),
    }
    for method, prefix in (
        ("Joint", "stress_018/joint_cold_seed"),
        ("OA-Independent", "stress_018/oaind_cold_seed"),
    ):
        seeds = []
        for seed in (0, 1, 2):
            recs = by_label.get(f"{prefix}{seed}", [])
            if recs:
                r = recs[0]
                rec = {
                    "method": method,
                    "seed": seed,
                    "perturbation": "root-xy 0.05 m (pre-registered)",
                }
                for key in KEYS:
                    rec[key] = r.get(key)
                rob.append(rec)
                seeds.append(r)
        if base[method]:
            rec = {"method": method, "seed": "none", "perturbation": "unperturbed"}
            for key in KEYS:
                rec[key] = base[method][0].get(key)
            rob.append(rec)
        if seeds:
            rec = {
                "method": method,
                "seed": f"mean_sd_over_{len(seeds)}",
                "perturbation": "root-xy 0.05 m",
            }
            rec.update(aggregate(seeds, KEYS))
            if base[method]:
                for key in KEYS:
                    b = fnum(base[method][0].get(key))
                    vals = [fnum(s.get(key)) for s in seeds]
                    vals = [v for v in vals if v is not None]
                    if b is not None and vals:
                        rec[f"{key}_max_abs_dev_from_unperturbed"] = max(abs(v - b) for v in vals)
            rob.append(rec)
    write_csv(rob, ROOT / "coordination_robustness.csv")

    # ---- paper-ready table -----------------------------------------------------------
    table = []

    def add(method, protocol, scope, recs):
        if not recs:
            return
        rec = {"method": method, "protocol": protocol, "scope": scope, "n": len(recs)}
        agg = aggregate(recs, KEYS)
        for key, _, _ in PRIMARY + SECONDARY:
            m, s = agg.get(f"{key}_mean"), agg.get(f"{key}_sd")
            rec[key] = m
            rec[f"{key}_sd"] = s if len(recs) > 1 else None
        table.append(rec)

    add("Joint (frozen v41 config)", "cold", "018", by_label.get("018/joint_J4_cold"))
    add("OA-Independent", "cold", "018", by_label.get("018/oaind_J4_cold"))
    add("EXT joint-contact w10 (coupled)", "cold", "018", by_label.get("ext_018/ext_jc_w10_cold"))
    add(
        "EXT w10 separable surrogate",
        "cold",
        "018",
        by_label.get("ext_018/ext_jc_w10_surrogate_cold"),
    )
    add(
        "Joint (frozen v41)",
        "staged",
        "018",
        by_label.get("frozen/v41", [])[:1]
        and [r for r in by_label["frozen/v41"] if r["sequence_id"] == "20231108/018"],
    )
    add("OA-Independent", "staged", "018", by_label.get("018/oaind_v41_staged"))
    add(
        "EXT joint-contact w40 (coupled)",
        "staged",
        "018",
        by_label.get("ext_018/ext_jc_w40_staged"),
    )
    add(
        "EXT w40 separable surrogate",
        "staged",
        "018",
        by_label.get("ext_018/ext_jc_w40_surrogate_staged"),
    )
    for method, labels in method_labels.items():
        recs = [r for r in rows if r["label"] in labels]
        add(method, "staged", "six takes (mean ± SD)", recs)
        add(
            method,
            "staged",
            "test 109/110 (mean ± SD)",
            [r for r in recs if SPLIT.get(str(r["sequence_id"])) == "test"],
        )
    add(
        "Joint",
        "cold+perturbation",
        "018, 3 seeds",
        [
            by_label[f"stress_018/joint_cold_seed{s}"][0]
            for s in (0, 1, 2)
            if f"stress_018/joint_cold_seed{s}" in by_label
        ],
    )
    add(
        "OA-Independent",
        "cold+perturbation",
        "018, 3 seeds",
        [
            by_label[f"stress_018/oaind_cold_seed{s}"][0]
            for s in (0, 1, 2)
            if f"stress_018/oaind_cold_seed{s}" in by_label
        ],
    )
    write_csv(table, ROOT / "paper_ready_results_table.csv")

    # ---- figure ----------------------------------------------------------------------
    panels = [
        (
            "valid_joint_coverage_tau_0.5cm",
            "Geometry-valid joint coverage\n(τ = 0.5 cm) ↑",
            (0, 1.05),
        ),
        ("joint_f1", "Joint-contact F1 ↑", (0, 1.05)),
        ("bilateral_max_error_cm", "Bilateral anchor error (cm) ↓", None),
        ("paper_hand_object_penetration_max_cm", "Hand penetration max (cm) ↓", None),
        ("paper_joint_jerk_rms_rad_s3", "Joint jerk RMS (rad/s³) ↓", None),
        ("paper_planted_foot_speed_mean_cm_s", "Planted-foot speed (cm/s) ↓", None),
    ]
    groups = [
        (
            "018 cold start",
            [
                ("Joint", by_label.get("018/joint_J4_cold", [])),
                ("OA-Ind.", by_label.get("018/oaind_J4_cold", [])),
                ("EXT w10", by_label.get("ext_018/ext_jc_w10_cold", [])),
                ("EXT w10\nsurrogate", by_label.get("ext_018/ext_jc_w10_surrogate_cold", [])),
            ],
        ),
        (
            "018 staged (v40 init)",
            [
                (
                    "Joint\n(frozen v41)",
                    [
                        r
                        for r in by_label.get("frozen/v41", [])
                        if r["sequence_id"] == "20231108/018"
                    ],
                ),
                ("OA-Ind.", by_label.get("018/oaind_v41_staged", [])),
                ("EXT w40", by_label.get("ext_018/ext_jc_w40_staged", [])),
                ("EXT w40\nsurrogate", by_label.get("ext_018/ext_jc_w40_surrogate_staged", [])),
            ],
        ),
        (
            "019–110 staged, mean ± SD (fixed configs)",
            [
                (
                    "Joint\n(frozen v41)",
                    [
                        r
                        for r in by_label.get("frozen/v41", [])
                        if r["sequence_id"] != "20231108/018"
                    ],
                ),
                ("OA-Ind.", by_label.get("multiseq/oaind_v41_staged", [])),
                ("EXT w10", by_label.get("multiseq/ext_jc_w10_staged", [])),
                ("EXT w10\nsurrogate", by_label.get("multiseq/ext_jc_w10_surrogate_staged", [])),
            ],
        ),
        (
            "018 cold + 0.05 m root perturbation, 3 seeds",
            [
                (
                    "Joint",
                    [
                        by_label[f"stress_018/joint_cold_seed{s}"][0]
                        for s in (0, 1, 2)
                        if f"stress_018/joint_cold_seed{s}" in by_label
                    ],
                ),
                (
                    "OA-Ind.",
                    [
                        by_label[f"stress_018/oaind_cold_seed{s}"][0]
                        for s in (0, 1, 2)
                        if f"stress_018/oaind_cold_seed{s}" in by_label
                    ],
                ),
            ],
        ),
    ]
    colors = {"Joint": "#4c72b0", "OA-Ind.": "#dd8452", "EXT": "#55a868", "surrogate": "#8c8c8c"}
    fig, axes = plt.subplots(
        len(panels), len(groups), figsize=(4.2 * len(groups), 2.6 * len(panels)), squeeze=False
    )
    for gi, (gtitle, members) in enumerate(groups):
        for pi, (key, ptitle, ylim) in enumerate(panels):
            ax = axes[pi][gi]
            names, means, sds, cols = [], [], [], []
            for name, recs in members:
                vals = [fnum(r.get(key)) for r in recs]
                vals = [v for v in vals if v is not None]
                names.append(name)
                means.append(np.mean(vals) if vals else np.nan)
                sds.append(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)
                c = (
                    colors["surrogate"]
                    if "surrogate" in name
                    else colors["EXT"]
                    if name.startswith("EXT")
                    else colors["OA-Ind."]
                    if "OA" in name
                    else colors["Joint"]
                )
                cols.append(c)
            x = np.arange(len(names))
            ax.bar(
                x,
                means,
                yerr=sds if any(s > 0 for s in sds) else None,
                color=cols,
                capsize=3,
                width=0.65,
            )
            for xi, m, s in zip(x, means, sds, strict=True):
                if not np.isnan(m):
                    ax.text(
                        xi,
                        m
                        + (s if s else 0)
                        + (0.01 if ylim else 0.02 * max(1e-9, np.nanmax(means))),
                        f"{m:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                    )
            ax.set_xticks(x)
            ax.set_xticklabels(names, fontsize=7)
            if ylim:
                ax.set_ylim(*ylim)
            else:
                tops = [m + s for m, s in zip(means, sds, strict=True) if not np.isnan(m)]
                ax.set_ylim(0, (max(tops) if tops else 0) * 1.35 + 1e-9)
            if gi == 0:
                ax.set_ylabel(ptitle, fontsize=8)
            if pi == 0:
                ax.set_title(gtitle, fontsize=9)
            ax.tick_params(axis="y", labelsize=7)
            ax.grid(axis="y", alpha=0.3)
    fig.suptitle(
        "Object-aware independent baseline vs joint solver (frozen v41 config) vs research-extension coupling term\n"
        "same inputs, init, budget, smoothing and frozen evaluator; EXT = joint-contact continuity (not part of v41)",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(ROOT / "paper_ready_comparison.png", dpi=180)
    fig.savefig(ROOT / "paper_ready_comparison.pdf")
    print("wrote tables and figure;", len(table), "table rows")
    (ROOT / "paper_ready_results_table.json").write_text(
        json.dumps(table, indent=2, default=str) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()

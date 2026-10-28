"""Evaluate every finished variant of the object-aware baseline extension.

Discovers ``runs_*/<tag>/raw/<stem>/robot{1,2}.npz`` under the extension
root, applies the frozen smoothing recipe + frozen paper evaluator + the
post-hoc coordination metrics, and writes the comparison CSVs.  Frozen v41
(`outputs/balanced_clearance_v41_smooth`) is evaluated read-only as the
reference.  Re-running is idempotent (smoothing / paper metrics are cached).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from core4d_dual_retarget.oa_extension import (
    PRIMARY_COLUMNS,
    SECONDARY_COLUMNS,
    aggregate,
    compare_raw,
    evaluate_variant,
    term_log_summary,
    write_csv,
)

ROOT = Path("outputs/paper_eval/object_aware_baseline_extension")
FROZEN_SMOOTH = Path("outputs/balanced_clearance_v41_smooth")
FROZEN_RAW = Path("outputs/balanced_clearance_v41")
SEQUENCES = [
    "20231108/018",
    "20231108/019",
    "20231108/020",
    "20231108/108",
    "20231108/109",
    "20231108/110",
]
SPLIT = {
    "20231108/018": "train",
    "20231108/019": "train",
    "20231108/108": "train",
    "20231108/020": "val",
    "20231108/109": "test",
    "20231108/110": "test",
}


def discover(root: Path) -> list[tuple[str, str, Path, str]]:
    found = []
    for runs_dir in sorted(root.glob("runs_*")):
        group = runs_dir.name.removeprefix("runs_")
        for variant_root in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
            raw = variant_root / "raw"
            if not raw.exists():
                continue
            for stem_dir in sorted(raw.iterdir()):
                if (stem_dir / "robot1.npz").exists() and (stem_dir / "robot2.npz").exists():
                    seq = stem_dir.name.replace("_", "/", 1)
                    found.append((f"{group}/{variant_root.name}", group, variant_root, seq))
    return found


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--skip-frozen", action="store_true")
    args = parser.parse_args()
    root = args.root
    log = root / "evaluation_commands.log"
    rows: list[dict[str, object]] = []

    if not args.skip_frozen:
        for seq in SEQUENCES:
            stem = seq.replace("/", "_")
            if not (FROZEN_SMOOTH / stem / "robot1.npz").exists():
                continue
            row = evaluate_variant(
                label="frozen/v41",
                group="frozen",
                variant_root=None,
                smooth_root=FROZEN_SMOOTH,
                sequence_id=seq,
                output_dir=root,
                log=log,
            )
            row["split"] = SPLIT[seq]
            rows.append(row)

    for label, group, variant_root, seq in discover(root):
        row = evaluate_variant(
            label=label,
            group=group,
            variant_root=variant_root,
            smooth_root=variant_root / "smooth",
            sequence_id=seq,
            output_dir=root,
            log=log,
        )
        row["split"] = SPLIT.get(seq)
        config_path = variant_root / "config.json"
        if not config_path.exists():
            config_path = variant_root / "config_robot1.json"
        if config_path.exists():
            payload = json.loads(config_path.read_text(encoding="utf-8"))
            row["baseline_root"] = payload.get("baseline_root")
            row["overrides"] = json.dumps(payload.get("overrides", {}))
            row["initial_root_offsets_xy"] = json.dumps(payload.get("initial_root_offsets_xy"))
        rows.append(row)

    write_csv(rows, root / "all_variant_metrics.csv")
    (root / "all_variant_metrics.json").write_text(
        json.dumps(rows, indent=2, default=str) + "\n", encoding="utf-8"
    )

    # Raw-trajectory comparisons between Joint and OA-Independent (same init).
    comparisons = []
    pairs = [
        ("018/joint_J4_cold", "018/oaind_J4_cold"),
        ("018/joint_v41_staged", "018/oaind_v41_staged"),
    ]
    for seed in (0, 1, 2):
        pairs.append((f"stress_018/joint_cold_seed{seed}", f"stress_018/oaind_cold_seed{seed}"))
    for a, b in pairs:
        ra = root / f"runs_{a}" / "raw"
        rb = root / f"runs_{b}" / "raw"
        if (ra / "20231108_018" / "robot1.npz").exists() and (
            rb / "20231108_018" / "robot2.npz"
        ).exists():
            comparisons.append(compare_raw(ra, rb, "20231108/018", label_a=a, label_b=b))
    # Reproduction checks against frozen references.
    checks = [
        ("018/joint_v41_staged", FROZEN_RAW, "20231108/018", "frozen/v41_raw"),
        (
            "018/joint_J4_cold",
            Path("outputs/paper_eval_join/ablation_018/J4/raw"),
            "20231108/018",
            "paper_eval_join/J4_raw",
        ),
    ]
    multiseq = root / "runs_multiseq"
    if multiseq.exists():
        for seq in SEQUENCES[1:]:
            checks.append(("multiseq/joint_v41_staged", FROZEN_RAW, seq, "frozen/v41_raw"))
            pairs_ms = ("multiseq/joint_v41_staged", "multiseq/oaind_v41_staged")
            ra = root / f"runs_{pairs_ms[0]}" / "raw"
            rb = root / f"runs_{pairs_ms[1]}" / "raw"
            stem = seq.replace("/", "_")
            if (ra / stem / "robot1.npz").exists() and (rb / stem / "robot2.npz").exists():
                comparisons.append(
                    compare_raw(ra, rb, seq, label_a=pairs_ms[0], label_b=pairs_ms[1])
                )
    for label, ref_root, seq, ref_label in checks:
        ra = root / f"runs_{label}" / "raw"
        stem = seq.replace("/", "_")
        if (ra / stem / "robot1.npz").exists() and (ref_root / stem / "robot1.npz").exists():
            comparisons.append(compare_raw(ra, ref_root, seq, label_a=label, label_b=ref_label))
    write_csv(comparisons, root / "raw_trajectory_comparisons.csv")

    # Term logs.
    logs = []
    for path in sorted(root.glob("runs_*/*/term_log_*.csv")):
        summary = term_log_summary(path)
        summary["variant"] = f"{path.parent.parent.name.removeprefix('runs_')}/{path.parent.name}"
        logs.append(summary)
    (root / "loss_activation_logs.json").write_text(
        json.dumps(logs, indent=2) + "\n", encoding="utf-8"
    )
    flat_logs = []
    for s in logs:
        rec = {
            "variant": s["variant"],
            "term_log": s["path"],
            "iterations": s["iterations"],
            "loss_first": s.get("loss_first"),
            "loss_last": s.get("loss_last"),
            "loss_min": s.get("loss_min"),
        }
        for name in ("inter_robot_raw", "joint_contact_raw"):
            for k, v in s.get(name, {}).items():
                rec[f"{name}_{k}"] = v
        flat_logs.append(rec)
    write_csv(flat_logs, root / "loss_activation_logs.csv")

    # Per-group aggregates.
    agg_rows = []
    by_label: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_label.setdefault(str(row["label"]), []).append(row)
    for label, group_rows in by_label.items():
        rec = {"label": label, "sequences": ",".join(str(r["sequence_id"]) for r in group_rows)}
        rec.update(aggregate(group_rows, PRIMARY_COLUMNS + SECONDARY_COLUMNS))
        agg_rows.append(rec)
    write_csv(agg_rows, root / "aggregates_by_label.csv")
    print(f"evaluated {len(rows)} (variant, sequence) rows; {len(comparisons)} raw comparisons")


if __name__ == "__main__":
    main()

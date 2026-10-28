"""Part A of the EXT-JC visual refinement: locate, hash and document the frozen inputs.

Writes ``artifact_hashes.csv`` and ``provenance.md`` under the output directory.
Nothing is modified; every listed file is only read.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

SEQUENCES = ("018", "019", "020", "108", "109", "110")
SPLIT = {"018": "train", "019": "train", "108": "train", "020": "val", "109": "test", "110": "test"}
OA = Path("outputs/paper_eval/object_aware_baseline_extension")


def md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stem(seq: str) -> str:
    return f"20231108_{seq}"


def init_root(seq: str) -> str:
    return (
        "outputs/balanced_clearance_v40_smooth"
        if seq == "018"
        else "outputs/balanced_clearance_v20_smooth"
    )


def method_roots(seq: str, refinement_root: Path) -> dict[str, Path]:
    if seq == "018":
        return {
            "frozen_v41": Path("outputs/balanced_clearance_v41"),
            "oa_independent": OA / "runs_018/oaind_v41_staged",
            "ext_jc_w10": refinement_root / "reference_runs_018/ext_jc_w10_staged",
            "ext_jc_w10_surrogate": refinement_root / "reference_runs_018/ext_jc_w10_surrogate_staged",
        }
    return {
        "frozen_v41": Path("outputs/balanced_clearance_v41"),
        "oa_independent": OA / "runs_multiseq/oaind_v41_staged",
        "ext_jc_w10": OA / "runs_multiseq/ext_jc_w10_staged",
        "ext_jc_w10_surrogate": OA / "runs_multiseq/ext_jc_w10_surrogate_staged",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("outputs/paper_eval/ext_jc_visual_refinement"))
    args = parser.parse_args()
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []

    def add(kind: str, method: str, seq: str, path: Path) -> None:
        rows.append(
            {
                "kind": kind,
                "method": method,
                "sequence": seq,
                "split": SPLIT.get(seq, "-"),
                "path": str(path),
                "exists": str(path.exists()),
                "md5": md5(path) if path.is_file() else "",
            }
        )

    for seq in SEQUENCES:
        s = stem(seq)
        # Source (CORE4D) trajectories and derived inputs.
        src = Path("data/core4d/motions/20231108") / seq
        for name in ("person1_poses.npz", "person2_poses.npz", "smooth_objposes.npy", "object_metadata.json"):
            add("source", "core4d", seq, src / name)
        add("source", "contact_analysis", seq, Path("outputs/contact_analysis") / f"{s}.npz")
        for p in (1, 2):
            add("init", f"staged_init({init_root(seq).split('/')[-1]})", seq, Path(init_root(seq)) / s / f"robot{p}.npz")
        for method, root in method_roots(seq, out).items():
            raw = root / s if method == "frozen_v41" else root / "raw" / s
            smooth = (
                Path("outputs/balanced_clearance_v41_smooth") / s
                if method == "frozen_v41"
                else root / "smooth" / s
            )
            for p in (1, 2):
                add("raw_trajectory", method, seq, raw / f"robot{p}.npz")
                add("smooth_trajectory", method, seq, smooth / f"robot{p}.npz")
            if method != "frozen_v41":
                for cfg in ("config.json", "config_robot1.json", "config_robot2.json"):
                    if (root / cfg).exists():
                        add("config", method, seq, root / cfg)
        add("frozen_metrics", "frozen_v41", seq, Path("outputs/metrics/balanced_clearance_v41_smooth") / f"{s}.json")
        add("frozen_video", "frozen_v41", seq, Path("videos/retarget") / f"{seq}_clearance_v41.mp4")

    for name in ("all_variant_metrics.csv", "per_sequence_comparison.csv", "paper_ready_results_table.csv", "REPRODUCE.md"):
        add("prior_report", "object_aware_baseline_extension", "all", OA / name)
    for name in ("src/core4d_dual_retarget/retarget/coupled.py", "src/core4d_dual_retarget/retarget/paper_presets.py",
                 "src/core4d_dual_retarget/coordination_eval.py", "src/core4d_dual_retarget/oa_extension.py",
                 "src/core4d_dual_retarget/visual_refinement.py", "src/core4d_dual_retarget/mujoco_scene.py"):
        add("code", "repo", "all", Path(name))

    with (out / "artifact_hashes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    ext_cfg = json.loads((OA / "runs_multiseq/ext_jc_w10_staged/config.json").read_text())
    cfg = ext_cfg["config"]
    lines = [
        "# Provenance of the frozen inputs (Part A)",
        "",
        "All paths are relative to the repository root. `artifact_hashes.csv` lists the md5 of every file.",
        "None of these artifacts were modified by the visual-refinement work; new results live only under",
        "`outputs/paper_eval/ext_jc_visual_refinement/`.",
        "",
        "## Methods and where their trajectories live",
        "",
        "| method | raw solve | smoothed (evaluated/rendered) | config | note |",
        "|---|---|---|---|---|",
        "| Frozen v41 (base, joint solver) | `outputs/balanced_clearance_v41/<stem>` | `outputs/balanced_clearance_v41_smooth/<stem>` | J4 preset (`paper_presets.py`) | polished base demonstration; metrics `outputs/metrics/balanced_clearance_v41_smooth`, videos `videos/retarget/<seq>_clearance_v41.mp4` |",
        "| OA-Independent | `.../object_aware_baseline_extension/runs_{018,multiseq}/oaind_v41_staged/raw` | `.../oaind_v41_staged/smooth` | `config_robot{1,2}.json` | J4 terms, each robot solved alone (`solve_robots=(r,)`) |",
        "| EXT-JC w10 (coupled) | `.../runs_multiseq/ext_jc_w10_staged/raw` (019-110); `ext_jc_visual_refinement/reference_runs_018/ext_jc_w10_staged/raw` (018) | `.../smooth` | `config.json` (`joint_contact_continuity_weight=10`) | frozen reference for this task |",
        "| EXT w10 separable surrogate | `.../ext_jc_w10_surrogate_staged/raw` | `.../smooth` | `config.json` (`joint_contact_separable_surrogate=true`) | control |",
        "",
        "The 018 EXT-JC w10 / surrogate reference runs were missing from the earlier report (it used `w10_cold` and",
        "`w40_staged` on 018); they are produced here with exactly the 019-110 configuration (staged init from",
        "`balanced_clearance_v40_smooth`, 600 iterations) and no tuning.",
        "",
        "## Initialisation protocol (identical for every method on a given take)",
        "",
        "| take | split | init root | robots |",
        "|---|---|---|---|",
    ]
    for seq in SEQUENCES:
        lines.append(f"| {seq} | {SPLIT[seq]} | `{init_root(seq)}/{stem(seq)}` | robot1.npz, robot2.npz |")
    lines += [
        "",
        "Optimizer: Adam, lr {lr}, {it} iterations, global `clip_grad_norm_` 5.0 over both robots, best-loss iterate kept.".format(
            lr=cfg["learning_rate"], it=cfg["iterations"]
        ),
        "",
        "## EXT-JC w10 configuration (frozen reference)",
        "",
        "```json",
        json.dumps({k: cfg[k] for k in sorted(cfg) if k.startswith("joint_contact")}, indent=2),
        "```",
        "",
        f"All other fields equal the J4 preset (= frozen v41); md5 of the config file: `{md5(OA / 'runs_multiseq/ext_jc_w10_staged/config.json')}`.",
        "",
        "## Evaluation contract (frozen)",
        "",
        "* `smooth-trajectories --clearance-recipe` → `evaluate-paper-join` (paper metrics) → `coordination_eval.coordination_metrics`.",
        "* Robot contact hysteresis 0.04/0.055 m on 512 rubber-hand samples per hand, source 0.05/0.07 m, stable onset ≥ 5 frames, 15 fps.",
        "* Valid joint coverage counts a frame only if both robots touch AND max hand penetration ≤ 0.5 cm.",
        "* No threshold was changed for this task.",
        "",
        "## Prior exposure of test takes",
        "",
        "109 and 110 results for v41, OA-Independent, EXT-JC w10 and the surrogate were already reported in",
        "`object_aware_baseline_extension/per_sequence_comparison.csv`.  New configurations are developed and",
        "selected on 018/019/108 (020 = validation) only; 109/110 are run once for the selected configuration.",
    ]
    (out / "provenance.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    missing = [r for r in rows if r["exists"] == "False"]
    print(f"wrote {out / 'artifact_hashes.csv'} ({len(rows)} rows, {len(missing)} missing)")
    for r in missing:
        print("  missing:", r["path"])


if __name__ == "__main__":
    main()

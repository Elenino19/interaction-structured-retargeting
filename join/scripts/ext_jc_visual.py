"""Driver for the EXT-JC visual refinement study (render / diagnose / timeline).

Usage:
  ext_jc_visual.py render   --sequence 20231108/020 --out DIR --pair v41 --pair oaind --pair surrogate [--candidate NAME=ROOT ...]
  ext_jc_visual.py diagnose --sequence 20231108/020 --out DIR [--candidate NAME=ROOT ...]
  ext_jc_visual.py timeline --sequences 018,020 --out DIR [--candidate NAME=ROOT ...]

Method roots default to the frozen / previously evaluated results; candidates
are extra (label, smooth_root) pairs.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from core4d_dual_retarget.visual_refinement import (
    FrameDiagnostics,
    load_frame_diagnostics,
    phase_windows,
    render_comparison_video,
    render_keyframes,
)

OA = Path("outputs/paper_eval/object_aware_baseline_extension")
VR = Path("outputs/paper_eval/ext_jc_visual_refinement")
FROZEN = Path("outputs/balanced_clearance_v41_smooth")


def default_roots(sequence_id: str) -> dict[str, Path]:
    short = sequence_id[-3:]
    if short == "018":
        return {
            "v41": FROZEN,
            "ext": VR / "reference_runs_018/ext_jc_w10_staged/smooth",
            "surrogate": VR / "reference_runs_018/ext_jc_w10_surrogate_staged/smooth",
            "oaind": OA / "runs_018/oaind_v41_staged/smooth",
        }
    return {
        "v41": FROZEN,
        "ext": OA / "runs_multiseq/ext_jc_w10_staged/smooth",
        "surrogate": OA / "runs_multiseq/ext_jc_w10_surrogate_staged/smooth",
        "oaind": OA / "runs_multiseq/oaind_v41_staged/smooth",
    }


LABELS = {
    "v41": "Frozen v41 (joint solver)",
    "ext": "EXT-JC w10 (coupled)",
    "surrogate": "EXT w10 separable surrogate",
    "oaind": "OA-Independent",
    "ext_repaired": "EXT-JC w10 + smoothing repair",
    "surrogate_repaired": "surrogate + same smoothing repair",
}


def parse_candidates(items: list[str]) -> dict[str, Path]:
    out = {}
    for item in items:
        name, root = item.split("=", 1)
        out[name] = Path(root)
    return out


def load_all(
    sequence_id: str, candidates: dict[str, Path]
) -> dict[str, tuple[str, Path, FrameDiagnostics]]:
    roots = default_roots(sequence_id)
    roots.update(candidates)
    out = {}
    for key, root in roots.items():
        if not (root / sequence_id.replace("/", "_") / "robot1.npz").exists():
            print(f"skip {key}: {root} missing")
            continue
        label = LABELS.get(key, key)
        out[key] = (label, root, load_frame_diagnostics(sequence_id, root, label=label))
    return out


def worst_frames(diag: FrameDiagnostics, k: int = 3) -> dict[str, object]:
    rec: dict[str, object] = {}
    for p in (1, 2):
        order = np.argsort(-diag.hand_pen_cm[p])[:k]
        rec[f"robot{p}_worst_penetration"] = [
            {
                "frame": int(i),
                "cm": round(float(diag.hand_pen_cm[p][i]), 2),
                "phase": str(diag.phases[i]),
                "softmin_m": round(float(diag.soft_min_signed_m[p][i]), 4),
                "true_min_m": round(float(diag.true_min_signed_m[p][i]), 4),
                "indicator": round(float(diag.soft_indicator[p][i]), 4),
                "source_joint": bool(diag.expected_joint[i]),
            }
            for i in order
        ]
        for name, arr in (
            ("arm_jerk", diag.arm_jerk_max[p]),
            ("waist_jerk", diag.waist_jerk_max[p]),
            ("leg_jerk", diag.leg_jerk_max[p]),
            ("root_jerk", diag.root_jerk_m_s3[p]),
            ("root_acc", diag.root_acc_m_s2[p]),
            ("joint_vel", diag.joint_vel_max[p]),
        ):
            order = np.argsort(-arr)[:k]
            rec[f"robot{p}_worst_{name}"] = [
                {"frame": int(i), "value": round(float(arr[i]), 2), "phase": str(diag.phases[i])}
                for i in order
            ]
        planted = diag.foot_zmin_m[p][1:] < 0.03
        speed = diag.foot_speed_cm_s[p]
        skating = np.where(planted, speed, 0.0)
        order = np.argsort(-skating.max(axis=1))[:k]
        rec[f"robot{p}_worst_planted_foot_speed"] = [
            {"frame": int(i) + 1, "cm_s": round(float(skating[i].max()), 1)} for i in order
        ]
        rec[f"robot{p}_sole_float_frames_gt_2cm"] = int(
            (diag.foot_zmin_m[p].min(axis=1) > 0.02).sum()
        )
    exp, pred = diag.expected_joint, diag.predicted_joint
    inside = np.flatnonzero(exp)
    if inside.size:
        window = pred[inside[0] : inside[-1] + 1]
        gaps = []
        t = 0
        while t < len(window):
            if not window[t]:
                s = t
                while t < len(window) and not window[t]:
                    t += 1
                gaps.append([int(inside[0] + s), int(inside[0] + t)])
            else:
                t += 1
        rec["joint_contact_gaps_in_source_window"] = gaps
    return rec


def deviation_from_v41(diag: FrameDiagnostics, ref: FrameDiagnostics) -> dict[str, object]:
    rec: dict[str, object] = {}
    names = [str(n) for n in diag.joint_names]
    arm = np.array([any(t in n for t in ("shoulder", "elbow", "wrist")) for n in names])
    leg = np.array([any(t in n for t in ("hip", "knee", "ankle")) for n in names])
    waist = np.array(["waist" in n for n in names])
    for p in (1, 2):
        dq = np.abs(diag.joint_positions[p] - ref.joint_positions[p])
        droot = (
            np.linalg.norm(diag.root_translation[p][:, :2] - ref.root_translation[p][:, :2], axis=1)
            * 100
        )
        dyaw = np.degrees(np.abs(diag.root_yaw[p] - ref.root_yaw[p]))
        rec[f"robot{p}_arm_dq_max_deg"] = round(float(np.degrees(dq[:, arm].max())), 1)
        rec[f"robot{p}_leg_dq_max_deg"] = round(float(np.degrees(dq[:, leg].max())), 1)
        rec[f"robot{p}_leg_dq_rms_deg"] = round(
            float(np.degrees(np.sqrt((dq[:, leg] ** 2).mean()))), 2
        )
        rec[f"robot{p}_waist_dq_max_deg"] = (
            round(float(np.degrees(dq[:, waist].max())), 1) if waist.any() else None
        )
        rec[f"robot{p}_root_xy_dev_max_cm"] = round(float(droot.max()), 2)
        rec[f"robot{p}_root_xy_dev_rms_cm"] = round(float(np.sqrt((droot**2).mean())), 2)
        rec[f"robot{p}_yaw_dev_max_deg"] = round(float(dyaw.max()), 1)
        rec[f"robot{p}_root_z_dev_max_cm"] = round(
            float(
                np.abs(diag.root_translation[p][:, 2] - ref.root_translation[p][:, 2]).max() * 100
            ),
            2,
        )
    return rec


def summary_row(diag: FrameDiagnostics) -> dict[str, object]:
    row: dict[str, object] = {"label": diag.label, "sequence_id": diag.sequence_id}
    for p in (1, 2):
        row[f"r{p}_hand_pen_max_cm"] = round(float(diag.hand_pen_cm[p].max()), 2)
        row[f"r{p}_frames_pen_gt_0p5"] = int((diag.hand_pen_cm[p] > 0.5).sum())
        row[f"r{p}_arm_jerk_p95"] = round(float(np.percentile(diag.arm_jerk_max[p], 95)), 1)
        row[f"r{p}_arm_jerk_max"] = round(float(diag.arm_jerk_max[p].max()), 1)
        row[f"r{p}_waist_jerk_max"] = round(float(diag.waist_jerk_max[p].max()), 1)
        row[f"r{p}_root_jerk_max"] = round(float(diag.root_jerk_m_s3[p].max()), 1)
        row[f"r{p}_root_speed_max"] = round(float(diag.root_speed_m_s[p].max()), 3)
        row[f"r{p}_joint_vel_max"] = round(float(diag.joint_vel_max[p].max()), 2)
        planted = diag.foot_zmin_m[p][1:] < 0.03
        sp = np.where(planted, diag.foot_speed_cm_s[p], 0.0)
        row[f"r{p}_planted_foot_speed_max_cm_s"] = round(float(sp.max()), 1)
        row[f"r{p}_softmin_bias_mean_cm"] = round(
            float(np.mean(diag.true_min_signed_m[p] - diag.soft_min_signed_m[p]) * 100), 2
        )
    exp, pred = diag.expected_joint, diag.predicted_joint
    row["source_joint_frames"] = int(exp.sum())
    row["robot_joint_frames"] = int(pred.sum())
    row["joint_recall"] = round(float((exp & pred).sum() / max(exp.sum(), 1)), 3)
    return row


def cmd_diagnose(args) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    methods = load_all(args.sequence, parse_candidates(args.candidate))
    report = {
        "sequence_id": args.sequence,
        "phases": phase_windows(methods["v41"][2].phases),
        "methods": {},
    }
    rows = []
    for key, (label, root, diag) in methods.items():
        rec = {"label": label, "smooth_root": str(root), "worst": worst_frames(diag)}
        if key != "v41":
            rec["deviation_from_v41"] = deviation_from_v41(diag, methods["v41"][2])
        report["methods"][key] = rec
        rows.append(summary_row(diag))
    stem = args.sequence.replace("/", "_")
    (out / f"diagnostics_{stem}.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    with (out / f"diagnostics_summary_{stem}.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    # per-frame CSV for plotting
    with (out / f"per_frame_{stem}.csv").open("w", newline="", encoding="utf-8") as handle:
        keys = list(methods)
        header = ["frame", "phase", "source_joint"]
        for k in keys:
            header += [
                f"{k}_joint",
                f"{k}_r1_pen",
                f"{k}_r2_pen",
                f"{k}_r1_s",
                f"{k}_r2_s",
                f"{k}_r1_armjerk",
                f"{k}_r2_armjerk",
                f"{k}_r1_rootjerk",
                f"{k}_r2_rootjerk",
            ]
        writer = csv.writer(handle)
        writer.writerow(header)
        ref = methods["v41"][2]
        for t in range(len(ref.phases)):
            row = [t, int(ref.phases[t]), int(ref.expected_joint[t])]
            for k in keys:
                d = methods[k][2]
                row += [
                    int(d.predicted_joint[t]),
                    f"{d.hand_pen_cm[1][t]:.2f}",
                    f"{d.hand_pen_cm[2][t]:.2f}",
                    f"{d.soft_indicator[1][t]:.3f}",
                    f"{d.soft_indicator[2][t]:.3f}",
                    f"{d.arm_jerk_max[1][t]:.0f}",
                    f"{d.arm_jerk_max[2][t]:.0f}",
                    f"{d.root_jerk_m_s3[1][t]:.1f}",
                    f"{d.root_jerk_m_s3[2][t]:.1f}",
                ]
            writer.writerow(row)
    print(json.dumps(rows, indent=1))


def cmd_render(args) -> None:
    out = Path(args.out)
    methods = load_all(args.sequence, parse_candidates(args.candidate))
    stem = args.sequence.replace("/", "_")
    short = stem[-3:]
    meta = {}
    pairs = args.pair or ["v41", "oaind", "surrogate"]
    ext_key = args.ext_key
    for other in pairs:
        if other not in methods or ext_key not in methods:
            print(f"skip pair {other} vs {ext_key}")
            continue
        name = out / "videos" / f"{short}_{other}_vs_{ext_key}.mp4"
        meta[name.name] = render_comparison_video(
            args.sequence,
            [methods[other], methods[ext_key]],
            name,
            reference_root=methods["v41"][1],
        )
        print("wrote", name)
    if args.full:
        keys = [k for k in ("v41", "oaind", "surrogate", ext_key) if k in methods]
        name = out / f"sequence_{short}_comparison.mp4"
        meta[name.name] = render_comparison_video(
            args.sequence,
            [methods[k] for k in keys],
            name,
            reference_root=methods["v41"][1],
            tile_width=640,
            tile_height=480,
        )
        print("wrote", name)
    if args.phase_clips:
        windows = phase_windows(methods["v41"][2].phases)
        keys = [k for k in ("v41", ext_key) if k in methods]
        for phase in ("approach", "contact_transition", "joint_carry"):
            if phase not in windows:
                continue
            s, e = windows[phase]
            s = max(0, s - 5)
            e = min(len(methods["v41"][2].phases), e + 5)
            name = out / "videos" / f"{short}_{phase}_{ext_key}_slow3x.mp4"
            meta[name.name] = render_comparison_video(
                args.sequence,
                [methods[k] for k in keys],
                name,
                reference_root=methods["v41"][1],
                frames=(s, e),
                slow_factor=3,
            )
            print("wrote", name)
    if args.keyframes:
        windows = phase_windows(methods["v41"][2].phases)
        ext = methods[ext_key][2]
        worst = int(np.maximum(ext.hand_pen_cm[1], ext.hand_pen_cm[2]).argmax())
        onset = int(np.flatnonzero(ext.expected_joint)[0]) if ext.expected_joint.any() else 0
        release = int(np.flatnonzero(ext.expected_joint)[-1]) if ext.expected_joint.any() else 0
        frames = [
            ("approach", windows.get("approach", (0, 0))[0] + 2),
            ("source joint onset", onset),
            ("worst EXT penetration", worst),
            ("mid joint-carry", (onset + release) // 2),
            ("release", release),
        ]
        keys = [k for k in ("v41", "oaind", "surrogate", ext_key) if k in methods] + [
            k for k in methods if k not in ("v41", "oaind", "surrogate", ext_key)
        ]
        name = out / f"qualitative_keyframes_{short}.png"
        render_keyframes(
            args.sequence,
            [methods[k] for k in keys],
            frames,
            name,
            reference_root=methods["v41"][1],
            zoom=args.zoom,
        )
        meta[name.name] = {"frames": frames, "camera": "object-centred close-up", "zoom": args.zoom}
        print("wrote", name)
    (out / f"render_manifest_{short}.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )


def cmd_timeline(args) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    seqs = [f"20231108/{s}" for s in args.sequences.split(",")]
    candidates = parse_candidates(args.candidate)
    fig, axes = plt.subplots(len(seqs), 1, figsize=(14, 3.0 * len(seqs)), squeeze=False)
    for ax, seq in zip(axes[:, 0], seqs, strict=True):
        methods = load_all(seq, candidates)
        order = [k for k in ("v41", "oaind", "surrogate", "ext") if k in methods] + [
            k for k in methods if k not in ("v41", "oaind", "surrogate", "ext")
        ]
        ref = methods["v41"][2]
        lanes = [("Source (both persons in contact)", ref.expected_joint, None, "#444444")]
        for k in order:
            d = methods[k][2]
            lanes.append(
                (
                    methods[k][0],
                    d.predicted_joint,
                    np.maximum(d.hand_pen_cm[1], d.hand_pen_cm[2]),
                    "#1f77b4",
                )
            )
        for li, (name, mask, pen, color) in enumerate(lanes):
            y = len(lanes) - 1 - li
            for t in np.flatnonzero(mask):
                ok = pen is None or pen[t] <= 0.5
                ax.broken_barh(
                    [(t, 1)],
                    (y + 0.15, 0.7),
                    facecolors=color if ok else "#d62728",
                    edgecolors="none",
                )
            if pen is not None:
                for t in np.flatnonzero(pen > 0.5):
                    ax.broken_barh(
                        [(t, 1)], (y + 0.02, 0.1), facecolors="#d62728", edgecolors="none"
                    )
        ax.set_yticks([len(lanes) - 1 - i + 0.5 for i in range(len(lanes))])
        ax.set_yticklabels([n for n, *_ in lanes], fontsize=9)
        windows = phase_windows(ref.phases)
        for pname, (s, e) in windows.items():
            ax.axvspan(
                s,
                e,
                color={
                    "approach": "#fff3cd",
                    "contact_transition": "#ffe0b2",
                    "joint_carry": "#e3f2fd",
                }.get(pname, "white"),
                alpha=0.5,
                zorder=0,
            )
            ax.text(
                (s + e) / 2, len(lanes) + 0.05, pname.replace("_", " "), ha="center", fontsize=8
            )
        ax.set_xlim(0, len(ref.phases))
        ax.set_ylim(0, len(lanes) + 0.4)
        ax.set_xlabel("frame (15 fps)")
        ax.set_title(
            f"Sequence {seq[-3:]}: simultaneous-contact timeline (red = frame with hand penetration > 0.5 cm)",
            fontsize=10,
        )
    fig.tight_layout()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "contact_timeline_comparison.png", dpi=170)
    print("wrote", out / "contact_timeline_comparison.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("render", "diagnose"):
        p = sub.add_parser(name)
        p.add_argument("--sequence", required=True)
        p.add_argument("--out", default=str(VR))
        p.add_argument("--candidate", action="append", default=[])
        if name == "render":
            p.add_argument("--pair", action="append", default=[])
            p.add_argument("--ext-key", default="ext")
            p.add_argument("--full", action="store_true")
            p.add_argument("--phase-clips", action="store_true")
            p.add_argument("--keyframes", action="store_true")
            p.add_argument("--zoom", type=float, default=0.55, help="camera distance multiplier for keyframes")
    p = sub.add_parser("timeline")
    p.add_argument("--sequences", default="018,020")
    p.add_argument("--out", default=str(VR))
    p.add_argument("--candidate", action="append", default=[])
    args = parser.parse_args()
    {"render": cmd_render, "diagnose": cmd_diagnose, "timeline": cmd_timeline}[args.cmd](args)


if __name__ == "__main__":
    main()

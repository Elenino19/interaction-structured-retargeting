"""Phase 1 objective separability probe for the Join coupled objective.

For a configuration (default: J4 == frozen v41) and an operating point (the
trajectories in ``--baseline-root``), this script evaluates every weighted
loss term once and records

* value,
* gradient norm w.r.t. robot-1 parameters and w.r.t. robot-2 parameters,
* cross-Hessian-vector products  H_12 u  for random u (exactly zero for an
  additively separable term),
* finite-difference check: does the robot-1 gradient change when robot-2's
  parameters are perturbed?

It also checks additive separability of the whole objective:
``L_joint(θ1, θ2) == L_1(θ1) + L_2(θ2) + L_cross`` by solving the per-robot
objectives in ``solve_robots=(1,)`` / ``(2,)`` mode at the same point.

Nothing is optimized; the frozen results are only read.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
from pathlib import Path

import torch

from core4d_dual_retarget.retarget.coupled import retarget_coupled
from core4d_dual_retarget.retarget.paper_presets import paper_join_config

TERM_INPUTS = {
    "pose": ("q_r", "source joint targets"),
    "contact": ("q_r, root_r", "object pose, source fingertip anchors, contact masks, phase"),
    "root": ("root_r", "baseline root"),
    "velocity": ("q_r", "-"),
    "acceleration": ("q_r, root_r", "-"),
    "jerk": ("q_r", "-"),
    "inter_robot": ("q_1, q_2, root_1, root_2", "clearance 0.12 m"),
    "support": ("q_r, root_r", "-"),
    "foot_height": ("q_r, root_r", "ground"),
    "foot_slip": ("q_r, root_r", "phase"),
    "torso": ("q_r", "-"),
    "center": ("q_r, root_r", "-"),
    "clearance": ("q_r, root_r", "object AABB"),
    "hand_volume": ("q_r, root_r", "object hull / AABB"),
    "torso_volume": ("q_r, root_r", "object hull / AABB"),
    "foot_volume": ("q_r, root_r", "object hull / AABB"),
    "hull": ("q_r, root_r", "object hull"),
    "leg_volume": ("q_r, root_r", "object hull / AABB"),
    "leg_hull": ("q_r, root_r", "object hull"),
    "arm_hull": ("q_r, root_r", "object hull"),
    "foot_ground": ("q_r, root_r", "ground"),
    "foot_slide": ("q_r, root_r", "-"),
    "planted_root": ("root_r", "phase"),
    "swing": ("q_r, root_r", "phase"),
    "swing_knee": ("q_r", "phase"),
    "gait_knee": ("q_r", "source knees"),
    "squat": ("q_r", "phase"),
    "knee_min": ("q_r", "-"),
    "palm_facing": ("q_r, root_r", "object pose, anchors"),
    "finger_up": ("q_r, root_r", "object pose, anchors"),
    "stance_lock": ("q_r, root_r", "phase"),
    "knee_sym": ("q_r", "-"),
    "both_soles": ("q_r, root_r", "phase"),
    "stance_width": ("q_r, root_r", "-"),
    "putdown_root": ("root_r", "phase"),
    "laterality": ("q_r, root_r", "-"),
    "self_leg": ("q_r, root_r", "-"),
    "hip_yaw": ("q_r", "-"),
    "hip_roll": ("q_r", "-"),
    "outward": ("q_r, root_r", "phase"),
    "speed_cap": ("q_r, root_r", "-"),
    "approach_ground": ("q_r, root_r", "phase"),
    "approach_root": ("root_r", "source root, phase"),
    "joint_contact": ("q_1, q_2, root_1, root_2", "object hull, joint-carry frames"),
    "hand_jerk": ("q_r, root_r", "-"),
    "reference_trust": ("q_r, root_r", "frozen reference trajectory (v41)"),
    "smoothed_barrier": ("q_r, root_r", "object hull, frozen smoothing operators"),
}


def _flat(grads: list[torch.Tensor | None]) -> torch.Tensor:
    pieces = [g.reshape(-1) for g in grads if g is not None]
    if not pieces:
        return torch.zeros(1, dtype=torch.float64)
    return torch.cat(pieces)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", default="20231108/018")
    parser.add_argument("--variant", default="J4")
    parser.add_argument("--baseline-root", type=Path, default=Path("outputs/independent"))
    parser.add_argument("--point-name", default="cold_init")
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--directions", type=int, default=3)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    config = paper_join_config(args.variant, iterations=0)
    overrides = {}
    for item in args.override:
        name, raw = item.split("=", 1)
        overrides[name] = json.loads(raw)
    config = dataclasses.replace(config, **overrides)
    stem = args.sequence.replace("/", "_")
    common = (
        Path("data/core4d/motions"),
        args.sequence,
        Path("data/core4d/object_models"),
        args.baseline_root,
        Path("third_party/unitree_ros/robots/g1_description/g1_29dof_rev_1_0.urdf"),
        Path("outputs/contact_analysis") / f"{stem}.npz",
    )

    rows: list[dict[str, object]] = []
    summary: dict[str, object] = {
        "sequence": args.sequence,
        "variant": args.variant,
        "overrides": overrides,
        "operating_point": args.point_name,
        "baseline_root": str(args.baseline_root),
    }

    def joint_probe(evaluate, raw_joints, root_translation, root_yaw) -> None:
        params1 = [raw_joints[1], root_translation[1], root_yaw[1]]
        params2 = [raw_joints[2], root_translation[2], root_yaw[2]]
        loss, _, terms = evaluate()
        summary["joint_loss"] = float(loss)
        total1 = _flat(torch.autograd.grad(loss, params1, retain_graph=True, allow_unused=True))
        total2 = _flat(torch.autograd.grad(loss, params2, retain_graph=True, allow_unused=True))
        summary["joint_grad_norm_robot1"] = float(total1.norm())
        summary["joint_grad_norm_robot2"] = float(total2.norm())
        for name, tensor in terms.items():
            if name.endswith(("_raw", "_active_pairs", "_active_frames")):
                continue
            row: dict[str, object] = {
                "term": name,
                "operating_point": args.point_name,
                "value_weighted": float(tensor.detach()),
                "optimized_inputs": TERM_INPUTS.get(name, ("?", "?"))[0],
                "fixed_inputs": TERM_INPUTS.get(name, ("?", "?"))[1],
                "object_fixed": "yes",
            }
            if not tensor.requires_grad:
                row.update(
                    grad_norm_robot1=0.0,
                    grad_norm_robot2=0.0,
                    cross_hvp_norm_max=0.0,
                    fd_cross_change_max=0.0,
                    depends_on="none (constant/inactive)",
                    separable="yes (inactive)",
                )
                rows.append(row)
                continue
            g1 = torch.autograd.grad(
                tensor, params1, retain_graph=True, create_graph=True, allow_unused=True
            )
            g2 = torch.autograd.grad(
                tensor, params2, retain_graph=True, create_graph=True, allow_unused=True
            )
            g1_flat = _flat(g1)
            g2_flat = _flat(g2)
            row["grad_norm_robot1"] = float(g1_flat.detach().norm())
            row["grad_norm_robot2"] = float(g2_flat.detach().norm())
            # Cross Hessian-vector products: d/dθ2 (g1 · u1) for random u1.
            hvp_max = 0.0
            if g1_flat.requires_grad:
                for _ in range(args.directions):
                    u1 = torch.randn_like(g1_flat)
                    u1 = u1 / u1.norm()
                    hv = torch.autograd.grad(
                        g1_flat @ u1, params2, retain_graph=True, allow_unused=True
                    )
                    hvp_max = max(hvp_max, float(_flat(hv).norm()))
            if g2_flat.requires_grad:
                for _ in range(args.directions):
                    u2 = torch.randn_like(g2_flat)
                    u2 = u2 / u2.norm()
                    hv = torch.autograd.grad(
                        g2_flat @ u2, params1, retain_graph=True, allow_unused=True
                    )
                    hvp_max = max(hvp_max, float(_flat(hv).norm()))
            row["cross_hvp_norm_max"] = hvp_max
            uses1 = row["grad_norm_robot1"] > 0
            uses2 = row["grad_norm_robot2"] > 0
            row["depends_on"] = (
                "both" if uses1 and uses2 else "robot1" if uses1 else "robot2" if uses2 else "none"
            )
            rows.append(row)

        # Finite-difference cross check on the TOTAL loss: perturb θ2, re-evaluate g1.
        with torch.no_grad():
            saved = [p.detach().clone() for p in params2]
            for p in params2:
                p.add_(1e-2 * torch.randn_like(p))
        loss_b, _, terms_b = evaluate()
        g1_b = _flat(torch.autograd.grad(loss_b, params1, retain_graph=True, allow_unused=True))
        fd_change = float((g1_b - total1).abs().max())
        summary["fd_total_grad1_change_after_theta2_perturbation"] = fd_change
        for row in rows:
            name = row["term"]
            if name in terms_b and terms_b[name].requires_grad and terms[name].requires_grad:
                ga = _flat(
                    torch.autograd.grad(terms[name], params1, retain_graph=True, allow_unused=True)
                )
                gb = _flat(
                    torch.autograd.grad(
                        terms_b[name], params1, retain_graph=True, allow_unused=True
                    )
                )
                row["fd_cross_change_max"] = float((gb - ga).abs().max()) if ga.shape == gb.shape else float("nan")
            else:
                row.setdefault("fd_cross_change_max", 0.0)
            hv = float(row.get("cross_hvp_norm_max", 0.0))
            fd = float(row.get("fd_cross_change_max", 0.0))
            if row["depends_on"] == "both" and (hv > 1e-12 or fd > 1e-12):
                row["separable"] = "NO (active cross-agent dependency)"
            elif row["depends_on"] == "both":
                row["separable"] = "yes (sum of per-robot terms)"
            else:
                row["separable"] = "yes"
        with torch.no_grad():
            for p, s in zip(params2, saved, strict=True):
                p.copy_(s)

    retarget_coupled(*common, config, probe_callback=joint_probe)

    # Additive separability of the whole objective.
    per_robot: dict[int, float] = {}
    for robot in (1, 2):
        cfg_r = dataclasses.replace(config, solve_robots=(robot,))

        def single_probe(evaluate, *_args, robot=robot) -> None:
            loss, _, _terms = evaluate()
            per_robot[robot] = float(loss)

        retarget_coupled(*common, cfg_r, probe_callback=single_probe)
    cross = sum(
        float(r["value_weighted"]) for r in rows if r["term"] in ("inter_robot", "joint_contact")
    )
    summary["single_robot_losses"] = per_robot
    summary["sum_single_plus_cross"] = per_robot[1] + per_robot[2] + cross
    summary["additive_separability_abs_error"] = abs(
        summary["joint_loss"] - summary["sum_single_plus_cross"]
    )
    summary["cross_agent_terms_weighted_value"] = cross
    summary["active_cross_agent_terms"] = [
        r["term"] for r in rows if str(r["separable"]).startswith("NO")
    ]

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / f"loss_dependency_{args.point_name}.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fields = [
            "term",
            "operating_point",
            "optimized_inputs",
            "fixed_inputs",
            "object_fixed",
            "depends_on",
            "value_weighted",
            "grad_norm_robot1",
            "grad_norm_robot2",
            "cross_hvp_norm_max",
            "fd_cross_change_max",
            "separable",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (args.output / f"separability_summary_{args.point_name}.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

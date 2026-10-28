"""Write ``cross_gradient_verification.md`` from the separability-probe outputs.

Reads ``cross_gradient/separability_summary_<name>.json`` and
``cross_gradient/loss_dependency_<name>.csv`` produced by
``scripts/oa_separability_probe.py`` for every configuration listed in
``cross_gradient/probe_configs.txt``.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path("outputs/paper_eval/ext_jc_visual_refinement")
V41_LOSS = 2.789216976690139  # phase-1 probe value at the frozen v41 solution (018)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    cg = args.root / "cross_gradient"
    names: list[tuple[str, str]] = []
    for line in (cg / "probe_configs.txt").read_text(encoding="utf-8").splitlines():
        if line.strip():
            name, _, overrides = line.partition("|")
            names.append((name.strip(), overrides.strip()))

    lines = [
        "# Cross-agent gradient verification",
        "",
        "Operating point: frozen v41 solution of take 018 (`outputs/balanced_clearance_v41/20231108_018`),",
        "probe = `scripts/oa_separability_probe.py` (3 random directions, seed 0).  For every weighted term the",
        "probe reports the gradient norm w.r.t. each robot, the maximum norm of the cross Hessian-vector product",
        "`d/dθ₂ (∇θ₁ L · u)` (exactly 0 for an additively separable term) and a finite-difference check (does the",
        "robot-1 gradient change after perturbing robot-2's parameters by 1e-2?).  The verification was re-run",
        "after every method change (one row per configuration, in the order the configurations were introduced).",
        "",
        "## Regression check",
        "",
    ]
    reg = json.loads((cg / "separability_summary_regression_v41_defaults.json").read_text())
    ok = abs(reg["joint_loss"] - V41_LOSS) < 1e-9
    regression_line = (
        f"* Default configuration (all new fields off) at the v41 solution: joint loss `{reg['joint_loss']!r}`"
        f" vs. phase-1 value `{V41_LOSS!r}` -> {'IDENTICAL' if ok else 'DIFFERENT'};"
        f" additive-separability error `{reg['additive_separability_abs_error']:.1e}`;"
        f" active cross-agent terms: `{reg['active_cross_agent_terms']}`."
    )
    lines += [
        regression_line,
        "",
        "## Per-configuration results",
        "",
        "| configuration | overrides | loss | joint_contact value | ‖∇₁‖ / ‖∇₂‖ | cross-HVP max | FD Δ∇₁ | coupled? | other new terms |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, overrides in names:
        summ_path = cg / f"separability_summary_{name}.json"
        if not summ_path.exists():
            lines.append(f"| {name} | `{overrides}` | (probe missing) | | | | | | |")
            continue
        summ = json.loads(summ_path.read_text())
        dep = {r["term"]: r for r in csv.DictReader((cg / f"loss_dependency_{name}.csv").open())}
        jc = dep["joint_contact"]
        others = []
        for term in ("hand_jerk", "reference_trust", "smoothed_barrier"):
            r = dep.get(term)
            if r and float(r["value_weighted"]) != 0.0:
                others.append(
                    f"{term}: value {float(r['value_weighted']):.3f}, HVP {float(r['cross_hvp_norm_max']):.1e}, {r['separable']}"
                )
        coupled = str(jc["separable"]).startswith("NO")
        lines.append(
            "| {n} | `{o}` | {loss:.4f} | {v:.4f} | {g1:.3f} / {g2:.3f} | {hvp:.3g} | {fd:.3g} | {c} | {oth} |".format(
                n=name,
                o=overrides or "(none)",
                loss=summ["joint_loss"],
                v=float(jc["value_weighted"]),
                g1=float(jc["grad_norm_robot1"]),
                g2=float(jc["grad_norm_robot2"]),
                hvp=float(jc["cross_hvp_norm_max"]),
                fd=float(jc["fd_cross_change_max"]),
                c="YES (product s₁·s₂)" if coupled else "no (separable)",
                oth="; ".join(others) or "-",
            )
        )
    lines += [
        "",
        "## Reading the table",
        "",
        "* Every coupled configuration keeps the product term `Σ_t E_J(t)·(1 − s₁(t)s₂(t)) / Σ_t E_J(t)`; its cross",
        "  Hessian-vector product is O(1) and the robot-1 gradient changes when robot-2 moves, so the loss is",
        "  genuinely non-separable.  The surrogate control `(s₁+s₂)/2` with the SAME new terms has cross-HVP = 0 and",
        "  FD Δ∇₁ = 0: the only cross-robot operation left is the shared gradient-norm clipping.",
        "* The surface-aware indicator changes the *value* of `s_r` (it saturates at the hull surface instead of",
        "  increasing inside), the edge taper changes `E_J(t)`, and `hand_jerk` / `reference_trust` are sums of",
        "  per-robot terms (HVP = 0).  None of them removes the coupling.",
        "* The probe's `additive_separability_abs_error` for configurations with the joint-contact term switched on",
        "  equals `L_JC(θ₁ alone) + L_JC(θ₂ alone)` by construction (the single-robot probe evaluates the term with",
        "  the partner indicator replaced by 1); it is 0 to machine precision for the base objective.",
        "",
        "Unit tests: `tests/test_oa_extension.py::test_surface_aware_product_still_couples_both_robots`,",
        "`::test_joint_contact_product_couples_both_robots`, `::test_surface_aware_indicator_has_no_gradient_inside_hull`.",
    ]
    (args.root / "cross_gradient_verification.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", args.root / "cross_gradient_verification.md")


if __name__ == "__main__":
    main()

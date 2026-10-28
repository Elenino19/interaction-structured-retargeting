# Cross-agent gradient verification

Operating point: frozen v41 solution of take 018 (`outputs/balanced_clearance_v41/20231108_018`),
probe = `scripts/oa_separability_probe.py` (3 random directions, seed 0).  For every weighted term the
probe reports the gradient norm w.r.t. each robot, the maximum norm of the cross Hessian-vector product
`d/dθ₂ (∇θ₁ L · u)` (exactly 0 for an additively separable term) and a finite-difference check (does the
robot-1 gradient change after perturbing robot-2's parameters by 1e-2?).  The verification was re-run
after every method change (one row per configuration, in the order the configurations were introduced).

## Regression check

* Default configuration (all new fields off) at the v41 solution: joint loss `2.789216976690139` vs. phase-1 value `2.789216976690139` -> IDENTICAL; additive-separability error `1.8e-15`; active cross-agent terms: `[]`.

## Per-configuration results

| configuration | overrides | loss | joint_contact value | ‖∇₁‖ / ‖∇₂‖ | cross-HVP max | FD Δ∇₁ | coupled? | other new terms |
|---|---|---|---|---|---|---|---|---|
| regression_v41_defaults | `(none)` | 2.7892 | 0.0000 | 0.000 / 0.000 | 0 | 0 | no (separable) | - |
| ext_jc_w10 | `--override joint_contact_continuity_weight=10.0` | 3.4847 | 0.6955 | 6.474 / 6.580 | 1.27 | 0.0471 | YES (product s₁·s₂) | - |
| refined_v1_surface | `--override joint_contact_continuity_weight=10.0 --override joint_contact_surface_aware=true` | 3.6824 | 0.8932 | 6.235 / 6.502 | 1.27 | 0.0422 | YES (product s₁·s₂) | - |
| refined_v2_surface_temporal | `--override joint_contact_continuity_weight=10.0 --override joint_contact_surface_aware=true --override joint_contact_edge_taper_frames=3 --override hand_cartesian_jerk_weight=20.0` | 3.8561 | 0.9176 | 6.418 / 6.844 | 1.33 | 1.2 | YES (product s₁·s₂) | hand_jerk: value 0.149, HVP 0.0e+00, yes (sum of per-robot terms) |
| refined_v3_surface_temporal_trust | `--override joint_contact_continuity_weight=10.0 --override joint_contact_surface_aware=true --override joint_contact_edge_taper_frames=3 --override hand_cartesian_jerk_weight=20.0 --override reference_trust_weight=200.0 --override reference_trust_root="\"outputs/balanced_clearance_v41_smooth\""` | 3.8645 | 0.9176 | 6.418 / 6.844 | 1.33 | 0.0716 | YES (product s₁·s₂) | hand_jerk: value 0.149, HVP 0.0e+00, yes (sum of per-robot terms); reference_trust: value 0.008, HVP 0.0e+00, yes (sum of per-robot terms) |
| refined_v3_surrogate_control | `--override joint_contact_continuity_weight=10.0 --override joint_contact_separable_surrogate=true --override joint_contact_surface_aware=true --override joint_contact_edge_taper_frames=3 --override hand_cartesian_jerk_weight=20.0 --override reference_trust_weight=200.0 --override reference_trust_root="\"outputs/balanced_clearance_v41_smooth\""` | 3.4725 | 0.5257 | 4.074 / 3.567 | 0 | 0 | no (separable) | hand_jerk: value 0.149, HVP 0.0e+00, yes (sum of per-robot terms); reference_trust: value 0.008, HVP 0.0e+00, yes (sum of per-robot terms) |
| refined_v1b_surface_lowbeta | `--override joint_contact_continuity_weight=10.0 --override joint_contact_surface_aware=true --override joint_contact_softmin_beta_m=0.002` | 7.2532 | 4.4640 | 30.401 / 15.665 | 11.5 | 1.04 | YES (product s₁·s₂) | - |
| refined_v4_surface_smoothbarrier | `--override joint_contact_continuity_weight=10.0 --override joint_contact_surface_aware=true --override smoothed_hand_barrier_weight=800.0` | 3.6985 | 0.8932 | 6.235 / 6.502 | 1.27 | 1.14 | YES (product s₁·s₂) | smoothed_barrier: value 0.016, HVP 0.0e+00, yes (sum of per-robot terms) |
| refined_v5_surface_smoothbarrier_temporal | `--override joint_contact_continuity_weight=10.0 --override joint_contact_surface_aware=true --override smoothed_hand_barrier_weight=800.0 --override joint_contact_edge_taper_frames=3 --override hand_cartesian_jerk_weight=20.0` | 3.8722 | 0.9176 | 6.418 / 6.844 | 1.33 | 0.0716 | YES (product s₁·s₂) | hand_jerk: value 0.149, HVP 0.0e+00, yes (sum of per-robot terms); smoothed_barrier: value 0.016, HVP 0.0e+00, yes (sum of per-robot terms) |
| refined_v5_surrogate_control | `--override joint_contact_continuity_weight=10.0 --override joint_contact_separable_surrogate=true --override joint_contact_surface_aware=true --override smoothed_hand_barrier_weight=800.0 --override joint_contact_edge_taper_frames=3 --override hand_cartesian_jerk_weight=20.0` | 3.4802 | 0.5257 | 4.074 / 3.567 | 0 | 0 | no (separable) | hand_jerk: value 0.149, HVP 0.0e+00, yes (sum of per-robot terms); smoothed_barrier: value 0.016, HVP 0.0e+00, yes (sum of per-robot terms) |

## Reading the table

* Every coupled configuration keeps the product term `Σ_t E_J(t)·(1 − s₁(t)s₂(t)) / Σ_t E_J(t)`; its cross
  Hessian-vector product is O(1) and the robot-1 gradient changes when robot-2 moves, so the loss is
  genuinely non-separable.  The surrogate control `(s₁+s₂)/2` with the SAME new terms has cross-HVP = 0 and
  FD Δ∇₁ = 0: the only cross-robot operation left is the shared gradient-norm clipping.
* The surface-aware indicator changes the *value* of `s_r` (it saturates at the hull surface instead of
  increasing inside), the edge taper changes `E_J(t)`, and `hand_jerk` / `reference_trust` are sums of
  per-robot terms (HVP = 0).  None of them removes the coupling.
* The probe's `additive_separability_abs_error` for configurations with the joint-contact term switched on
  equals `L_JC(θ₁ alone) + L_JC(θ₂ alone)` by construction (the single-robot probe evaluates the term with
  the partner indicator replaced by 1); it is 0 to machine precision for the base objective.

Unit tests: `tests/test_oa_extension.py::test_surface_aware_product_still_couples_both_robots`,
`::test_joint_contact_product_couples_both_robots`, `::test_surface_aware_indicator_has_no_gradient_inside_hull`.

# Objective separability audit — Join coupled retargeting (frozen v41 / J4 preset)

Code audited: `src/core4d_dual_retarget/retarget/coupled.py::retarget_coupled`
(objective construction in the nested `evaluate_objective`, lines ~700–1700),
`retarget/paper_presets.py` (J4 == frozen v41 configuration),
`cli.py::retarget-paper-join` / `retarget-coupled`.
Probe script: `scripts/oa_separability_probe.py`; raw outputs in
`phase1_separability/`.

## 1. Statement of the result

**The frozen Join objective is additively separable across the two robots.**

\[
\mathcal{L}(\theta_1,\theta_2)=\mathcal{L}_1(\theta_1)+\mathcal{L}_2(\theta_2)+
w_{\text{inter}}\,\mathcal{L}_{\text{inter}}(\theta_1,\theta_2),\qquad
\mathcal{L}_{\text{inter}}\equiv 0 \text{ on every evaluated trajectory.}
\]

* 44 term slots, 37 active in v41 runs; every active term is a *sum* of two
  per-robot sub-terms that read only that robot's joints/root/yaw plus fixed
  inputs (object trajectory, source anchors, contact masks, phase labels,
  ground).
* The single structurally cross-agent term in v41, the inter-robot keypoint
  clearance `mean(relu(0.12 − ‖kp₁−kp₂‖)²)`, has **zero value and zero
  gradient** at the cold-start point, at the staged-initialization point
  (v40 result) and at the frozen v41 solution: the minimum inter-robot
  keypoint distance in all evaluated results is ≥ 0.41 m, far above 0.12 m.
  Its activation count over all 600 iterations of the logged joint runs is
  zero (`loss_activation_logs.csv`, `inter_robot_raw_iterations_nonzero = 0`).
* Therefore **we do not claim any direct cross-agent optimization benefit
  from the objective**; there is no active cross-agent dependency.

## 2. Loss dependency table

`loss_dependency_table.csv` lists for every term: weight field and value in
the J4/v41 preset, optimized inputs, fixed inputs, whether the object pose is
fixed (always yes — the object trajectory is a fixed input), and the three
probe quantities at three operating points (cold init, staged init, frozen
v41 solution):

| quantity | meaning | result for all 42 per-robot terms |
|---|---|---|
| `grad_r1`, `grad_r2` | gradient norm w.r.t. robot‑1 / robot‑2 parameters | non-zero for 26–28 terms per point (the rest are exactly zero: `both_soles`, `putdown_root`, `self_leg`, `hip_yaw`, `hip_roll`, `outward`, `squat`, `swing_knee` are inactive at the probed points) |
| `cross_hvp` | max over 3 random directions of ‖∂²T/∂θ₂∂θ₁·u‖ (cross Hessian–vector product, double backward) | **0.0 exactly** |
| `fd_cross` | max change of ∂T/∂θ₁ after perturbing θ₂ by N(0, 0.01²) | **0.0 exactly** |

Additive separability of the total objective (`separability_summary_*.json`):

| operating point | L_joint | L₁ + L₂ + L_cross | abs error |
|---|---|---|---|
| cold init (`outputs/independent`) | 275.1335 | 275.1335 | 5.7e‑14 |
| staged init (`balanced_clearance_v40_smooth`) | 7.5467 | 7.5467 | 1.8e‑15 |
| frozen v41 solution (`balanced_clearance_v41`) | 2.7892 | 2.7892 | 1.8e‑15 |

Positive controls (the test detects coupling when it exists):

| control | cross-agent term value | FD change of ∂L/∂θ₁ after θ₂ perturbation | flagged |
|---|---|---|---|
| `inter_robot_clearance = 1.0 m` | 1.16e‑3 | 1.5e‑4 | yes (`inter_robot`) |
| `joint_contact_continuity_weight = 50` (research extension) | 3.48 | 0.236 | yes (`joint_contact`) |

## 3. Why "joint optimization" still does not reproduce two independent solves

The *objective* is separable, but the historical *solver* is not bitwise
equivalent to two independent solvers, for three reasons
(`phase1_separability/split_optimizer_reproduction.json`,
`*.loss_histories.csv`):

1. **Global gradient-norm clipping.** `clip_grad_norm_(all 6 parameter
   tensors, 5.0)` rescales both robots' gradients by one coefficient.
   At cold start the per-robot gradient norms are 187 (robot 1) and 462
   (robot 2), so robot 1 is throttled by robot 2.  The joint and split loss
   histories diverge from iteration 1 (|Δ| > 1e‑3 immediately).
2. **Best-iterate selection by the summed loss.**  The returned trajectory
   is the iterate with minimal L₁+L₂ (iteration 581), whereas the
   independent solves return their own minima (585 and 591).
3. **Chaotic sensitivity.**  With per-robot clipping
   (`per_robot_grad_clip=true`, a diagnostic control added for this audit)
   the gradients are bitwise identical to the split solve except 6 of 11 000
   entries differing by ≤ 1.8e‑15 (autograd accumulation order).  The loss
   histories agree to 1.7e‑13 for 50 iterations, to 1e‑9 for 85 iterations,
   then diverge (|Δ| > 1e‑3 at iteration 103, max 1.27).  Thread count is
   not the cause (8 vs 10 threads reproduce bitwise).

Consequences for the trajectories (cold start, 018, 600 iterations):

| pair | max |Δq| (rad) | max |Δroot| (cm) | max |Δkeypoint| (cm) | contact masks |
|---|---|---|---|---|
| Joint (global clip) vs OA‑Independent | 0.41 / 0.50 | 6.0 / 5.7 | 24.7 / 14.2 | identical (fixed inputs) |
| Joint (per-robot clip) vs OA‑Independent | 0.34 / 0.50 | 3.0 / 5.7 | 21.5 / 14.2 | identical |

Objective values: Joint (global clip) best total 4.779; OA‑Independent
L₁*+L₂* = 4.455; Joint (per-robot clip) 4.530.  **The independent solves reach
a lower value of the same objective** — the joint solver's global clipping
is a hindrance, not a coupling mechanism.

## 4. Other code paths checked

* Object-relative palm anchors (`_contact_anchors`): computed once from the
  *source* fingertips and the object trajectory; identical inputs for both
  solve modes; not optimized.
* Phase / contact weights: `contact_weights[person] * phase_weights` — fixed
  per-robot tensors.
* Staged initialization: the warm start is a per-robot file read
  (`baselines[person]`); no cross-robot operation.
* Smoothing (`smooth-trajectories --clearance-recipe`): per-robot Gaussian
  filters and per-robot sole lift; no cross-robot operation.
* Reproduction checks: `runs_018/joint_J4_cold` is bit-identical to
  `outputs/paper_eval_join/ablation_018/J4/raw`; `runs_018/joint_v41_staged`
  (J4 preset, init from v40) is bit-identical to the frozen
  `outputs/balanced_clearance_v41/20231108_018` raw result.  The refactor
  used for this audit (nested `evaluate_objective`, `solve_robots`,
  default-off extension fields) therefore preserves the historical solver.

## 5. Conclusion

The current objective is separable.  No active cross-agent dependency exists
in frozen v41.  The only cross-robot operations are numerical (global
gradient clipping and summed-loss iterate selection), and they make the joint
solver *worse* on its own objective than two independent solves.  Any
performance difference between "Joint" and "OA‑Independent" must therefore be
attributed to optimizer numerics, not to interaction-aware coupling.

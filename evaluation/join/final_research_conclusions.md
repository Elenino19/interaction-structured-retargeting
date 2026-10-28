# Final research conclusions — object-aware independent baseline extension

Scope: CORE4D Join takes 018/019/020/108/109/110, dual G1, frozen v41 method.
All numbers come from the frozen paper evaluator (`evaluate-paper-join`) and
the post-hoc coordination evaluator with the paper contact contract
(0.04/0.055 m hysteresis, stable onset ≥ 5 frames, 15 fps).  Files referenced
are in this directory.  Nothing under `outputs/balanced_*`,
`outputs/metrics/*`, `outputs/paper_eval_join/*`, `videos/*` was modified
(md5 of frozen 018 robot1/robot2 unchanged:
`bdcf43e656cc3df846f12baa740a6a1f` / `6ec5e5d65a0226a69004f1db139b9002`).

Noise floor to keep in mind: a single take of this non-smooth problem is
sensitive to 1e‑15‑level numerical perturbations (Phase 1); on 018 that alone
moves joint F1 by ≈0.1.  Single-take differences of that size are not
individually conclusive; multi-take means ± SD and paired per-take deltas are
the evidence used below.

---

## Q1. Is the current multi-robot objective mathematically separable?

**Yes.**  `objective_separability_audit.md`, `loss_dependency_table.csv`.

* L(θ₁,θ₂) = L₁(θ₁) + L₂(θ₂) + w·L_inter, with L_inter ≡ 0 on every
  evaluated trajectory.  Additive error ≤ 5.7e‑14 at cold init, staged init
  and the v41 solution.
* All 42 per-robot term slots: cross Hessian–vector products **exactly 0**,
  finite-difference cross-gradient change **exactly 0**, at all three
  operating points.  The two positive controls (clearance 1.0 m; the
  research-extension term) were detected by the same test.
* The inter-robot clearance term (0.12 m) never activated in any of the
  41 logged runs (`loss_activation_logs.csv`,
  `inter_robot_raw_iterations_nonzero = 0`); minimum inter-robot keypoint
  distance over all results ≥ 0.40 m.

## Q2. Does joint optimization provide any benefit over independent object-aware retargeting?

**No measurable benefit; the opposite on most takes.**
(`oa_independent_metrics.csv`, `per_sequence_comparison.csv`,
`per_sequence_summary_mean_sd.csv`, `raw_trajectory_comparisons.csv`.)

Matched protocols, same inputs/init/budget/smoothing/evaluator
(`baseline_config_comparison.md`, `fairness_checklist.md`):

| protocol / scope | metric | Joint (frozen v41 config) | OA‑Independent |
|---|---|---|---|
| 018 cold | valid joint coverage @0.5 cm | 0.400 | **0.683** |
| 018 cold | joint F1 | 0.571 | **0.812** |
| 018 cold | bilateral anchor error (cm) | 14.40 | **13.24** |
| 018 cold | hand penetration max (cm) | 0.00 | 0.00 |
| 018 staged (v40 init) | valid joint coverage | 0.650 | **0.783** |
| 018 staged | joint F1 | 0.788 | **0.879** |
| 018 staged | bilateral error (cm) | 9.67 | **9.09** |
| 018 staged | hand penetration (cm) | 0.00 | 0.00 |
| 6 takes staged, mean ± SD | valid joint coverage | 0.553 ± 0.173 | 0.619 ± 0.201 |
| 6 takes | joint F1 | 0.707 ± 0.149 | 0.762 ± 0.156 |
| 6 takes | bilateral error (cm) | 12.44 | 12.03 |
| 6 takes | joint onset error (frames) | 11.7 | **4.5** |
| 6 takes | hand penetration max (cm) | **0.79** | 1.18 |
| 6 takes | joint jerk RMS (rad/s³) | 182 | **168** |
| test 109/110 | valid joint coverage | 0.381 ± 0.116 | 0.462 ± 0.158 |

Per take, OA‑Independent is better on coverage/F1 for 018, 019, 108, 109,
110 and worse on 020 (−0.27 coverage; OA‑Independent robot 2 onset 17 frames
late on that take).  The only consistent cost of the independent baseline is
slightly higher hand penetration on 3 of 6 takes (mean 1.18 vs 0.79 cm).

Mechanism: the joint solver's only cross-robot operations are global
gradient-norm clipping and best-iterate selection by the summed loss; these
make it reach a *higher* value of the same objective than two independent
solves (018 cold: 4.78 vs 4.46; staged: 2.82 vs 2.66 — see
`split_optimizer_reproduction.json` and `raw_trajectory_comparisons.csv`).

## Q3. Is the frozen v41 result reproducible under independent solving?

**Not within numerical tolerance.**  Objective and gradients are identical
(gradient tensors bitwise equal except ≤ 1.8e‑15 roundoff in 6 of 11 000
entries), but the trajectories diverge: with the historical global clipping
the loss histories differ from iteration 1; with per-robot clipping they
agree to 1e‑13 for 50 iterations and to 1e‑9 for 85 iterations, then
diverge (|Δloss| > 1e‑3 at iteration 103).  Final max |Δq| 0.32–0.50 rad,
max |Δroot| 3–11 cm, max keypoint difference 14–25 cm; contact masks
identical (they are fixed inputs).  The *frozen* v41 trajectories are,
however, reproduced bit-exactly by the joint solver with the instrumented
code (018 from v40 init, 019 from v20 init), and the cold J4 run is
bit-identical to `paper_eval_join/ablation_018/J4`.

## Q4. What exactly distinguishes our method from an independent object-aware baseline?

In the frozen v41 pipeline, **only solver numerics**: (i) one shared
gradient-norm clipping coefficient over both robots and (ii) returning the
iterate with minimal summed loss.  There is no active interaction-structured
term.  The components that *do* produce the paper's gains over independent
IK (object-relative contact anchors, phase-dependent weights, geometry/gait
terms, staged initialization) are per-robot and are **shared by the
OA‑Independent baseline**, which is why it matches or exceeds v41.

## Q5. Does the research-extension cross-agent coupling term improve coordination beyond the surrogate and the independent baseline?

**Yes, on the pre-registered primary metrics, with a cost.**
(`coupling_ablation_metrics.csv`,
`research_extension_joint_contact_continuity.md`.)  The term
L_JC = mean over source joint-carry frames of (1 − s₁·s₂) with differentiable
hull-distance contact indicators s_r; weight 10 selected on 018 only, then
fixed; control = separable surrogate (1 − (s₁+s₂)/2) at the same weight.
Gradient coupling was proven analytically and numerically (cross‑HVP 6.4,
FD cross-change 0.24, vs 0 for every v41 term).

Five non-development takes (019/020/108/109/110), staged protocol:

| metric | Joint (frozen v41) | OA‑Independent | EXT w10 **coupled** | EXT w10 separable surrogate |
|---|---|---|---|---|
| valid joint coverage @0.5 cm | 0.534 ± 0.185 | 0.586 ± 0.206 | **0.818 ± 0.079** | 0.752 ± 0.070 |
| joint F1 | 0.690 ± 0.161 | 0.738 ± 0.162 | **0.927 ± 0.047** | 0.887 ± 0.040 |
| valid joint IoU | 0.534 | 0.586 | **0.818** | 0.752 |
| longest valid simultaneous span (frames) | 10.8 | 11.4 | **19.6** | 13.0 |
| joint onset error (frames) | 13.6 | 4.2 | **1.2** | 2.6 |
| bilateral anchor error (cm) | **13.00** | 12.62 | 13.38 | 13.31 |
| hand penetration max (cm) | **0.95** | 1.42 | 1.84 | 2.32 |
| geometry violation (cm, hand/foot/ground max) | **7.53** | 7.72 | 7.76 | 7.95 |
| joint jerk RMS (rad/s³) | 194.7 | **180.0** | 216.1 | 213.9 |
| planted-foot speed (cm/s) | 22.0 | **21.7** | 23.2 | 22.3 |
| keypoint error (cm) | 12.9 | 12.9 | **12.7** | 12.8 |
| grasp-order preserved | 5/5 | 5/5 | 5/5 | 5/5 |
| test 109/110 coverage | 0.381 | 0.462 | **0.738 ± 0.023** | 0.693 ± 0.037 |

Paired per-take deltas, coupled − surrogate: coverage +0.062, +0.061,
+0.116, +0.056, +0.035 (**5/5 positive**); joint F1 +0.023, +0.012, +0.123,
0.000, +0.044 (4/5 positive, 1 tie); longest valid span +8, +9, +7, −2, +11
frames (4/5).  The coupled term also lowers the variance across takes
(coverage SD 0.079 vs 0.185 for v41).  Caveats: (a) **both** extension
variants raise jerk (~+11%) and hand penetration (+0.4–0.9 cm, driven by one
take, 020, with 3.7 cm); the coupled version is better than the surrogate on
penetration (1.84 vs 2.32) but worse than v41; (b) n = 5 takes, single
deterministic run each; (c) bilateral anchor error is unchanged — the term
improves *when* both robots hold, not *where*.

## Q6. Where does independent object-aware retargeting fail?

Observed failure modes of OA‑Independent (vs source):

1. **Asymmetric onset timing** — robot 2 grasps 9–17 frames late on 019/020
   (robot 1 is on time), giving the 020 collapse (coverage 0.43 vs 0.69
   for v41).  Nothing in a per-robot objective ties one robot's grasp
   timing to the other's; the coupled term fixed this on 020
   (coverage 0.84, onset lag error 0).
2. **Short valid simultaneous spans** — mean 11.4 frames (0.76 s) vs 19.6
   for the coupled extension: each robot drifts in and out of contact
   independently so the AND-mask fragments.
3. **Release-order errors** on 019 (robot 1 releases before robot 2 while
   the source is the reverse) — shared with v41.
4. Slightly higher hand penetration than v41 on 3 of 6 takes.

What it does *not* fail at: object-relative grasp placement (bilateral error
equal or better than v41), per-robot contact F1 (0.85–0.92 on train takes),
footwork/jerk (better than v41).  Under the pre-registered 0.05 m root
perturbations (`coordination_robustness.csv`), Joint and OA‑Independent are
indistinguishable: coverage 0.52 ± 0.09 vs 0.55 ± 0.03, F1 0.68 ± 0.08 vs
0.71 ± 0.03, max deviation from the unperturbed run 0.18 vs 0.17 — as
pre-registered for a separable objective.

## Q7. What is the honest novelty claim?

Supported by evidence:

* The paper's **interaction-structured formulation** — object-relative
  contact anchors from source fingertips, phase-labelled weighting,
  geometry/support terms around a shared object — produces the reported
  gains over independent IK.  These components are per-robot and
  transferable; "multi-party" enters through the shared object trajectory
  and the shared interaction phase labels, not through cross-robot
  optimization.
* **Not supported:** any claim that *jointly optimizing* the two humanoids
  in v41 yields coordination that independent object-aware retargeting
  cannot.  The v41 objective is separable, its one cross-agent term is
  inert, and a fair independent baseline is equal or better on 5 of 6 takes.
* **Newly supported (research extension, not in frozen v41):** a
  differentiable joint-contact continuity term with a product coupling
  improves geometry-valid simultaneous-contact coverage from 0.59 (independent)
  / 0.53 (v41) to 0.82 on five held-out takes, beats its separable surrogate
  on all five (paired), and halves across-take variance, at the cost of
  ~11% more jerk and ~0.4 cm more hand penetration.  This is the first
  component in the pipeline whose benefit is attributable to cross-agent
  coupling, and it should be presented as such (with the surrogate control)
  rather than folded into the v41 claims.

Recommended paper positioning: present v41 as an *object-aware,
phase-structured* retargeter whose two-robot solve is a convenience, add
OA‑Independent as the primary baseline (it is stronger than independent IK
and must be reported), and present the joint-contact term as an explicit
multi-party coordination contribution with its own ablation.

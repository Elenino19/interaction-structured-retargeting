# Proposed extension — joint transfer‑event optimisation (NOT implemented)

Status: proposal only. Nothing below exists in the code base; the accepted Full pipeline is unchanged. It is written
because the audit found no genuine cross‑agent coupling (`dependency_audit.md`), and a coordination claim needs one.

## Idea

Make each kick→receive transfer a *coupled* sub‑problem: the sender's release pose, the receiver's reception pose,
the transfer time and the ball segment are decided together, so that a change in one robot's feasible region moves
the other robot's targets through shared variables.

## Optimised variables (per transfer \(i\), sender \(s\), receiver \(r\))

| symbol | meaning | currently |
|---|---|---|
| \(t^{\text{rel}}_i, t^{\text{arr}}_i\) | release and arrival frames (continuous, bounded around the annotation) | fixed annotated frames |
| \(b_i(\tau)\) | ball segment on \([t^{\text{rel}}_i, t^{\text{arr}}_i]\), low‑order spline on the ground plane (e.g. 4 control points, \(z = 0.11\)) | closed‑form post‑process from both robots |
| \(q_s, r_s, q_r, r_r\) | both robots' joint/root trajectories (as now) | independent |
| optional \(y_i\) | lane crossing Y (bounded to the annotated gap ± half‑gap) | fixed lane schedule |

## Objective

\[
\min\ L_A + L_B
+ \sum_i \Big[
\underbrace{w_{\text{rel}}\, d\big(\text{foot}_s(t^{\text{rel}}_i),\, b_i(t^{\text{rel}}_i)\big)^2}_{\text{sender release geometry}}
+ \underbrace{w_{\text{arr}}\, d\big(\text{foot}_r(t^{\text{arr}}_i),\, b_i(t^{\text{arr}}_i)\big)^2}_{\text{receiver reception geometry}}
+ \underbrace{w_{\text{dir}}\,\big\|\hat v_s(t^{\text{rel}}_i) - \hat{\dot b}_i(t^{\text{rel}}_i)\big\|^2}_{\text{release direction = kick‑foot velocity}}
+ \underbrace{w_t\,\big((t^{\text{rel}}_i-\hat t^{\text{rel}}_i)^2 + (t^{\text{arr}}_i-\hat t^{\text{arr}}_i)^2\big)}_{\text{stay near source timing}}
+ \underbrace{w_b\!\int\!\|\ddot b_i\|^2}_{\text{ball path regularity}}
\Big]
\]

with \(d(\cdot,\cdot)\) the signed distance between the official foot mesh (or a capsule approximation) and the ball
sphere, \(L_A, L_B\) the existing per‑robot objectives (source fidelity, support, smoothness). The coupling is
explicit: \(b_i\) and \(t_i\) appear in terms involving **both** robots, so \(\partial L/\partial q_s\) depends on
\(q_r\) through \(b_i\) and vice versa.

## Feasibility constraints

* Order: \(t^{\text{rel}}_i < t^{\text{arr}}_i < t^{\text{rel}}_{i+1}\); bounded drift \(|t - \hat t| \le 4\) frames.
* Receiver reachability: at \(t^{\text{arr}}_i\), \(b_i\) must lie inside the receiver's annotated‑foot reachable set
  (leg‑length ball around the hip, foot not crossing the stance leg), expressed as a soft barrier.
* Sender release: ball starts at the kick‑foot surface, initial direction within a cone of the foot velocity.
* Scene: ball segment clears cones by ≥ 0.06 + 0.11 m; stays within the ladder; speed within a plausible range for a
  ground pass (e.g. 2–8 m/s); ball speed monotonically non‑increasing between release and arrival (rolling).
* Non‑penetration: signed foot–ball distance ≥ 0 at all frames except within ±1 frame of release/arrival (soft).
* Continuity across transfers: \(b_i(t^{\text{arr}}_i) = b_{i+1}(t^{\text{rel}}_{i+1})\) (ball is carried by the
  receiver between events, as now, or held still).

## Required code changes

1. `retarget_soccer` → a joint `retarget_pair` that holds both robots' parameters and the per‑transfer variables in
   one optimiser; ball spline evaluation and foot‑mesh (capsule) distance in torch.
2. Replace `snap_ball_to_g1_feet` by exporting the optimised \(b_i\) (keep the current function as the baseline
   generator).
3. Soft time variables: sample foot poses at continuous \(t\) by linear interpolation of FK outputs (differentiable).
4. Evaluation: reuse `evaluate_event_chain.py`; add release‑direction consistency and per‑transfer feasibility
   residuals; keep the fixed ±4‑frame timing rule or report both.
5. Determinism / reproducibility harness as in `run_fixed_ball_event_ablation.py`.

## Comparison baselines (all with the same reference, annotations, initialisation and budget)

* B0: accepted Full (independent IK + post‑hoc ball).
* B1: independent object‑aware solvers on the B0 ball (already run: `independent_object_aware/obj_w30_event_on`).
* B2: *sequential* coupling — sender solved first, ball from sender release, receiver solved to that ball (one‑way
  information flow). Needed to show that *joint* solving, not merely information flow, is what helps.
* B3: proposed joint optimisation.
Metrics: solver‑sensitive set only (mesh‑gated chains, surface deviation, penetration, jerk, support sliding,
velocity fidelity, release‑direction consistency) — the specification‑determined set will be constant by design.

## Risks

* **Source fidelity**: freeing \(t_i\) and \(b_i\) lets the solver move feet/time away from the human performance;
  bound drift tightly and report keypoint error vs. the source reference per variant.
* **Smoothness**: extra contact terms at a few frames already raised joint jerk 17 % in the independent baseline;
  joint terms on both robots will do the same unless the contact window is spread (±2 frames) and the smoothness
  weights are kept.
* **Non‑convexity / timing variables**: continuous time with interpolated FK is non‑smooth at frame boundaries; expect
  local minima; initialise from the annotation and use a short schedule.
* **Evaluation circularity**: if the ball is optimised against the same foot‑distance metric used for evaluation, the
  gain is partly tautological — report held‑out metrics (release direction, penetration, jerk, source fidelity) and
  the sequential baseline B2.
* **Still kinematic**: a rolling‑speed constraint is not physics; no claim of physical contact should follow.
* **n = 1**: the method would still be validated on one sequence unless a second licensed clip is obtained.

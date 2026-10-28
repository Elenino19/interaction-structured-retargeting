# Novelty assessment and answers to the six questions

Basis: `dependency_audit.md`, `objective_separability.md`, `strong_baseline_feasibility.md`,
`event_chain_comparison.csv`, `metric_sensitivity.csv`, `separability_probe.json`, and the frozen evaluation data in
`../event_coordination_extension/`. One 144‑frame monocular sequence; 6 receive + 7 kick events; 6 transfer pairs;
all metrics are kinematic proxies on a prescribed object path (no physics, no task success).

## Task 3 — multi‑agent event metrics, reported separately (`event_chain_comparison.csv`)

| metric | accepted Full | range over the 7 solvers that share the frozen ball + stationed reference | range over all 14 variants | determined by |
|---|---|---|---|---|
| correct actor association (recv / kick) | 6/6, 7/7 | constant 1.00 | constant 1.00 | ball path + annotations |
| correct active‑foot association (recv / kick) | 6/6, 7/7 | constant 1.00 | 0.83–1.00 / 0.57–1.00 | ball generation + stationing |
| source‑consistent kick→receive order | 6/6 | constant 1.00 | constant 1.00 | ball path (+ ±4‑frame rule) |
| event‑chain consistency, role‑only | 6/6 | constant 1.00 | 0.50–1.00 | ball generation + stationing |
| event‑chain, centre‑gated τ=0.05 | 6/6 | constant 1.00 | 0.00–1.00 | ball generation + stationing |
| geometry‑valid association, mesh‑gated τ=0.05 | 3/6 | 1/6 – 6/6 | 0/6 – 6/6 | **computed robot trajectories** |
| mean \|surface deviation\| | 0.022 m | 0.0001–0.028 m | 0.0001–0.319 m | **robot trajectories** |
| foot–ball penetration (signed mesh gap / events) | −0.043 m, 13/13 | −0.044…−0.019 m, 12–13/13 | −0.044…+0.29 m, 2–13/13 | **robot trajectories** |
| kick‑foot velocity fidelity (RMSE) | 0.483 m/s | 0.18–0.94 m/s | 0.14–0.94 m/s | **robot trajectories** |
| timing error (recv / kick) | 0.17 / 0.57 f | 0.17–0.83 / 0.29–0.57 f | 0–2.7 / 0.3–1.9 f | mostly ball path; weakly robot |

Which results follow directly from the fixed object path + annotations: all actor, foot, order and role‑chain
results, and the centre‑gated chains. For the accepted Full they hold **by construction** — every ball waypoint is
placed 0.075/0.095 m from the annotated foot of the annotated robot at the annotated frame
(`snap_ball_to_g1_feet`), so the nearest foot at that frame is that foot unless the other robot stands within ≈0.1 m
of it. Across seven different per‑robot solvers on the same frozen ball these numbers did not move at all.
They break only when the *specification* changes: a ball not snapped to the robots (`basic_ik`, kick foot 4/7) or a
mismatched stationing (`current_code`, receive 5/6, chain 4/6 → 5/6).

Which results are sensitive to computed robot trajectories: everything geometric at the contact (surface
deviation, mesh gap, penetration count, mesh‑gated chain), jerk, support sliding, kick‑velocity fidelity, and the
sub‑frame part of timing. These are the metrics on which solvers can legitimately be compared — and on them the
independent object‑aware baseline beats Full on geometry while Full's event terms win on kick‑velocity fidelity.

## Answers

**1. What is genuinely shared across agents?**
Three things, none of them optimisation variables: (i) the frozen interaction annotation \(\mathcal{E}\) (actor,
receive/kick frame, active foot), produced offline by a DP that *does* look at both players' feet; (ii) the lane
schedule / scene layout derived from \(\mathcal{E}\) and the cone geometry, which both per‑robot planners read;
(iii) the ball path, computed after the fact from both robots' solutions. The per‑robot solvers share no state.

**2. Which operations are mathematically coupled?**
Only the ball post‑process \(g(p_A^\*, p_B^\*, \mathcal{E})\) (reads both robots; one‑directional, no feedback) and
the offline annotation DP. The objective \(L_A + L_B\) is additively separable; the probe shows zero change in any
robot‑A stage under large perturbations of robot B. There is no class‑(4) joint term.

**3. Does current Soccer Full provide a benefit over two strong independent object‑aware solvers?**
No. It is computationally identical to two independent solvers with identical inputs (equivalence, not benefit).
Against a strictly stronger independent solver that also sees the frozen ball, Full is worse on every foot–ball
geometry metric (surface deviation 0.022 vs 0.002 m; mesh‑gated chains 3/6 vs 6/6) and equal on every
actor/foot/order/chain metric; its only advantage is kick‑foot velocity fidelity (0.48 vs 0.21 m/s — and the
independent solver with the Full recipe reaches 0.18).

**4. What exactly is supported by the existing results?**
* A monocular two‑person clip can be turned into two G1 motions plus an event‑consistent object path such that all
  13 annotated events and 6 transfers are reproduced with the correct actor, foot and order (kinematically).
* Event‑conditioned IK terms improve kick‑foot velocity fidelity (0.66 → 0.20 m/s) at the cost of ~2× jerk and deeper
  interpenetration; they do not change any association/chain outcome.
* The event/lane specification (stationing + object path tied to annotated feet) is what produces the chain
  consistency: removing it on the object side (event_semantics → full_without_position) or mismatching it
  (current_code) is the only thing that moved those metrics.
* Not supported: coordination benefit from joint optimisation; physical validity; generalisation beyond one sequence.

**5. Is Soccer best positioned as an interaction‑representation case study rather than a standalone coupled
optimisation algorithm?**
Yes. The defensible contribution is the *representation and its compilation*: an interaction‑state annotation
(actors, limbs, event times, transfer order, lanes) that is compiled into per‑robot stationing/limb targets and an
event‑consistent object path, with an evaluation protocol (actor/foot/order/chain, geometry gates, sensitivity
classes) that separates specification‑determined from solver‑determined outcomes. Presenting Soccer as a coupled
multi‑agent optimiser would be contradicted by the code.

**6. What minimal new method would be necessary to make a stronger coordination claim?**
Introduce at least one shared decision variable with constraints from both agents and solve for it jointly — e.g. a
per‑transfer release/arrival time and bounded ball segment constrained by the sender's release geometry and the
receiver's reachability, with gradients flowing into both robots (`proposed_extension.md`). The claim would then be
tested against the independent object‑aware baseline above on the solver‑sensitive metrics, and against a
"sequential" variant (sender first, receiver second) to show that joint solving matters.

## Novelty position (summary)

| claim | status |
|---|---|
| Multi‑party interaction‑state representation for retargeting + object path synthesis | supported (case study, n = 1) |
| Event‑conditioned IK improves limb‑motion fidelity at events | supported, with jerk/penetration cost |
| Coordination benefit from a coupled multi‑agent optimisation | **not supported — no coupling exists** |
| Benefit over independent object‑aware solvers | **not supported — equivalent or worse on geometry** |
| Physical foot–ball interaction | not claimed, not supported |

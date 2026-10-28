# Strong independent object‑aware baseline (Task 2)

## 1. Equivalence statement

Given identical ball trajectory, identical per‑robot event / active‑foot annotations, identical object‑relative
targets (the stationed reference), identical source motion, geometry, support constraints, stationing, initialisation
and solver budget, **two independent object‑aware robot solvers and the accepted Soccer Full solver are the same
computation** for the robot trajectories: the Full IK is already two independent per‑robot minimisations
(`objective_separability.md`, probe Δ = 0 at every robot‑side stage), and it contains no object term at all. The only
remaining difference is how the ball is obtained — Full derives it from the two solutions (`snap_ball_to_g1_feet`),
whereas an independent baseline must be handed it. With the ball prescribed identically, the outputs coincide
exactly. **We therefore report equivalence, not a coordination improvement.**

The row `fixed_ball/frozen_reference/event_on_full_recipe` in `event_chain_comparison.csv` is exactly this
"independent solver with identical inputs" (one process, robots solved in sequence with no communication); it
reproduces the accepted Full's event‑chain results (13/13, 6/6, same centre‑gated chains) up to the documented
IK‑reproduction drift (`../event_coordination_extension/pipeline_dependency_audit.md` §5).

## 2. A *stronger* independent baseline: explicit object term

Because the Full IK never references the ball, a fair "object‑aware" independent baseline is strictly richer than
Full. We added an optional per‑robot term (default off; accepted pipeline unchanged):

\[
L_k \mathrel{+}= w \sum_{(t, f)\in \mathcal{E}_k} \big(\|p_k^{\text{toe}_f}(t) - \text{ball}(t)\| - 0.11\big)^2 ,
\]

over this robot's own annotated receive and kick frames only (no information about the other robot). Everything
else is identical to `event_on_full_recipe`: frozen ball (SHA‑256 `b608f0a9…f47f`, copied verbatim), shared accepted
reference targets, same initialisation, 340 Adam iterations. Weights were predefined as \(w\in\{10, 30, 100\}\) with
30 primary; an event‑IK‑off version with \(w = 30\) was also run. Script:
`scripts/run_independent_object_aware_baseline.py`; outputs `independent_object_aware/<variant>/`.

## 3. Results (frozen ball + shared reference; 6 receive, 7 kick, 6 pairs, one sequence)

| metric | accepted Full | independent = Full recipe | independent + object term w=30 | w=10 | w=100 | event IK off + w=30 |
|---|---|---|---|---|---|---|
| correct actor (recv / kick) | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |
| correct foot (recv / kick) | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |
| kick‑before‑receive order | 6/6 | 6/6 | 6/6 | 6/6 | 6/6 | 6/6 |
| timing error recv / kick [frames] | 0.17 / 0.57 | 0.33 / 0.43 | 0.83 / 0.29 | 0.83 / 0.29 | 0.33 / 0.43 | 0.33 / 0.43 |
| chain, role‑only | 6/6 | 6/6 | 6/6 | 6/6 | 6/6 | 6/6 |
| chain, centre‑gated τ=0.05 | 6/6 | 6/6 | 6/6 | 6/6 | 6/6 | 6/6 |
| chain, mesh‑gated τ=0.05 | 3/6 | 1/6 | **6/6** | **6/6** | **6/6** | **6/6** |
| mean \|surface deviation\| [m] | 0.0220 | 0.0280 | **0.0018** | 0.0036 | 0.0004 | 0.0001 |
| mean signed mesh gap [m] | −0.0428 | −0.0435 | −0.0270 | −0.0277 | −0.0273 | **−0.0194** |
| mesh ball‑penetration events /13 | 13 | 12 | 13 | 13 | 13 | 13 |
| support‑foot displacement [m] | 0.461 | 0.439 | 0.436 | 0.438 | 0.447 | 0.468 |
| mesh floor penetration max [m] | 0.0270 | 0.0226 | 0.0228 | 0.0227 | 0.0227 | 0.0239 |
| root jerk RMS [m/s³] | 784 | 797 | 835 | 820 | 843 | **458** |
| joint jerk RMS [rad/s³] | 1242 | 1278 | 1459 | 1367 | 1533 | 1377 |
| kick‑foot velocity RMSE [m/s] | 0.483 | 0.184 | 0.211 | 0.206 | 0.225 | 0.936 |

Reading:
* All actor/foot/order/role‑chain metrics are identical across every column (they are fixed by the ball + annotation
  + stationed reference; see `metric_sensitivity.csv`).
* The independent object‑aware solver **dominates Full on every geometry metric** (surface deviation 0.022 → 0.002 m,
  mesh‑gated chains 3/6 → 6/6) at a 17 % joint‑jerk cost and ~+0.03 m/s velocity RMSE. The foot mesh still enters the
  ball at all 13 events because the term places the toe *keypoint* on the sphere while the shoe mesh extends past it; a
  mesh‑aware distance would be the natural fix and is equally available to an independent solver.
* Removing event IK while keeping the object term gives the best mesh gap and root jerk but destroys kick‑velocity
  fidelity (0.94 m/s RMSE): the object term alone pins the toe at one frame and the smooth solver reaches it slowly.

## 4. Conclusion

The accepted Soccer Full does not provide a benefit over two strong independent object‑aware solvers; it *is* two
independent solvers, and an independent solver that additionally sees the (identical) ball is better on foot–ball
geometry. The baseline was not weakened: it received the same reference, ball, annotations, initialisation and
budget, plus one extra term. No claim of coordination improvement from the optimisation is supportable.

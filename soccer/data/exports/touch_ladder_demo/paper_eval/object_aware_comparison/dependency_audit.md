# Dependency audit — what is shared across the two robots (Task 1)

Code traced: `scripts/recover_ball_cones.py`, `scripts/pack_touch_ladder_demo.py`,
`scripts/retarget_dual_g1_soccer.py` (stationing, windows, IK, post-alignment, ball snapping),
`scripts/run_ablation_experiments.py`, `scripts/evaluate_*.py`. Empirical confirmation:
`separability_probe.json` (produced by `scripts/check_objective_separability.py`): robot B's source joints were
perturbed (N(0, 0.05 m) noise + 0.4 m lateral shift) or B's IK output shifted by 0.10 m, and every robot‑A stage was
re‑computed and compared bit‑for‑bit.

Classification key — **(1)** independent per‑robot term · **(2)** prescribed common object trajectory ·
**(3)** shared interaction annotation · **(4)** genuinely joint cross‑agent optimisation term.

| # | Mechanism | Where | Inputs | Reads state of BOTH robots? | Updates a shared variable from both? | Class | Probe result |
|---|---|---|---|---|---|---|---|
| 1 | Two‑person detection / crop / GVHMR | `prepare_touch_ladder_demo.py`, `run_gvhmr_pair.py` | video | no (two independent runs) | no | (1) | — |
| 2 | Shared‑ground placement (A faces +X, B −X, nominal 2.8 m gap) | `pack_touch_ladder_demo.py::place_on_shared_ground` | both GVHMR results | yes (both placed in one frame) | yes — a fixed layout constant, not optimised | (3) scene convention | — |
| 3 | Contact annotation (alternating 7‑contact DP) | `recover_ball_cones.py::detect_contacts` | ball UV, both players' foot UV | **yes** (DP over A/B alternation) | yes → `ball_cones.json` | (3) shared annotation, frozen offline | — |
| 4 | Sender/receiver role assignment | `ball_cones.json` order; `_contact_stations`, `transfer_pairs` | annotation | no robot state | no | (3) | — |
| 5 | Pass lines / lane schedule `_pass_lines` | `retarget_dual_g1_soccer.py` | event frames, lane geometry | no | shared Y schedule derived from annotation only | (3)+(2) | — |
| 6 | Stationing `_ready_shift`, event‑frame foot grounding, step plant (`_apply_player_plan`) | same | this player's joints, events, lanes | **no** (`station_pair_on_ladder` only shares `n = min(len)`) | no | (1) conditioned on (3) | A targets Δ = 0, A plant Δ = 0 |
| 7 | Contact‑limb targets (ankle→0.05 m, toe→0.02 m at annotated frames) | `_apply_player_plan` | this player's joints, own events | no | no | (1) | Δ = 0 |
| 8 | Event windows `contact_windows` (kick ramp, plant mask) | same | own events | no | no | (1) | Δ = 0 |
| 9 | Event‑dependent weights (boost ×(1+2.4·kick), smooth relaxation, kick‑velocity 24) | `retarget_soccer` | own `kick` | no | no | (1) | — |
| 10 | Whole‑body IK (all loss terms) | `retarget_soccer` | one robot's targets/windows | **no** — signature takes one robot; loss has no term on the other robot **or on the ball** | no | (1) | A joints/root Δ = 0 (solved alone vs after B) |
| 11 | Post‑IK alignment `align_second_pass_stations` | same | A: own f0 right toe; B: own f15/f21 toes | no (separate offsets, separate files) | no | (1) | A root/keypoints Δ = 0 under B perturbation |
| 12 | Ball trajectory `snap_ball_to_g1_feet` | same | **both** robots' keypoints at every annotated frame (toucher's toe/ankle + other robot's pelvis for direction), mean of both roots for cone X, lane schedule | **yes** | yes → `ball_cones.npz` | (2) prescribed path computed *from* both outputs; no feedback to robots | B shifted 0.10 m ⇒ ball changes up to 0.099 m over f46–143 (98 frames); re‑snap of frozen robots reproduces frozen ball exactly (Δ = 0) |
| 13 | Baseline ball `baseline_ball_from_contacts` (ablations only) | `run_ablation_experiments.py` | both players' source ankles + mean X of the other | yes | yes | (2) | — |
| 14 | Evaluation: 4‑way actor assignment, transfer pairs | `evaluate_event_chain.py` | both robots + ball | yes | no (read‑only) | metric | — |

## Findings

* **No class‑(4) term exists.** Nothing in the optimisation couples the two robots: there is no shared decision
  variable (no common transfer time, no common ball variable inside the solver), no constraint that reads A and B
  simultaneously, and no gradient path from one robot to the other. The IK can be, and in `run_fixed_ball_event_ablation.py`
  literally is, run per robot in any order with identical results.
* The only computations that read both robots are **offline annotation (3)** and **post‑hoc ball generation (2)**.
  The ball is a deterministic function `ball = g(robot_A, robot_B, annotations, lanes)`; robots never see it.
  The pipeline is therefore a *feed‑forward composition*: annotations → per‑robot stationing → per‑robot IK →
  per‑robot alignment → ball from both.
* "Coordination" in the accepted output is realised by (a) the shared annotation and lane schedule that both per‑robot
  planners obey, and (b) generating the object path from both robots' resulting feet. Both are specification/
  composition mechanisms, not joint optimisation.
* The ball depends on the *other* robot only through a unit direction (pelvis‑to‑pelvis) and the cone X position;
  the touch locations themselves are read from the toucher only.

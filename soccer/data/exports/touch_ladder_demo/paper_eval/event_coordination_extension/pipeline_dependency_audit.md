# Pipeline dependency audit (Phase 1)

Scope: `scripts/retarget_dual_g1_soccer.py`, `scripts/run_ablation_experiments.py`,
`scripts/run_position_correction_ablation.py`, `scripts/evaluate_*.py`, frozen artifacts under
`data/exports/touch_ladder_demo/`. Written before the fixed-ball ablation was implemented;
amended with the reproduction findings discovered while running it (section 5).

## 1. Frozen inputs (read-only, hash-verified)

| Artifact | Path | SHA-256 |
|---|---|---|
| Event annotations | `data/exports/touch_ladder_demo/ball_cones.json` | `cad51d64…2156` |
| Source SMPL-X joints (GVHMR, 144 f, 24 fps) | `data/exports/touch_ladder_demo/smplx_joints.npz` | `f03a8c5b…3b4b` |
| Accepted Full robot A / B | `data/exports/touch_ladder_demo/g1/robot_{a,b}.npz` | `ef5277f2…d96b` / `eb5646a3…568b` |
| Accepted Full ball + cones | `data/exports/touch_ladder_demo/ball_cones.npz` | `b608f0a9…f47f` |
| Accepted preview | `data/exports/touch_ladder_demo/comparison/current_hop_timed_lateral.mp4` | `c904d2ed…039e` |

Annotation content (7 contacts → 6 receive + 7 kick events, 6 kick→receive transfer pairs, all cross-agent):

| idx | player | receive (frame, foot) | kick (frame, foot) |
|---|---|---|---|
| 0 | a | — (clip starts on the kick) | f0, right |
| 1 | b | f15, right | f21, left |
| 2 | a | f42, right | f45, right |
| 3 | b | f71, right | f71, right |
| 4 | a | f90, left | f90, left |
| 5 | b | f110, right | f110, right |
| 6 | a | f132, left | f132, left |

Coordinates: exports are Y-up; robot frame is Z-up via `[x, y, z]_yup → [x, −z, y]`. A faces +X, B faces −X.
Cones at robot Y `[1, 0.5, 0, −0.5, −1]` (spacing 0.5 m, radius 0.06 m); gap lanes `[1.25, 0.75, 0.25, −0.25, −0.75, −1.25]`;
pass lane order `[3, 4, 5, 4, 3, 2]`; ball radius 0.11 m.

## 2. What each stage depends on

### A. Robot joint motion (`retarget_soccer`, Adam lr 0.04, 340 it, float64, CPU)
Deterministic initialisation: `raw_joints = atanh(−mid/half)` (joint-limit midpoints), `root_delta = 0`,
`root_yaw = _yaw_from_shoulders(source)`. No warm start from any previous solution.

| Term | Weight | Event-conditioned? | Through what |
|---|---|---|---|
| keypoint tracking (16 keypoints, `BASE_WEIGHTS`) | 1.0 | **yes** (boost) | legs ×(1+2.4·kick), ankle/toe ×(1+1.6·kick); `kick` from `contact_windows` |
| root regularisation (`root_delta²`) | 0.35 | no | |
| yaw prior | 0.4 | no | |
| hip-yaw prior | 0.06 | no | |
| joint velocity / acceleration | 0.02 / 0.04 | **yes** (relaxation) | ×(1 − 0.75·swing), swing from `kick` |
| plant slip (world ankle velocity on planted foot) | 6 | **yes via mask** | `plant` mask: `contact_windows` (kick−3..kick+1 support foot) ∪ step plant (`_apply_player_plan`, event ±2) |
| sole floor (ankle z ≥ 0.012) | 6 | no | |
| pelvis bounce / bounce velocity | 18 / 40 | no | |
| upper-body velocity, chest-up, shoulder line | 8 / 6 / 4 | no | |
| kick velocity (toe/ankle velocity vs reference in kick phase) | 24 | **yes** | `left/right_phase` from `kick` |

`SoccerRetargetWeights(support_plant, support_floor, kick_velocity, event_keypoint_boost, event_smooth_relaxation)`
toggles the event terms; `kick = 0` removes boost, relaxation and kick-velocity simultaneously (they are gated by the
same window), so "event IK OFF" = `kick = 0` + `kick_velocity = 0` + flags off + a non-event plant mask.

### B. Root / stationing
1. Pre-IK, on the SMPL-X reference (`station_pair_on_ladder` → `_apply_player_plan`):
   * `_ready_shift`: world-Y offset per frame placing the **annotated striking toe** on the live pass line
     (`_pass_lines`, from `_contact_stations`, which uses the event frames and `kick_side`). **Event-timed.**
   * event-frame foot grounding: annotated foot ankle→0.05 m, toe→0.02 m blended over ±5 frames. **Event-conditioned
     modification of the IK targets.**
   * step plant: support foot ±2 frames at each event. **Event-conditioned.**
2. In IK: `root = base_root + root_delta`, `base_root = pelvis target`.
3. Post-IK (`align_second_pass_stations`): world-Y offsets for A (f0–f15) and B (f6–f35) computed from the **IK output
   keypoints** at f0 / f15 / f21 and the lane geometry. Depends on each variant's own solution → variant-dependent.

### C. Ball trajectory (`snap_ball_to_g1_feet`)
Generated **after** IK from the retargeted robots' own toe/ankle keypoints at the annotated frames (offset 0.095 m kick /
0.075 m receive along the pass direction), passes 1 and 3 foot-to-foot, the others straight along the lane Y
(`straight_gap_pass`). Therefore the frozen Full ball is a function of the frozen Full robots. It is a prescribed,
event/scene-conditioned path, not the result of foot–ball contact physics.

### D. Event and active-foot constraints
`contact_windows(contacts, player, n, radius=8, step_plant)`: kick foot active on `[kick−8, kick+3)` with ramp
0.55→1.25 and 1.7 at the kick frame; plant mask on the opposite foot `[kick−3, kick+2)`. Uses `kick_side`/`kick_frame`
(receive foot labels enter only through stationing and grounding).

## 3. Existing variants (what they do and do not share)

| Variant | stationing (B.1) | post-align (B.3) | event IK terms (A) | ball (C) |
|---|---|---|---|---|
| basic_ik | **none** | none | off, no support | source-contact interpolation baseline |
| support | none | none | generic support only | baseline |
| event_semantics | none | none | on (generic + event windows) | baseline |
| full_without_position | none (= event_semantics robots) | none | on | re-snapped to its own robots |
| full (accepted) | yes (Oct‑1 code) | yes (two manual passes) | on | snapped to accepted robots |

Consequence: the three non-stationed variants cannot be evaluated against the frozen Full ball (they stand on different
lanes), which is why a new controlled pair with a *shared* reference was required.

## 4. Existing evaluators
* `evaluate_ablation_experiments.py`: toe-centre distances at annotated frames, surface error |d−0.11|, support sliding
  kick−3..kick+1, keypoint floor proxy 0.012 m, pelvis vertical acceleration, lane crossing / cone clearance (ball
  properties).
* `evaluate_soccer_paper.py`: receive/kick split, timing (earliest minimum in ±4 frames), stance interval kick−8..kick+2,
  jerk (forward differences × fps^n), mesh foot–ground and mesh foot–ball gap (`*_ankle_roll_link` mesh vertices).
* No cross-agent assignment, no transfer-chain logic, no threshold sensitivity existed → added in
  `scripts/evaluate_event_chain.py`.

## 5. Findings that constrain the new ablation

1. **The accepted Full robots are not regenerated by the current scripts.** The archive trail
   (`g1/archive/robot_*_20261001_*.npz`) shows the accepted joints equal the IK output of 2026‑10‑01 ≈20:17, followed
   by two in-place world-Y re-stationing passes (`pre_opening_alignment` 22:43, `pre_hop_shift` 22:57). The stationing
   code has since changed (`_pass_lines` settle logic now places the striking toe on the *previous* lane at the event
   frame: e.g. B right toe at f71 Y = −1.29 m with current code vs −0.81 m in the accepted run). Re-running the current
   code therefore yields references up to 0.78 m away from the accepted targets and 0.4–0.5 m from the frozen ball at
   8 of 13 events (`fixed_ball_ablation/current_code/*`).
2. **Shared reference used for the primary comparison.** Each accepted NPZ stores the exact IK targets
   (`target_keypoints`, 16 keypoints, robot frame, including stationing, event-frame grounding and the post-alignment
   offsets). Feeding these verbatim to both variants (`retarget_soccer(..., precomputed_targets=...)`) gives a reference
   that is identical across variants and consistent with the frozen ball. Shared (non-isolated) event information in
   this reference: event-timed stationing and event-frame foot grounding. Isolated factor: event-conditioned IK
   refinement (foot-role windows, keypoint boost, smoothness relaxation, kick-velocity term, event plant mask).
3. **Reproduction gap of the IK itself.** `event_on_full_recipe` (accepted recipe on the accepted targets) vs accepted
   joints: mean |Δq| 0.013 rad (A) / 0.018 rad (B), keypoint mean |Δ| 4–9 mm, final loss 0.0529 vs 0.0517 (A),
   0.0357 vs 0.0343 (B); frame-localised differences coincide with the post-alignment ramp windows (the stored targets
   carry time-varying world-Y offsets the original run did not see). The frozen Full row is therefore reported
   separately and never replaced.

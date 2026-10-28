# Method to code map

Draft v6.1 is not in this repository. The formulas below are the ones the code implements. The 2026-10-28 audit in `outputs/paper_eval/ext_jc_visual_refinement/FINAL_MATH_AUDIT.md` checked them against the frozen trajectories. It is complete. It did not find that the soft-min indicator equals the evaluator's contact predicate.

Preset J4 (`src/core4d_dual_retarget/retarget/paper_presets.py`, `paper_join_config("J4")`) is the frozen v41 component stack: Adam, learning rate 0.01, 600 iterations, global gradient clip 5, best-loss iterate. `joint_contact_continuity_weight` defaults to 0, so J4 alone is not EXT-JC.

## 1. Interaction phase and contact extraction

Source fingertips are transformed into the object frame and compared with the object mesh.

| Piece | Code |
|---|---|
| Fingertip distances | `contact.py` `fingertips_in_object_space`, `fingertip_distances_to_object` |
| Hysteresis | `hysteresis_contact`, default enter 0.05 m, exit 0.07 m |
| Six phase labels | `join_phase_labels`, `analyze_join_sequence` |
| CLI | `core4d-retarget analyze-join 20231108/018` writes `outputs/contact_analysis/20231108_018.npz` |
| Joint-carry mask used by `L_JC` | `coupled.py` lines 536–541: AND of the two persons' per-hand source masks from `_contact_anchors` |

Those source masks are inputs. They are not the robot evaluator's predicted contact.

## 2. Human to G1 retargeting

| Method | Entry | Objective |
|---|---|---|
| A, Independent IK | `retarget/independent.py` `retarget_independent`; CLI `retarget-independent` | Per-person keypoint IK. J0 of `retarget-paper-join` only copies `outputs/independent`. |
| B–F | `retarget/coupled.py` `retarget_coupled`; CLI `retarget-paper-join --variant J4` | Pose and root fidelity, contact, velocity, acceleration, jerk, plus the geometry stack in section 4. |

URDF: `third_party/unitree_ros/robots/g1_description/g1_29dof_rev_1_0.urdf`. Forward kinematics: `pytorch-kinematics`.

Saved `robotN.npz` fields: `joint_names` (29), `joint_positions` (T, 29), `root_translation` (T, 3), `root_yaw` (T,), `robot_keypoints` (T, 14, 3), `target_keypoints`, `contact_anchors` (T, 2, 3), `contact_mask` (T, 2), `config_json`, `loss_history` (600,).

## 3. Object-relative contact anchors

`coupled.py` `_contact_anchors` (lines 254–312).

Local anchor: nearest object vertex to the closest fingertip, then `palm_offset_m = 0.052` along the outward direction. `grasp_up_frac = 0.4` clamps that point along object-up and snaps it back to the mesh. `anchor_smoothing_sigma = 1.5` smooths the local anchor.

World target, confirmed by reconstructing the stored anchors to 0 error:

```text
a_core = R_obj @ a_local + t_obj
a_robot = C @ a_core
```

`C` is `CORE4D_TO_ROBOT` in `coordinates.py` (y-up to z-up: `(x, y, z) -> (x, -z, y)`).

The residual J4 / EXT-JC actually optimizes, because `palmar_contact` is true (`coupled.py` lines 1138–1141):

```text
r = p_palmar - a_robot
loss = sum ||r||^2 * contact_weight * phase_weight / sum(weights)
```

`p_palmar` is one point per hand: rubber-hand origin plus `(0.05, -0.03, 0)` left and `(0.05, +0.03, 0)` right (`balance.py` lines 673–674). It is not the wrist keypoint. The 512-sample soft-min residual is the other branch and is not taken.

Recommended manuscript formula:

```latex
\mathbf{a}^{w}=R_{\mathrm{obj}}\mathbf{a}+\mathbf{t}_{\mathrm{obj}},
\qquad
\mathbf{r}=\mathbf{p}_{\mathrm{palmar}}-\mathbf{a}^{w}
```

with `R` and `t` in the same frame as `p`. The expression `p - R a + t` equals this residual plus `2 t`.

## 4. Geometry and support constraints

All of these are inside `retarget_coupled` and are configured by `_geometry_gait()` plus the J4 extras in `paper_presets.py`. They are per-robot. The separability audit measured cross Hessian-vector products of 0 for every one of them.

| Term | Weight in J4 | Role |
|---|---:|---|
| `hand_volume_clearance` | 1200 | Rubber-hand samples outside the object hull, margin 8 mm |
| `object_hull_clearance` | 800 | Same hull, margin 8 mm, slack 0 |
| `arm_hull` | 120 | Arm links versus the hull |
| `leg_hull` / `leg_volume` | 1200 / 400 | Legs versus the object |
| `foot_volume`, `foot_ground`, `foot_slide`, `hard_plant` | 250–1200 | Feet on the ground, planted when the source foot is slow |
| `com_support`, `com_center`, `torso_upright` | 40 / 20 / 18 | Support and upright torso |
| `swing_lift`, `gait_knee_*`, `leg_laterality`, `self_leg_clearance` | various | Gait shape |
| `palm_facing`, `finger_up` | 50 / 60 | Palm orientation |
| `inter_robot` | 8, clearance 0.12 m | Partner keypoint clearance. Measured inactive: minimum distance on the logged runs is at least 0.40 m, so this term is identically zero on the paper trajectories. It does not make v41 coupled. |
| `source_approach_root` | 240 | J4 only. Approach root toward the source pelvis. |

Hand samples: 512 farthest-point samples on each official rubber-hand STL (`balance.py` `_rubber_hand_mesh_offsets`). Both hands are concatenated when the soft-contact indicator runs (N = 1024).

## 5. Cross-agent joint-contact continuity (E and F)

Config fields, default off: `joint_contact_continuity_weight`, `joint_contact_tau_m = 0.04`, `joint_contact_beta_m = 0.01`, `joint_contact_separable_surrogate = false`. Frozen EXT-JC sets only the weight to 10. Surface-aware mode, edge taper, hand jerk, trust region, and the smoothed-hand barrier stay at 0 / false. They exist in the file from a later sweep and are **not** part of E or F.

Implementation: `torch_soft_contact_indicator` (`coupled.py` lines 315–332) and the loss at lines 1786–1817.

```text
d_n = max_j (a_j · x_n + b_j)          # negative inside the convex hull
d   = -β logsumexp_n(-d_n / β)
s   = sigmoid((τ - d) / β)
L_JC = sum_t E_J(t) * (1 - s1(t) * s2(t)) / sum_t E_J(t)
```

`E_J` is the hard source joint-carry mask (`joint_contact_edge_taper_frames = 0`).

Cross-agent gradient:

```text
∂L_JC/∂θ1 = -sum_t E_J(t) * s2(t) * ∂s1/∂θ1 / sum_t E_J
```

`s2` depends on robot 2, so the mixed Hessian is nonzero when both indicators have slope. Measured at the v41 solution of 018 with weight 10 (`cross_gradient_verification.md`): cross Hessian-vector product 1.27, finite-difference change of the robot-1 gradient 0.047. With the weight at 0 the same probe is separable to 1e-15 and the joint loss matches `2.789216976690139`.

## 6. Separable contact surrogate (D)

Same function, same weight, `joint_contact_separable_surrogate = true`. Lines 1812–1813 replace the product by `(s1 + s2) / 2`.

```text
L_sep = sum_t E_J(t) * (1 - (s1 + s2) / 2) / sum_t E_J
```

The cross Hessian of this term is 0. A comment at `coupled.py` lines 213–216 says the surrogate replaces the partner by the constant 1. That comment does not match the code. The paper must describe the average, not "partner replaced by 1".

## 7. Staged initialization

| Run | Initial `baseline_root` |
|---|---|
| Frozen v41, 018 | Historical continuation from `outputs/balanced_clearance_v40_smooth`. Not regenerated by `retarget-paper-join` from independent IK. |
| C, D, E on 018 | `--baseline-root outputs/balanced_clearance_v40_smooth` |
| C, D, E on 019, 020, 108, 109, 110 | `--baseline-root outputs/balanced_clearance_v20_smooth` |
| J0–J4 ablation on 018 | `outputs/independent` (the CLI default) |

OA-Independent is two processes, `--solve-robot 1` and `--solve-robot 2`, same objective, same init, same 600 iterations. There is no random seed in the optimizer. The separability probe uses seed 0 for its finite-difference directions only.

## 8. Post-processing

Shared recipe, `smooth-trajectories --clearance-recipe` (`cli.py` `smooth_trajectories_command`, `smooth.py` `resolve_smooth_sigmas`, `smooth_robot_npz`, `lift_root_for_sole_clearance`):

- Joint Gaussian σ = 1.0
- Arm joints (name contains shoulder, elbow, or wrist) σ = 0.35
- Root σ = 0
- Sole lift so ankle-roll samples are not underground

This recipe, not the raw EXT-JC solve, creates the 3–4 cm hand penetration on 020. Diagnosis: `outputs/paper_eval/ext_jc_visual_refinement/postprocess/DECISION.md`.

Repair, method F only, `scripts/ext_jc_postprocess.py`: on each frame, blend waist and arm joints from the smoothed pose toward the raw pose by the smallest factor on a 21-point grid in [0, 1] that brings convex-hull depth to ≤ 0.05 cm, then a triangular ramp of 3 frames. Legs and the lifted root stay. The same script is run on the surrogate. 109 robot 1 still reaches 0.57 cm after a full blend back to the raw arms. The rule was not changed to remove that.

## 9. Geometrically valid simultaneous contact

Not part of the loss.

| Evaluator | Command | Definition |
|---|---|---|
| Paper contact and geometry | `evaluate-paper-join` → `metrics.evaluate_independent_sequence` with `paper_metric_mode=True` | Rubber-hand sample distance to object vertices. Hysteresis enter 0.04 m, exit 0.055 m. 15 fps. |
| Coordination spans | `coordination_eval.py`, also `scripts/oa_run_evaluation.py` and `scripts/ext_jc_evaluate.py` | A frame counts toward the valid joint span only when both robots' predicted masks are on and hand penetration is ≤ 0.5 cm. `longest_valid_joint_run_frames` is that measured run. |

## Soft-min issues the manuscript still has to acknowledge

These were measured. They are not open questions about what the code does.

1. Sample-count bias. On the frozen smoothed poses of 018, 020, 109, and 110 the soft-min is 3.45–3.59 cm below the true minimum of the plane distances. Halving N, without moving the robot, drops the bias by about 0.6 cm, close to `β ln 2`.
2. `τ = 0.04` is applied to the biased soft-min. `s = 0.5` at about 7 cm of true hull clearance (observed crossings 6.98, 7.09, 7.21 cm).
3. Agreement of `s > 0.5` with the evaluator mask is 88.9–97.5% of frames, and every mismatch is soft-score on, evaluator off. When the evaluator is on, `s` is already about 0.99.
4. Inside the hull, `s` still increases (0.982 on the surface, 0.9997 at 4 cm inside). The frozen trajectories are almost never inside before smoothing. The surface-aware variant that would flatten this reward was not used for E or F.

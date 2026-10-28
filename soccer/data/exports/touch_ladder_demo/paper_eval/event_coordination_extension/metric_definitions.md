# Metric definitions — event-coordination extension

All metrics are **kinematic event-consistency metrics** computed on retargeted G1 keypoints / meshes and a prescribed,
event- and scene-conditioned ball path. They are not physical soccer task success rates, not contact forces, and not
evidence of physically generated ball motion. Sample size: one 144-frame, 24 fps sequence; 6 receive events, 7 kick
events, 6 kick→receive transfer pairs. The 13 events are dependent samples from that single sequence.

Coordinates: robot frame (Z-up). Toe keypoint = `*_ankle_roll_link` origin + local offset `[0.12, 0, −0.03]`.
Ball radius r = 0.11 m. fps = 24.

## Per-event metrics (`event_metrics.csv`, 13 rows per variant)
| Column | Definition |
|---|---|
| `phase` | `receive` (annotated `frame`,`side`) or `kick` (annotated `kick_frame`,`kick_side`); the first contact has no receive. |
| `ball_center_to_target_toe_m` | ‖ball(f) − toe_target(f)‖ at the annotated frame f. |
| `ball_center_to_opposite_toe_m` | same for the other foot of the same robot. |
| `target_toe_surface_deviation_signed_m` | d_target − r. Negative = toe keypoint inside the ball sphere. Smaller |·| is better; smaller *distance* alone is **not**. |
| `toe_center_inside_ball` | surface deviation < 0. |
| `correct_foot_association` | d_target < d_opposite (strict, same robot). Existing rule, kept for comparability. |
| `nearest_of_four_feet` / `actor_assignment_correct` / `foot_assignment_correct_4way` | the nearest of the four toes {A‑L, A‑R, B‑L, B‑R} at frame f; actor correct if its robot is the annotated player; 4‑way correct if robot **and** foot match. |
| `other_agent_min_toe_distance_m`, `actor_margin_m` | nearest toe of the *other* robot; margin = that − nearest toe of the annotated robot (>0 = annotated actor is closer). |
| `predicted_frame`, `timing_error_frames` | earliest frame with minimum d_target in [f−4, f+4] (existing rule); error = |predicted − f|. |
| `mesh_target_foot_gap_signed_m` | min over the official G1 `*_ankle_roll_link` foot-mesh vertices (MuJoCo, posed) of ‖v − ball‖ − r. Negative = mesh vertex inside the ball (vertex-sampled proxy, not exact triangle distance). |
| `mesh_opposite_foot_gap_signed_m`, `mesh_nearest_of_four_feet`, `mesh_foot_assignment_correct_4way`, `mesh_correct_foot_association` | same assignments using mesh gaps. |
| `mesh_target_foot_ground_depth_m` | max depth of that foot mesh below Z = 0 at frame f. |
| `center_gate_pass_tauX`, `mesh_gate_pass_tauX` | |surface deviation| ≤ τ, resp. |mesh gap| ≤ τ, for τ ∈ {0.03, 0.05, 0.08} m (predefined; primary τ = 0.05 m). |

## Transfer-pair metrics (`transfer_chain_metrics.csv`, 6 rows per variant)
Pair *i* = annotated kick of contact *i* → annotated receive of contact *i+1* (all pairs are cross-agent).

| Column | Definition |
|---|---|
| `sender_assigned` / `receiver_assigned` | 4‑way nearest toe at the kick / receive frame belongs to the annotated sender / receiver robot. |
| `sender_foot_matched` / `receiver_foot_matched` | 4‑way nearest toe equals the annotated (robot, foot) at the kick / receive frame. |
| `order_ok` | predicted kick frame < predicted receive frame. |
| `role_chain_consistent` | **foot-role-only chain consistency** = all four assignment checks ∧ `order_ok`. |
| `{center,mesh}_geometry_ok_tauX` | gate passes at **both** endpoints. |
| `{center,mesh}_gated_chain_consistent_tauX` | **geometry-gated chain consistency** = `role_chain_consistent` ∧ geometry ok. |

## Per-variant summary (`event_chain_summary.csv`, `fixed_ball_ablation.csv`)
Rates are means over 6 receive / 7 kick / 13 events / 6 pairs. Additional columns:

| Column | Definition |
|---|---|
| `support_foot_displacement_mean_m` | cumulative horizontal displacement of the toe opposite the annotated kick foot over kick−8..kick+2 (existing definition). |
| `keypoint_floor_penetration_max_m` | max depth of ankle/toe keypoints below 0.012 m (existing proxy). |
| `mesh_floor_penetration_max_m` | max depth of any of the four foot meshes below Z = 0 over all frames. |
| `root_jerk_rms_m_s3`, `joint_jerk_rms_rad_s3` | RMS of third forward difference × fps³ (root: XYZ magnitude; joints: pooled 29 DoF), both robots pooled. |
| `*_BALL_PROPERTY` | lane-crossing error / cone clearance of the ball path. These describe the prescribed object path, **not** robot behaviour; identical by construction across fixed-ball variants. |
| `*_MOTION_FIDELITY` | supplement, not an event-consistency metric: RMSE between robot and reference (`target_keypoints`) kick-foot toe velocity over kick−8..kick+2, and mean peak toe speed (robot vs reference). |
| `threshold_sensitivity.csv` | event-level gate pass rate and geometry-gated chain consistency for every (gate, τ). |

## Threshold choice
τ values were fixed before running: 0.03 m (tight, below the mesh-vertex sampling error), 0.05 m (primary; ≈ half ball
radius, comparable to the 0.075–0.095 m snapping offsets minus the toe offset), 0.08 m (loose). Results are reported for
all three; no threshold was selected after seeing the numbers.

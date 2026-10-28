# Soccer method ↔ code map

Scientific boundary: two independent IK problems, then a ball post-process. No EXT-JC coupling loss. No shared optimization variable.

| Stage | What happens | Code | Frozen / config artifact |
|---|---|---|---|
| 1. Original video | Local 7MLC / MZC CONTROLLO WeChat forward, 203.0–209.0 s. Original file is 960×544, 24 fps. **Not redistributable.** | `scripts/prepare_touch_ladder_demo.py` (`SOURCE_VIDEO`, `CLIP_START_S=203`, `CLIP_END_S=209`) | Private path `$SOCCER_SOURCE_VIDEO`. Packed local clip `data/clips/touch_ladder_demo_source.mp4` (SHA-256 `40b7e859…d601`, 144 frames, 24 fps). Metadata `data/clips/touch_ladder_demo_prep.json`. |
| 2. Human-motion recovery | YOLOv8x person boxes, 1.8× crops, other person masked, two **independent** official GVHMR static-camera runs. | `prepare_touch_ladder_demo.py`; `scripts/run_gvhmr_pair.py` → `GVHMR/tools/demo/demo.py -s` | Crops `data/clips/touch_ladder_demo_player_{a,b}.mp4`. GVHMR outputs `data/gvhmr/touch_ladder_demo/player_{a,b}/…/hmr4d_results.pt`. Weights live under `$GVHMR_ROOT`, not in this repo. |
| 3. Preprocessing and robot adaptation | SMPL-X FK, feet to y=0, A faces +X, B faces −X, nominal 2.8 m gap. This is geometric placement of two monocular tracks, not a metric shared-world reconstruction. | `scripts/export_smplx_joints.py`; `scripts/pack_touch_ladder_demo.py` (`LADDER_GAP_M = 2.8`) | `data/exports/touch_ladder_demo/smplx_joints.npz` SHA-256 `f03a8c5b…3b4b`. |
| 4. Interaction annotations | Ball UV from YOLO; cones from HSV; alternating-contact DP proposes 7 contacts. Final receive/kick feet and frames were corrected against the video and then **frozen**. The solver does not discover them. | `scripts/recover_ball_cones.py` (`detect_contacts`); hand notes inside JSON | `data/exports/touch_ladder_demo/ball_cones.json` SHA-256 `cad51d64…2156`. |
| 5. Actor roles and active feet | Each contact: `player`, `side` (receive foot), `kick_side`, `kick_frame`. First contact is a kick only (clip starts on the kick). Transfers are kick *i* → receive *i+1*, all cross-agent. | JSON schema; `evaluate_event_chain.py::annotated_events`, `transfer_pairs` | Same JSON. 6 receive + 7 kick = 13 events. 6 pairs. |
| 6. Event-conditioned independent IK | Per robot: Adam, lr 0.04, 340 iterations, float64. Event windows boost leg keypoints, relax smoothness, add kick-velocity term (weight 24), and plant the opposite foot. Loss never reads the other robot or the ball. | `scripts/retarget_dual_g1_soccer.py` `contact_windows`, `SoccerRetargetWeights`, `retarget_soccer` | Accepted joints: `g1/robot_a.npz` `ef5277f2…d96b` (scale 0.9470989763); `g1/robot_b.npz` `eb5646a3…568b` (scale 0.9246479386). `g1/scales.json`. Config stored inside each NPZ (`iterations=340`). |
| 7. Root and stationing | Pre-IK `_ready_shift` / `_apply_player_plan` place the annotated striking toe on the lane and ground the event-frame foot. Post-IK `align_second_pass_stations` adds world-Y offsets from each robot’s own toes (A at f0, B at f15/f21). Still per-robot. **The stationing code has changed since 2026-10-01; re-running it does not regenerate these NPZ.** | `station_pair_on_ladder`, `align_second_pass_stations` | Offsets are already baked into the frozen `root_translation` and `robot_keypoints`. |
| 8. Post-IK ball synthesis | `snap_ball_to_g1_feet` reads both robots’ feet at annotated frames, offsets the ball 0.095 m (kick) / 0.075 m (receive), and rolls straight through lanes `[3,4,5,4,3,2]`. Method tag `straight_gap_pass`. Robots do not see the ball. Cone spacing 0.50 m is a scene convention, not a calibrated measurement. | `snap_ball_to_g1_feet`, `ladder_geometry` | `ball_cones.npz` SHA-256 `b608f0a9…f47f`. Radius 0.11 m. Re-snapping the frozen robots reproduces this file exactly; do not rewrite it. |
| 9. Rendering | Dual G1 from `g1_29dof_rev_1_0.xml`, collision geoms hidden, ball is a mocap body, five cones. EGL, 960×720, 24 fps. Camera: free, look-at mean root, distance 4.4, azimuth 75, elevation −24. | `scripts/render_g1_soccer.py` | Accepted video SHA-256 `c904d2ed…039e` at `comparison/current_hop_timed_lateral.mp4` (same bytes as `g1_ball_cones_preview.mp4` and `paper_eval/videos/full.mp4`). Scene XML `g1/soccer_scene.xml` is generated at render time. |
| 10. Evaluation | Kinematic proxies only. | `evaluate_ablation_experiments.py`, `evaluate_soccer_paper.py`, `evaluate_event_chain.py` | `paper_eval/summary.csv`, `receive_kick_summary.csv`, `event_coordination_extension/`, `object_aware_comparison/`. |

## What is derived vs specified

| From the video (estimated, not mocap) | Explicitly specified or synthesized |
|---|---|
| Two monocular GVHMR body motions | 2.8 m gap, facing, ground alignment |
| Image-space ball detections (129/144 frames) and foot tracks | Cone spacing 0.50 m and lane order |
| DP contact *proposal* | Final receive/kick feet and frames in `ball_cones.json` |
| | IK targets, event weights, support windows |
| | Entire ball path after IK |

## IK terms that are event-conditioned (still per robot)

`SoccerRetargetWeights` defaults: `support_plant=6`, `support_floor=6`, `kick_velocity=24`, `event_keypoint_boost=True`, `event_smooth_relaxation=True`. Shared terms that stay on in ablations: keypoint tracking, root regularization 0.35, yaw, hip-yaw, pelvis bounce, upper-body velocity, chest and shoulder alignment.

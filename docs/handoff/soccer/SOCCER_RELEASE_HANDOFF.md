# Soccer release handoff

Paper: **Interaction-Structured Motion Retargeting for Multi-Party Humanoid–Object Interactions**.

Audience: the integration Agent that will merge this Soccer case study with CORE4D Join / EXT-JC. **Do not publish from this folder. Do not rerun or overwrite the accepted Full trajectories, event annotations, or ball path.**

Soccer Full is **not** Join EXT-JC. It has no coupled two-robot IK objective and no cross-agent contact loss. The two G1 solvers are separable. The ball is synthesized after IK.

## Read these files in order

| File | Use |
|---|---|
| `METHOD_CODE_MAP.md` | Pipeline stages → scripts and frozen outputs |
| `FINAL_ARTIFACT_MANIFEST.csv` | Paths, roles, SHA-256 |
| `PAPER_RESULT_MAP.csv` | Which claim each number supports |
| `DEPENDENCIES.md` | External code, weights, paths |
| `THIRD_PARTY_ASSETS_SOCCER.md` | Licenses; no Sketchfab field |
| `REPRODUCE_SOCCER.md` | Levels 1–4 |
| `RELEASE_INCLUDE_EXCLUDE.csv` | What may enter the public repo |
| `SOCCER_DEMO_MANIFEST.md` | Website copy |
| `SOCCER_SMOKE_TESTS.md` | Checks that do not rerun IK |
| `KNOWN_LIMITATIONS.md` | Claim boundary |
| `public_export/` | Sanitized display copies only |

## Frozen Full, in one paragraph

One 144-frame, 24 fps monocular clip (original 203–209 s) is split into two independent GVHMR tracks, geometrically placed 2.8 m apart, and annotated with 7 contacts → 6 receives + 7 kicks and 6 kick→receive transfers. Each G1 is retargeted by its own Adam IK (340 iterations, lr 0.04). Stationing and event windows use the annotation. After both robots are solved, `snap_ball_to_g1_feet` writes the ball. The accepted motion is the 2026-10-01 NPZ pair plus that ball, rendered at 960×720, camera distance 4.4, azimuth 75, elevation −24. Current scripts do **not** bit-reproduce those joints; replay the frozen NPZ.

## Answers the integration Agent needs

1. **Pipeline.** Video → YOLO crops → two GVHMR runs → SMPL-X joints → supplied/corrected events → independent event-conditioned IK + stationing → post-IK ball → MuJoCo render → kinematic evaluation. See `METHOD_CODE_MAP.md`.
2. **Best demo files.** `data/exports/touch_ladder_demo/comparison/current_hop_timed_lateral.mp4` (SHA-256 `c904d2ed…039e`), identical to `g1_ball_cones_preview.mp4` and `paper_eval/videos/full.mp4`. Driven by `g1/robot_a.npz`, `g1/robot_b.npz`, `ball_cones.npz`.
3. **To recreate event-conditioned IK and the ball** you need `smplx_joints.npz`, `ball_cones.json`, the G1 URDF, and `scripts/retarget_dual_g1_soccer.py`. Running it **overwrites** outputs and will not match the frozen joints. Treat frozen NPZ as the paper result; treat the script as the method, not as a bit-exact regenerator.
4. **Minimum public example.** Replay Level 1: frozen `robot_a.npz`, `robot_b.npz`, `ball_cones.npz` + G1 MJCF + `render_g1_soccer.py`. No source video.
5. **Without the source video.** Replay, re-evaluate metrics, and show robot-only figures and the timeline. Cannot show source-to-robot frames or rerun GVHMR.
6. **Redistributable.** Our code, tests, docs, frozen robot/ball NPZ, robot-only renders and figures, metric CSV/JSON. **Not** the human source video, GVHMR checkpoints, SMPL-X body models, or raw candidate videos under `data/raw/`.
7. **Supported claims.** Interaction-state *specification* compiled into per-robot targets and a post-IK ball; 13/13 limb assignment and 6/6 role chains on this one sequence; event IK improves kick-foot velocity fidelity and increases jerk; independent object-aware baseline matches chain metrics. **Not** supported: coupled multi-agent optimization, automatic event discovery, physics, mocap, real-robot execution, multi-sequence statistics.
8. **Copy.** Everything in `RELEASE_INCLUDE_EXCLUDE.csv` marked `include`, plus `public_export/` for the site. Point CORE4D/G1/GVHMR at external checkouts; do not vendor private paths.
9. **Blocked.** Source-video license (not granted). GVHMR non-commercial research license and its weights. SMPL-X license. YOLO weights. G1 meshes must ship with the Unitree upstream license (local checkout had no LICENSE file at the inspected path). No Sketchfab stadium exists to attribute or include.

Absolute handoff directory: `$SOCCER_ROOT/release_handoff/soccer/`.

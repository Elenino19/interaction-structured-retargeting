# Method: object-mediated dual-G1 retargeting

Kinematic `join + box` freeze (2026-09-14): CLI `--clearance --omni-geom`
(v41). Artifacts and split: `docs/progress.md` Status. Older gated
release remains `outputs/balanced_smooth` (`docs/paper_table.md`).

## 1. Inputs and normalization

For each CORE4D sequence, the pipeline merges:

- CORE4D-Real V1: object metadata, object pose trajectory, frame alignment.
- CORE4D-Real V2: corrected SMPL-X joints, vertices, and pose parameters.
- Official Unitree G1 `g1_29dof_rev_1_0` URDF: kinematic tree and limits.

CORE4D's y-up frame is converted to the G1 z-up frame by the proper rotation

```text
[x_r, y_r, z_r] = [x_c, -z_c, y_c].
```

No global rescaling is applied to the shared object. Human root trajectories
remain in scene scale; human-relative skeletal targets are scaled by the ratio
between G1 and human median leg lengths.

## 2. Contact topology and phases

Ten SMPL-X fingertip vertices are transformed into the canonical object frame.
Nearest object-mesh distances are converted into contact signals with 5 cm
enter and 7 cm exit hysteresis thresholds.

The participant with the earlier sustained contact is inferred as the
incumbent; the participant with the later contact is the joiner. This avoids
assuming that CORE4D's person indices always encode fixed interaction roles.

Full clips receive six labels:

1. `pre_interaction`
2. `single_carry`
3. `approach`
4. `contact_transition`
5. `joint_carry`
6. `post_interaction`

The middle four labels are the core `join` action. The outer labels prevent
waiting and release frames from contaminating the collaboration metrics.

## 3. Independent baseline

Each human is independently fitted to a 29-DoF G1 trajectory. Joint limits are
enforced by a bounded `tanh` parameterization. The loss contains:

- semantic keypoint fidelity, with larger wrist and elbow weights;
- root-trajectory regularization;
- joint velocity regularization;
- joint acceleration regularization.

This baseline has no shared-object or partner awareness.

## 4. Coupled refinement

For each active human hand, the nearest source fingertip contact is projected
onto the object mesh. A 3 cm offset toward the source fingertip defines a G1
palm anchor outside the object surface.

Both G1 trajectories are then optimized together with:

- fidelity to the independent trajectories;
- phase-weighted palm-to-object-anchor losses;
- root fidelity;
- velocity, acceleration, and jerk regularization;
- a lightweight inter-robot keypoint-clearance penalty.

The object trajectory remains fixed in this MVP, making differences against
the independent baseline directly attributable to robot trajectory coupling.

Optional `--balance` refinement adds:

- mass-weighted CoM inside the stance support (double- or single-support);
- contact-aware ankle height and stance-foot slip;
- torso z-axis upright after root yaw.

A final Gaussian σ=1.4 is applied to joints, root translation, and yaw, then
keypoints are recomputed with FK. This recovers jerk below the coupled
baseline without leaving the quality gate.

## 5. Current limitations

- Palm contact uses link origins rather than full G1 hand collision geometry.
- Object penetration and robot self-collision need mesh/SDF metrics.
- Phase `approach` currently uses a fixed temporal look-back from join onset.
- Free-base joint-PD still falls without a locomotion/balance controller.
  Balance-aware kinematics pass a CoM/foot quality gate (6/6 clips) but are
  not dynamically executable; pinned-root PD saturates G1 torque limits.
- Inverse-dynamics **median** joint torque is ~1.6 Nm with both robots posed.
  p95 is ~1.2×10³ Nm. Two-foot LS contact forces are kN-scale and mostly
  tensile; forbidding tension leaves a large residual floating-base wrench.
  Do not treat unconstrained ID peaks as an executability metric. Earlier
  10⁴–10⁵ Nm peaks were inflated when the partner robot was left at the
  default pose.

# Claims that are currently supported

Scope: six CORE4D-Real `join + box + obs0` clips retargeted to two Unitree G1 29-DoF robots. Object pose is held fixed.

**2026-09-14 freeze:** the accepted *visual* kinematic recipe is v41
(`outputs/balanced_clearance_v41_smooth`, videos `*_clearance_v41.mp4`).
Numbers below for Contact F1 0.98 / gate 6/6 refer to the older
`outputs/balanced_smooth` release, not v41 (v41 mean F1 ≈ 0.83; 108 join
and 109/110 foot hull remain open). Do not cite both as one dataset.

## Supported

1. **Object-mediated coupling improves contact over independent IK.**
   Mean Contact F1 0.868 → 0.980; palm distance 4.72 cm → 2.13 cm; onset error 3.5 → 1.0 frames.
2. **Balance + temporal smoothing is the dataset default.**
   After CoM/foot constraints, Gaussian σ=1.4 on joints/root (FK refreshed) cuts jerk 23.1 → 9.74, below coupled (14.3), with Contact F1 0.984 and 6/6 quality-gate pass. See `docs/paper_table.md`.
3. **The quality split does not leak frames across train/val/test.**
   Train `018/019/108`, val `020`, test `109/110`.
4. **Naive free-base joint-PD is not a controller.**
   Robots fall (~95% frames). Pinned-root PD still saturates G1 torque limits (~80%) and does not retain 8 cm palm contact.
5. **Gravity compensation, Jacobian CoM PD, time-scaling, and ID upsampling do not produce standing.**
   On clip 018, first-frame hold still falls at ~0.40–0.53 s (pelvis from 0.80 m to ~0.06 m) with gravity-only bias, Jacobian CoM wrench (k_p=2500–4000), leg k_p=80–800, 1 ms stepping, and higher floor friction. Slow-motion playback (time_scale=2) also falls.
6. **Two-foot contact ID does not make the motion a stance equilibrium.**
   On all six box clips (12 robots, q̈=0, both robots posed): unconstrained median |τ| 1.61 Nm, p95 1211 Nm, 17.7% limit violations. Least-squares 3D foot forces average 3650 N with **62% tensile** normals. Clipping tension leaves base-wrench RMS 4320 vs 128. Joint violations do not improve (17.7% → 19.9%). CLI: `contact-id`.

## Not supported yet

- Dynamic executability, walking balance, or sim-to-real.
- Inverse-dynamics numbers are diagnostic only. Median joint torque is small;
  unconstrained peaks remain huge even with zero acceleration.
- Other CORE4D actions or heterogeneous robots. Six holdout clips (barrel,
  board, desk; two each) pass the same gate and are not mixed into the box split.
- Object-trajectory correction.

## Recommended paper framing

Primary contribution: **object-mediated, phase-aware, balance-constrained dual-humanoid retargeting** that preserves join contact topology. Physics is a **negative result / limitation**, not a success claim. Downstream: residual RL or WBC on this kinematic dataset.

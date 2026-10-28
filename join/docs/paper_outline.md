# Paper outline (kinematic dual-G1 join retargeting)

Status: outline only. Numbers must match `docs/paper_claims.md`.

## Title direction

Object-mediated retargeting of two-person join actions onto dual Unitree G1 robots, with balance-aware kinematics and a documented physics limitation.

## Story

1. CORE4D has two humans plus a shared object; most retargeting papers fit one avatar.
2. Independent IK loses join contact when the object is the coupling.
3. Coupling through object-frame palm anchors and join phases restores contact.
4. CoM/foot terms make the motion kinematically standable (quality gate).
5. Smoothing recovers jerk without breaking the gate.
6. The same stack transfers to barrel/board/desk holdouts (not mixed into the box split).
7. Free-base PD, gravity+CoM Jacobian, and unconstrained ID **fail**. Execution is future work on `outputs/residual_obs`.

## Suggested sections

- Intro: dual-humanoid collaboration data; CORE4D join.
- Related: human-to-humanoid retargeting; object-centric HOI; WBC/RL tracking (cite as downstream, not this paper’s result).
- Method: coordinates, phases, independent, coupled, balance, smooth.
- Dataset: 6 box clips + split; 6 holdout objects; quality gate.
- Experiments: table in `docs/paper_table.md`; ablations already in `outputs/ablation` if used.
- Limitation: Stage 6 controller probe.
- Conclusion: kinematic dataset + negative physics.

## Do not claim

Sim-to-real, walking, successful PD tracking, or that ID peaks measure feasibility.

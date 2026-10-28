# Known limitations

## Algorithm

- Soccer Full is two separable per-robot IK problems. Perturbing robot B does not change robot A’s stationing, windows, IK, or post-alignment (`object_aware_comparison/separability_probe.json`). There is no EXT-JC coupling term to merge with Join.
- The ball is a closed-form function of both robots’ feet, the annotation, and the lane schedule. It is applied after IK. Robots never receive a ball residual in the accepted loss.
- An independent solver with an explicit toe–sphere term on the same frozen ball matches every actor/foot/chain rate and beats Full on surface deviation. A “coordination benefit over independent object-aware retargeting” is not supported.
- Event-conditioned IK changes motion fidelity (kick-foot speed) and smoothness (jerk, foot–ball penetration). It does not change 13/13 or 6/6 when the ball and reference are fixed.
- 13/13 and 6/6 on Full are largely guaranteed by placing the ball on the annotated feet. Say “annotation-consistent kinematic chain,” not “task success.”
- Mesh vertices of the G1 foot enter the 0.11 m ball at most Full events (signed mesh gap about −0.04 m). Do not treat centre distance as a contact quality score by itself.
- Current `retarget_dual_g1_soccer.py` does not bit-reproduce the accepted joints. The paper motion is the frozen NPZ, not a fresh 340-iteration run.
- Cone spacing 0.50 m and the 2.8 m player gap are scene choices. They are not calibrated metric reconstructions.
- One sequence, 9 distinct event frames. No population variance.

## Data and licensing

- The human video cannot be redistributed. Source-to-robot figures stay in `local_only/`.
- GVHMR is research/non-profit. Its checkpoints and SMPL-X are not in this license grant.
- No Sketchfab or stadium asset exists. Do not add one and describe it as part of the accepted experiment.
- G1 meshes are Unitree’s. A LICENSE file was not in the inspected description folder; ship upstream license text with any vendored meshes.

## What the paper may say

Soccer is an interaction-representation case study: a supplied actor/foot/time annotation is compiled into per-robot targets, support constraints, and a post-IK ball, then checked with actor, foot, order, and geometry-gated chain metrics. It is not a coupled multi-agent controller and it does not discover the event roles.

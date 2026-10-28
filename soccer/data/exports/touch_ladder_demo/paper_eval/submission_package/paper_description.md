# Paper-ready text — Soccer interaction-state instantiation

Do not claim joint cross-agent optimization. Do not claim that the solver discovers the event annotations. The accepted Full pipeline was not rerun.

## Suggested figure captions

**Fig. source-to-robot (use `local_only/fig_source_to_robot.pdf` only if source permission is obtained; otherwise use `fig_robot_events.pdf`).**
Selected frames from a 144-frame monocular passing clip (24 fps) and the frozen dual-G1 Soccer Full result at the same pipeline index. Columns: A right-foot opening kick (f0); B right-foot receive (f15); B left-foot kick after a limb-role switch (f21); ball in the annotated lane during the B→A transfer (f26); A right-foot receive (f42); A same-foot redirect (f45). Source original, packed clip, and robot preview are all 24 fps; frame \(i\) is original time \(203 + i/24\) s. No frame-rate conversion. Yellow/cyan marks on source frames are the detected ball and the annotated toe. Per-frame views are qualitative; they are not aggregate scores. The object path is synthesized after IK and is not physical foot–ball contact.

**Fig. timeline (`fig_interaction_timeline.pdf`).**
Supplied interaction-state timeline for the same clip: actor, active limb, receive/kick time, and annotated transfer direction. Receive and kick labels are frozen annotations, not outputs of the solver. RK denotes a receive and kick on the same annotated frame. One sequence; 6 receive events, 7 kick events, 6 kick→receive transfers.

**Fig. results (`fig_results_reuse.pdf`).**
Reused kinematic event-consistency numbers from the frozen evaluation. Event-conditioned IK ON vs OFF, with the ball and reference held fixed, does not change actor/limb/chain assignment. It improves kick-foot velocity tracking and increases jerk. An independent object-aware solver with no cross-robot communication matches the chain metrics and improves foot–ball surface error.

## How the interaction-state abstraction is instantiated

Soccer compiles a frozen interaction annotation \(\mathcal{E}\) into two per-robot retargets and a post-hoc object path. Each contact in \(\mathcal{E}\) names an actor \(\in\{A,B\}\), a receive frame/foot, and a kick frame/foot. Consecutive kick→receive pairs define the six annotated transfers (all cross-agent). A shared lane schedule is derived from those event times and the cone geometry. Each robot is stationed independently onto that schedule: the annotated striking toe is placed on the live pass line, the event-frame foot is grounded, and event windows modulate keypoint weights, smoothness, kick-velocity tracking, and the support-foot plant mask. The two G1 robots are then solved **independently** (separable objectives \(L_A+L_B\); no shared decision variable, no gradient from one robot to the other). After IK, `snap_ball_to_g1_feet` builds a prescribed ball trajectory from both robots’ feet at the annotated frames and the lane schedule. The ball is therefore event/scene-conditioned object synthesis, not a physics residual of contact.

Two mechanisms must be kept distinct in the paper:

1. **Event-conditioned IK** (per robot). Uses \(\mathcal{E}\) to set limb targets, windows, and loss weights. On a *fixed* ball it does not change 13/13 limb assignment or 6/6 transfer-chain consistency. It does improve kick-foot velocity RMSE (0.66 → 0.20 m/s) and roughly doubles root/joint jerk.
2. **Post-IK ball trajectory synthesis** (reads both robots, no feedback). This is what makes the accepted Full ball meet the annotated feet. Removing it (object path not snapped to the robots) is what previously broke kick association (4/7) and role-only chain consistency (3/6). It is a composition step, not a coupled optimizer.

## Claims supported by the existing numbers (n = 1 sequence)

- Accepted Full: 13/13 correct active-limb associations; 6/6 role-consistent annotated transfer chains; centre-gated geometry at \(\tau=0.05\) m is 6/6. Timing mean error 0.17 (receive) / 0.57 (kick) frames under the existing \(\pm 4\)-frame rule.
- Fixed-ball Event IK ON vs OFF: assignment and chain metrics unchanged; kick-velocity fidelity improves; jerk and foot–ball interpenetration worsen.
- Independent object-aware baseline (explicit toe–sphere term, no communication): 13/13 and 6/6, better surface deviation than Full (0.002 vs 0.022 m). Soccer Full is computationally two independent solvers plus post-IK ball synthesis.

## Claims that must not be made

- Joint cross-agent optimization, or any gradient coupling between the two robots.
- Automatic discovery of receive/kick labels, actor roles, or transfer order.
- Physical contact, impulse, or real-robot execution.
- Population generalization (one 144-frame clip; events are dependent samples).
- That a smaller toe–ball centre distance is always better: the toe or foot mesh may lie inside the 0.11 m sphere.

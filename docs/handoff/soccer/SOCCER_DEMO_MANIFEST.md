# Soccer website copy

Use files in `public_export/`. Do not swap in source-to-robot images.

## Task (short)

Two people pass a ball through a line of cones. We retarget that interaction onto two Unitree G1 humanoids: who receives, who kicks, which foot is active, and when the ball should cross.

## Algorithm (short)

A frozen interaction annotation assigns each touch to an actor and a foot. Each humanoid is solved by its own event-conditioned inverse kinematics. The ball path is built afterward from those feet and the cone lanes. The two solvers do not share a coupling loss.

## Demo video

`public_export/soccer_full_demo_web.mp4` (display encode of the accepted Full preview).

Label: “Frozen Soccer Full, 144 frames, 24 fps. Kinematic reference, not physical ball dynamics.”

Canonical scientific file, if the site can host the original encode: `data/exports/touch_ladder_demo/comparison/current_hop_timed_lateral.mp4`.

## Event strip

`public_export/fig_robot_events.png` — six frames: A kick, B receive, B kick (other foot), ball in the lane, A receive, A redirect.

## Timeline

`public_export/fig_interaction_timeline.png` — actor, foot, receive/kick time, transfer arrows. Caption must say the labels are supplied.

## Source comparison

Do **not** publish `local_only/fig_source_to_robot.png` or `source_robot_side_by_side.mp4`. Permission for the human video was not granted.

## Metrics to show, with definitions

From `public_export/summary.csv` and `fig_results_reuse.png`, for this one sequence:

- Active-limb association 13/13: at each annotated frame the annotated foot is closer to the ball centre than the other foot of the same robot, and (in the 4-way check) closer than either foot of the other robot.
- Transfer-chain consistency 6/6: each annotated kick is followed by the next annotated receive with the correct sender, receiver, and feet, and the predicted closest approach of the kick is before that of the receive.
- These two numbers are largely fixed once the ball is placed on the annotated feet. They are not a coupled-optimization score.
- Event IK, ball fixed: kick-foot velocity RMSE about 0.66 → 0.20 m/s, with higher jerk.
- Independent object-aware baseline: same 13/13 and 6/6, smaller surface error than Full.

## Disclaimer (required)

The output is kinematic reference motion for two humanoids plus a prescribed ball trajectory. It is not a physics simulation of football, not motion-capture ground truth, and not a real-robot execution. The interaction roles are annotations, not a result of automatic multi-agent discovery.

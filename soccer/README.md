# Soccer case study

One 144-frame sequence retargeted to two Unitree G1 robots.

The accepted result is **Frozen Full**: independent event-conditioned inverse
kinematics, then a ball path written by `snap_ball_to_g1_feet`. There is no
cross-agent coupling loss and no EXT-JC term.

The annotation has 13 interaction events: 6 receives, 7 kicks, and 6 transfer chains.

Bundled, and checked by hash:

- `data/exports/touch_ladder_demo/g1/robot_a.npz`
- `data/exports/touch_ladder_demo/g1/robot_b.npz`
- `data/exports/touch_ladder_demo/ball_cones.npz`
- `data/exports/touch_ladder_demo/ball_cones.json`
- `data/exports/touch_ladder_demo/comparison/current_hop_timed_lateral.mp4`

`smplx_joints.npz` is a derived GVHMR / SMPL-X joint input, not the frozen G1 result. Inclusion was approved by the release author only if upstream redistribution is allowed. Those terms were read and remain unresolved, so the file is **pending** and is not in this tree. Level 1 replay does not need it. Level 3 reads it from `ISR_SOCCER_JOINTS`. Provenance: `docs/smplx_joints_provenance.md`.

Scripts refuse to write inside `data/exports/touch_ladder_demo/`. Replay and
any rerun must use a new directory. The current IK script does not bit-reproduce
the frozen joints. Its default iteration count is 260; the accepted NPZ was
produced with 340 iterations on 2026-10-01, and stationing changed afterward.

The human source video, GVHMR weights, and G1 meshes are not in this tree.

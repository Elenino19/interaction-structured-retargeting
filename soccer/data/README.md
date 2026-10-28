# Soccer data included in this release

`exports/touch_ladder_demo/` contains the frozen Full robots, the frozen ball,
the frozen event annotation, the accepted robot-only preview, GVHMR joint
pseudo-labels, ablation trajectories, and metric tables.

Not included, and not cleared for redistribution:

- the human source video and player crops
- GVHMR checkpoints and `hmr4d_results.pt`
- generated MuJoCo scene XML (it embeds a local mesh directory)
- historical IK archives
- source-to-robot figures (`local_only`)

G1 meshes are obtained separately. See `THIRD_PARTY_LICENSES.md`.

# Join package

Object-mediated retargeting for two Unitree G1 robots on CORE4D `join` clips.

Run commands from this directory after the external CORE4D tree and the G1
description are available. See the repository `README.md` and `REPRODUCE.md`.

Method identities:

- **B, frozen v41** is the visual reference baseline.
- **F, EXT-JC w10 plus smoothing repair** is the final coupled method.
- **C** is the strong OA-Independent baseline.
- **D** is the separable surrogate with the same repair.
- Cold-start J4 is an ablation. It is not frozen v41 and not EXT-JC.

`docs/paper_claims.md` and `docs/paper_table.md` describe an older
`balanced_smooth` quality gate (Contact F1 about 0.98). Do not cite those
numbers as v41 or as EXT-JC. The v41 aggregate is Contact F1 mean 0.827.
The repaired EXT-JC table is `../evaluation/join/postprocess_metrics.csv`.

No project license is selected.

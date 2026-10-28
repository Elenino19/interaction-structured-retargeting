# `smplx_joints.npz` provenance and redistribution status

Status: **pending**. The file is not in this tree and is not in `PUBLIC_RELEASE_MANIFEST.csv`.

The person preparing this release approved inclusion on 2026-10-28, only if the applicable GVHMR and SMPL-X terms allow redistribution. Those terms were read. They do not clearly permit offering this file to a third party. Author approval is recorded. It is not treated as legal clearance.

## What the file is

A local copy was inspected outside this repository. It was not edited.

| Field | Value |
|---|---|
| Bytes | 391214 |
| SHA-256 | `f03a8c5bb6e3e658c92b6ac7dc3915754100c07ef5cbfb010fc149cdb0ae3b4b` |
| Keys | `player_a_joints`, `player_b_joints`, `fps`, `gap_m`, `label_type` |
| Joint arrays | `(144, 127, 3)` `float32` for each player |
| `fps` | 24.0 |
| `gap_m` | 2.8 |
| `label_type` | `gvhmr_pseudo_label_not_mocap` |

Role: derived input to Soccer inverse kinematics. It is not motion capture, not the SMPL-X model, and not the frozen G1 result (`robot_a.npz` / `robot_b.npz`). Level 1 replay of the frozen robots does not read it. Level 3 does.

## How it was produced

`soccer/scripts/export_smplx_joints.py` does the following. The weights and the source video are not in this tree.

1. Two independent GVHMR static-camera runs write `hmr4d_results.pt`. The script reads `smpl_params_global`.
2. It builds GVHMR's `make_smplx("supermotion")` model and stores `out.joints`.
3. It drops the feet to `y = 0`, turns player A toward `+X` and player B toward `-X`, and separates the tracks by 2.8 m.
4. It saves the compressed NPZ above.

The 127-wide joint axis is the model's joint output, not a 29-DoF G1 vector.

## Terms that were read

GVHMR software license, <https://github.com/zju3dv/GVHMR/blob/main/LICENSE>: use, copy, modification, and distribution of **that software** are limited to educational, research, and non-profit purposes. Modifications of that software must stay open source and non-commercial. Commercial use of the software requires permission from the GVHMR authors. This is not a grant to redistribute SMPL-X model outputs. GVHMR's install notes say the body models are obtained by registration and are not redistributed by that project.

SMPL-X model license, <https://smpl-x.is.tue.mpg.de/modellicense.html>: a single-user, non-transferable, non-commercial research license. Derivative works may be created by the licensee. The same text says the model and software may not be reproduced or made available in any form to any third party without Max-Planck's prior written permission, and that the grant shall not be copied, shared, or distributed except for one archive copy. The defined materials include animations.

SMPL-X Body under CC BY 4.0, <https://smpl-x.is.tue.mpg.de/bodylicense.html>, is a different artifact: a body mesh without shape blendshapes, shared as mesh and rig data. This NPZ was not exported under that grant. CC BY 4.0 is not applied to it.

## Why it stays out

A private GitHub repository still makes the bytes available to the host and to anyone later given access. The SMPL-X text does not contain an exception for joint coordinates, for a derived NPZ, or for a private research repository. No written permission from Max-Planck is in this tree.

Until that permission is obtained, or counsel determines that these joint coordinates fall outside the no-distribution clause, keep the file outside the proposed commit. `.gitignore` lists its release-tree path so a normal add does not pick it up.

Do not describe the file as redistributable because this pipeline wrote it.

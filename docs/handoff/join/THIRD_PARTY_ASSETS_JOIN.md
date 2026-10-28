# Third-party assets — Join

No license was selected for this project. The notes below are what is actually in the tree. They are not permission to upload.

## Own code

`src/core4d_dual_retarget/`, `scripts/`, `tests/`, `pyproject.toml`, `uv.lock`, and the docs written for this paper.

Status: **own code, license unresolved**. There is no `LICENSE` file and no git commit that would show an earlier choice. Do not stamp MIT or Apache-2.0 onto it while it imports Unitree meshes and is meaningless without CORE4D.

## CORE4D instructions repository

Path: `third_party/CORE4D-Instructions/`.

Upstream: <https://github.com/leolyliu/CORE4D-Instructions>. The README in this checkout says "This work is licensed under a CC BY 4.0 license." That sentence is about the instructions repository. It is not a license file placed on `data/core4d/`.

Status: **third-party code, CC BY 4.0 identified for the instructions repo only**. Prefer a link and a citation, not a full copy of the upstream benchmarks.

Citation, from that README:

```text
Liu, Zhang, Xing, Tang, Yang, Yi. CORE4D: A 4D Human-Object-Human Interaction
Dataset for Collaborative Object REarrangement. CVPR 2025.
```

Dataset pages linked there:

- <https://core4d.github.io/>
- <https://huggingface.co/datasets/leolyliu/CORE4D/tree/main>

## CORE4D motions and meshes

Path: `data/core4d/` (about 460 MB). Gitignored.

`person1_poses.npz` / `person2_poses.npz` contain `betas`, `expression`, `global_orient`, `transl`, `body_pose`, `left_hand_pose`, `right_hand_pose`, `joints`, and `vertices` (T, 10475, 3). Those are SMPL-X body parameters and fitted vertices. `data/README.md` says SMPL-X model files require registration and must not be redistributed. The parameter files are in the same category until someone checks the CORE4D dataset card and the SMPL-X license together.

Object meshes used by the six takes: `object_models/object_models/box/box024_m.obj` (018–020) and `box001_m.obj` (108–110). The rest of the object-model tree is not needed for Join.

Status: **cannot safely be published** from this checkout. There is no dataset license file next to the meshes. Human body parameters are present. Use an external download after the license is read. Do not assume academic data are redistributable because the code repo says CC BY 4.0.

`outputs/contact_analysis/` and every rendered video that shows the box are derived from these files. Hold them for the same reason.

## SMPL-X model

`data/smplx/` is not present. No model weights are in the tree. Status: **requires external download and a separate registration**. Not included.

## Unitree G1

Path: `third_party/unitree_ros/` (about 255 MB), gitignored. The file the solver loads is `robots/g1_description/g1_29dof_rev_1_0.urdf` plus the STL meshes it references (167 STL files under `g1_description`).

The README identifies Unitree Robotics as the author and points at <https://www.unitree.com/g1/>. This checkout contains **no LICENSE, COPYING, or package.xml license tag** that was found by search. Upstream `unitree_ros` is often distributed as BSD-3-Clause; that was not verified against a commit id in this tree, so it is not adopted here.

Status: **license unresolved. Third-party asset requiring attribution once the license file is recovered.** Until then, **do not upload** `third_party/unitree_ros`. Tell readers to clone the upstream description that matches `g1_29dof_rev_1_0`.

## Python packages

Installed from PyPI via `uv.lock`. Their licenses travel with the wheels and are not re-licensed by this project. Notable ones: PyTorch, MuJoCo (Apache-2.0 upstream), pytorch-kinematics, NumPy, SciPy, Trimesh. Confirm SPDX headers from the installed distributions before a NOTICE file is written. This handoff did not copy those license texts out.

## Private paths and secrets

No API keys, tokens, or private keys were found in `src/`, `scripts/`, `tests/`, or `docs/`.

`config.json` files written by `retarget-paper-join` store the absolute command line, including `$ISR_JOIN_ROOT/...`. Those JSON files and the term logs must be stripped or omitted. `MUJOCO_LOG.TXT` at the repository root is a local simulator log. Exclude it.

A local account name appears in that absolute path. Do not publish the raw config JSON.

## What a public tree may point at

- Link CORE4D and cite the CVPR paper.
- Link the Unitree G1 description and ship the license file that belongs to the revision you pin.
- Ship this project's code only after its license is chosen and it does not embed the assets above.

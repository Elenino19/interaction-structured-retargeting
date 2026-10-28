# Third-party licenses

No license has been selected for this project's own code. Do not add MIT, Apache-2.0, or any other project license until the obligations below are resolved and the authors approve one.

This file records what was found in the source checkouts. It is not permission to upload.

## Own code

`join/src/`, `join/scripts/`, `join/tests/`, `soccer/scripts/`, `soccer/tests/`, and the documents written for this release.

Status: license unresolved. There was no `LICENSE` file and no git commit that would show an earlier choice.

The preliminary preference for this original code is MIT. It is not applied. MIT is a reasonable fit for code the authors own, because it is permissive and compatible with using Apache-2.0, BSD, and MIT libraries as dependencies. It is not a fit for the repository as a whole until the checks in `RELEASE_LICENSE_AUDIT.md` pass. GVHMR's software license is non-commercial; SMPL-X, CORE4D, Unitree meshes, and the source video remain under their own terms even if the original code later becomes MIT.

## CORE4D

The instructions repository at <https://github.com/leolyliu/CORE4D-Instructions> states that the instructions repository is CC BY 4.0. That sentence is not a license file on the motion archive.

Join uses six CORE4D-Real clips (`20231108/018, 019, 020, 108, 109, 110`), action `join`, obstacle `box`. `person*_poses.npz` contains SMPL-X coefficients and fitted vertices. Object meshes used are `box024_m.obj` (018–020) and `box001_m.obj` (108–110).

Status: **not included**. Download from the dataset distribution after reading its card and the SMPL-X license together. Do not treat the code repository's CC BY 4.0 line as a license for the motions. Rendered videos that show the box are held for the same reason.

Citation:

> Liu, Zhang, Xing, Tang, Yang, Yi. CORE4D: A 4D Human-Object-Human Interaction Dataset for Collaborative Object REarrangement. CVPR 2025.

## SMPL-X

Model files require registration and must not be redistributed. They are not in this tree.

`smplx_joints.npz` is the Soccer IK input produced by running GVHMR's SMPL-X forward pass and then placing the two skeletons. It is not the model file. The SMPL-X model license still says the model and software, including animations, may not be made available to a third party without Max-Planck's prior written permission. That permission is not on file. The release author's approval to include the NPZ was conditional on those terms and does not clear it. Status: **pending**. Provenance and the inspected SHA-256 are in `docs/smplx_joints_provenance.md`. Do not put this file under a future MIT grant.

## Unitree G1

The solver expects `g1_29dof_rev_1_0.urdf`, the matching MJCF, and the STL meshes it references.

The local checkout identified Unitree Robotics as the author and contained **no LICENSE, COPYING, or package.xml license tag**. Upstream `unitree_ros` is often distributed as BSD-3-Clause. That was not verified against a commit id, so it is not adopted here.

Status: **not vendored**. Clone the upstream description that matches `g1_29dof_rev_1_0`, keep the upstream LICENSE next to the meshes, and set `UNITREE_G1_ROOT`.

Attribution when a render is shown: “Robot geometry: Unitree G1 description (unitree_ros).”

## GVHMR, YOLO, and related weights

GVHMR (ZJU 3D Vision Group, <https://github.com/zju3dv/GVHMR>) is educational, research, and non-profit as audited from the local license text. Commercial use needs permission. Checkpoints under that checkout (`gvhmr`, `hmr2`, `vitpose`, YOLO, body models, DPVO) have their own terms. Ultralytics YOLOv8 weights are not ours to relicense.

Status: **not included**. Link and cite. Do not copy weights into this repository.

Attribution when the method text mentions recovery: “Human motion: GVHMR (Shen et al. / ZJU 3DV), two independent static-camera runs.”

## Soccer source video

A private training video. Redistribution was **not** granted. The packed clip, player crops, and source-to-robot figures are excluded. There is no Sketchfab, stadium, or turf mesh in the study. Do not add one and describe it as part of the accepted experiment.

## Python packages

Installed from PyPI. Their licenses travel with the wheels. Notable ones: PyTorch, MuJoCo (Apache-2.0 upstream), pytorch-kinematics, NumPy, SciPy, Trimesh. SPDX texts were not copied out of the installed distributions. Confirm them before writing a NOTICE file.

## What a public tree may point at

- This repository's own code, after a license is chosen.
- Frozen Soccer robot and ball trajectories, robot-only renders, and metric tables.
- Not `smplx_joints.npz`. Author approval was conditional. Upstream redistribution is pending. See `docs/smplx_joints_provenance.md`.
- Links to CORE4D, Unitree, and GVHMR. Not their data or weights.

# Release license audit

This audit is conservative. It does not choose a license. Unresolved items stay unresolved.

## Decision for this tree

| Item | Bundled? | Why |
|---|---|---|
| Join and Soccer source, tests, and docs written for the paper | Yes | Own code. License still unselected. |
| Join frozen trajectories, warm starts, contact NPZ, and mp4 demos | No | `copy-private-bundle` or `hold`. Videos show CORE4D geometry. |
| CORE4D motions, SMPL-X coefficients, object meshes | No | No dataset license file in the checkout. CC BY 4.0 on the instructions repo does not cover `data/core4d`. |
| Unitree G1 URDF, MJCF, STL | No | No LICENSE file was in the inspected description directory. Upstream terms were not pinned to a commit. |
| GVHMR code, checkpoints, YOLO weights, SMPL-X model | No | Research / non-commercial or registration terms. Not ours to relicense. |
| Soccer source video and crops | No | Redistribution not granted. |
| Soccer `robot_a.npz`, `robot_b.npz`, `ball_cones.npz`, `ball_cones.json`, robot-only mp4, ablation NPZ, metric tables, public-export figures | Yes | Handoff marks them include or include-selective, and they do not contain source frames. |
| `smplx_joints.npz` | No. Status **pending** | Inspected outside the tree. SHA-256 `f03a8c5bb6e3e658c92b6ac7dc3915754100c07ef5cbfb010fc149cdb0ae3b4b`. The release author approved inclusion only if GVHMR and SMPL-X redistribution terms allow it. The SMPL-X model license does not clearly allow offering this derivative to a third party. See `docs/smplx_joints_provenance.md`. |
| Sketchfab / pitch mesh | No | None exists. None was downloaded. |

## How a user obtains what was left out

- **CORE4D.** Dataset page and Hugging Face distribution linked from the CORE4D-Instructions README. After the license is accepted, place motions and the two box meshes as in `join/data/README.md`.
- **G1.** Upstream Unitree description for `g1_29dof_rev_1_0`, including whatever LICENSE that revision ships. Set `UNITREE_G1_ROOT` to the directory that contains the URDF, MJCF, and `meshes/`.
- **GVHMR and YOLO.** Official checkout and its checkpoint license. Set `GVHMR_ROOT` and, if needed, `YOLO_WEIGHTS`.
- **SMPL-X.** Register and download the model into a path GVHMR expects. Do not commit it.
- **Soccer video.** Only an authorized local file, via `SOCCER_SOURCE_VIDEO`. There is no substitute clip in this release.
- **Join motions.** A private bundle via `ISR_JOIN_PRIVATE_ROOT`. Do not copy it here until the CORE4D-derived-video question is closed.

## What must not be done

- Stamping MIT or Apache-2.0 on this repository before approval
- Vendoring `third_party/unitree_ros` without the upstream license file
- Replacing missing meshes or motions with primitive shapes and calling the run a reproduction
- Publishing Join renders other than the three 018 files in `figures/join/`
- Adding a stadium model and describing it as part of the accepted soccer experiment

## Own-code status

There was no `LICENSE` file and no commit in either source checkout. This release also has no `LICENSE` file, on purpose.

Python dependencies keep the licenses of their wheels. MuJoCo's upstream license is Apache-2.0. That was not copied into a NOTICE file. Confirm SPDX headers from the installed distributions before publishing a notice.

## Preliminary preference for original code

MIT is the stated preference for original project code. No `LICENSE` file is added.

Add MIT only after all of the following are confirmed:

- Buwen Zhang has confirmed sole authorship of the original code. Agreement to apply MIT is still required. Do not add `LICENSE` before that approval.
- Other contributors' rights are confirmed, including anyone who wrote code that landed in `join/` or `soccer/` without a prior license.
- University or other institutional obligations are checked where they apply.
- Third-party license compatibility is reviewed. MIT would cover the original code only. It would not relicense CORE4D, SMPL-X, the pending joint NPZ, Unitree meshes, GVHMR weights, or the source video. GVHMR's own software license is non-commercial; that code is not vendored here. A later copy of GVHMR source into this tree could not be placed under MIT.

The private repository may remain without a `LICENSE` until that approval is explicit.

## Manuscript

Draft v6.2 is the current reference and is not frozen. Later wording, formatting, citation, and figure-number edits follow `PAPER_VERSION_SYNC.md`. The PDF was read from a copy kept outside this tree. It is not a file in the proposed commit. `PAPER_CLAIM_TRACEABILITY.csv` maps Table I, the abstract aggregates, and Figs. 1–4. The 018 comparison videos are in `figures/join/`. The source-video half of Fig. 4 stays out. The sentence that an earlier 018 ladder moved bilateral anchor error from 18.28 cm to 8.94 cm is untraced in the bundled tables.

# Dependencies

Nothing here is vendored inside `soccer_retarget` except our scripts. User-specific absolute paths must be turned into environment variables by the integration Agent. Do not commit `.venv`, checkpoints, or `data/raw/`.

## This repository

- Python 3 scripts under `scripts/`. Tests under `tests/` run with system `python3` plus numpy for the metric tests; event-chain structure tests need only the standard library plus numpy imported by the evaluator. `python3 -m unittest discover -s tests` passed in this workspace (8 tests) without loading the frozen NPZ.
- Rendering and IK need the CORE4D environment: `$ISR_JOIN_ROOT/.venv` (torch CPU, pytorch_kinematics, mujoco, matplotlib, imageio, scipy). `MUJOCO_GL=egl`.
- Video I/O and YOLO need the GVHMR environment: `$GVHMR_ROOT/.venv` (opencv, ultralytics).
- `ffmpeg` for figure frame grabs and the web encode. Figures use DejaVu (`/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf`), a standard Debian font, not a project asset.

## CORE4D dual-retarget (external)

Path: `$ISR_JOIN_ROOT`.

Imported symbols:

- `core4d_dual_retarget.coordinates.points_core4d_to_robot`
- `core4d_dual_retarget.retarget.independent.IndependentRetargetConfig`, `_apply_root_transform`, `_yaw_from_shoulders`
- `core4d_dual_retarget.robot_model.load_actuated_joint_limits`
- `core4d_dual_retarget.mujoco_scene._set_robot_qpos`, `_prefix_names`

Soccer does **not** import the Join EXT-JC coupling loss.

## Unitree G1

URDF: `$ISR_JOIN_ROOT/third_party/unitree_ros/robots/g1_description/g1_29dof_rev_1_0.urdf`

MJCF: `.../g1_29dof_rev_1_0.xml` plus `meshes/`.

The local checkout’s description README identifies Unitree Robotics as the author. A LICENSE file was **not** found next to that README during this audit. Upstream `unitree_ros` is published by Unitree; ship the upstream LICENSE with any meshes. Do not strip copyright notices.

## GVHMR and body models

Checkout: `$GVHMR_ROOT`. License text (head): educational, research, and non-profit only; commercial use needs permission from the ZJU 3D Vision Group; cite the project. Weights under `inputs/checkpoints/` (`gvhmr`, `hmr2`, `vitpose`, `yolo/yolov8x.pt`, `body_models`, `dpvo`) have **their own** licenses (SMPL-X is not Apache). Do not copy weights into the public Soccer repo.

YOLO is used only to crop people and detect the ball. Ultralytics YOLOv8 weights are not ours to relicense.

## Private paths that must not ship

- `$SOCCER_SOURCE_VIDEO`
- `$GVHMR_ROOT`
- `$ISR_JOIN_ROOT`
- Any `HF_TOKEN` or checkpoint path

`prepare_touch_ladder_demo.py` and `export_smplx_joints.py` currently hardcode some of these. The integration Agent should replace them with `SOCCER_SOURCE_VIDEO`, `GVHMR_ROOT`, and `CORE4D_ROOT` before a public commit, without changing numerical defaults.

## No football-field asset

Search of this repo found no Sketchfab, stadium, or turf mesh. The accepted render is the MuJoCo grid plus five cones and a ball. Do not add scenery and present it as part of the scientific result. A later visual-only field would be a new asset with its own license and must not change `robot_*.npz` or `ball_cones.npz`.

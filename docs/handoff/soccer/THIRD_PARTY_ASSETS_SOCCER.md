# Third-party assets (Soccer)

| Asset | Where it lives | Author / URL | License as audited | In public Soccer repo? |
|---|---|---|---|---|
| Human source video | Private file; packed copy `data/clips/touch_ladder_demo_source.mp4` | 7MLC / MZC CONTROLLO training video, WeChat forward. No license grant on file. | Redistribution **not** granted | No |
| Player crops | `data/clips/touch_ladder_demo_player_{a,b}.mp4` | Derived from the same video | Same restriction | No |
| GVHMR code | `$GVHMR_ROOT` | ZJU 3D Vision Group, https://github.com/zju3dv/GVHMR | Research / non-profit; commercial use requires permission | No. Link and cite. |
| GVHMR / HMR2 / ViTPose / SMPL-X / DPVO weights | `GVHMR/inputs/checkpoints/` | Multiple upstream authors | Each weight has its own non-commercial or registration license | No |
| YOLOv8x | `GVHMR/inputs/checkpoints/yolo/yolov8x.pt` | Ultralytics | Ultralytics license (AGPL for the package; do not relicense) | No |
| Unitree G1 URDF/MJCF/meshes | `core4d-dual-retarget/third_party/unitree_ros/robots/g1_description/` | Unitree Robotics, https://www.unitree.com/g1/ ; model README in that folder | LICENSE file **not present** in the inspected checkout. Carry the upstream `unitree_ros` license with the meshes. | Only if that license is copied alongside |
| MuJoCo | Python package in the CORE4D venv | Google DeepMind | Apache-2.0 library; not vendored | Dependency, not an asset blob |
| pytorch_kinematics, torch, numpy, matplotlib, OpenCV | venvs | Upstream | Their respective licenses | No |
| DejaVu fonts | OS package | Bitstream Vera derivative | Public-domain-style Bitstream Vera / DejaVu license | Not copied; figures already rasterize text |
| Sketchfab / stadium / pitch mesh | **None in this project** | — | — | Do not invent or download one for the release |
| Candidate public videos under `data/raw/` and `data/review/` | Mixkit, Pexels, DVIDS, MUVS | Various | Not the accepted demo. Mixkit/Pexels labels were not snapshotted as license files. Fail closed. | No |
| Our code, frozen NPZ, robot-only renders, metric tables | this repo | This project | Research release alongside the paper, subject to the exclusions above | Yes, per `RELEASE_INCLUDE_EXCLUDE.csv` |

Attribution sentence if G1 renders are shown: “Robot geometry: Unitree G1 description (unitree_ros).”

Attribution sentence if the method text mentions recovery: “Human motion: GVHMR (Shen et al. / ZJU 3DV), two independent static-camera runs.”

Do not attribute a football pitch. The scientific scene has no pitch texture.

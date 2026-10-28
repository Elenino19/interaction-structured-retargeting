# Dependencies

Versions below were read from the existing `.venv` on this machine on 2026-10-28. They are not the result of a clean install in an empty environment. `uv.lock` is the lockfile that produced this environment; a fresh machine was not tested.

## Tested stack

| Item | Value |
|---|---|
| OS | Linux 6.8.0-138-generic, x86_64 |
| CPU / RAM | 32 cores, 15 GiB RAM. More than about four simultaneous 600-iteration solves exhausts memory. |
| Python | 3.10.12. `pyproject.toml` allows `>=3.9`. Only 3.10 was used. |
| PyTorch | 2.14.0+cpu |
| MuJoCo | 3.12.0. Rendering used `MUJOCO_GL=egl`. |
| pytorch-kinematics | 0.10.0 |
| NumPy | 2.2.6 |
| SciPy | 1.15.3 |
| Trimesh | 5.1.0 |
| GPU | Not used. The installed torch build is CPU. |

## Install that matches this checkout

From the repository root, with [uv](https://docs.astral.sh/uv/) available:

```bash
uv venv .venv
uv pip install --python .venv/bin/python -e ".[dev,torch,visualization,simulation]"
```

Direct dependencies declared in `pyproject.toml`:

- Always: numpy, scipy, trimesh, typer, rich, pydantic, remotezip, requests
- `dev`: pytest, ruff
- `torch`: torch, pytorch-kinematics
- `visualization`: matplotlib, imageio, imageio-ffmpeg
- `simulation`: mujoco, PyOpenGL

System packages used and not pinned: a C compiler is not required for the wheels that were installed here. EGL is required only to render. `ffmpeg` is pulled in by `imageio-ffmpeg` for mp4.

## Robot assets

Required URDF, and the STL meshes it references:

`third_party/unitree_ros/robots/g1_description/g1_29dof_rev_1_0.urdf`

This checkout has no `LICENSE` file next to those meshes. See `THIRD_PARTY_ASSETS_JOIN.md`. Do not vendor them into a public repository until that file is recovered from the exact upstream revision.

## CORE4D

Required for anything above playback of an already rendered mp4:

```text
data/core4d/metadata/action_labels.json
data/core4d/motions/20231108/{018,019,020,108,109,110}/
    person1_poses.npz
    person2_poses.npz
    smooth_objposes.npy
    smooth_objposes_so3.npy    # 018 uses the repaired SO(3) track when present
    object_metadata.json
data/core4d/object_models/object_models/box/box024_m.obj   # 018, 019, 020
data/core4d/object_models/object_models/box/box001_m.obj   # 108, 109, 110
```

`person*_poses.npz` contains SMPL-X coefficients (`betas`, `expression`, `body_pose`, hand poses) and fitted vertices. `data/README.md` already says SMPL-X model files must not be redistributed. The motion npz are the same problem. Download CORE4D from the upstream distribution after checking its license; do not commit this `data/` tree.

Acquisition commands, only after the license is accepted:

```bash
.venv/bin/core4d-retarget fetch-sequence 20231108/018
.venv/bin/core4d-retarget repair-object-poses \
  data/core4d/motions/20231108/018/smooth_objposes.npy \
  data/core4d/motions/20231108/018/smooth_objposes_so3.npy
```

## Randomness

The Adam retargeter has no seed. Results are deterministic given the same initial npz, the same thread count was checked by the OA notes (8 vs 10 threads) for the separability probe, and `retarget-paper-join` refuses to overwrite. The separability probe's finite-difference directions use NumPy seed 0. The post-process repair is a deterministic grid search.

# Reproduction

Three levels. Level 3 is documented so it can be attempted. It is not claimed to be bit-exact, and it was not rerun for this release.

Do not write into bundled result directories. Soccer scripts exit if an output path lies under `soccer/data/exports/touch_ladder_demo/`. Join's `retarget-paper-join` refuses to overwrite an existing result directory. Point new runs at `/tmp` or another fresh root.

## Environment

Tested stack, read from an existing workstation environment on 2026-10-28, not from a clean install:

- Linux, Python 3.10.12
- PyTorch 2.14.0+cpu, MuJoCo 3.12.0, `MUJOCO_GL=egl`
- pytorch-kinematics 0.10.0, NumPy 2.2.6, SciPy 1.15.3, Trimesh 5.1.0

```bash
# working directory: repository root
cd join
uv venv .venv
uv pip install --python .venv/bin/python -e ".[dev,torch,visualization,simulation]"
cd ..
# working directory is the repository root again
export PYTHONPATH="$PWD/join/src:${PYTHONPATH:-}"
export UNITREE_G1_ROOT=/path/to/g1_29dof_description
export MUJOCO_GL=egl
```

Soccer unit tests do not need that environment. Run them from the repository root so the shell is not left inside `soccer/`:

```bash
# working directory: repository root
python3 -m unittest discover -s soccer/tests -v
python3 -m py_compile soccer/scripts/*.py
```

## Level 1 — Replay distributable frozen results

### Soccer

Bundled files are enough to load the motion. No source video.

```bash
# working directory: repository root
python3 examples/load_soccer_frozen.py
```

Expected: five hashes match the handoff, both robots are `(144, 29)`, and the annotation expands to 6 receives, 7 kicks, and 6 transfer chains. `smplx_joints.npz` must be absent. Its redistribution status is pending (`docs/smplx_joints_provenance.md`). Level 1 does not need it.

A MuJoCo replay needs the G1 MJCF and meshes, which are not bundled. Copy the NPZ to a scratch directory first. The renderer writes `soccer_scene.xml` beside the robots and must not receive the bundled preview as `--out`.

```bash
mkdir -p /tmp/soccer_replay
cp soccer/data/exports/touch_ladder_demo/g1/robot_a.npz /tmp/soccer_replay/
cp soccer/data/exports/touch_ladder_demo/g1/robot_b.npz /tmp/soccer_replay/
MUJOCO_GL=egl python3 soccer/scripts/render_g1_soccer.py \
  --g1-dir /tmp/soccer_replay \
  --ball-cones soccer/data/exports/touch_ladder_demo/ball_cones.npz \
  --xml "$UNITREE_G1_ROOT/g1_29dof_rev_1_0.xml" \
  --out /tmp/soccer_full_replay.mp4
```

Camera inside the script: distance 4.4, azimuth 75, elevation −24, 960×720, 24 fps. A byte match with `c904d2ed…` was not re-proven by the original handoff. Compare visually.

The display encode used on the local website is `docs/media/soccer_full_demo_web.mp4` (SHA-256 `3d5fcef1a8500c2a591f6a5a51661f01a1b482bb9cc95b4e6e0f8482af6c7801`). It is a CRF-28 re-encode of the same frames, not a new trajectory.

### Join

Not bundled. `examples/check_join_private_bundle.py` exits with status 2 until `ISR_JOIN_PRIVATE_ROOT` points at a local read-only bundle. Playback of existing mp4 files requires that bundle. Publishing those mp4 files is blocked because the CORE4D box is in frame.

Frozen v41 identity recorded by the handoff, for `20231108/018` smooth NPZ:

- robot1 MD5 `bdcf43e656cc3df846f12baa740a6a1f`
- robot2 MD5 `6ec5e5d65a0226a69004f1db139b9002`

## Level 2 — Recompute metrics from frozen artifacts

No optimizer. Write only to a new directory.

### Soccer

Prefer the CSVs already in the tree if you only need the paper numbers. To recompute, copy the ablation tree to a scratch directory. The evaluator writes `metrics.json` next to the variants, and the bundled copy is protected.

```bash
# working directory: repository root
cp -a soccer/data/exports/touch_ladder_demo/experiments/ablation /tmp/soccer_ablation
python3 soccer/scripts/evaluate_ablation_experiments.py \
  --root /tmp/soccer_ablation \
  --contacts soccer/data/exports/touch_ladder_demo/ball_cones.json
```

Event-chain and paper evaluators likewise need `--out` outside the bundled `paper_eval/` directory. Mesh metrics need a scene XML and the G1 meshes. Without those, the evaluators should report the mesh probe as unavailable rather than invent a score.

### Join

Requires `join/data/core4d` (motions and `box001` / `box024` meshes), contact-analysis NPZ, and the G1 URDF. Run from `join/` so the relative data paths resolve. Example, one take, output outside the frozen tree:

```bash
cd join
.venv/bin/core4d-retarget evaluate-paper-join 20231108/020 \
  --result-root "$ISR_JOIN_PRIVATE_ROOT/outputs/balanced_clearance_v41_smooth" \
  --output-path /tmp/join_repro/v41_020.json
```

Do not point `--output-path` at an existing metrics directory. Repaired EXT-JC was scored by `scripts/ext_jc_evaluate.py`. The numbers to cite are already in `evaluation/join/postprocess_metrics.csv`. This release did not recompute them against a fresh CORE4D download, so a numerical match after you download the dataset is unverified here.

## Level 3 — Rerun the pipeline

This level is not fully reproduced. Exact upstream files and the historical initializations are not all in this repository, and a successful empty-disk run was not executed.

### Join

1. Obtain CORE4D and the G1 description under their own licenses.
2. Independent IK (method A), only if you need it: `core4d-retarget retarget-independent 20231108/018`.
3. Frozen v41 is **not** that command followed by `retarget-paper-join --variant J4`. Take 018 was warm-started from v40. A cold J4 run is the ablation in the paper tables, and the report refuses to substitute it for v41.
4. OA-Independent, the surrogate, and EXT-JC were warm-started from `v40_smooth` (018) or `v20_smooth` (019–110). Those directories are previous optimization outputs. Reproducing them means replaying older clearance recipes, which are not a single pinned command.
5. Illustrative EXT-JC invocation for 020, which must use a **new** `--output-root`:

```bash
.venv/bin/core4d-retarget retarget-paper-join 20231108/020 --variant J4 \
  --iterations 600 \
  --output-root /tmp/join_level3 \
  --tag ext_jc_w10_staged \
  --baseline-root /path/to/balanced_clearance_v20_smooth \
  --term-log \
  --override joint_contact_continuity_weight=10.0
```

Surrogate: add `--override joint_contact_separable_surrogate=true`. OA-Independent: omit the joint-contact override and run twice with `--solve-robot 1` and `--solve-robot 2`.

6. Smooth with the clearance recipe, then run `scripts/ext_jc_postprocess.py` with a new `--out-root`.

One 600-iteration J4 solve took about 13 minutes on a 32-core, 15 GiB machine. Keep at most three or four solves at once. Do not resume rejected configs under `runs_dev/`.

`scripts/run_join_paper_experiments.sh` is the cold J0–J4 ablation driver. It refuses to overwrite its output root. It does not regenerate frozen v41.

### Soccer

From prepared joints and the frozen annotation, into a new directory. The prepared joints are a derived SMPL-X artifact. Author approval to include them was conditional, and upstream redistribution is still pending, so they are **not** in this repository. Set `ISR_SOCCER_JOINTS` to a local copy you are allowed to use. Without that variable the command below stops. The expected SHA-256 of the inspected copy is `f03a8c5bb6e3e658c92b6ac7dc3915754100c07ef5cbfb010fc149cdb0ae3b4b`.

```bash
# working directory: repository root
test -n "${ISR_SOCCER_JOINTS:-}" || { echo "BLOCKED: prepared joints are not in this tree"; exit 2; }
cp soccer/data/exports/touch_ladder_demo/ball_cones.json /tmp/ball_cones_copy.json
python3 soccer/scripts/retarget_dual_g1_soccer.py \
  --joints "$ISR_SOCCER_JOINTS" \
  --contacts /tmp/ball_cones_copy.json \
  --out-dir /tmp/soccer_level3 \
  --iterations 340
```

This run will not match `robot_a.npz` / `robot_b.npz`. The accepted files are the 2026-10-01 IK plus two later in-place world-Y alignments. The script's default `--out-dir` is still the bundled `g1/` directory; the write guard rejects that default. After IK it also writes a ball NPZ beside `--contacts`, which is why the command copies the JSON first.

Level 4, from the human video, needs `SOCCER_SOURCE_VIDEO`, `GVHMR_ROOT`, YOLO weights, and a registered SMPL-X model. The clip window is 203.0–209.0 s. `recover_ball_cones.py` writes a **proposal**. The paper events are the frozen JSON, including manual corrections. Do not replace `ball_cones.json` with a detector run and call it the paper annotation.

There is no public substitute clip. Without the video, the weights, and SMPL-X, Level 4 cannot be run.

## What not to run

- Any optimizer aimed at the bundled or frozen result paths
- `ISR_ALLOW_FROZEN_OVERWRITE=1` on the packaged tree
- A fresh CORE4D download presented as a completed Level 3 reproduction of v41 or EXT-JC
- Rejected Join refinement sweeps (`refined_v1` through `v5`)

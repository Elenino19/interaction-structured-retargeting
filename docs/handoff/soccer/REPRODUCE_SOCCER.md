# Reproduce Soccer

Do not overwrite `data/exports/touch_ladder_demo/g1/robot_a.npz`, `robot_b.npz`, `ball_cones.npz`, or `ball_cones.json`. Fresh IK writes those paths if you point `--out-dir` at `g1/`. Always use a new directory.

Workstation pythons (not portable):

- `PY=$ISR_JOIN_ROOT/.venv/bin/python` for IK, render, metrics.
- `PYV=$GVHMR_ROOT/.venv/bin/python` for video and GVHMR.
- `export MUJOCO_GL=egl`

## Level 1 — Replay the accepted animation

Needs: frozen `robot_a.npz`, `robot_b.npz`, `ball_cones.npz`, G1 `g1_29dof_rev_1_0.xml` and meshes, `scripts/render_g1_soccer.py`, CORE4D on `PYTHONPATH` (the script inserts `$ISR_JOIN_ROOT/src` today).

```bash
MUJOCO_GL=egl "$PY" scripts/render_g1_soccer.py \
  --g1-dir data/exports/touch_ladder_demo/g1 \
  --ball-cones data/exports/touch_ladder_demo/ball_cones.npz \
  --out /tmp/soccer_full_replay.mp4
```

`--out` must not be the accepted mp4. `render_g1_soccer.py` archives an existing `--out` before writing; still do not pass the canonical path.

Limitation: a replay is deterministic given MuJoCo and the NPZ, but this audit did not re-render, so byte identity with `c904d2ed…` is not re-proven here. Compare visually. Camera is hard-coded (distance 4.4, azimuth 75, elevation −24, 960×720, 24 fps).

No source video is required.

## Level 2 — Recompute metrics

Needs Level 1 inputs plus `ball_cones.json`. Does not refit IK.

```bash
"$PY" scripts/evaluate_ablation_experiments.py \
  --root data/exports/touch_ladder_demo/experiments/ablation \
  --contacts data/exports/touch_ladder_demo/ball_cones.json

MUJOCO_GL=egl "$PY" scripts/evaluate_soccer_paper.py

MUJOCO_GL=egl "$PY" scripts/evaluate_event_chain.py \
  --out data/exports/touch_ladder_demo/paper_eval/event_coordination_extension
```

The last command’s default entry list skips `independent_object_aware/`. To refresh the comparison table:

```bash
MUJOCO_GL=egl "$PY" scripts/evaluate_event_chain.py \
  --out data/exports/touch_ladder_demo/paper_eval/object_aware_comparison \
  --entries COMPARISON
python3 data/exports/touch_ladder_demo/paper_eval/object_aware_comparison/logs/build_comparison.py
```

These rewrite metric JSON/CSV under `paper_eval/`, not the robot NPZ. Prefer reading the existing CSV if you only need the paper numbers.

## Level 3 — Recreate event-conditioned motion and the ball from prepared joints

Needs: `smplx_joints.npz`, `ball_cones.json`, G1 URDF, `scripts/retarget_dual_g1_soccer.py`.

```bash
"$PY" scripts/retarget_dual_g1_soccer.py \
  --joints data/exports/touch_ladder_demo/smplx_joints.npz \
  --contacts data/exports/touch_ladder_demo/ball_cones.json \
  --out-dir /tmp/soccer_level3 \
  --iterations 340
```

Limitations, which are mandatory to state:

- This run **will not** match `robot_a.npz` / `robot_b.npz`. The accepted files are the 2026-10-01 IK plus two later in-place world-Y alignments. Stationing code changed afterward. See `paper_eval/event_coordination_extension/pipeline_dependency_audit.md` section 5.
- The script’s default `--out-dir` is the frozen `g1/` directory. Do not use that default.
- After IK the script calls `snap_ball_to_g1_feet` on `ball_cones.npz` next to `--contacts`. Point `--contacts` at a **copy** if you need a new ball file, or the frozen ball will be archived and replaced.
- 340 iterations, CPU float64, on the order of tens of seconds per robot on this machine. Not a coupled solve.

## Level 4 — From the human video

Only an authorized local copy of the source. There is no substitute clip.

```bash
export SOCCER_SOURCE_VIDEO="/path/to/authorized_source.mp4"   # not provided in the repo
# prepare_touch_ladder_demo.py currently hardcodes the private path and
# CLIP_START_S=203 / CLIP_END_S=209. Change only that path, then:
"$PYV" scripts/prepare_touch_ladder_demo.py
"$PYV" scripts/run_gvhmr_pair.py touch_ladder_demo \
  --player-a data/clips/touch_ladder_demo_player_a.mp4 \
  --player-b data/clips/touch_ladder_demo_player_b.mp4 \
  --gvhmr-root /path/to/GVHMR \
  --python "$PYV"
"$PYV" scripts/pack_touch_ladder_demo.py
"$PYV" scripts/export_smplx_joints.py --gvhmr-root /path/to/GVHMR
"$PYV" scripts/export_source_foot_tracks.py --gvhmr-root /path/to/GVHMR
"$PYV" scripts/recover_ball_cones.py
```

Then Level 3. GVHMR also needs its checkpoints and a registered SMPL-X model.

If `data/gvhmr/touch_ladder_demo/` already exists, official GVHMR may reuse it. Delete that directory before a real rerun. Do not delete frozen `data/exports/touch_ladder_demo/g1/` or `ball_cones.*`.

`recover_ball_cones.py` writes a **proposal**. The paper events are the frozen JSON, including manual receive/kick corrections. Do not replace `ball_cones.json` with a fresh detector run and call it the paper annotation.

## What “complete video-to-robot reproducibility” is not

Without the source file, GVHMR weights, and SMPL-X, Level 4 cannot be run. Level 1 is the public demonstration. Level 3 is the algorithmic recreation from prepared joints and will differ numerically from the frozen demo.

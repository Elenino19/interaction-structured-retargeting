# Reproduce Join

Commands were taken from code and from notes that match files already on disk. A from-scratch run in a new environment was **not** performed for this handoff. Where a command was executed during the handoff, it is listed in `JOIN_SMOKE_TESTS.md`.

Working directory is the repository root. Python is `.venv/bin/python`.

## Level 1 — Read frozen motions and show a Join demo

Prerequisites: the mp4 files below. No CORE4D download, no MuJoCo, no optimization.

Frozen v41, one clip per take. These are method B, not EXT-JC.

```text
videos/retarget/018_clearance_v41.mp4
videos/retarget/019_clearance_v41.mp4
videos/retarget/020_clearance_v41.mp4
videos/retarget/108_clearance_v41.mp4
videos/retarget/109_clearance_v41.mp4
videos/retarget/110_clearance_v41.mp4
```

EXT-JC after the smoothing repair (method F), already rendered:

```text
outputs/paper_eval/ext_jc_visual_refinement/postprocess/demo_018_ext_jc_repaired.mp4
outputs/paper_eval/ext_jc_visual_refinement/postprocess/demo_020_ext_jc_repaired.mp4
outputs/paper_eval/ext_jc_visual_refinement/postprocess/videos/020_v41_vs_ext_repaired.mp4
outputs/paper_eval/ext_jc_visual_refinement/postprocess/videos/018_v41_vs_ext_repaired.mp4
outputs/paper_eval/ext_jc_visual_refinement/postprocess/videos/020_surrogate_repaired_vs_ext_repaired.mp4
outputs/paper_eval/ext_jc_visual_refinement/postprocess/videos/018_surrogate_repaired_vs_ext_repaired.mp4
```

The comparison clips put v41 and repaired EXT-JC in separate tiles. Do not retitle the v41 tile.

To re-render method F from the frozen repaired npz (this was how the demos were made; it was not re-run for this handoff):

```bash
export MUJOCO_GL=egl
# scripts/ext_jc_demo.py reads the repaired npz and the v41 camera.
# See the script's --help. Expected outputs stay under postprocess/.
```

Blockers: EGL and the G1 meshes are required to re-render. Playback of the existing mp4 is not blocked. The demos contain the CORE4D box mesh, so they are not cleared for a public upload. See `THIRD_PARTY_ASSETS_JOIN.md`.

Identity check that was run:

```bash
md5sum outputs/balanced_clearance_v41_smooth/20231108_018/robot1.npz
# bdcf43e656cc3df846f12baa740a6a1f
md5sum outputs/balanced_clearance_v41_smooth/20231108_018/robot2.npz
# 6ec5e5d65a0226a69004f1db139b9002
```

## Level 2 — Recompute paper metrics from frozen trajectories

Prerequisites:

- Level 1 trajectories
- `data/core4d` motions and `box001` / `box024` meshes
- `outputs/contact_analysis/20231108_{018,019,020,108,109,110}.npz`
- G1 URDF
- the installed `.venv`

No optimization. Write outputs to a new directory so the frozen JSON is not replaced.

Frozen v41 (method B), one take:

```bash
.venv/bin/core4d-retarget evaluate-paper-join 20231108/020 \
  --result-root outputs/balanced_clearance_v41_smooth \
  --output-path /tmp/join_repro/v41_020.json
```

Coordination spans, including the 0.5 cm valid-run length:

```bash
.venv/bin/core4d-retarget evaluate-coordination 20231108/020 \
  --result-root outputs/balanced_clearance_v41_smooth \
  --output-path /tmp/join_repro/v41_020_coord.json
```

The flags above match `evaluate-paper-join` and `evaluate-coordination` in `cli.py`. Confirm `--help` if the integration tree renames them. Do not point `--output-path` at `outputs/metrics/balanced_clearance_v41_smooth/`.

Repaired EXT-JC and the repaired surrogate (methods F and D) were scored by `scripts/ext_jc_evaluate.py` into:

```text
outputs/paper_eval/ext_jc_visual_refinement/postprocess/postprocess_metrics.csv
outputs/paper_eval/ext_jc_visual_refinement/postprocess/eval/paper_metrics/
outputs/paper_eval/ext_jc_visual_refinement/postprocess/eval/per_result/
```

Recomputing them is valid if the output path is new. The numbers to cite are the ones already in `postprocess_metrics.csv`.

Unrepaired OA / EXT / surrogate aggregates:

```text
outputs/paper_eval/object_aware_baseline_extension/paper_ready_results_table.csv
outputs/paper_eval/object_aware_baseline_extension/all_variant_metrics.csv
```

Those unrepaired EXT rows are not method F.

Blockers: without `data/core4d` this level cannot run. This handoff did not recompute the metrics, so a numerical match after a fresh data download is unverified here. The evaluator code and the frozen npz are the ones that produced the CSV.

## Level 3 — Reproduce retargeting and EXT-JC from source data

Prerequisites: everything in Level 2, plus independent IK outputs and the warm-start trajectories.

This level is not a single command from a raw CORE4D download to method F.

1. Independent IK (method A), if `outputs/independent` is absent:

```bash
.venv/bin/core4d-retarget retarget-independent 20231108/018
```

2. The published v41 motions are **not** produced by that command followed by `retarget-paper-join --variant J4`. They are the historical clearance stack. 018 was warm-started from v40. Re-running J4 from `outputs/independent` produces the cold ablation in `outputs/paper_eval_join/ablation_018/J4/`, which the paper report explicitly refuses to substitute for frozen v41.

3. OA-Independent, surrogate, and EXT-JC were warm-started:

```bash
# 020 shown. 019, 108, 109, 110 use the same baseline root.
# 018 uses outputs/balanced_clearance_v40_smooth instead.
.venv/bin/core4d-retarget retarget-paper-join 20231108/020 --variant J4 \
  --iterations 600 \
  --output-root outputs/paper_eval/object_aware_baseline_extension/runs_multiseq \
  --tag ext_jc_w10_staged \
  --baseline-root outputs/balanced_clearance_v20_smooth \
  --term-log \
  --override joint_contact_continuity_weight=10.0
```

Surrogate: add `--override joint_contact_separable_surrogate=true` and `--tag ext_jc_w10_surrogate_staged`.

OA-Independent: omit the joint-contact override, `--tag oaind_v41_staged`, and run twice with `--solve-robot 1` and `--solve-robot 2`.

4. Smooth with the clearance recipe, then repair. The existing repaired files must not be overwritten. `retarget-paper-join` already refuses to overwrite a raw directory. `ext_jc_postprocess.py` should be pointed at a new `--out-root` if a byte-level reproduction is required.

One 600-iteration J4 solve took about 13 minutes on this 32-core / 15 GiB machine. Keep at most three or four solves at once.

Blockers:

- `outputs/balanced_clearance_v20_smooth` and `v40_smooth` are themselves optimized artifacts. Reproducing them means replaying the older clearance recipes, which are not a single pinned command in this handoff.
- The git history cannot help: there are no commits.
- A successful Level 3 run from an empty disk was not executed for this handoff.
- Do not resume `outputs/paper_eval/ext_jc_visual_refinement/runs_dev/`. Those configs were rejected.

## Expected paths for the selected result

| Method | Smooth or repaired motions | Table |
|---|---|---|
| B v41 | `outputs/balanced_clearance_v41_smooth/20231108_*/` | `outputs/paper_release_join_v2/tables/join_six_sequence_aggregate.md` |
| C OA-Independent | `runs_018/oaind_v41_staged/smooth` and `runs_multiseq/oaind_v41_staged/smooth` | `paper_ready_results_table.csv` |
| D surrogate repaired | `postprocess/ext_jc_w10_surrogate/` | `postprocess_metrics.csv` rows `surrogate_repaired` |
| E unrepaired EXT-JC | `reference_runs_018/ext_jc_w10_staged/smooth` and `runs_multiseq/ext_jc_w10_staged/smooth` | `all_variant_metrics.csv` plus the 018 reference evaluation. Not the final visual result. |
| F repaired EXT-JC | `postprocess/ext_jc_w10/` | `postprocess_metrics.csv` rows `ext_repaired` |

# Join release handoff

Paper: **Interaction-Structured Motion Retargeting for Multi-Party Humanoid–Object Interactions**.

This folder is a handoff for an integration agent. It is not a GitHub publication. Do not push this tree, and do not overwrite anything under `outputs/balanced_clearance_v41*`, `outputs/metrics/`, `videos/retarget/`, or `outputs/paper_eval/`.

The local git repository has **no commits** on `main`. The working tree is the source of truth. A small early subset is staged; almost all paper code and every result file are untracked. Do not treat `git show HEAD` as the paper code.

Draft v6.1 of the manuscript is **not in this repository**. Figure and table numbers in the Word/LaTeX draft cannot be checked against a local tex file. `PAPER_RESULT_MAP.csv` maps the numbers this repository actually computed.

## What the Join paper is

Six CORE4D-Real clips, action `join`, obstacle `box`, `obs0`:

`20231108/{018,019,020,108,109,110}`

Two Unitree G1 29-DoF robots. The object pose is an input, not an optimized variable. Split used by the later evaluations: train/dev `018/019/108`, val `020`, test `109/110`.

All released motions are kinematic references. Free-base MuJoCo PD does not stand these up. That is a limitation, not a result.

## Final selected methods

Keep these identities separate. Never label a v41 video or metric as EXT-JC.

| ID | Name | What it is | Where the publication motions live |
|---|---|---|---|
| A | Independent IK | Per-person inverse kinematics. No object anchor, no partner term. Paper ablation J0 copies these files. | `outputs/independent/20231108_*/robot{1,2}.npz` |
| B | Frozen v41 | Visually accepted Join recipe. Loss components match preset J4. The objective is additively separable: the inter-robot clearance term never activates on these takes. Warm-started historically (018 from v40), not the cold-start J4 ablation. | Raw `outputs/balanced_clearance_v41/`. Smoothed, which is what the videos and paper metrics use: `outputs/balanced_clearance_v41_smooth/`. Videos `videos/retarget/{018,019,020,108,109,110}_clearance_v41.mp4`. |
| C | OA-Independent | Same J4 objective and budget as v41, but each robot is optimized alone (`--solve-robot 1` then `2`). The strong independent baseline. | 018: `outputs/paper_eval/object_aware_baseline_extension/runs_018/oaind_v41_staged/smooth/`. 019–110: `.../runs_multiseq/oaind_v41_staged/smooth/`. |
| D | Separable contact surrogate | Same as EXT-JC except the product `s1*s2` is replaced by `(s1+s2)/2`. Weight 10. | Unrepaired smooth: 018 in `outputs/paper_eval/ext_jc_visual_refinement/reference_runs_018/ext_jc_w10_surrogate_staged/smooth/`; 019–110 in `.../runs_multiseq/ext_jc_w10_surrogate_staged/smooth/`. Repaired: `outputs/paper_eval/ext_jc_visual_refinement/postprocess/ext_jc_w10_surrogate/`. |
| E | EXT-JC w10 | J4 plus `L_JC = mean_t (1 - s1 s2)` on source joint-carry frames, weight 10. This is the coupled objective. | Unrepaired smooth: 018 in `reference_runs_018/ext_jc_w10_staged/smooth/`; 019–110 in `runs_multiseq/ext_jc_w10_staged/smooth/`. |
| F | EXT-JC w10 + smoothing repair | E, then the frozen clearance smoothing recipe, then `scripts/ext_jc_postprocess.py`, which blends waist/arm joints back toward the raw solve where smoothing drives the hand into the object. The loss is not changed. The same repair is applied to D. | `outputs/paper_eval/ext_jc_visual_refinement/postprocess/ext_jc_w10/`. Metrics `postprocess/postprocess_metrics.csv`. Demos `postprocess/demo_{018,020}_ext_jc_repaired.mp4`. |

**Which result to call the final Join research result:** F for the coupled method that should appear as EXT-JC in the paper, with D repaired by the same script as the matched control, C as the strong independent baseline, and B as the frozen visual Join base. A is the historical IK baseline and the J0 ablation, not the strong baseline.

The unrepaired EXT-JC table in `paper_ready_results_table.csv` is an earlier aggregate. Its staged EXT-JC row has **n = 5** because 018 w10 staged was produced later. Do not replace `postprocess_metrics.csv` with that row.

Controlled cold-start J0–J4 on 018 (`outputs/paper_eval_join/ablation_018/`) is a component ablation. It is not frozen v41 and not EXT-JC. `outputs/paper_release_join_v2/tables/FINAL_REPORT.md` says this explicitly: the accepted v41/018 demo was warm-started from v40 and must stay a separate reference.

## Repository navigation

```text
src/core4d_dual_retarget/          own Join code
  cli.py                           every command-line entry
  contact.py                       phase and source contact
  retarget/independent.py          A
  retarget/coupled.py              B–F objective
  retarget/paper_presets.py        J0–J4; J4 is the v41 component stack
  retarget/smooth.py               clearance smoothing recipe
  metrics.py                       evaluate-paper-join
  coordination_eval.py             simultaneous-contact spans
  coordinates.py                   CORE4D y-up to robot z-up
scripts/ext_jc_postprocess.py      F, the repair
scripts/oa_*.py                    separability and OA evaluation
third_party/unitree_ros/           G1 URDF and meshes; no LICENSE file in this checkout
data/core4d/                       CORE4D motions and object meshes; not in git
outputs/balanced_clearance_v41*_   frozen B; do not modify
outputs/paper_eval/object_aware_baseline_extension/   C, D, E for 019–110 and the OA report
outputs/paper_eval/ext_jc_visual_refinement/          018 of D and E, repair F, math audit
outputs/paper_eval_join/            J0–J4 ablation and the v41 re-evaluation that fed the release
outputs/paper_release_join_v2/      copied v41 paper bundle (tables identical to paper_release_join)
videos/retarget/*_clearance_v41.mp4 frozen B videos
```

Obsolete and not part of the release: `outputs/balanced_clearance_v9` through `v40` except as **initialization inputs** (`v40_smooth` for 018 of C/D/E, `v20_smooth` for 019–110), `outputs/paper_eval/ext_jc_visual_refinement/runs_dev/` (stopped sweep; not selected), `outputs/controller_probe/`, and the historical `videos/retarget/*_clearance_v2.mp4` … `v40.mp4`.

## Reproduction entry points

Details, blockers, and the three levels are in `REPRODUCE_JOIN.md`.

```bash
# Environment that was actually used. This is not a fresh-machine test.
uv venv .venv
uv pip install --python .venv/bin/python -e ".[dev,torch,visualization,simulation]"

# B metrics from frozen files (needs CORE4D data and the G1 URDF)
.venv/bin/core4d-retarget evaluate-paper-join 20231108/018 \
  --result-root outputs/balanced_clearance_v41_smooth \
  --output-path /tmp/v41_018.json

# F is already on disk. Do not rerun the optimizer to "refresh" it.
# Repair command, only if the repaired npz are missing:
.venv/bin/python scripts/ext_jc_postprocess.py --sequence 20231108/020 \
  --raw-root outputs/paper_eval/object_aware_baseline_extension/runs_multiseq/ext_jc_w10_staged/raw \
  --smooth-root outputs/paper_eval/object_aware_baseline_extension/runs_multiseq/ext_jc_w10_staged/smooth \
  --out-root outputs/paper_eval/ext_jc_visual_refinement/postprocess/ext_jc_w10
```

`retarget-paper-join` refuses to overwrite an existing result directory.

## Mathematical status

`outputs/paper_eval/ext_jc_visual_refinement/FINAL_MATH_AUDIT.md` is a completed read-only audit (2026-10-28). It verified the implemented product, the anchor map `R a + t`, and the numerical soft-min behaviour on the frozen 018/020/109/110 trajectories.

It also found issues the manuscript must not paper over:

- The soft-min sits about 3.5 cm below the true hull minimum at N = 1024 and β = 0.01 m, so `s = 0.5` is about 7 cm of hull clearance, not the 4 cm evaluator threshold.
- `s` is not the evaluator contact mask.
- The contact residual that J4 actually optimizes is the palmar sample minus `(R a + t)`, after the y-up to z-up change of basis. The unparenthesized expression `p - R a + t` differs by `2 t`.
- Deep penetration increases `s`, but on these trajectories the extra gain inside the hull is about 1e-5 per 2 cm. The surface-aware clamp was not used.
- `L_JC` has no temporal derivative. Long contact spans are an evaluator outcome.

## License status

No project license has been chosen. Do not add MIT, Apache-2.0, or any other license in the integration step until the obligations in `THIRD_PARTY_ASSETS_JOIN.md` are resolved. Human SMPL-X parameters in `person*_poses.npz` and the CORE4D meshes are not cleared for upload.

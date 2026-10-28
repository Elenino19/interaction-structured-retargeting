# Known limitations

## Kinematic, not a controller

Every Join trajectory is a sequence of joint angles and a free root. `docs/paper_claims.md` records that free-base joint PD falls (about 95% of frames on the clips that were simulated) and that contact-space inverse dynamics does not yield a stance equilibrium. Do not describe method B or F as dynamically stable or as a sim-to-real result.

Foot–object penetration on frozen v41 remains large on some takes (the v2 aggregate reports foot penetration max mean 5.0 cm, max 10.8 cm). Hand penetration was the quantity the EXT-JC repair targeted. Feet were not.

Planted-foot speed is still high after smoothing (v41 aggregate mean about 21 cm/s). The repair does not retune the gait.

## Contact fidelity

The quantity `L_JC` optimizes is a soft hull score, not the metric in the tables.

- Soft-min bias is about 3.5 cm at the frozen sample count. `s = 0.5` means roughly 7 cm of hull clearance.
- The evaluator turns contact on with a 4 cm / 5.5 cm hysteresis on distance to mesh vertices.
- On 018/020/109/110, whenever the two disagree, the soft score is on and the evaluator is off.
- Inside the hull the soft score still rises, by a negligible amount once the hand is already near the surface. Raw EXT-JC is outside the hull except one frame of 109 robot 1. Smoothing, not the product term, creates the visible penetration. The repair puts most of it back under 0.05 cm. 109 robot 1 remains at 0.57 cm and the rule was not changed.

Bilateral anchor error on the repaired takes is still on the order of 10 cm (`postprocess_metrics.csv`, `bilateral_max_error_cm`). Simultaneous contact improved relative to v41. It is not a perfect grasp.

018 repaired EXT-JC valid coverage (0.883) does not beat the repaired surrogate (0.967). That was already true before the repair. The dev mean still favours repaired EXT-JC (0.940 vs 0.869) because of 019 and 108. Say so if the paper claims a uniform win.

109 repaired longest valid run is 13 frames for EXT-JC and 16 for the surrogate. 018 and 109 are the takes where the surrogate is ahead on that span. 020 is 36 vs 13.

## Post-processing and version boundaries

| Claim | Use these files | Do not use |
|---|---|---|
| Frozen visual Join, method B | `videos/retarget/*_clearance_v41.mp4`, `outputs/balanced_clearance_v41_smooth/` | Any `*_clearance_vN.mp4` for N < 41. `outputs/balanced_smooth` (older gate, Contact F1 0.98). |
| v41 paper numbers under the rubber-hand contract | `outputs/paper_release_join_v2/tables/join_six_sequence_aggregate.md` (contact F1 mean 0.827) | README's "mean Contact F1 is 0.83" is the same family, rounded. The older `docs/paper_table.md` F1 of 0.984 is a different release. |
| Coupled method, method F | `postprocess/ext_jc_w10/`, `postprocess_metrics.csv` label `ext_repaired`, demos `demo_*_ext_jc_repaired.mp4` | Unrepaired EXT renders. v41 footage relabeled as EXT-JC. `paper_ready_results_table.csv` staged EXT row (n = 5, unrepaired, includes penetration from smoothing). |
| Strong baseline, method C | `oaind_v41_staged` | Method A, `outputs/independent` |
| Matched control, method D | `postprocess/ext_jc_w10_surrogate/` | A surrogate that was not passed through `ext_jc_postprocess.py` |
| Cold ablation J0–J4 | `outputs/paper_eval_join/ablation_018/` | Do not call J4 "frozen v41". The report says v41/018 was warm-started from v40. |

`runs_dev/refined_v1` through `v5` changed the loss (surface-aware score, edge taper, hand jerk, trust region, smoothed barrier). They were not selected. v1b lowered coverage. v2 lost the 020 coordination edge. v3 and v4 did not beat the repaired baseline on the pre-registered comparison. Leave them out of the paper and out of the public tree.

The object track is not smoothed with the robot. On 020, `smooth_objposes` repeats frames and then jumps 12–22 cm. The arm swing at that jump is still in the repaired video (arm jerk unchanged, about 532 and 742 rad/s³). v41 has the same jump (robot 2 arm jerk 659 at frame 71).

## Mathematical uncertainty that remains

The implementation audit is finished. What remains is manuscript text, not an unverified function:

- Parentheses on the anchor residual, and the name palmar sample rather than an unspecified palm point.
- Do not equate τ = 4 cm with the evaluator threshold.
- Do not describe `L_JC` as a temporal-span or derivative loss.
- The code comment that says the surrogate replaces the partner by 1 is wrong. The code averages the two indicators.

## Reproducibility uncertainty

- No git commits. The integration agent cannot pin a revision hash from this repository.
- Level 1 playback of existing mp4 files does not need a new run. Level 2 was not re-executed against a fresh CORE4D download in this handoff. Level 3 depends on v20 and v40 warm starts that are previous optimization outputs.
- `retarget-paper-join` is deterministic given those inputs, but re-running it into the existing directories will be refused. A reproduction must use a new `--output-root`.
- 15 GiB RAM is a real constraint. The earlier six-way sweep was killed by the OOM killer.

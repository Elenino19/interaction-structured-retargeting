# Decision: keep EXT-JC w10 and repair the smoothing step

Selected result: the frozen EXT-JC w10 trajectories (product term, weight 10, unchanged) followed by the existing `smooth-trajectories --clearance-recipe`, then a post-hoc blend of waist and arm joints back toward the raw solve on the few frames where that recipe pushes the rubber-hand samples into the object. The coupled loss, its weight, the raw solves, frozen v41, and the metric thresholds are unchanged.

Selection used 018 (dev) and 020 (val) only. 019 and 108 were repaired with the same fixed rule after that choice. 109 and 110 were repaired and scored once afterwards and were not used to change the rule.

## 1. Diagnosis, checked with the evaluator's own geometry

Same object poses (`smooth_objposes*.npy`), same rubber-hand volume samples, same convex-hull test as `metrics.hull_inside_max_cm`.

| 020, EXT-JC w10 | robot 1 max | robot 2 max |
|---|---|---|
| raw solve | 0.00 cm | 0.00 cm |
| smoothed joints, raw root | 3.17 cm | 4.53 cm |
| raw joints, lifted root | 0.00 cm | 0.00 cm |
| published smooth (recipe) | 3.17 cm | 4.29 cm |

The sole lift does not create the penetration. Joint smoothing does, because the object track is not smoothed with the robot: on 020 the object translation is duplicated on frames 66, 68 and 70 and then jumps 12–22 cm. v41 robot 2 has the same mechanism at a smaller depth (0.00 raw → 1.95 cm after smoothing). 018 raw is also 0.00 cm; the recipe adds at most 0.35 cm.

The soft contact score does not drive the raw solve inside the hull. At a 4 cm depth its extra reward is about 1e-4 in the weighted loss, against a barrier cost of order 1.

## 2. The repair

`scripts/ext_jc_postprocess.py`. For each frame, blend only waist and arm joints from the smoothed pose toward the raw pose by the smallest factor on a 0.05 grid that brings hull depth back to ≤ 0.05 cm, then fade that factor over 3 frames so the correction is not a one-frame pop. Legs, root and the sole lift stay as the recipe left them. The same code is applied to the separable surrogate.

On 020 this uses at most 70% of the raw arm pose (robot 1) and 50% (robot 2), on 14 and 18 frames. Arm-jerk max stays 532 and 741–742 rad/s³: the repair removes penetration and does not remove the arm swing that comes from the object jump. That swing is already present in v41 (659 rad/s³ at frame 71).

## 3. Frozen-evaluator numbers

`postprocess_metrics.csv`. Coverage is valid joint coverage at 0.5 cm. Penetration is the paper hand-penetration max (mean of the two robots). Unrepaired numbers are from `object_aware_baseline_extension/all_variant_metrics.csv` plus the 018 reference runs scored in this task.

| take | split | v41 cov / pen | EXT unrepaired cov / pen | EXT repaired cov / pen | surrogate repaired cov / pen | repaired longest run, EXT vs surrogate |
|---|---|---|---|---|---|---|
| 018 | dev | 0.650 / 0.00 | 0.883 / 0.18 | 0.883 / 0.00 | 0.967 / 0.00 | 37 vs 34 |
| 019 | dev | 0.750 / 0.50 | 0.917 / 0.74 | 0.938 / 0.00 | 0.896 / 0.02 | 20 vs 20 |
| 108 | dev | 0.465 / 0.09 | 0.860 / 1.85 | 1.000 / 0.03 | 0.744 / 0.04 | 43 vs 21 |
| 020 | val | 0.694 / 0.98 | 0.837 / 3.73 | 0.857 / 0.02 | 0.837 / 0.00 | 36 vs 13 |
| 109 | test | 0.463 / 2.16 | 0.722 / 2.10 | 0.796 / 0.29 | 0.759 / 0.14 | 13 vs 16 |
| 110 | test | 0.298 / 1.00 | 0.754 / 0.75 | 0.825 / 0.01 | 0.754 / 0.04 | 29 vs 14 |

Dev mean valid coverage: repaired EXT 0.940, repaired surrogate 0.869. 020 (the visual failure) goes from 3.73 cm to 0.02 cm and keeps a longer valid contact run than the surrogate (36 vs 13 frames). 018 coverage does not beat the surrogate; that was already true before the repair (0.883 vs 0.967) and is one take.

109 robot 1 still reaches 0.57 cm after blending all the way back to the raw arms, because the lifted root plus raw arms is not clearance-free on that take. The rule was not changed to chase it.

## 4. What is essential, and what was stopped

Essential and done:

* Diagnosis ablation and `visual_diagnosis.md`.
* Repair of the already-finished EXT-JC w10 and surrogate trajectories on 018, 019, 020, 108, 109, 110.
* Frozen-metric table `postprocess_metrics.csv`.
* v41-camera renders: `videos/020_v41_vs_ext_repaired.mp4`, `videos/018_v41_vs_ext_repaired.mp4`, the surrogate pairs, `qualitative_keyframes_{018,020}.png`, `frame68_v41_ext_repaired.png`.
* Labeled full demos: `demo_020_ext_jc_repaired.mp4`, `demo_018_ext_jc_repaired.mp4` (960×720, 15 fps, libx264 q8, camera taken from v41).

Not part of the selected result. Four solves were already running when the queues were paused (`refined_v3_surrogate` 019, `refined_v1` 108, `refined_v1_surrogate` 108, `refined_v1b_surrogate` 019). They are left to write their outputs. The queue processes are stopped, so they will not start the remaining v1/v2/v3/v4/v5 jobs. Those jobs can be skipped: v1b lowered coverage, v2 lost the 020 coordination edge, v3 traded the edge for a partial penetration cut, and v4's 018 surrogate already beat v4. None of them is needed once the smoothing repair holds the 020 penetration at 0.02 cm.

## 5. What to show

Primary demo: `demo_020_ext_jc_repaired.mp4`, labeled EXT-JC w10 plus smoothing repair, next to frozen v41 (`videos/retarget/020_clearance_v41.mp4` and `videos/020_v41_vs_ext_repaired.mp4`). Do not present unrepaired EXT footage as the repaired method. Frozen v41 stays the polished base. The research claim stays the coupled product at weight 10; the repair is post-processing, documented above, and applied identically to the surrogate.

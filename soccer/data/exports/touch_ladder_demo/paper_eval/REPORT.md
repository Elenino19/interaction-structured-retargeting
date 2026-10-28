# Soccer paper-stage evaluation report

Date: 2026-10-04  
Frozen sequence count: 1  
Methods: Basic IK, + Support, + Event Semantics, Full

## A. Implemented

New scripts:

- `scripts/evaluate_soccer_paper.py`
  - separates receive, kick, and overall interaction metrics;
  - adds fixed-window timing error;
  - adds broad stance displacement/speed;
  - adds root and 29-DoF velocity, acceleration, and jerk;
  - adds foot-mesh ground penetration and mesh-vertex ball-gap proxies;
  - writes single-sequence, aggregate, motion, and cross-task outputs.
- `scripts/make_soccer_paper_figure.py`
  - creates a publication-safe Basic/Event/Full figure;
  - creates a watermarked local-only figure with source frames.
- `scripts/package_soccer_paper_eval.py`
  - copies frozen ablation videos into `paper_eval/videos/`;
  - records hashes and intentionally excludes the unlicensed source.
- `tests/test_paper_metrics.py`
  - verifies receive/kick separation and finite-difference smoothness.

Existing frozen ablation trajectories and videos were read only. The accepted Full video hash
remains `c904d2ed28760466df36261f35f6fd281049faf9618a1f266c51cad3dd46039e`.

## B. Executed

Sequence:

- `touch_ladder_demo`: 144 frames, 24 fps, 6 receive events and 7 kick events.

Methods:

- `basic_ik`
- `support`
- `event_semantics`
- `full`

No additional sequence was accepted. Licensed local candidates were re-screened for generic
two-person passing; detailed decisions are in `data_search_report.json` and `DATA_SEARCH.md`.

## C. Main numerical results

### Per-sequence summary

```text
method             target_d  surface_err  foot_assoc  recv_t  kick_t  stance_disp  speed_p95  floor_max  root_jerk  joint_jerk  lane_err  cone_clear
                   (m)       (m)          rate        (frame) (frame) (m)          (m/s)      (m)        (m/s^3)    (rad/s^3)   (m)       (m)
basic_ik           0.2087    0.1024       0.7692      0.000   1.000   0.4941       3.111       0.01294    259.63      407.70       0.7038   -0.1321
support            0.2092    0.1026       0.7692      0.000   1.000   0.4898       3.133       0.01310    259.19      507.56       0.7038   -0.1321
event_semantics    0.2009    0.0941       0.7692      0.000   1.286   0.4827       3.244       0.01565    395.38     1086.83       0.7038   -0.1321
full               0.0898    0.0220       1.0000      0.167   0.571   0.4607       3.131       0.01355    784.32     1241.59       0.01545   0.01520
```

Full improves interaction distance, correct-foot association, kick timing, lane error, and cone
clearance. Full also reduces broad stance displacement relative to Basic IK by about 6.8%.
However, root jerk is about 3.0× and joint jerk about 3.0× Basic IK.

### Receive versus kick

```text
method             phase    count  target_d  surface_err  foot_assoc  timing_mean_frames  timing_mean_s
basic_ik           receive  6      0.1378    0.0359       1.0000      0.000               0.0000
basic_ik           kick     7      0.2695    0.1595       0.5714      1.000               0.0417
support            receive  6      0.1385    0.0359       1.0000      0.000               0.0000
support            kick     7      0.2698    0.1598       0.5714      1.000               0.0417
event_semantics    receive  6      0.1295    0.0264       1.0000      0.000               0.0000
event_semantics    kick     7      0.2620    0.1520       0.5714      1.286               0.0536
full               receive  6      0.0867    0.0233       1.0000      0.167               0.0069
full               kick     7      0.0924    0.0208       1.0000      0.571               0.0238
```

The fixed timing rule is the earliest minimum ball-center/target-toe distance inside
`annotation ± 4` frames. It was selected before comparing method outcomes and is applied
identically to every method.

### Motion and mesh geometry

```text
method             root_accel  root_jerk  joint_accel  joint_jerk  mesh_floor_mean  mesh_floor_max  mesh_ball_gap_abs
                   (m/s^2)     (m/s^3)    (rad/s^2)   (rad/s^3)  (m)              (m)             (m)
basic_ik           8.432        259.63     17.820       407.70     0.000624         0.02118         0.07634
support            8.415        259.19     19.420       507.56     0.000613         0.01971         0.07613
event_semantics   11.450        395.38     31.504      1086.83     0.000527         0.02232         0.07274
full              22.187        784.32     35.574      1241.59     0.000713         0.02699         0.04275
```

The mesh metric transforms the actual G1 foot-mesh vertices. Ground penetration is vertex depth
below Z=0. Ball-foot distance uses the nearest mesh vertex minus ball radius, so it is a
mesh-sampled proxy rather than exact triangle distance.

### Aggregate status

There is only one qualifying sequence. Mean and median therefore equal the single-sequence
values; standard deviation is explicitly unavailable. Deterministic reruns were not treated as
independent samples.

### Position-correction isolation

An additional `Full - Position Correction` variant keeps support/event semantics, ball contact
correction, straight lane segments, and canonical cones, but removes pre-IK ladder stationing
and post-IK lateral station alignment.

```text
metric                              Full - Position     Full
target-foot distance (m)            0.1122              0.0898
surface contact error (m)            0.0173              0.0220
correct-foot association             1.0000              1.0000
timing error (frames)                 0.3077              0.3846
lane crossing error (m)               0.2010              0.01545
contact-foot lane error (m)           0.6475              0.2244
minimum cone surface clearance (m)   -0.1286             +0.0152
stance displacement (m)               0.4827              0.4607
root jerk RMS (m/s^3)               395.38              784.32
joint jerk RMS (rad/s^3)           1086.83             1241.59
mesh floor penetration max (m)        0.02232             0.02699
```

Position correction reduces lane-crossing error by about 92.3%, contact-foot lane error by
65.3%, and changes cone overlap into positive clearance. It is therefore necessary for the
current direct-pass/lane geometry. The trade-off is substantial: root jerk almost doubles,
joint jerk rises about 14.2%, and maximum mesh penetration becomes worse. Ball correction
alone is sufficient for correct-foot association but not for globally consistent foot/lane/cone
alignment.

Outputs:

- `paper_eval/position_ablation/summary.csv`
- `paper_eval/position_ablation/metrics.json`
- `paper_eval/position_ablation/comparison.mp4`
- `paper_eval/position_ablation/comparison_frame_f26.png`

## D. Files

Primary data:

- `paper_eval/summary.csv`
- `paper_eval/aggregate.csv`
- `paper_eval/aggregate.json`
- `paper_eval/receive_kick_summary.csv`
- `paper_eval/motion_geometry_summary.csv`
- `paper_eval/cross_task_metrics.json`
- `paper_eval/sequences/touch_ladder_demo/metrics.json`

Figures:

- publication-safe:
  `paper_eval/figures/soccer_ablation_qualitative.png`
- publication-safe PDF:
  `paper_eval/figures/soccer_ablation_qualitative.pdf`
- local review with source frames, not redistributable:
  `paper_eval/figures/soccer_ablation_qualitative_local_only.png`

The middle column is titled `+ Support & Event Semantics`, because it includes both
components. Figure caption:

> Selected interaction frames are shown for qualitative comparison; per-frame annotations
> are not aggregate scores. The event-aware variant includes support constraints.

Videos:

- `paper_eval/videos/comparison_2x2.mp4`
- `paper_eval/videos/basic_ik.mp4`
- `paper_eval/videos/support.mp4`
- `paper_eval/videos/event_semantics.mp4`
- `paper_eval/videos/full.mp4`
- hashes: `paper_eval/videos/video_manifest.json`

Data audit and provenance:

- `paper_eval/data_search_report.json`
- `paper_eval/DATA_SEARCH.md`
- `paper_eval/frozen_inputs.json`
- `paper_eval/task_brief_provenance.json`

All paths above are relative to `data/exports/touch_ladder_demo/`.

## E. Limitations

- Only one sequence passed. No cross-sequence variance or generalization result is available.
- Mixkit 43486 was closest, but the second player median height was 240.68 px and the action is
  aerial juggling rather than the evaluated ground-pass/lane task.
- Mixkit 2927 had a far-player median height of 154.56 px; Mixkit 43517 measured 70.76 px for
  the smaller player. Other local legal candidates were basketball, cropped, distant, or
  multi-player footage.
- Mixkit/Pexels license names are manifest declarations only; no license text, download receipt,
  or source-page snapshot is archived locally, so legal provenance is not independently
  verifiable offline.
- MUVS licensing is internally inconsistent: the audit labels it CC BY 4.0 while the local
  README says `License: TO BE ADDED`; it remains fail-closed.
- SoccerTrack-v2 remains blocked without an authorized `HF_TOKEN`.
- The current source lacks redistribution permission. It is excluded from the packaged videos;
  the source-column figure is watermarked local-only.
- Full has substantially worse root/joint jerk and worse maximum mesh foot penetration than
  Basic IK.
- Timing is measured against manually fixed event annotations but uses prescribed method-specific
  ball paths; it is not independent sensor or motion-capture ground truth.
- Mesh ball distance is vertex-sampled. Exact triangle distance is not implemented.
- Robust self-collision was not reported because the rendering scene filters relevant collision
  pairs and no independent non-adjacent mesh-distance implementation is available.
- Results remain kinematic/event-conditioned geometry, not rigid-body contact physics or real
  robot execution.

## F. Reproduction

From `$HOME/Projects/soccer_retarget`:

```bash
# Main paper metrics, including MuJoCo mesh checks
MUJOCO_GL=egl join/.venv/bin/python \
  scripts/evaluate_soccer_paper.py

# Figures; GVHMR environment provides OpenCV and Matplotlib
$GVHMR_ROOT/.venv/bin/python \
  scripts/make_soccer_paper_figure.py

# Copy frozen ablation videos and write hashes
python3 scripts/package_soccer_paper_eval.py

# Verification
python3 -m py_compile scripts/*.py tests/*.py
python3 -m unittest discover -s tests -v
```

The existing four-way ablation does not need to be rerun. If it must be regenerated separately:

```bash
join/.venv/bin/python \
  scripts/run_ablation_experiments.py --force
python3 scripts/evaluate_ablation_experiments.py
```

## Extension: event-chain consistency and fixed-ball event-IK ablation

Added after this report was accepted; it does not alter any number above. See
`paper_eval/event_coordination_extension/evaluation_report.md` (answers to the six event-coordination questions),
`pipeline_dependency_audit.md`, `metric_definitions.md`, `fixed_ball_ablation.csv`, `event_metrics.csv`,
`transfer_chain_metrics.csv`, `interaction_timeline.{png,pdf}`, `qualitative_comparison.{png,pdf}` and `REPRODUCE.md`.
Headline: the frozen Full output is role-consistent on 13/13 events and 6/6 kick→receive pairs; with the ball and
reference held fixed, event-conditioned IK ON vs OFF changes no assignment, timing or chain metric, improves kick-foot
velocity fidelity, and increases jerk and foot–ball interpenetration. The accepted Full robots are not regenerated
bit-exactly by the current scripts (documented in the audit); they remain the frozen reference.

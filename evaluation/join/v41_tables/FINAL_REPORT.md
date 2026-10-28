# Join Paper Experiments — Final Report

Generated: 2026-10-04  
Paper evaluation root: `outputs/paper_eval_join/`  
Frozen release: `outputs/paper_release_join/`

## 1. Evaluation contract

- No accepted/frozen v41 files were overwritten.
- Controlled J0–J4 use the same independent-IK initialization.
- J1–J4 use 600 Adam iterations and common optimizer/fidelity settings.
- Every row uses identical post-processing: Gaussian σ=1, arm σ=0.35,
  root σ=0, and sole-lift correction.
- Every row uses the same paper metrics: official rubber-hand/foot geometry,
  robot contact hysteresis 0.04/0.055 m, and 15 fps derivatives.
- The accepted v41/018 demo was warm-started from v40. It is reported as a
  separate frozen reference, not substituted for the controlled cold-start J4.
- Object trajectories remain fixed to CORE4D.

## 2. J0–J4 implementation

| Variant | Object anchors | Phase weighting | Partner clearance | Geometry/gait | Put-down support | Source approach + idle dual support |
|---|---:|---:|---:|---:|---:|---:|
| J0 Independent IK | off | off | off | off | off | off |
| J1 + Object Contact | on | off | off | off | off | off |
| J2 + Phase-Aware Coupling | on | on | on | off | off | off |
| J3 + Geometry / Gait | on | on | on | on | on | off |
| J4 Full v41 Components | on | on | on | on | on | on |

Exact resolved configs:

- `outputs/paper_eval_join/summary/join_ablation_configs.json`
- `outputs/paper_eval_join/ablation_018/J{0,1,2,3,4}/config.json`

J1 explicitly disables the hardcoded historical phase boost and partner loss.
J2 restores contact-transition weighting (1.5) and inter-robot clearance.
J3 adds the complete hand/body/foot geometry, support, ground, gait, leg
laterality/self-clearance, and final put-down stack. J4 additionally adds
source-pelvis-guided approach (weight 240), phase-1 slip boost, and low-speed
dual-foot support (0.15 m/s).

## 3. Controlled 018 ablation

Both robots are averaged. Lower is better except Contact F1.

| Variant | Contact F1 | \|Onset\| frames | Palm cm | JPE cm | Root error cm | Joint jerk | Root jerk | Hand pen. cm | Foot pen. cm | Foot p95 cm/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| J0 | 0.946 | 1.5 | 1.574 | 7.019 | 3.598 | 22.689 | 62.846 | 9.108 | 0.000 | 115.413 |
| J1 | **0.965** | **0.5** | **0.951** | 8.744 | 3.882 | 11.051 | 65.025 | 9.358 | 0.000 | 107.105 |
| J2 | **0.965** | **0.5** | **0.951** | 8.743 | 3.880 | **11.023** | 65.216 | 9.373 | 0.000 | 107.032 |
| J3 | 0.740 | 12.5 | 4.524 | 9.748 | 5.450 | 156.832 | **155.768** | **0.000** | 1.240 | 85.345 |
| J4 | 0.728 | 13.5 | 5.000 | 9.610 | 4.900 | 148.569 | 161.450 | **0.000** | 1.406 | **70.259** |

Machine-readable / paper formats:

- `outputs/paper_eval_join/summary/join_ablation_018.csv`
- `outputs/paper_eval_join/summary/join_ablation_018.md`
- `outputs/paper_eval_join/summary/join_ablation_018.tex`

### Ablation interpretation

1. Object anchors (J1) improve F1, timing, palm distance, and joint jerk over J0.
2. J2 is numerically almost identical to J1 on this clip. The phase transition
   weight and partner clearance have little marginal effect under this setup.
3. J3 eliminates maximum hand hull penetration (about 9.37 cm → 0), and
   reduces foot-slide p95, but contact/timing and smoothness become much worse.
4. J4 further lowers foot-slide p95 and root error versus J3, but does not
   recover contact F1 or onset timing.
5. The method is multi-objective: geometry/gait feasibility is not a monotonic
   improvement in interaction metrics.

### Frozen v41/018 reference (not a controlled row)

The visually accepted warm-started demo has:

- average Contact F1: 0.842
- average absolute onset error: 2.5 frames
- average palm distance: 3.750 cm
- average semantic JPE: 13.958 cm
- average joint jerk: 121.201 rad/s³
- average root jerk: 146.632 m/s³
- average hand hull penetration: 0 cm
- average foot hull penetration: 1.019 cm
- average planted-foot p95: 62.091 cm/s

Artifacts:

- `videos/retarget/018_clearance_v41.mp4`
- `outputs/balanced_clearance_v41_smooth/20231108_018/`
- copied paper reference: `outputs/paper_eval_join/final_six/018/`

The difference between cold-start J4 and frozen v41 demonstrates optimizer
initialization/local-minimum sensitivity. It must be disclosed.

## 4. Final six-sequence v41 evaluation

Values are averages of the two robots in each sequence, evaluated with the
same paper metric mode.

| Seq. | Split | Box | F1 | \|Onset\| frames | Palm cm | JPE cm | Joint jerk | Root jerk | Geometry violation cm | Foot p95 cm/s |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 018 | train | Box024 | 0.842 | 2.5 | 3.750 | 13.958 | 121.201 | 146.632 | 1.546 | 62.091 |
| 019 | train | Box024 | 0.922 | 3.0 | 3.332 | 10.632 | 139.193 | 150.502 | 1.708 | 52.243 |
| 020 | val | Box024 | 0.857 | 11.0 | 3.958 | 12.891 | 145.857 | 198.025 | 6.016 | 82.951 |
| 108 | train | Box001 | 0.808 | 11.5 | 4.475 | 11.437 | 251.878 | 156.614 | 6.149 | 78.722 |
| 109 | test | Box001 | 0.810 | 10.0 | 4.321 | 12.839 | 258.415 | 204.127 | 10.555 | 80.290 |
| 110 | test | Box001 | 0.725 | 0.5 | 4.536 | 16.654 | 177.916 | 346.662 | 13.223 | 130.321 |

Across six sequence averages (mean ± population std):

- Contact F1: **0.827 ± 0.060**
- absolute onset error: **6.417 ± 4.504 frames**
- expected-contact palm distance: **4.062 ± 0.429 cm**
- semantic JPE: **13.069 ± 1.929 cm**
- joint jerk: **182.410 ± 54.119 rad/s³**
- root jerk: **200.427 ± 69.174 m/s³**
- combined geometry violation: **6.533 ± 4.272 cm**
- planted-foot p95: **81.103 ± 24.585 cm/s**

Full rows and aggregate statistics:

- `outputs/paper_eval_join/summary/join_sequence_summary.csv`
- `outputs/paper_eval_join/summary/join_six_sequence_aggregate.json`
- `outputs/paper_eval_join/summary/join_six_sequence_aggregate.md`

## 5. Figures and videos

Main-paper figures use the accepted final result rather than the controlled
cold-start J4 row:

- source / independent IK / frozen v41 qualitative comparison:
  `outputs/paper_eval_join/figures/join_final_qualitative_comparison.png`
- frozen v41 phase/contact timeline:
  `outputs/paper_eval_join/figures/join_frozen_v41_contact_timeline.png`

Controlled-ablation supplemental figures:

- source / J0 / J2 / J4 qualitative comparison:
  `outputs/paper_eval_join/figures/join_ablation_qualitative_comparison.png`
- cold-start J4 phase/contact timeline:
  `outputs/paper_eval_join/figures/join_ablation_J4_contact_timeline.png`

Controlled ablation videos:

- `outputs/paper_eval_join/ablation_018/J0/preview.mp4`
- `outputs/paper_eval_join/ablation_018/J1/preview.mp4`
- `outputs/paper_eval_join/ablation_018/J2/preview.mp4`
- `outputs/paper_eval_join/ablation_018/J3/preview.mp4`
- `outputs/paper_eval_join/ablation_018/J4/preview.mp4`

Final six previews:

- `outputs/paper_eval_join/final_six/{018,019,020,108,109,110}/preview.mp4`

The videos are kinematic `mj_forward` replays, not free-base tracking success.

## 6. Reproduction commands

Full clean run under a new root:

```bash
PAPER_ROOT=outputs/paper_eval_join_repro \
  bash scripts/run_join_paper_experiments.sh
```

One controlled variant:

```bash
.venv/bin/python -m core4d_dual_retarget.cli retarget-paper-join \
  20231108/018 --variant J4 \
  --output-root outputs/paper_eval_join_repro/ablation_018 \
  --iterations 600

.venv/bin/python -m core4d_dual_retarget.cli smooth-trajectories \
  20231108/018 --clearance-recipe \
  --result-root outputs/paper_eval_join_repro/ablation_018/J4/raw \
  --output-root outputs/paper_eval_join_repro/ablation_018/J4/smooth

.venv/bin/python -m core4d_dual_retarget.cli evaluate-paper-join \
  20231108/018 \
  --result-root outputs/paper_eval_join_repro/ablation_018/J4/smooth \
  --output-path outputs/paper_eval_join_repro/ablation_018/J4/metrics.json
```

Tables/figures:

```bash
.venv/bin/python -m core4d_dual_retarget.cli plot-paper-join \
  --paper-root outputs/paper_eval_join_repro

.venv/bin/python -m core4d_dual_retarget.cli summarize-paper-join \
  --paper-root outputs/paper_eval_join_repro
```

Freeze:

```bash
.venv/bin/python -m core4d_dual_retarget.cli freeze-paper-join \
  --paper-root outputs/paper_eval_join \
  --output-root outputs/paper_release_join
```

## 7. Limitations and failure cases

- 020 robot 2 joins 19 frames late.
- 108 robot 1 has F1 0.677, joins 22 frames late, and reaches 10.49 cm foot
  hull penetration.
- 109 robot 1 has 16.78 cm foot hull penetration; robot 2 has 4.33 cm hand
  hull penetration.
- 110 has the weakest sequence-average F1 (0.725), severe foot/ground
  violations, and 130.3 cm/s sequence-average planted-foot p95.
- All final sequences exceed the 10 cm/s foot-skating target.
- Cold-start J3/J4 trade contact and smoothness for geometry/gait constraints.
- The accepted v41 demo depends on staged warm-start optimization.
- Root/foot trajectories are kinematic; no stable free-base PD, dynamic
  co-carrying, sim-to-real, or direct real-robot executability is claimed.
- The six clips are six takes of one Join category and two box instances, not
  six action categories.

Detailed per-robot failure notes:

`outputs/paper_eval_join/summary/failure_cases.md`

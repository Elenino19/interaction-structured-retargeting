# Interaction-Structured Motion Retargeting

**Interaction-Structured Motion Retargeting for Multi-Party Humanoid–Object Interactions**

Sole paper author and sole developer of the original code: Buwen Zhang. No affiliation, email, or ORCID is recorded here.

Kinematic reference motion for two Unitree G1 humanoids, with the interaction state made explicit: who is in contact, with which end-effector, and how an object or a ball moves between them.

This repository is a local academic release of two studies that share that question and do not share an optimizer.

| Study | What was optimized | What to call the result |
|---|---|---|
| **Join** | A coupled retargeting objective on six CORE4D carry clips, plus a documented smoothing repair | **EXT-JC w10 + smoothing repair** |
| **Soccer** | Two independent inverse-kinematics problems, then a ball path | **Frozen Full**, a sequential-interaction case study |

Frozen v41 is the visual Join baseline. It is not EXT-JC. Soccer Full is not EXT-JC either: its solvers never share a coupling loss, and the ball is written after both robots are solved.

No project-wide license has been chosen. CORE4D motions, SMPL-X parameters, Unitree meshes, GVHMR weights, and the soccer source video are not in this tree. See [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) and [RELEASE_LICENSE_AUDIT.md](RELEASE_LICENSE_AUDIT.md).

## Motivation

Multi-party interaction is not a pair of independent poses plus a nearby object. On a collaborative carry, both hands have to arrive on the same object at the same time. On a short passing drill, the annotation says which person kicks, which foot receives, and when the ball should be in the lane. We retarget those constraints onto 29-DoF humanoids and evaluate the kinematic references. We do not claim that the robots can execute the motion, that the contacts were discovered automatically, or that the ball obeys impact physics.

## Contributions

1. **EXT-JC**, a cross-agent joint-contact term on top of an object-aware dual-humanoid retargeter. On source frames where both people are carrying, the loss uses the product of two soft contact indicators, `1 - s1 s2`, with weight 10. A separable surrogate replaces that product by `(s1 + s2) / 2` and is the matched control.
2. **A verified post-process**, applied identically to EXT-JC and to the surrogate. Clearance smoothing was pushing the rubber hand into the object. The repair blends waist and arm joints back toward the raw solve on those frames. It does not change the loss.
3. **Baselines that stay distinct.** Independent IK is the historical J0 ablation. OA-Independent, each robot solved alone under the v41 objective, is the strong baseline. Frozen v41 is the visually accepted reference and was warm-started; a cold-start J4 ablation is not a substitute.
4. **A soccer case study** in which a supplied actor, foot, and time annotation is compiled into per-robot targets and a post-IK ball. On this one sequence the frozen result matches the annotation at 13/13 events and 6/6 kick-to-receive role chains. Those rates largely follow from placing the ball on the annotated feet.

## Method overview

```text
Join (six CORE4D clips)
  source fingertips and object mesh
        -> contact phase and palmar anchors
        -> per-robot geometry, gait, and contact terms (J4 / v41 stack)
        -> optional L_JC on joint-carry frames          # EXT-JC only
        -> clearance smoothing
        -> waist/arm repair where smoothing penetrates  # final EXT-JC

Soccer (one 144-frame clip)
  supplied events
        -> robot A IK, alone
        -> robot B IK, alone
        -> ball snapped to the annotated feet and rolled through the lanes
```

The Join object pose is an input. The Soccer ball is not an optimization variable. Neither study produces a dynamically stable controller: free-base joint PD does not stand the Join motions up, and the soccer ball is a prescribed trajectory.

### Join: EXT-JC

Preset J4 is the frozen v41 component stack (Adam, learning rate 0.01, 600 iterations). `joint_contact_continuity_weight` defaults to 0, so J4 alone is not EXT-JC. The inter-robot clearance term in v41 is inactive on the paper takes (the robots stay at least 0.40 m apart), so v41 is additively separable. EXT-JC is the same stack with the product term at weight 10. The cross Hessian of that term is nonzero. The surrogate's cross Hessian is zero.

The anchor actually optimized is a palmar sample minus `R a + t` after the CORE4D y-up to robot z-up change of basis. The soft contact score is not the evaluator's contact mask. At the frozen sample count the soft-min sits about 3.5 cm below the true hull minimum, so `s = 0.5` is about 7 cm of hull clearance, not the 4 cm evaluator threshold. Details are in `evaluation/join/FINAL_MATH_AUDIT.md`.

Final repaired motions are not bundled, because the rendered demos contain CORE4D geometry and the trajectory files are a private bundle. The numbers below are the bundled table.

### Soccer: sequential interaction

Two monocular tracks are placed 2.8 m apart. That placement is a scene choice, not a metric reconstruction. The frozen annotation contains 13 interaction events: 6 receives, 7 kicks, and 6 kick-to-receive transfer chains. The JSON stores these as 7 contact records; the opening record is a kick only, and each later record pairs a receive with a kick. Each G1 is solved by its own Adam problem (accepted run: 340 iterations, learning rate 0.04). Event windows boost the active leg, relax smoothness, and add a kick-velocity term. The loss never reads the other robot or the ball. `snap_ball_to_g1_feet` then writes the ball 0.095 m (kick) or 0.075 m (receive) off the foot and rolls it through lanes `[3, 4, 5, 4, 3, 2]`.

An independent solver that also sees the same frozen ball matches 13/13 and 6/6 and improves surface deviation. A coordination benefit over independent object-aware retargeting is not supported.

The script in this repository does **not** regenerate those joints. Stationing changed after 2026-10-01, and the script's default iteration count is 260. Treat the NPZ files as the paper result and the script as the method.

## Main results

Manuscript figure and table numbers below were checked against Draft v6.2, which is kept outside this tree. The PDF is not part of the proposed commit. Values are the bundled tables, not new measurements.

### Join, Table I (repaired EXT-JC versus repaired surrogate)

Draft v6.2 Table I is the selected repaired comparison. `C0.5` is geometry-valid joint-contact coverage in percent, `Tmax` is the longest valid simultaneous-contact span in frames, and `Dmax` is the mean of the two robots' maximum sampled hand–hull depth in centimetres. The same rows are in `evaluation/join/DECISION.md` and `evaluation/join/postprocess_metrics.csv`.

| Take | C0.5 separable | C0.5 EXT-JC | Tmax separable | Tmax EXT-JC | Dmax EXT-JC |
|---|---:|---:|---:|---:|---:|
| 018 dev | 96.7 | 88.3 | 34 | 37 | 0.00 |
| 019 dev | 89.6 | 93.8 | 20 | 20 | 0.00 |
| 108 dev | 74.4 | 100.0 | 21 | 43 | 0.03 |
| 020 val | 83.7 | 85.7 | 13 | 36 | 0.02 |
| 109 test | 75.9 | 79.6 | 16 | 13 | 0.29 |
| 110 test | 75.4 | 82.5 | 14 | 29 | 0.01 |

The manuscript states three aggregates of this table, excluding 018 from the first two:

- Five additional takes (019, 108, 020, 109, 110): coverage 79.8% to 88.3%, mean longest span 16.8 to 28.2 frames.
- Held-out 109 and 110: coverage 81.1% versus 75.7%.
- Take 020 hand penetration after the shared repair: 0.02 cm, from 3.73 cm before repair.

018 does not beat the surrogate on coverage. 109 does not beat it on span. Those exceptions are in Table I.

Fig. 2 is the 018 v41 versus repaired EXT-JC rendering. It is not in this tree because the frames contain the CORE4D box. Fig. 3 plots the Table I series; the plot image is not bundled, and the numbers are.

Frozen v41 remains the visual baseline (Fig. 2, top). Its six-take contact F1 mean of 0.827 is in `evaluation/join/v41_tables/join_six_sequence_aggregate.md`. Draft v6.2 does not use that F1 as the coupling result. Foot penetration on v41 stays large (mean of maxima 5.019 cm).

### Soccer, Frozen Full (one sequence)

| Check | Value | How to read it |
|---|---|---|
| Active-limb association | 13/13 | Annotated foot is the closest foot at the annotated frame |
| Role-chain consistency | 6/6 | Annotated kick then the next annotated receive, correct robots and feet |
| Centre-gated chain at 0.05 m | 6/6 | Centre distance |
| Mesh-gated chain at 0.05 m | 3/6 | The foot mesh enters the 0.11 m ball |
| Receive / kick timing error | 0.17 / 0.57 frames | Earliest minimum inside ±4 frames, not a force sensor |
| Event IK, ball fixed | kick-foot velocity RMSE about 0.66 → 0.20 m/s | Higher joint jerk (about 585 → 1292 rad/s³) |

Event conditioning changes fidelity and smoothness. It does not change 13/13 or 6/6 when the ball and the reference are held fixed.

## Installation

Python 3.10 is the version that was used for Join. The declared requirement is `>=3.9`. Soccer metric tests need only the standard library and NumPy.

Every command in this file starts at the **repository root** unless the block says otherwise. Do not chain a later block onto a shell that is still inside `join/` or `soccer/`.

```bash
# working directory: repository root
cd join
uv venv .venv
uv pip install --python .venv/bin/python -e ".[dev,torch,visualization,simulation]"
cd ..
```

That lockfile matches one workstation. It is not a fresh-machine proof. Join was run on CPU PyTorch. Rendering uses `MUJOCO_GL=egl`. Soccer IK and rendering import this package; they do not vendor a second copy of the solver.

After returning to the repository root, `join/src` is the package path. `$PWD/join/src` is correct only at the root. Inside `join/`, the same directory is `$PWD/src`.

```bash
# working directory: repository root
export PYTHONPATH="$PWD/join/src:${PYTHONPATH:-}"
export UNITREE_G1_ROOT=/path/to/g1_description
export MUJOCO_GL=egl
```

`configs/assets.env.example` lists every external path.

## Quick start

Soccer Level 1 needs no robot mesh, no prepared human joints, and does not render.

```bash
# working directory: repository root
python3 examples/load_soccer_frozen.py
```

Join trajectories are not in the repository:

```bash
# working directory: repository root
python3 examples/check_join_private_bundle.py
```

Unit tests that do not need data, still from the repository root:

```bash
# working directory: repository root
python3 -m unittest discover -s soccer/tests -v
```

Join tests import the rubber-hand mesh at collection time. Point `UNITREE_G1_ROOT` at a local G1 description first, with `PYTHONPATH` set from the repository root as above. See [REPRODUCE.md](REPRODUCE.md).

## External data and models

| Asset | In this repo? | What to do |
|---|---|---|
| CORE4D motions, SMPL-X coefficients, object meshes | No | Download from the dataset authors after reading the license. Layout: `join/data/README.md` |
| Unitree G1 URDF, MJCF, STL | No | Clone the upstream `g1_29dof_rev_1_0` description and keep its LICENSE. Set `UNITREE_G1_ROOT` |
| GVHMR code and checkpoints, YOLO, SMPL-X model | No | Research / non-commercial or registration terms. Set `GVHMR_ROOT` |
| Soccer source video | No | Permission was not granted. Level 4 cannot be run from this repo |
| `smplx_joints.npz` | No. Pending | Derived SMPL-X joints. Author approval was conditional. Upstream redistribution is unresolved. `docs/smplx_joints_provenance.md` |
| Join mp4 demos | No | They contain the CORE4D box |

Do not replace a missing dataset with a synthetic stand-in and treat the output as a reproduction of the paper.

## Reproduction levels

1. **Replay what is legally in the tree.** Soccer NPZ and the robot-only preview. Join playback waits on a private bundle and is not cleared for public upload.
2. **Recompute metrics from frozen files.** Possible once CORE4D and the G1 description are local. Write to a new directory. The numbers to cite are the CSVs already here.
3. **Rerun optimization.** Documented, not claimed as bit-exact. Join Level 3 depends on earlier v20 and v40 warm starts. Soccer Level 3 from prepared joints will differ from the frozen NPZ. Soccer Level 4 needs the source video, GVHMR, and SMPL-X.

Full commands: [REPRODUCE.md](REPRODUCE.md).

## Demo media

The project page under `docs/` uses the sanitized Soccer encode and three robot-only figures. The scientific preview is:

`soccer/data/exports/touch_ladder_demo/comparison/current_hop_timed_lateral.mp4`

Label it as kinematic reference motion with a prescribed ball, 144 frames, 24 fps. It is not physical football. Join comparison videos are withheld.

## Citation

Draft v6.2 has one paper author, Buwen Zhang, who is also the sole developer of the original code. `CITATION.cff` cites that name. Affiliation, email, ORCID, venue, DOI, and a repository URL are not recorded. There is no `preferred-citation` until the paper is published. See [AUTHORS_APPROVAL.md](AUTHORS_APPROVAL.md).

If a venue requires double-blind review, omit this name from the submitted manuscript and from any public page. The local project page under `docs/` is unpublished.

MIT remains a candidate for original code only. It is not in the tree. `RELEASE_LICENSE_AUDIT.md` lists the checks that remain.

CORE4D, as used by Join:

> Liu, Zhang, Xing, Tang, Yang, Yi. CORE4D: A 4D Human-Object-Human Interaction Dataset for Collaborative Object REarrangement. CVPR 2025.

Robot geometry, when a G1 render is shown: Unitree G1 description (`unitree_ros`).

Soccer human motion, when the method text mentions recovery: GVHMR (Shen et al., ZJU 3D Vision), two independent static-camera runs. There is no football-pitch asset to attribute.

## Known limitations

- Motions are joint angles and a free root. They are not sim-to-real results.
- Join foot penetration on v41 remains on the order of several centimetres. The EXT-JC repair targets the hands.
- `L_JC` has no temporal derivative. Long contact spans are an evaluator outcome.
- The code comment that says the surrogate replaces the partner by 1 is wrong. The code averages the two indicators.
- Soccer is one sequence. The 13 events are dependent samples. There is no cross-sequence standard deviation.
- Cone spacing (0.50 m) and the player gap (2.8 m) are scene choices.
- Both source checkouts had no git commits, so this release cannot cite an upstream revision hash.
- Draft v6.2 is the current manuscript reference and may still change in wording, formatting, citations, and figure numbers. `PAPER_VERSION_SYNC.md` says which of those edits touch the README. Frozen numbers are not rewritten to match a later draft.
- Draft v6.2 Table I matches the repaired Join CSV. Fig. 2 and the source-video half of Fig. 4 are not in this tree.
- One manuscript sentence, bilateral anchor error from 18.28 cm to 8.94 cm on an earlier 018 ladder, is not in the bundled tables. It is left untraced rather than reassigned.

## Layout

```text
join/            CORE4D dual-retarget package, tests, and paper scripts
soccer/          Soccer scripts, tests, and frozen Full artifacts
evaluation/      Bundled Join tables and the math audit
configs/         Asset variables and the method registry
examples/        Level-1 loaders
scripts/         Reserved for release helpers
figures/         Soccer figures cleared for display
docs/            Local project page (not deployed)
```

Handoff notes, with local account paths removed, are in `docs/handoff/`.

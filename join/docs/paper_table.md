# Paper metric table (six `join + box + obs0` clips, 12 robot trajectories)

Release method: **balanced then Gaussian-smooth σ=1.4** (`outputs/balanced_smooth`).

| method | Contact F1 | palm cm | onset err | CoM excess m | ankle MAE m | jerk | keypoint cm | gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| independent | 0.868 | 4.72 | 3.50 | 0.002 | 0.029 | 30.4 | 5.64 | — |
| coupled | 0.980 | 2.13 | 1.00 | 0.005 | 0.035 | 14.3 | 7.99 | 1/6 |
| balanced | 0.982 | 2.13 | 0.75 | 0.000 | 0.011 | 23.1 | 9.62 | 6/6 |
| **smooth σ=1.4** | **0.984** | **2.18** | **0.58** | **0.000** | **0.010** | **9.74** | 9.69 | **6/6** |

Split: train 018/019/108, val 020, test 109/110. No clip appears in two splits.

Holdout objects (kept out of the box split): 2 barrel + 2 board + 2 desk.
Same release stack, **6/6 pass**. Mean F1 0.990, palm 1.69 cm, jerk 9.31, CoM excess 0.

Physics (not in the success table): free-base PD falls within 0.5 s on 018.
Contact ID on six clips: median |τ| 1.61 Nm, 62% tensile foot normals, unilateral
base residual RMS 4320. See `docs/paper_claims.md` items 5–6.

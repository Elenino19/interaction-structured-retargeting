# Reproduction — object‑aware comparison

Run from `$HOME/Projects/soccer_retarget`. Python `join/.venv/bin/python`
(torch CPU, pytorch_kinematics, mujoco). `MUJOCO_GL=egl`. Frozen inputs (`g1/robot_*.npz`, `ball_cones.npz`,
`ball_cones.json`, accepted preview) are read only; re‑running writes only under
`data/exports/touch_ladder_demo/paper_eval/object_aware_comparison/`.

```bash
PY=join/.venv/bin/python
OUT=data/exports/touch_ladder_demo/paper_eval/object_aware_comparison

# Task 1/2 — separability probe (perturb B, re-run every A-side stage, re-snap ball). ~3 s.
$PY scripts/check_objective_separability.py                      # -> $OUT/separability_probe.json

# Task 2 — strong independent object-aware baseline on frozen ball + shared reference. ~1 min incl. renders.
MUJOCO_GL=egl $PY scripts/run_independent_object_aware_baseline.py   # -> $OUT/independent_object_aware/<variant>/

# Task 3 — event-chain metrics for the comparison set (reuses frozen Full + fixed-ball runs).
MUJOCO_GL=egl $PY scripts/evaluate_event_chain.py --out $OUT --entries COMPARISON
#   -> $OUT/event_metrics.csv transfer_chain_metrics.csv event_chain_summary.csv
#      fixed_ball_ablation.csv threshold_sensitivity.csv event_chain_metrics.json
# event_chain_comparison.csv / metric_sensitivity.csv were assembled from the two event_chain_summary.csv files
# (this directory and ../event_coordination_extension) by the inline script recorded in logs/build_comparison.py.

# Integrity
sha256sum data/exports/touch_ladder_demo/g1/robot_a.npz data/exports/touch_ladder_demo/g1/robot_b.npz \
          data/exports/touch_ladder_demo/ball_cones.npz data/exports/touch_ladder_demo/ball_cones.json
python3 -m unittest discover -s tests -v
```

Prerequisite for Task 3 rows `fixed_ball/*`: `../event_coordination_extension/REPRODUCE.md`.

Code changes made for this task (defaults preserve accepted behaviour): `retarget_soccer(..., ball_contact=None)`
optional object term; new scripts `check_objective_separability.py`, `run_independent_object_aware_baseline.py`;
`evaluate_event_chain.py` gained the `independent_object_aware/*` entries and the `COMPARISON` entry set.

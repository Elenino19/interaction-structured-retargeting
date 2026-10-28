# Evaluation tables bundled with this release

These are copied numeric results. They are not a new experiment.

## Join

| File | What it is | What it is not |
|---|---|---|
| `join/postprocess_metrics.csv` | Final repaired EXT-JC (`ext_repaired`) and repaired surrogate | Unrepaired EXT-JC |
| `join/DECISION.md` | Selection note for the smoothing repair | A license to relabel v41 as EXT-JC |
| `join/v41_tables/join_six_sequence_aggregate.md` | Frozen v41 six-take aggregate, Contact F1 mean 0.827 | EXT-JC |
| `join/paper_ready_results_table.csv` | Unrepaired OA / EXT aggregate. The staged EXT row has n = 5 | Method F |
| `join/FINAL_MATH_AUDIT.md` | Read-only audit of the implemented product and anchor map | A claim that the soft score is the evaluator mask |

Cold-start J0–J4 numbers, where present in `v41_tables/`, are an ablation on 018.
They are not frozen v41.

## Soccer

Metric CSV and JSON files live under
`soccer/data/exports/touch_ladder_demo/paper_eval/` and
`experiments/ablation/`. The summary row for Frozen Full is also in
`soccer/public_export/summary.csv`.

13/13 limb assignment and 6/6 role chains are annotation-consistent kinematic
checks on one sequence. They are not a coupled-optimization score and not a
physical task-success rate.

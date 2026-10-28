# Submission package (frozen Soccer Full; IK not rerun)

All new files live here. Frozen robots, ball, annotations, and the accepted preview were read only.

## Figures (PNG + PDF, 300 dpi, PDF text as Type 42)

| File | Use in paper |
|---|---|
| `fig_robot_events.{png,pdf}` | Publication-safe qualitative strip (no source frames) |
| `fig_interaction_timeline.{png,pdf}` | Interaction-state timeline |
| `fig_results_reuse.{png,pdf}` | Reused quantitative card |
| `local_only/fig_source_to_robot.{png,pdf}` | Source-to-robot 2×6; **needs source permission** |
| `local_only/fig_source_to_robot_watermarked.{png,pdf}` | Same, with a local-review banner |

Captions: `paper_description.md`. Chinese briefing: `CHINESE_SUMMARY.md`. Event table: `event_metadata.json`.

## Videos (paths only; Full preview not copied)

See `video_paths.json`. The source|robot hstack is local-only and must not be packaged for distribution.

## Reproduce figures

```bash
join/.venv/bin/python \
  scripts/make_soccer_submission_figures.py
```

Requires ffmpeg. Does not call the IK solver.

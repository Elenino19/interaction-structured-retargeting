# Join smoke tests

Executed on 2026-10-28 in `$ISR_JOIN_ROOT` with the existing `.venv`. No optimizer was started. No frozen npz, metric JSON, or video was written.

## Passed

### Frozen v41 identity

```bash
md5sum outputs/balanced_clearance_v41_smooth/20231108_018/robot1.npz \
       outputs/balanced_clearance_v41_smooth/20231108_018/robot2.npz
```

```text
bdcf43e656cc3df846f12baa740a6a1f  robot1.npz
6ec5e5d65a0226a69004f1db139b9002  robot2.npz
```

These match the hashes recorded in `outputs/paper_eval/object_aware_baseline_extension/final_research_conclusions.md`.

### J4 preset does not silently enable EXT-JC

```bash
.venv/bin/python -c "
from core4d_dual_retarget.retarget.paper_presets import paper_join_config
c = paper_join_config('J4', iterations=600)
assert c.joint_contact_continuity_weight == 0.0
assert c.palmar_contact is True
assert c.inter_robot_weight == 8.0
print('J4 preset ok', c.iterations, c.hand_volume_clearance_weight, c.palm_offset_m)
"
```

```text
J4 preset ok 600 1200.0 0.052
```

### Extension unit tests

```bash
.venv/bin/python -m pytest tests/test_oa_extension.py -q --tb=line
```

```text
8 passed, 1 warning in 2.27s
```

The warning is a PyTorch note inside `test_surface_aware_indicator_has_no_gradient_inside_hull`. The test passed. The surface-aware indicator is not part of frozen EXT-JC w10.

### Manifest

`FINAL_ARTIFACT_MANIFEST.csv` was built by hashing the publication set. 181 rows, 0 missing paths.

## Not run, and not claimed

- A clean `uv venv` in an empty directory.
- `evaluate-paper-join` or `evaluate-coordination` (they need the CORE4D meshes and would only repeat numbers already in the CSV).
- Any `retarget-paper-join` solve, including a 1-iteration smoke solve. The CLI refuses to overwrite the frozen directories, and a new solve was not required to audit the files.
- MuJoCo re-render of the demos. The mp4 files are on disk; EGL was not exercised in this handoff.
- The full `tests/` suite. Only `tests/test_oa_extension.py` was run.
- License retrieval from GitHub or Hugging Face. The local tree has no Unitree license file and no CORE4D dataset license next to `data/core4d`.

## Blocked for a public smoke test

Level 1 playback of the mp4 files works locally and was not given a fresh viewer check in this session. Publishing those mp4 files is blocked by the CORE4D object geometry inside them, not by a missing file.

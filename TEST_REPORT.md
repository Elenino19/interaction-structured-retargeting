# Test report

Date: 2026-10-28. Local release tree only. No optimizer was started. No file in the Join or Soccer source checkouts was written. Outputs that had to be created went to `/tmp`.

Manuscript Draft v6.2 was checked from a copy kept outside this tree. Figure and table IDs below are that PDF's, not guesses.

## Environment

- OS: Linux 6.8.0-138-generic
- Soccer unit tests: system `python3` (3.10) plus NumPy already installed
- Join tests, metric recompute, and Soccer replay: the existing Join virtualenv (CPU PyTorch, MuJoCo, NumPy). Not a clean install.
- `PYTHONPATH` pointed at `join/src` in this release, confirmed by printing `core4d_dual_retarget.__file__`
- `UNITREE_G1_ROOT` pointed at a local G1 description that is **not** copied into this tree
- `MUJOCO_GL=egl` for the replay and the Join metric command

## Results

### T1. Soccer unit tests

- Command: `python3 -m unittest discover -s soccer/tests -v` from the repository root
- Input: synthetic arrays inside the tests. Frozen NPZ not loaded.
- Expected: 8 tests OK
- Actual, re-run in the final pass: `Ran 8 tests in 0.001s` / `OK`
- Status: **PASS**

### T2. Soccer script compile

- Command: `python3 -m py_compile soccer/scripts/*.py`
- Expected: exit 0
- Actual: exit 0
- Status: **PASS**

### T3. Soccer frozen identity and shapes

- Command: `python3 examples/load_soccer_frozen.py`
- Input: bundled NPZ, JSON, and the accepted preview
- Expected: five SHA-256 values from the Soccer handoff; joints `(144, 29)`; 7 contact records expanding to 6 receives, 7 kicks, and 6 transfers; `smplx_joints.npz` absent
- Actual, final pass: all five hashes matched; shapes `(144, 29)` and ball `(144, 3)`; `contact_records 7 receives 6 kicks 7 transfers 6`; `EXCLUDED smplx_joints.npz`; `PASS soccer frozen load`; exit 0
- Status: **PASS**

### T4. Display-encode identity

- Command: `sha256sum docs/media/soccer_full_demo_web.mp4`
- Expected: `3d5fcef1a8500c2a591f6a5a51661f01a1b482bb9cc95b4e6e0f8482af6c7801`
- Actual: that digest
- Status: **PASS**

### T5. Frozen-write guard

- Command: `python scripts/retarget_dual_g1_soccer.py --iterations 1` from `soccer/`, using the Join virtualenv
- Input: default paths, which point at the bundled Full directory
- Expected: exit before any solve, no change to bundled NPZ
- Actual: `Refusing to write inside the bundled Soccer result tree` naming `ball_cones.npz`, exit 1. No `retarget robot` log line. T3 hashes were checked before this and the guard does not open the NPZ for writing.
- Status: **PASS**

### T6. Join unit tests with an external G1 mesh

- Command: `pytest join/tests -q --tb=line -k "not test_refine_clip_equilibrium_two_frames"`
- Working directory: the Join source checkout, so tests that build a MuJoCo scene can read CORE4D box meshes and the G1 XML. Code imported from this release.
- Excluded on purpose: `test_refine_clip_equilibrium_two_frames`, which runs a 2-frame equilibrium optimization.
- Expected: the handoff's extension tests pass; asset-backed scene tests pass when the mesh is available
- Actual: `109 passed, 1 deselected, 1 warning in 4.65s`. The warning is the known PyTorch note inside `test_surface_aware_indicator_has_no_gradient_inside_hull`. That indicator is not part of frozen EXT-JC w10.
- Status: **PASS**

### T7. Join tests with no G1 mesh

- Command: the same pytest without `UNITREE_G1_ROOT`, and without `join/third_party/`
- Expected: collection cannot load the rubber-hand STL
- Actual: `ValueError: string is not a file: .../join/third_party/unitree_ros/.../left_rubber_hand.STL` on all three test modules
- Status: **BLOCKED** without the external description. This is the public-tree condition.

### T8. J4 does not enable EXT-JC

- Covered by `test_paper_presets_are_cumulative_and_isolate_final_v41_terms` inside T6
- Expected: `joint_contact_continuity_weight == 0` for J4
- Actual: that test passed
- Status: **PASS**

### T9. Join private bundle absent

- Command: `python3 examples/check_join_private_bundle.py` with `ISR_JOIN_PRIVATE_ROOT` unset
- Expected: exit 2
- Actual: `BLOCKED ISR_JOIN_PRIVATE_ROOT is unset.`, exit 2
- Status: **BLOCKED** (expected). Join motions are not in the release.

### T10. Level 2, one frozen v41 take

- Command: `python -m core4d_dual_retarget.cli evaluate-paper-join 20231108/018 --result-root <private v41 smooth root> --output-path /tmp/join_repro/v41_018.json`
- Input: read-only frozen smooth NPZ in the Join checkout, local CORE4D data, external G1 URDF. Nothing under the frozen root was the output path.
- Expected: a JSON file whose contact F1 matches `evaluation/join/v41_tables/join_sequence_summary.csv` for 018
- Actual: exit 0. Robot 1 contact F1 `0.8301886792452831`. Robot 2 contact F1 `0.8535031847133758`. Both match the bundled per-robot summary rows. URDF parser printed the known `dont_collapse` / `mujoco` tag warnings.
- Status: **PASS** for take 018 only. The other five takes and repaired EXT-JC were not recomputed.

### T11. Soccer Level 1 replay

- Command: `render_g1_soccer.py --g1-dir /tmp/soccer_replay --ball-cones <bundled ball_cones.npz> --xml "$UNITREE_G1_ROOT/g1_29dof_rev_1_0.xml" --out /tmp/soccer_full_replay.mp4`
- Input: copies of the two frozen robot NPZ in `/tmp`, bundled ball, external G1 MJCF
- Expected: an mp4. Byte identity with `c904d2ed…` was not claimed by the handoff.
- Actual: `wrote /tmp/soccer_full_replay.mp4` (2,010,295 bytes). Canonical preview is 2,009,964 bytes. Not compared frame by frame.
- Status: **PASS** as a functional replay. **Not** a bit-exact video match.

### T12. Level 3 optimization

- Not run. Join warm starts and a from-scratch CORE4D download were not treated as available inside this repository. Soccer IK was refused by T5 rather than executed.
- Status: **BLOCKED** as a reproduction claim. The commands are documented in `REPRODUCE.md`.

### T13. Manuscript cross-check

- Draft v6.2 was read outside this tree. It was not copied into the release.
- Title matches. Table I matches `evaluation/join/DECISION.md` cell for cell.
- Abstract aggregates of the five takes excluding 018 are 79.8 to 88.3 and 16.8 to 28.2. Held-out coverage is printed as 81.1 versus 75.7; the mean of the printed one-decimal cells is 81.05 versus 75.65.
- Fig. 2 and the source half of Fig. 4 are absent on purpose.
- Bilateral anchor error 18.28 to 8.94 cm was not found in the bundled tables.
- Status: **PASS** for Table I and the soccer 13/6/7/6 wording. **UNTRACED** for the 18.28 cm sentence. **WITHHELD** for Fig. 2 and Fig. 4 source frames.

### T14. Final-pass commands, 2026-10-28

All of these start at the repository root. No optimizer was started. T10 and T11 were not repeated.

- `python3 examples/load_soccer_frozen.py`: exit 0, as T3.
- `python3 examples/check_join_private_bundle.py` with `ISR_JOIN_PRIVATE_ROOT` unset: `BLOCKED ISR_JOIN_PRIVATE_ROOT is unset.`, exit 2.
- `python3 -m unittest discover -s soccer/tests -v`: 8 tests, OK.
- `python3 -m py_compile soccer/scripts/*.py examples/load_soccer_frozen.py examples/check_join_private_bundle.py`: exit 0.
- `PYTHONPATH="$PWD/join/src"` then `import core4d_dual_retarget`: resolved to `join/src/core4d_dual_retarget/__init__.py`. `join/join/src` does not exist.
- `ISR_SOCCER_JOINTS` unset: `BLOCKED: prepared joints are not in this tree`.
- Text search for the account name, `/home/…` absolute home paths, and the manuscript download id: no matches.
- This pass did not write into the Join or Soccer source checkouts.

### T15. Metadata pass, 2026-10-28

No optimizer was started. The held joint file was hashed outside the tree and was not copied back.

- Held `smplx_joints.npz`: 391214 bytes, SHA-256 `f03a8c5bb6e3e658c92b6ac7dc3915754100c07ef5cbfb010fc149cdb0ae3b4b`, arrays `(144, 127, 3)`. It remains absent from this tree.
- `python3 examples/load_soccer_frozen.py`: exit 0. Five frozen hashes matched. Printed `PENDING smplx_joints.npz` and `receives 6 kicks 7 transfers 6`.
- `python3 examples/check_join_private_bundle.py`: exit 2, `BLOCKED ISR_JOIN_PRIVATE_ROOT is unset.`
- `python3 -m unittest discover -s soccer/tests -v`: 8 tests, OK.
- `python3 -m py_compile soccer/scripts/*.py examples/*.py`: exit 0.
- `PYTHONPATH="$PWD/join/src"` imports `core4d_dual_retarget` from this tree.
- `CITATION.cff` parses as YAML. No `LICENSE` file. This directory is not a git repository.

## Not claimed

- A clean `uv venv` install
- Bit-exact regeneration of frozen v41, repaired EXT-JC, or Soccer Full joints
- Public playback of Join mp4 files

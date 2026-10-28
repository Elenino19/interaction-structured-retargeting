# Release security audit

Scope: the local tree under this release directory after integration. The Join and Soccer source checkouts were read and were not modified. Git was not initialized. Nothing was pushed.

## Secrets

A text search of the release tree found no private keys, no filled tokens, and no PEM private-key blocks. The final pass repeated a text search for the workstation account name, absolute home-directory paths, and the manuscript download id. None of those strings remain in the release text files.

`HF_TOKEN` appears only as the name of an environment variable:

- `soccer/scripts/discover_soccertrack.py` reads it at runtime and exits if it is missing
- handoff and data-search notes say SoccerTrack-v2 is gated

No token value is stored. That script is not part of the paper result. Do not run it without a token you are allowed to use, and do not write its output into the frozen Soccer tree.

## Local paths and account names

Soccer scripts no longer contain absolute home-directory paths. Asset locations go through `soccer/scripts/isr_assets.py` (`UNITREE_G1_ROOT`, `GVHMR_ROOT`, `YOLO_WEIGHTS`, `SOCCER_SOURCE_VIDEO`, `SOCCER_GVHMR_RESULTS`).

Copied notes and experiment JSON files were rewritten so account-specific absolute paths become `soccer/…`, `join/…`, `$GVHMR_ROOT`, `$SOCCER_SOURCE_VIDEO`, or `$HOME`. A mention of the workstation account name in the Join third-party note was rephrased.

Frozen NPZ files were scanned as NumPy archives. String-like arrays did not contain the account name or the source-video filename. The NPZ bytes were not edited. Their SHA-256 values match the handoff.

Generated MuJoCo scene XML was not copied. Those files embed an absolute `meshdir`. The renderer builds a new scene in the directory you pass as `--g1-dir`.

## Data that could identify people

The human soccer video, player crops, and source-to-robot figures were not copied. `local_only` files and `*_local_only*` figures were excluded.

`smplx_joints.npz` is not in this tree. Redistribution is **pending**: the release author approved inclusion only if the upstream terms allow it, and the SMPL-X model license does not clearly allow offering the derived joints to a third party. Provenance, the inspected SHA-256, and the role of the file are in `docs/smplx_joints_provenance.md`. `.gitignore` still lists the path. The manuscript PDF stays outside this tree.

Join `config.json` files, which store absolute command lines, were not copied. Term logs were not copied.

## Write protection

`soccer/scripts/frozen_guard.py` refuses output paths inside `soccer/data/exports/touch_ladder_demo/` unless `ISR_ALLOW_FROZEN_OVERWRITE=1`. The default retarget command hits that guard before `np.load` and before Adam. Do not set the override on this tree.

Join `retarget-paper-join` still refuses to overwrite an existing result directory. `run_join_paper_experiments.sh` exits if its paper root already exists.

## What this audit did not do

- It did not fuzz the solvers.
- It did not review PyPI wheel provenance beyond the lockfile already in `join/`.
- It did not decide a project license.
- A later `git add -f` of `join/third_party`, `join/data/core4d`, or `soccer/data/clips` would undo the exclusions. `.gitignore` lists those paths. It is not a substitute for a license decision.

# Release decision

Local integration is complete. This directory is a reviewable academic tree. It is not a publication.

`PUBLIC_RELEASE_MANIFEST.csv` lists every other file in this tree and does not hash itself.

## Ready now

- Join package, tests, paper scripts, and the numeric tables that distinguish frozen v41, unrepaired EXT-JC, repaired EXT-JC, OA-Independent, and the separable surrogate.
- Soccer method, tests, Frozen Full trajectories, frozen events, robot-only media, and metric tables.
- Portable asset variables and a guard that refuses to overwrite bundled Soccer results.
- README, reproduction levels, third-party note, a software citation naming Buwen Zhang, and a local project page under `docs/`.
- Smoke tests in `TEST_REPORT.md`: Soccer unit tests, Join unit tests when a G1 mesh is supplied, Soccer hash check, one-take v41 metric recompute, and a Soccer replay to `/tmp`.

## Not ready

- A project `LICENSE`. MIT is only a candidate. Buwen Zhang is the sole paper author and the sole developer of the original code. Third-party materials stay under their own licenses. Do not add `LICENSE` until MIT is explicitly approved.
- Affiliation, email, ORCID, venue, DOI, paper URL, and repository URL. The checked PDF still prints an anonymous byline and was not edited.
- Fig. 2 and the source-video half of Fig. 4. Table I itself is mapped. Draft v6.2 may still change; `PAPER_VERSION_SYNC.md` says what to update.
- The 18.28 cm to 8.94 cm bilateral-anchor sentence. It is not in the bundled tables.
- Public redistribution of CORE4D data, G1 meshes, GVHMR weights, the soccer source video, Join demo videos, or `smplx_joints.npz`. The joint file is **pending**, not cleared.
- A claim that Level 3 reproduces frozen v41, repaired EXT-JC, or Soccer Full bit for bit.
- Git history. Neither source checkout had a commit, and this tree was not initialized.

## Approvals still required

Do not treat silence as a yes.

1. Project license. MIT is preferred for original code only. Do not add it until ownership, other contributors, institutional duties, and third-party compatibility are confirmed.
2. Explicit approval to add MIT. Sole authorship of the paper and of the original code is already recorded. Do not add affiliation, email, or ORCID unless they are supplied.
3. Venue, DOI, paper URL, and repository URL. Add `preferred-citation` only after publication. If the venue is double-blind, keep the submitted manuscript anonymized.
4. `smplx_joints.npz`: author approval is conditional and recorded. Upstream redistribution remains unresolved, so the file stays out. Written permission from Max-Planck, or a counsel determination that the joint coordinates are outside the no-distribution clause, would reopen this item.
5. Whether CORE4D motions, object meshes, G1 meshes, GVHMR weights, the source video, or Join demo videos may ever be vendored. Current answer used here: no.
6. Whether the source-video half of Fig. 4 may be published. Current answer used here: no.
7. Explicit approval to initialize Git and upload a private repository. This review does not give that approval.

## Private GitHub staging

Author metadata for a private repository is in place: Buwen Zhang is the sole paper author and the sole developer of the original code. Git has not been initialized. A private repository named `interaction-structured-retargeting` is the intended target. Do not create it, and do not commit, until the staged snapshot is explicitly approved. Visibility must stay private. GitHub Pages stays off.

Leave `smplx_joints.npz` out. Do not add a `LICENSE` until the ownership checks pass. Do not add `join/data/core4d`, `join/third_party`, Join mp4 files, GVHMR weights, or the source video. Do not enable GitHub Pages. If the venue is double-blind, do not publish the page in `docs/` with the author name.

Public release is **not** approved.

## Scientific boundaries preserved

- Final Join method: EXT-JC w10 plus the smoothing repair. Numbers: `evaluation/join/postprocess_metrics.csv` and `DECISION.md`.
- Frozen v41: visual baseline only. Six-take contact F1 mean 0.827.
- Cold-start J4 and unrepaired EXT-JC (n = 5) are labeled as such.
- Soccer: independent event-conditioned IK and a post-IK ball. No cross-agent joint optimization.
- Current Soccer IK does not regenerate the frozen NPZ.

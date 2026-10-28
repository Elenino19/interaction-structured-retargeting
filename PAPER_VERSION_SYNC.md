# Manuscript version sync

Draft v6.2 is the current manuscript reference. It is not frozen. Later drafts may change wording, formatting, citations, and figure numbers. Frozen experiments and the numerical tables already in this tree are not recomputed to follow those edits.

The PDF is kept outside this repository. It is not part of the proposed commit.

## Current reference

| Item | Value |
|---|---|
| Manuscript | Draft v6.2 |
| Title | Interaction-Structured Motion Retargeting for Multi-Party Humanoid–Object Interactions |
| Paper author | Buwen Zhang, sole author. Confirmed for this release. |
| Authors printed on the checked PDF | Anonymous. That file was not edited. |
| Software author | Buwen Zhang, sole developer of the original code, cited in `CITATION.cff`. |
| Join result table | Draft v6.2 Table I, matched to `evaluation/join/DECISION.md` and `evaluation/join/postprocess_metrics.csv` |
| Join method in the paper | EXT-JC at weight 10, then the shared smoothing repair. Frozen v41 is the visual baseline, not that comparison. |
| Soccer description | One 144-frame, 24 fps sequence. 13 annotated events: 6 receives, 7 kicks, 6 transfer chains. |
| Claim map | `PAPER_CLAIM_TRACEABILITY.csv` |

## What requires a README update

Update `README.md`, and `docs/index.html` when the page quotes the same sentence, if a later draft changes any of these:

- Title or the sole paper author
- A double-blind submission copy, which must omit the author name even when this repository keeps it
- A number printed in the README, including Table I cells and the five-take or held-out aggregates
- The Soccer event counts or the statement that the two robots are not jointly optimized
- The identity of frozen v41 versus repaired EXT-JC
- A figure or table number that the README cites

A formatting or citation edit that does not change those sentences does not require a README edit. Record the new draft label in this file either way.

## What requires new verification

Check the new PDF against `PAPER_CLAIM_TRACEABILITY.csv` before calling the map current. Verify again when any of these change:

- A reported number
- Which takes are included in an aggregate
- The definition of EXT-JC, the repair, or the Soccer Frozen Full pipeline
- Figure or table numbers, including a figure that splits or moves
- A claim that was previously untraced or withheld

Do not invent a figure number for the new draft. If a sentence moves and the number is unchanged, update only `manuscript_location`. If a number in the paper changes, mark the old row and point it at the bundled table that still has the previous value. Do not edit frozen NPZ files or `postprocess_metrics.csv` so the repository matches the new sentence.

## How to keep the correspondence

1. Store the new PDF outside this tree. Do not commit it here.
2. Set the current version at the top of this file.
3. Diff the claims in `PAPER_CLAIM_TRACEABILITY.csv` against the new text.
4. Update README quotes only where the rules above say so.
5. Note in `TEST_REPORT.md` which PDF was read and which rows changed status.
6. Leave `evaluation/join/` and the Soccer frozen exports untouched unless a separate scientific decision says an experiment was replaced.

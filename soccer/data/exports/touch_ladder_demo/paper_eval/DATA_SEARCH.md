# Additional-sequence search

Review date: 2026-10-04  
Outcome: no additional sequence passed both licensing and technical gates.

License status is based on labels in `data/manifests/source_audit.json`. The repository does
not contain Mixkit/Pexels license text, a download receipt, or a source-page snapshot, so those
labels cannot be independently verified offline and must not be presented as a complete legal
provenance package.

The existing source audit was re-screened under the broader paper-stage rule that allows
ordinary two-person passing, not only throw-in sequences. A candidate still had to provide:

- clear research/redistribution permission;
- one continuous two-person soccer interaction;
- both participants substantially full-body visible;
- approximately 250 px person height for monocular recovery;
- readable ball/foot contacts;
- a ground receive/kick task compatible with the evaluated method.

The closest licensed candidate was Mixkit 43486. YOLO review over 12 sampled frames found
median person heights of 254.49 px and 240.68 px. The second player remains below the gate,
and the action is primarily aerial juggling without the ground-pass lane task used by Full.

Mixkit 2927 contains ground passing, but the two median heights were 598.32 px and
154.56 px; the far participant was missing in 2 of 12 sampled detections, while the near
participant is intermittently cropped. Mixkit 43517 measured only 162.49 px and 70.76 px.

Several other locally available Mixkit/Pexels files were confirmed to be basketball rather
than soccer, leg-only close-ups, multi-player match footage, or sequences without a readable
two-person contact chain. SoccerTrack-v2 remains gated and requires an authorized
`HF_TOKEN`. MUVS also remains fail-closed: `source_audit.json` labels it CC BY 4.0, but the
local MUVS README says `License: TO BE ADDED`.

No window from the current 7MLC source was relabeled as a new sequence: it has no
redistribution permission and would not provide independent participants. Consequently:

- no additional Basic/Full result was fabricated;
- `summary.csv` contains one sequence and four existing ablation methods;
- cross-sequence standard deviation is reported as unavailable rather than estimated from
  repeated deterministic runs;
- broad generalization remains unsupported.

Machine-readable candidate details are in `data_search_report.json`.

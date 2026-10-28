#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

core=".venv/bin/python -m core4d_dual_retarget.cli"
paper_root="${PAPER_ROOT:-outputs/paper_eval_join}"
sequence="20231108/018"

if [[ -e "$paper_root" ]]; then
  echo "Refusing to overwrite existing paper root: $paper_root" >&2
  exit 2
fi

for variant in J0 J1 J2 J3 J4; do
  variant_root="$paper_root/ablation_018/$variant"
  $core retarget-paper-join "$sequence" \
    --variant "$variant" \
    --output-root "$paper_root/ablation_018" \
    --iterations 600
  $core smooth-trajectories "$sequence" \
    --clearance-recipe \
    --result-root "$variant_root/raw" \
    --output-root "$variant_root/smooth"
  $core evaluate-paper-join "$sequence" \
    --result-root "$variant_root/smooth" \
    --output-path "$variant_root/metrics.json"
  $core render-g1-meshes "$sequence" \
    --result-root "$variant_root/smooth" \
    --output-path "$variant_root/preview.mp4"
done

$core copy-paper-join-final --output-root "$paper_root/final_six"
for short in 018 019 020 108 109 110; do
  output="$paper_root/final_six/$short"
  $core evaluate-paper-join "20231108/$short" \
    --result-root outputs/balanced_clearance_v41_smooth \
    --output-path "$output/metrics.json"
  $core render-g1-meshes "20231108/$short" \
    --result-root outputs/balanced_clearance_v41_smooth \
    --output-path "$output/preview.mp4"
done

$core render-source "$sequence" \
  --output-path "$paper_root/figures/source_018.gif" \
  --frame-step 1
$core plot-paper-join --paper-root "$paper_root"
$core summarize-paper-join --paper-root "$paper_root"

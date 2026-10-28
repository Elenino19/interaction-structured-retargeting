#!/usr/bin/env bash
# Generic N-slot job queue for the object-aware baseline extension runs.
# Usage: scripts/run_oa_queue.sh JOBFILE [SLOTS]
# Each JOBFILE line:  TAG|OUTPUT_ROOT|SEQUENCE|extra retarget-paper-join args
# Lines starting with # are ignored.  Skips jobs whose output already exists.
set -uo pipefail
cd "$(dirname "$0")/.."
JOBS=$1; SLOTS=${2:-4}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export PY=.venv/bin/python
run_job() {
  local line=$1
  IFS='|' read -r tag root seq extra <<<"$line"
  local stem=${seq//\//_}
  local log="$root/${tag}_${stem}.log"
  if [[ "$extra" == *"--solve-robot 1"* ]]; then log="$root/${tag}_${stem}_r1.log"; fi
  if [[ "$extra" == *"--solve-robot 2"* ]]; then log="$root/${tag}_${stem}_r2.log"; fi
  mkdir -p "$root"
  # shellcheck disable=SC2086
  $PY -m core4d_dual_retarget.cli retarget-paper-join "$seq" --variant J4 --iterations 600 \
    --output-root "$root" --tag "$tag" $extra > "$log" 2>&1
  echo "done $tag $seq exit=$?"
}
export -f run_job
rg -v '^\s*(#|$)' "$JOBS" | xargs -P "$SLOTS" -I{} bash -c 'run_job "$@"' _ {}
echo "queue finished: $JOBS"

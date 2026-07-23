#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "${SCRIPT_DIR}/_common.sh"

# Sequential lane: run livemathematicianbench -> officeqa -> spreadsheetbench one
# at a time (NOT in parallel). Each dataset fully finishes — including its
# guard_dataset_results all-zero check — before the next one starts. With
# `set -e`, a failing dataset stops the chain (fail-fast); set CONTINUE_ON_ERROR=1
# to keep going and report the failures at the end instead.
#
# Both models run through the DeepSeek official API (target=deepseek-v4-flash,
# optimizer=deepseek-v4-pro); see _common.sh for the wiring.

export RUN_ID="${RUN_ID:-skillopt_tree_seq_livemath_officeqa_spreadsheet_dsv4_flash_pro_$(date +%Y%m%d_%H%M%S)}"
export OUT_BASE="${OUT_BASE:-${PROJECT_ROOT}/outputs/${RUN_ID}}"
export SEQ_WORKERS="${SEQ_WORKERS:-16}"
export SEQ_ANALYST_WORKERS="${SEQ_ANALYST_WORKERS:-16}"
export SEQ_EXEC_TIMEOUT="${SEQ_EXEC_TIMEOUT:-1800}"

require_layout
configure_models
trap cleanup_datasets EXIT INT TERM
print_comparison_header "07-seq-livemath-officeqa-spreadsheet" \
  "livemathematicianbench -> officeqa -> spreadsheetbench (sequential)"
echo "  order:              livemathematicianbench, officeqa, spreadsheetbench"
echo "  workers/dataset:    ${SEQ_WORKERS}"
echo "  analysts/dataset:   ${SEQ_ANALYST_WORKERS}"
echo "  request timeout:    ${SEQ_EXEC_TIMEOUT}s (livemath + spreadsheetbench)"

# Run one dataset in the foreground; optionally swallow its failure so the chain
# continues. Records failures for a summary at the end.
SEQ_FAILURES=()
run_dataset_sequential() {
  local dataset="$1"
  shift
  echo
  echo "############################################################"
  echo "# [seq] starting ${dataset}"
  echo "############################################################"
  if comparison_truthy "${CONTINUE_ON_ERROR:-0}"; then
    if ! run_dataset "${dataset}" "$@"; then
      echo "[seq] dataset=${dataset} FAILED — continuing (CONTINUE_ON_ERROR=1)" >&2
      SEQ_FAILURES+=("${dataset}")
    fi
  else
    run_dataset "${dataset}" "$@"
  fi
}

# officeqa follows 04/05: no --exec_timeout override (uses its config default);
# livemath + spreadsheetbench get the shared SEQ_EXEC_TIMEOUT.
run_dataset_sequential livemathematicianbench \
  --workers "${SEQ_WORKERS}" \
  --analyst_workers "${SEQ_ANALYST_WORKERS}" \
  --exec_timeout "${SEQ_EXEC_TIMEOUT}" \
  "$@"
run_dataset_sequential officeqa \
  --workers "${SEQ_WORKERS}" \
  --analyst_workers "${SEQ_ANALYST_WORKERS}" \
  "$@"
run_dataset_sequential spreadsheetbench \
  --workers "${SEQ_WORKERS}" \
  --analyst_workers "${SEQ_ANALYST_WORKERS}" \
  --exec_timeout "${SEQ_EXEC_TIMEOUT}" \
  "$@"

echo
echo "============================================================"
if (( ${#SEQ_FAILURES[@]} )); then
  echo "  [seq] completed with FAILURES: ${SEQ_FAILURES[*]}"
  echo "  output: ${OUT_BASE}"
  echo "============================================================"
  exit 1
fi
echo "  [seq] all datasets completed: livemathematicianbench, officeqa, spreadsheetbench"
echo "  output: ${OUT_BASE}"
echo "============================================================"

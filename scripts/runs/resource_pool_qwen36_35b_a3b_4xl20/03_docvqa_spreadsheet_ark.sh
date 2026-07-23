#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "${SCRIPT_DIR}/_common.sh"

export OPTIMIZER_SOURCE="${OPTIMIZER_SOURCE:-deepseek_official}"
export RUN_ID="${RUN_ID:-skillopt_tree_docvqa_spreadsheet_qwen36_35b_a3b_ds_official_$(date +%Y%m%d_%H%M%S)}"
export OUT_BASE="${OUT_BASE:-${PROJECT_ROOT}/outputs/${RUN_ID}}"
export GROUP_WORKERS="${GROUP_WORKERS:-16}"
export GROUP_ANALYST_WORKERS="${GROUP_ANALYST_WORKERS:-16}"
export GROUP_EXEC_TIMEOUT="${GROUP_EXEC_TIMEOUT:-1800}"

require_layout
configure_models
trap cleanup_vllm EXIT INT TERM
print_comparison_header "03-docvqa-spreadsheet" "docvqa spreadsheetbench (parallel)"
echo "  workers/dataset:    ${GROUP_WORKERS}"
echo "  analysts/dataset:   ${GROUP_ANALYST_WORKERS}"
echo "  request timeout:    ${GROUP_EXEC_TIMEOUT}s"
start_local_vllm
run_dataset_background docvqa \
  --workers "${GROUP_WORKERS}" \
  --analyst_workers "${GROUP_ANALYST_WORKERS}" \
  --exec_timeout "${GROUP_EXEC_TIMEOUT}" \
  "$@"
run_dataset_background spreadsheetbench \
  --workers "${GROUP_WORKERS}" \
  --analyst_workers "${GROUP_ANALYST_WORKERS}" \
  --exec_timeout "${GROUP_EXEC_TIMEOUT}" \
  "$@"
wait_for_datasets

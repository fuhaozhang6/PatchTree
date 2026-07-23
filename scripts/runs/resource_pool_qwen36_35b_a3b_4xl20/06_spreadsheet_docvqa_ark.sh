#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "${SCRIPT_DIR}/_common.sh"

export RUN_ID="${RUN_ID:-skillopt_tree_spreadsheet_docvqa_qwen36_35b_a3b_ark_$(date +%Y%m%d_%H%M%S)}"
export OUT_BASE="${OUT_BASE:-${PROJECT_ROOT}/outputs/${RUN_ID}}"
# spreadsheet codegen (very long generations) + docvqa (long prefills) is the
# heaviest paired workload. Use the same 4xL20 Qwen3.6-35B-A3B profile as the
# rest of this directory; override CUDA_VISIBLE_DEVICES only when the four-card
# allocation differs.
export VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-4}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
export GROUP_WORKERS="${GROUP_WORKERS:-16}"
export GROUP_ANALYST_WORKERS="${GROUP_ANALYST_WORKERS:-16}"
export GROUP_EXEC_TIMEOUT="${GROUP_EXEC_TIMEOUT:-1800}"

require_layout
configure_models
trap cleanup_vllm EXIT INT TERM
print_comparison_header "06-spreadsheet-docvqa" "spreadsheetbench docvqa (parallel)"
echo "  workers/dataset:    ${GROUP_WORKERS}"
echo "  analysts/dataset:   ${GROUP_ANALYST_WORKERS}"
echo "  request timeout:    ${GROUP_EXEC_TIMEOUT}s"
start_local_vllm
run_dataset_background spreadsheetbench \
  --workers "${GROUP_WORKERS}" \
  --analyst_workers "${GROUP_ANALYST_WORKERS}" \
  --exec_timeout "${GROUP_EXEC_TIMEOUT}" \
  "$@"
run_dataset_background docvqa \
  --workers "${GROUP_WORKERS}" \
  --analyst_workers "${GROUP_ANALYST_WORKERS}" \
  --exec_timeout "${GROUP_EXEC_TIMEOUT}" \
  "$@"
wait_for_datasets

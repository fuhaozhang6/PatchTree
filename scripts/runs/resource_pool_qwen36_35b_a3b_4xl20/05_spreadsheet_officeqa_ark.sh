#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "${SCRIPT_DIR}/_common.sh"

export RUN_ID="${RUN_ID:-skillopt_tree_spreadsheet_officeqa_qwen36_35b_a3b_ark_$(date +%Y%m%d_%H%M%S)}"
export OUT_BASE="${OUT_BASE:-${PROJECT_ROOT}/outputs/${RUN_ID}}"
# OfficeQA drives the target model with tool_choice="auto"; vLLM rejects that
# request unless the engine is started with --enable-auto-tool-choice, so the
# shared endpoint must enable it.  The parser default in _common.sh is
# qwen3_coder, which matches the XML tool-call grammar Qwen3.x emits
# (the hermes parser expects JSON and fails, leaving OfficeQA with zero
# retrieved evidence and near-zero scores).  SpreadsheetBench runs in codegen
# mode and does not use tool-calls, so it is unaffected by this flag.
export VLLM_ENABLE_AUTO_TOOL_CHOICE="${VLLM_ENABLE_AUTO_TOOL_CHOICE:-1}"
# Reserve activation headroom on the L20 for OfficeQA's longer prefills. The
# 35B-A3B target has much less spare memory than 4B, so keep this conservative.
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
export TARGET_QWEN_CHAT_TIMEOUT_SECONDS="${TARGET_QWEN_CHAT_TIMEOUT_SECONDS:-1800}"
export GROUP_WORKERS="${GROUP_WORKERS:-16}"
export GROUP_ANALYST_WORKERS="${GROUP_ANALYST_WORKERS:-16}"
export GROUP_EXEC_TIMEOUT="${GROUP_EXEC_TIMEOUT:-1800}"

require_layout
configure_models
trap cleanup_vllm EXIT INT TERM
print_comparison_header "05-spreadsheet-officeqa" "spreadsheetbench officeqa (parallel)"
echo "  workers/dataset:    ${GROUP_WORKERS}"
echo "  analysts/dataset:   ${GROUP_ANALYST_WORKERS}"
echo "  request timeout:    ${GROUP_EXEC_TIMEOUT}s"
start_local_vllm
run_dataset_background spreadsheetbench \
  --workers "${GROUP_WORKERS}" \
  --analyst_workers "${GROUP_ANALYST_WORKERS}" \
  --exec_timeout "${GROUP_EXEC_TIMEOUT}" \
  "$@"
run_dataset_background officeqa \
  --workers "${GROUP_WORKERS}" \
  --analyst_workers "${GROUP_ANALYST_WORKERS}" \
  "$@"
wait_for_datasets

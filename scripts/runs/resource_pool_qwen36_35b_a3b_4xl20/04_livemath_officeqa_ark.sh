#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "${SCRIPT_DIR}/_common.sh"

export OPTIMIZER_SOURCE="${OPTIMIZER_SOURCE:-deepseek_official}"
export RUN_ID="${RUN_ID:-skillopt_tree_livemath_officeqa_qwen36_35b_a3b_ds_official_$(date +%Y%m%d_%H%M%S)}"
export OUT_BASE="${OUT_BASE:-${PROJECT_ROOT}/outputs/${RUN_ID}}"
export VLLM_ENABLE_AUTO_TOOL_CHOICE="${VLLM_ENABLE_AUTO_TOOL_CHOICE:-1}"
# Reserve 15% of each L20 for transient activations. The 35B-A3B profile has
# much less KV headroom than 4B, so keep this conservative by default.
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
export TARGET_QWEN_CHAT_TIMEOUT_SECONDS="${TARGET_QWEN_CHAT_TIMEOUT_SECONDS:-1800}"
export GROUP_WORKERS="${GROUP_WORKERS:-16}"
export GROUP_ANALYST_WORKERS="${GROUP_ANALYST_WORKERS:-16}"
export GROUP_EXEC_TIMEOUT="${GROUP_EXEC_TIMEOUT:-1800}"

require_layout
configure_models
trap cleanup_vllm EXIT INT TERM
print_comparison_header "04-livemath-officeqa" "livemathematicianbench officeqa (parallel)"
echo "  workers/dataset:    ${GROUP_WORKERS}"
echo "  analysts/dataset:   ${GROUP_ANALYST_WORKERS}"
echo "  request timeout:    ${GROUP_EXEC_TIMEOUT}s"
start_local_vllm
run_dataset_background livemathematicianbench \
  --workers "${GROUP_WORKERS}" \
  --analyst_workers "${GROUP_ANALYST_WORKERS}" \
  --exec_timeout "${GROUP_EXEC_TIMEOUT}" \
  "$@"
run_dataset_background officeqa \
  --workers "${GROUP_WORKERS}" \
  --analyst_workers "${GROUP_ANALYST_WORKERS}" \
  "$@"
wait_for_datasets

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "${SCRIPT_DIR}/_common.sh"

configure_single_l20
configure_volcano_ark

export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-128}"
export QWEN_CHAT_TIMEOUT_SECONDS="${QWEN_CHAT_TIMEOUT_SECONDS:-300}"
export TARGET_QWEN_CHAT_TIMEOUT_SECONDS="${TARGET_QWEN_CHAT_TIMEOUT_SECONDS:-${QWEN_CHAT_TIMEOUT_SECONDS}}"
export NUM_EPOCHS="${NUM_EPOCHS:-4}"
export BATCH_SIZE="${BATCH_SIZE:-8}"
export API_MAX_CONCURRENCY="${API_MAX_CONCURRENCY:-48}"
export WORKERS="${WORKERS:-48}"
export ANALYST_WORKERS="${ANALYST_WORKERS:-16}"
export DOCVQA_WORKERS="${DOCVQA_WORKERS:-${WORKERS}}"
export DOCVQA_ANALYST_WORKERS="${DOCVQA_ANALYST_WORKERS:-${ANALYST_WORKERS}}"
export SPREADSHEETBENCH_WORKERS="${SPREADSHEETBENCH_WORKERS:-${WORKERS}}"
export SPREADSHEETBENCH_ANALYST_WORKERS="${SPREADSHEETBENCH_ANALYST_WORKERS:-${ANALYST_WORKERS}}"
export SPREADSHEETBENCH_EXEC_TIMEOUT="${SPREADSHEETBENCH_EXEC_TIMEOUT:-1200}"
export SPREADSHEETBENCH_LLM_TIMEOUT="${SPREADSHEETBENCH_LLM_TIMEOUT:-300}"

export TS="${TS:-resource_pool_docvqa_spreadsheet_ark_$(date +%Y%m%d_%H%M%S)}"

print_resource_header "03-docvqa-spreadsheet" "docvqa spreadsheetbench" "Volcano Ark"
exec bash "${PROJECT_ROOT}/scripts/runs/multi/run_docvqa_spreadsheetbench_l20_qwen35_4b_dsv4pro_ark.sh" "$@"

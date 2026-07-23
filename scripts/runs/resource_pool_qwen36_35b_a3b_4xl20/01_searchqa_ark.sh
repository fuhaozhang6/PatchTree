#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "${SCRIPT_DIR}/_common.sh"

export RUN_ID="${RUN_ID:-skillopt_tree_searchqa_qwen36_35b_a3b_ark_$(date +%Y%m%d_%H%M%S)}"
export OUT_BASE="${OUT_BASE:-${PROJECT_ROOT}/outputs/${RUN_ID}}"
SEARCHQA_WORKERS="${SEARCHQA_WORKERS:-24}"
SEARCHQA_EXEC_TIMEOUT="${SEARCHQA_EXEC_TIMEOUT:-600}"

require_layout
configure_models
trap cleanup_vllm EXIT INT TERM
print_comparison_header "01-searchqa" "searchqa"
echo "  workers:            ${SEARCHQA_WORKERS}"
echo "  request timeout:    ${SEARCHQA_EXEC_TIMEOUT}s"
start_local_vllm
run_dataset searchqa --workers "${SEARCHQA_WORKERS}" --exec_timeout "${SEARCHQA_EXEC_TIMEOUT}" "$@"

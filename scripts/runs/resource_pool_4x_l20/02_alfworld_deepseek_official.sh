#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "${SCRIPT_DIR}/_common.sh"

configure_single_l20
configure_deepseek_official

export DATASETS=alfworld
export MAX_PARALLEL=1
export WAIT_FOR_JOBS="${WAIT_FOR_JOBS:-1}"
export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-96}"
export MAX_MODEL_LEN="${ALFWORLD_MAX_MODEL_LEN:-32768}"
export QWEN_CHAT_TIMEOUT_SECONDS="${QWEN_CHAT_TIMEOUT_SECONDS:-300}"
export TARGET_QWEN_CHAT_TIMEOUT_SECONDS="${TARGET_QWEN_CHAT_TIMEOUT_SECONDS:-${QWEN_CHAT_TIMEOUT_SECONDS}}"
export QWEN_CHAT_MAX_TOKENS="${QWEN_CHAT_MAX_TOKENS:-2048}"
export TARGET_MAX_COMPLETION_TOKENS="${TARGET_MAX_COMPLETION_TOKENS:-2048}"

export NUM_EPOCHS="${NUM_EPOCHS:-4}"
export BATCH_SIZE="${BATCH_SIZE:-8}"
export ALFWORLD_BATCH_SIZE="${ALFWORLD_BATCH_SIZE:-${BATCH_SIZE}}"
export API_MAX_CONCURRENCY="${API_MAX_CONCURRENCY:-96}"
export WORKERS="${WORKERS:-24}"
export ANALYST_WORKERS="${ANALYST_WORKERS:-16}"
export ALFWORLD_WORKERS="${ALFWORLD_WORKERS:-${WORKERS}}"
export ALFWORLD_MAX_API_WORKERS="${ALFWORLD_MAX_API_WORKERS:-${WORKERS}}"
export ALFWORLD_ANALYST_WORKERS="${ALFWORLD_ANALYST_WORKERS:-${ANALYST_WORKERS}}"
export ALFWORLD_MAX_STEPS="${ALFWORLD_MAX_STEPS:-50}"

# Preserve the fuller ALFWorld tree while leaving the removed tail path off.
export TYPE_GUIDED_TREE_DEPTH="${TYPE_GUIDED_TREE_DEPTH:-3}"
export TYPE_GUIDED_MIN_SUPPORT="${TYPE_GUIDED_MIN_SUPPORT:-1}"
export TYPE_GUIDED_CLUSTERING="${TYPE_GUIDED_CLUSTERING:-false}"
export TYPE_GUIDED_TAIL_BANK="${TYPE_GUIDED_TAIL_BANK:-true}"
export TYPE_GUIDED_FALLBACK_TOP_K="${TYPE_GUIDED_FALLBACK_TOP_K:-4}"

export TS="${TS:-resource_pool_alfworld_official_$(date +%Y%m%d_%H%M%S)}"
export LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs/${TS}}"
export OUT_BASE="${OUT_BASE:-${PROJECT_ROOT}/outputs/${TS}}"

# ALFWorld needs the environment package + simulator data on the run machine.
# Check (and, by default, auto-install) the deps before launching so the run
# fails fast here instead of dying mid-training on `import gymnasium`. Set
# ALFWORLD_AUTO_INSTALL=0 to only self-check without installing.
require_alfworld_environment

print_resource_header "02-alfworld" "alfworld" "DeepSeek official"
exec bash "${PROJECT_ROOT}/scripts/runs/multi/run_v3_deepseek_local_qwen_parallel.sh" "$@"

#!/usr/bin/env bash
set -euo pipefail

# LiveMath 16-epoch launcher:
#   optimizer / teacher = DeepSeek through the official DeepSeek API
#   target / student    = local Qwen3.5-4B served by one single-GPU vLLM on L20
#   data                = existing project data/livemathematicianbench_split
#
# Typical usage on the L20 node:
#   cd /ai-car-vepfs1/ai_car/zhangfuhao/data/01/SkillOpt-Tree
#   export DEEPSEEK_API_KEY='...'
#   bash scripts/runs/livemath/run_livemath_l20_qwen35_4b_dsv4pro_16epoch.sh
#
# Useful overrides:
#   L20_GPU=1 bash scripts/runs/livemath/run_livemath_l20_qwen35_4b_dsv4pro_16epoch.sh
#   LIMIT=8 NUM_EPOCHS=1 bash scripts/runs/livemath/run_livemath_l20_qwen35_4b_dsv4pro_16epoch.sh
#   START_VLLM=0 QWEN_CHAT_BASE_URL=http://127.0.0.1:59317/v1 bash scripts/runs/livemath/run_livemath_l20_qwen35_4b_dsv4pro_16epoch.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_ROOT}"

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

truthy() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

# Optimizer: DeepSeek through the official OpenAI-compatible API.
export DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY:-${DS_API_KEY:-${DEEPSEEK_OFFICIAL_API_KEY:-}}}"
if [[ -z "${DEEPSEEK_API_KEY}" ]] && ! truthy "${DRY_RUN:-0}"; then
  fail "DEEPSEEK_API_KEY is required for the official DeepSeek optimizer."
fi
export AZURE_OPENAI_ENDPOINT="${AZURE_OPENAI_ENDPOINT:-https://api.deepseek.com}"
export AZURE_OPENAI_API_KEY="${AZURE_OPENAI_API_KEY:-${DEEPSEEK_API_KEY}}"
export AZURE_OPENAI_AUTH_MODE="${AZURE_OPENAI_AUTH_MODE:-openai_compatible}"
export AZURE_OPENAI_API_VERSION="${AZURE_OPENAI_API_VERSION:-openai-compat}"
export OPTIMIZER_AZURE_OPENAI_ENDPOINT="${OPTIMIZER_AZURE_OPENAI_ENDPOINT:-${AZURE_OPENAI_ENDPOINT}}"
export OPTIMIZER_AZURE_OPENAI_API_KEY="${OPTIMIZER_AZURE_OPENAI_API_KEY:-${AZURE_OPENAI_API_KEY}}"
export OPTIMIZER_AZURE_OPENAI_AUTH_MODE="${OPTIMIZER_AZURE_OPENAI_AUTH_MODE:-${AZURE_OPENAI_AUTH_MODE}}"
export OPTIMIZER_AZURE_OPENAI_API_VERSION="${OPTIMIZER_AZURE_OPENAI_API_VERSION:-${AZURE_OPENAI_API_VERSION}}"
export OPTIMIZER_MODEL="${OPTIMIZER_MODEL:-${DEEPSEEK_MODEL:-deepseek-v4-pro}}"
export DEEPSEEK_THINKING="${DEEPSEEK_THINKING:-enabled}"
export REASONING_EFFORT="${REASONING_EFFORT:-high}"

# Target: local Qwen3.5-4B served by vLLM on one L20 GPU.
export MODEL_PATH="${MODEL_PATH:-/ai-app-vepfs/modelscope/models/Qwen/Qwen3.5-4B}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen/Qwen3.5-4B}"
export TARGET_MODEL="${TARGET_MODEL:-${SERVED_MODEL_NAME}}"
export L20_GPU="${L20_GPU:-0}"
export QWEN_CUDA_VISIBLE_DEVICES="${QWEN_CUDA_VISIBLE_DEVICES:-${L20_GPU}}"
export VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-1}"
export VLLM_PORT="${VLLM_PORT:-59317}"
export QWEN_CHAT_BASE_URL="${QWEN_CHAT_BASE_URL:-http://127.0.0.1:${VLLM_PORT}/v1}"
export QWEN_CHAT_API_KEY="${QWEN_CHAT_API_KEY:-dummy}"
export QWEN_CHAT_MODEL="${QWEN_CHAT_MODEL:-${SERVED_MODEL_NAME}}"
export QWEN_CHAT_TIMEOUT_SECONDS="${QWEN_CHAT_TIMEOUT_SECONDS:-240}"
export QWEN_CHAT_MAX_TOKENS="${QWEN_CHAT_MAX_TOKENS:-16384}"
export TARGET_QWEN_CHAT_TIMEOUT_SECONDS="${TARGET_QWEN_CHAT_TIMEOUT_SECONDS:-${QWEN_CHAT_TIMEOUT_SECONDS}}"
export TARGET_QWEN_CHAT_MAX_TOKENS="${TARGET_QWEN_CHAT_MAX_TOKENS:-${QWEN_CHAT_MAX_TOKENS}}"
export TARGET_QWEN_CHAT_TEMPERATURE="${TARGET_QWEN_CHAT_TEMPERATURE:-0.2}"
export TARGET_QWEN_CHAT_ENABLE_THINKING="${TARGET_QWEN_CHAT_ENABLE_THINKING:-false}"

# L20-friendly vLLM defaults. Keep concurrency modest for stability.
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
export VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:---max-num-seqs 64 --max-num-batched-tokens 65536}"
export START_VLLM="${START_VLLM:-1}"
export STOP_VLLM_ON_EXIT="${STOP_VLLM_ON_EXIT:-1}"

# LiveMath-only training. Existing split data is used; no data download/materialization.
export DATASETS="livemath"
export NUM_EPOCHS="${NUM_EPOCHS:-16}"
export MAX_PARALLEL=1
export WAIT_FOR_JOBS="${WAIT_FOR_JOBS:-1}"
export LIVEMATH_SPLIT_DIR="${LIVEMATH_SPLIT_DIR:-${PROJECT_ROOT}/data/livemathematicianbench_split}"
export TRAIN_SIZE="${TRAIN_SIZE:-0}"
export LIMIT="${LIMIT:-0}"
export BATCH_SIZE="${BATCH_SIZE:-8}"
export LIVEMATH_BATCH_SIZE="${LIVEMATH_BATCH_SIZE:-${BATCH_SIZE}}"
export API_MAX_CONCURRENCY="${API_MAX_CONCURRENCY:-48}"
export WORKERS="${WORKERS:-48}"
export ANALYST_WORKERS="${ANALYST_WORKERS:-16}"
export LIVEMATH_WORKERS="${LIVEMATH_WORKERS:-${WORKERS}}"
export LIVEMATH_ANALYST_WORKERS="${LIVEMATH_ANALYST_WORKERS:-${ANALYST_WORKERS}}"
export LIVEMATH_SEL_ENV_NUM="${LIVEMATH_SEL_ENV_NUM:-18}"
export LIVEMATH_TEST_ENV_NUM="${LIVEMATH_TEST_ENV_NUM:-0}"
export EVAL_TEST="${EVAL_TEST:-false}"
export LIVEMATH_TARGET_MAX_COMPLETION_TOKENS="${LIVEMATH_TARGET_MAX_COMPLETION_TOKENS:-16384}"
export TARGET_MAX_COMPLETION_TOKENS="${TARGET_MAX_COMPLETION_TOKENS:-${LIVEMATH_TARGET_MAX_COMPLETION_TOKENS}}"

# Keep the type-guided path enabled, but avoid extra expensive self-checks.
export TYPE_GUIDED_TREE_DEPTH="${TYPE_GUIDED_TREE_DEPTH:-2}"
export TYPE_GUIDED_PATCH_RECORD_WORKERS="${TYPE_GUIDED_PATCH_RECORD_WORKERS:-${ANALYST_WORKERS}}"

export TS="${TS:-livemath_l20_qwen35_4b_dsv4pro_16ep_$(date +%Y%m%d_%H%M%S)}"
export LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs/${TS}}"
export OUT_BASE="${OUT_BASE:-${PROJECT_ROOT}/outputs/${TS}}"

[[ -d "${MODEL_PATH}" ]] || fail "MODEL_PATH not found: ${MODEL_PATH}"
[[ -d "${LIVEMATH_SPLIT_DIR}/train" ]] || fail "LiveMath split not found: ${LIVEMATH_SPLIT_DIR}"

echo "============================================================"
echo "  LiveMath L20 Qwen3.5-4B + DeepSeek Official"
echo "============================================================"
echo "  project:        ${PROJECT_ROOT}"
echo "  optimizer:      ${OPTIMIZER_MODEL}"
echo "  target:         ${TARGET_MODEL}"
echo "  model_path:     ${MODEL_PATH}"
echo "  gpu:            ${QWEN_CUDA_VISIBLE_DEVICES}"
echo "  qwen_url:       ${QWEN_CHAT_BASE_URL}"
echo "  epochs:         ${NUM_EPOCHS}"
echo "  workers:        ${WORKERS}"
echo "  analyst_workers:${ANALYST_WORKERS}"
echo "  batch_size:     ${BATCH_SIZE}"
echo "  split_dir:      ${LIVEMATH_SPLIT_DIR}"
echo "  out_base:       ${OUT_BASE}"
echo "  log_dir:        ${LOG_DIR}"
echo "============================================================"

exec bash "${PROJECT_ROOT}/scripts/runs/multi/run_v3_deepseek_local_qwen_parallel.sh" "$@"

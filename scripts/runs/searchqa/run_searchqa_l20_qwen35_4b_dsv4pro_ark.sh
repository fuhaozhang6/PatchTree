#!/usr/bin/env bash
set -euo pipefail

# SearchQA single-dataset launcher:
#   optimizer / teacher = DeepSeek V4 Pro through Volcano Ark
#   target / student    = local Qwen3.5-4B served by vLLM on one L20
#   concurrency         = one SearchQA job gets the full local endpoint
#
# Usage:
#   export ARK_API_KEY='...'
#   bash scripts/runs/searchqa/run_searchqa_l20_qwen35_4b_dsv4pro_ark.sh
#
# Smoke test:
#   LIMIT=4 NUM_EPOCHS=1 EVAL_TEST=false SEARCHQA_BATCH_SIZE=4 \
#     bash scripts/runs/searchqa/run_searchqa_l20_qwen35_4b_dsv4pro_ark.sh
#
# Reduce concurrency if the L20 starts preempting heavily:
#   WORKERS=96 VLLM_MAX_NUM_SEQS=96 \
#     bash scripts/runs/searchqa/run_searchqa_l20_qwen35_4b_dsv4pro_ark.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_ROOT}"

# DeepSeek optimizer through Ark's OpenAI-compatible endpoint.
export AZURE_OPENAI_ENDPOINT="${AZURE_OPENAI_ENDPOINT:-https://ark.cn-beijing.volces.com/api/v3}"
export AZURE_OPENAI_API_KEY="${AZURE_OPENAI_API_KEY:-${ARK_API_KEY:-}}"
export AZURE_OPENAI_AUTH_MODE="${AZURE_OPENAI_AUTH_MODE:-openai_compatible}"
export AZURE_OPENAI_API_VERSION="${AZURE_OPENAI_API_VERSION:-2024-12-01-preview}"
export OPTIMIZER_AZURE_OPENAI_ENDPOINT="${OPTIMIZER_AZURE_OPENAI_ENDPOINT:-${AZURE_OPENAI_ENDPOINT}}"
export OPTIMIZER_AZURE_OPENAI_API_KEY="${OPTIMIZER_AZURE_OPENAI_API_KEY:-${AZURE_OPENAI_API_KEY}}"
export OPTIMIZER_AZURE_OPENAI_AUTH_MODE="${OPTIMIZER_AZURE_OPENAI_AUTH_MODE:-${AZURE_OPENAI_AUTH_MODE}}"
export OPTIMIZER_AZURE_OPENAI_API_VERSION="${OPTIMIZER_AZURE_OPENAI_API_VERSION:-${AZURE_OPENAI_API_VERSION}}"
export OPTIMIZER_MODEL="${OPTIMIZER_MODEL:-deepseek-v4-pro-260425}"
export DEEPSEEK_THINKING="${DEEPSEEK_THINKING:-disabled}"
export REASONING_EFFORT="${REASONING_EFFORT:-}"

# Local Qwen target on one L20.
export MODEL_PATH="${MODEL_PATH:-/ai-app-vepfs/modelscope/models/Qwen/Qwen3.5-4B}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen/Qwen3.5-4B}"
export TARGET_MODEL="${TARGET_MODEL:-${SERVED_MODEL_NAME}}"
export L20_GPU="${L20_GPU:-0}"
export QWEN_CUDA_VISIBLE_DEVICES="${QWEN_CUDA_VISIBLE_DEVICES:-${L20_GPU}}"
export VLLM_TENSOR_PARALLEL_SIZE=1
export VLLM_PORT="${VLLM_PORT:-59317}"
export QWEN_CHAT_BASE_URL="${QWEN_CHAT_BASE_URL:-http://127.0.0.1:${VLLM_PORT}/v1}"
export QWEN_CHAT_API_KEY="${QWEN_CHAT_API_KEY:-dummy}"
export QWEN_CHAT_MODEL="${QWEN_CHAT_MODEL:-${SERVED_MODEL_NAME}}"
export QWEN_CHAT_TIMEOUT_SECONDS="${QWEN_CHAT_TIMEOUT_SECONDS:-300}"
export QWEN_CHAT_MAX_TOKENS="${QWEN_CHAT_MAX_TOKENS:-16384}"
export TARGET_QWEN_CHAT_TEMPERATURE="${TARGET_QWEN_CHAT_TEMPERATURE:-0.2}"
export QWEN_CHAT_ENABLE_THINKING=false
export TARGET_QWEN_CHAT_ENABLE_THINKING=false

# Aggressive single-L20 profile. vLLM may queue/preempt requests internally
# when all 128 sequences have long contexts; lower both values together if
# throughput drops or p95 latency grows sharply.
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-128}"
export VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-65536}"
export VLLM_ENABLE_CHUNKED_PREFILL="${VLLM_ENABLE_CHUNKED_PREFILL:-1}"
export START_VLLM="${START_VLLM:-1}"
export STOP_VLLM_ON_EXIT="${STOP_VLLM_ON_EXIT:-1}"

# SearchQA uses the whole L20 endpoint.
export DATASETS=searchqa
export MAX_PARALLEL=1
export WAIT_FOR_JOBS="${WAIT_FOR_JOBS:-1}"
export NUM_EPOCHS="${NUM_EPOCHS:-4}"
export TRAIN_SIZE="${TRAIN_SIZE:-0}"
export LIMIT="${LIMIT:-0}"
export BATCH_SIZE="${BATCH_SIZE:-40}"
export SEARCHQA_BATCH_SIZE="${SEARCHQA_BATCH_SIZE:-${BATCH_SIZE}}"

export API_MAX_CONCURRENCY="${API_MAX_CONCURRENCY:-128}"
export WORKERS="${WORKERS:-128}"
export ANALYST_WORKERS="${ANALYST_WORKERS:-48}"
export SEARCHQA_WORKERS="${SEARCHQA_WORKERS:-${WORKERS}}"
export SEARCHQA_ANALYST_WORKERS="${SEARCHQA_ANALYST_WORKERS:-${ANALYST_WORKERS}}"

# data/searchqa_id_split contains IDs only; use the materialized split below.
export SEARCHQA_SPLIT_DIR="${SEARCHQA_SPLIT_DIR:-${PROJECT_ROOT}/data/searchqa_split}"
export SEARCHQA_SEL_ENV_NUM="${SEARCHQA_SEL_ENV_NUM:-200}"
export SEARCHQA_TEST_ENV_NUM="${SEARCHQA_TEST_ENV_NUM:-400}"
export SEARCHQA_MAX_TURNS="${SEARCHQA_MAX_TURNS:-1}"
export SEARCHQA_TYPE_GUIDED_FALLBACK_SEL_ENV_NUM="${SEARCHQA_TYPE_GUIDED_FALLBACK_SEL_ENV_NUM:-80}"
export EVAL_TEST="${EVAL_TEST:-true}"

export TARGET_MAX_COMPLETION_TOKENS="${TARGET_MAX_COMPLETION_TOKENS:-16384}"
export SEARCHQA_TARGET_MAX_COMPLETION_TOKENS="${SEARCHQA_TARGET_MAX_COMPLETION_TOKENS:-${TARGET_MAX_COMPLETION_TOKENS}}"

# Match the compact PatchTree-v4 profile used by the paired launchers.
export TYPE_GUIDED_TREE_DEPTH="${TYPE_GUIDED_TREE_DEPTH:-2}"
export TYPE_GUIDED_CLUSTERING="${TYPE_GUIDED_CLUSTERING:-false}"
export TYPE_GUIDED_TAIL_BANK="${TYPE_GUIDED_TAIL_BANK:-true}"
export TYPE_GUIDED_FALLBACK_TOP_K="${TYPE_GUIDED_FALLBACK_TOP_K:-4}"
export TYPE_GUIDED_PATCH_RECORD_WORKERS="${TYPE_GUIDED_PATCH_RECORD_WORKERS:-${ANALYST_WORKERS}}"
export TYPE_GUIDED_LEAF_MERGE_WORKERS="${TYPE_GUIDED_LEAF_MERGE_WORKERS:-8}"
export TYPE_GUIDED_MID_MERGE_WORKERS="${TYPE_GUIDED_MID_MERGE_WORKERS:-4}"

export TS="${TS:-searchqa_l20_qwen35_4b_dsv4pro_ark_$(date +%Y%m%d_%H%M%S)}"
export LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs/${TS}}"
export OUT_BASE="${OUT_BASE:-${PROJECT_ROOT}/outputs/${TS}}"

echo "============================================================"
echo "  SearchQA: local Qwen3.5-4B + Ark DeepSeek V4 Pro"
echo "============================================================"
echo "  optimizer:       ${OPTIMIZER_MODEL}"
echo "  target:          ${TARGET_MODEL}"
echo "  model_path:      ${MODEL_PATH}"
echo "  L20 GPU:         ${QWEN_CUDA_VISIBLE_DEVICES}"
echo "  epochs/batch:    ${NUM_EPOCHS}/${SEARCHQA_BATCH_SIZE}"
echo "  target workers:  ${SEARCHQA_WORKERS}"
echo "  analyst workers: ${SEARCHQA_ANALYST_WORKERS}"
echo "  vLLM max seqs:   ${VLLM_MAX_NUM_SEQS}"
echo "  val/test items:  ${SEARCHQA_SEL_ENV_NUM}/${SEARCHQA_TEST_ENV_NUM}"
echo "  fallback top-k:  ${TYPE_GUIDED_FALLBACK_TOP_K}"
echo "  split:           ${SEARCHQA_SPLIT_DIR}"
echo "  output:          ${OUT_BASE}"
echo "============================================================"

exec bash "${PROJECT_ROOT}/scripts/runs/multi/run_v3_deepseek_local_qwen_parallel.sh" "$@"

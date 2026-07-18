#!/usr/bin/env bash
set -euo pipefail

# V3 training launcher for the current three-dataset run:
#   optimizer = DeepSeek V4 through Ark OpenAI-compatible API
#   target    = local Qwen3.6-35B-A3B served by vLLM
#   datasets  = docvqa, officeqa, spreadsheetbench
#   parallel  = 2 datasets at a time x 128 rollout workers per dataset
#   budget    = 16k target completion tokens
#
# This is intentionally a thin wrapper over run_v3_deepseek_local_qwen_parallel.sh
# so it inherits the common vLLM startup, dataset plumbing, V3 settings, and
# logging behavior from the maintained launcher.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_ROOT}"

# Model and service defaults.
export MODEL_PATH="${MODEL_PATH:-/ai-app-vepfs/models/Qwen3.6-35B-A3B}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3.6-35B-A3B}"
export TARGET_MODEL="${TARGET_MODEL:-${SERVED_MODEL_NAME}}"
export OPTIMIZER_MODEL="${OPTIMIZER_MODEL:-deepseek-v4-pro-260425}"

# Two H20s by default. Override QWEN_CUDA_VISIBLE_DEVICES / TP if needed.
export QWEN_CUDA_VISIBLE_DEVICES="${QWEN_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-0,1}}"
export VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-2}"
export QWEN_CHAT_MODEL="${QWEN_CHAT_MODEL:-${SERVED_MODEL_NAME}}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"

# Run scope.
export DATASETS="${DATASETS:-docvqa officeqa spreadsheetbench}"
export MAX_PARALLEL="${MAX_PARALLEL:-2}"
export WAIT_FOR_JOBS="${WAIT_FOR_JOBS:-1}"
export TS="${TS:-v3_deepseek_qwen36_35b_a3b_2x128_3datasets_$(date +%Y%m%d_%H%M%S)}"

# Training defaults. Keep self-check/support self-check off for cost.
export NUM_EPOCHS="${NUM_EPOCHS:-2}"
export TRAIN_SIZE="${TRAIN_SIZE:-0}"
export LIMIT="${LIMIT:-0}"
export TYPE_GUIDED_CLUSTERING="${TYPE_GUIDED_CLUSTERING:-true}"
export TYPE_GUIDED_TREE_DEPTH="${TYPE_GUIDED_TREE_DEPTH:-3}"
export TYPE_GUIDED_LEAF_MERGE_WORKERS="${TYPE_GUIDED_LEAF_MERGE_WORKERS:-8}"
export TYPE_GUIDED_MID_MERGE_WORKERS="${TYPE_GUIDED_MID_MERGE_WORKERS:-4}"
export TYPE_GUIDED_FALLBACK_RECONCILE="${TYPE_GUIDED_FALLBACK_RECONCILE:-deterministic}"
export TYPE_GUIDED_FALLBACK_TOP_K="${TYPE_GUIDED_FALLBACK_TOP_K:-2}"

# 2 x 128 means two dataset jobs in parallel, each with 128 target rollout
# workers. API_MAX_CONCURRENCY is per training process; total Qwen client
# concurrency can therefore reach roughly 256 when two datasets are active.
export API_MAX_CONCURRENCY="${API_MAX_CONCURRENCY:-128}"
export WORKERS="${WORKERS:-128}"
export DOCVQA_WORKERS="${DOCVQA_WORKERS:-128}"
export OFFICEQA_WORKERS="${OFFICEQA_WORKERS:-128}"
export SPREADSHEETBENCH_WORKERS="${SPREADSHEETBENCH_WORKERS:-128}"

# Keep analyst parallelism moderate; these calls go to the optimizer side and
# should not be confused with target rollout workers.
export ANALYST_WORKERS="${ANALYST_WORKERS:-24}"
export DOCVQA_ANALYST_WORKERS="${DOCVQA_ANALYST_WORKERS:-24}"
export OFFICEQA_ANALYST_WORKERS="${OFFICEQA_ANALYST_WORKERS:-24}"
export SPREADSHEETBENCH_ANALYST_WORKERS="${SPREADSHEETBENCH_ANALYST_WORKERS:-24}"

# Target token budget: 16k everywhere.
export QWEN_CHAT_MAX_TOKENS="${QWEN_CHAT_MAX_TOKENS:-16384}"
export TARGET_QWEN_CHAT_MAX_TOKENS="${TARGET_QWEN_CHAT_MAX_TOKENS:-16384}"
export TARGET_MAX_COMPLETION_TOKENS="${TARGET_MAX_COMPLETION_TOKENS:-16384}"
export DOCVQA_TARGET_MAX_COMPLETION_TOKENS="${DOCVQA_TARGET_MAX_COMPLETION_TOKENS:-16384}"
export OFFICEQA_TARGET_MAX_COMPLETION_TOKENS="${OFFICEQA_TARGET_MAX_COMPLETION_TOKENS:-16384}"
export SPREADSHEETBENCH_TARGET_MAX_COMPLETION_TOKENS="${SPREADSHEETBENCH_TARGET_MAX_COMPLETION_TOKENS:-16384}"
export QWEN_CHAT_TIMEOUT_SECONDS="${QWEN_CHAT_TIMEOUT_SECONDS:-240}"
export TARGET_QWEN_CHAT_TIMEOUT_SECONDS="${TARGET_QWEN_CHAT_TIMEOUT_SECONDS:-240}"
export TARGET_QWEN_CHAT_ENABLE_THINKING="${TARGET_QWEN_CHAT_ENABLE_THINKING:-false}"
export QWEN_CHAT_ROLLOUT_RETRIES="${QWEN_CHAT_ROLLOUT_RETRIES:-1}"
export TARGET_QWEN_CHAT_ROLLOUT_RETRIES="${TARGET_QWEN_CHAT_ROLLOUT_RETRIES:-${QWEN_CHAT_ROLLOUT_RETRIES}}"
export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-128}"
export VLLM_ENABLE_CHUNKED_PREFILL="${VLLM_ENABLE_CHUNKED_PREFILL:-1}"

# Local Qwen function-calling is deployment-sensitive. For this training
# profile, use the oracle parsed OfficeQA pages directly by default; set
# OFFICEQA_USE_LOCAL_TOOLS=true only after the startup tool smoke passes.
export OFFICEQA_USE_LOCAL_TOOLS="${OFFICEQA_USE_LOCAL_TOOLS:-false}"

# Dataset-specific defaults inherited from the main launcher unless overridden:
#   DocVQA batch=32, OfficeQA batch=16, SpreadsheetBench batch=16;
#   full split train/val/test because *_ENV_NUM=0.

echo "============================================================"
echo "  V3 2x128 Three-Dataset Training"
echo "============================================================"
echo "  project:          ${PROJECT_ROOT}"
echo "  datasets:         ${DATASETS}"
echo "  max_parallel:     ${MAX_PARALLEL}"
echo "  workers:          docvqa=${DOCVQA_WORKERS}, officeqa=${OFFICEQA_WORKERS}, spreadsheetbench=${SPREADSHEETBENCH_WORKERS}"
echo "  optimizer:        ${OPTIMIZER_MODEL}"
echo "  target:           ${TARGET_MODEL}"
echo "  qwen_model_path:  ${MODEL_PATH}"
echo "  qwen_gpus:        ${QWEN_CUDA_VISIBLE_DEVICES}"
echo "  qwen_tp:          ${VLLM_TENSOR_PARALLEL_SIZE}"
echo "  max_model_len:    ${MAX_MODEL_LEN}"
echo "  max_tokens:       ${TARGET_MAX_COMPLETION_TOKENS}"
echo "  rollout_retries:  ${TARGET_QWEN_CHAT_ROLLOUT_RETRIES}"
echo "  fallback_top_k:   ${TYPE_GUIDED_FALLBACK_TOP_K}"
echo "  merge_workers:    leaf=${TYPE_GUIDED_LEAF_MERGE_WORKERS}, mid=${TYPE_GUIDED_MID_MERGE_WORKERS}"
echo "  officeqa_tools:   ${OFFICEQA_USE_LOCAL_TOOLS}"
echo "  epochs:           ${NUM_EPOCHS}"
echo "============================================================"

exec bash "${SCRIPT_DIR}/run_v3_deepseek_local_qwen_parallel.sh" "$@"

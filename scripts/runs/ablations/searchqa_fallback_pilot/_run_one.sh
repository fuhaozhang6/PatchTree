#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../../resource_pool_4x_l20/_common.sh
source "${SCRIPT_DIR}/../../resource_pool_4x_l20/_common.sh"

: "${PILOT_NAME:?PILOT_NAME is required}"
: "${PILOT_TREE_DEPTH:?PILOT_TREE_DEPTH is required}"
: "${PILOT_FALLBACK:?PILOT_FALLBACK is required (true or false)}"

# Capture caller overrides before the shared helper fills its resource-pool
# defaults. Otherwise `${VAR:-pilot_default}` below would retain the helper's
# 65536-token / 0.90 settings instead of this pilot's safer L20 profile.
pilot_vllm_max_num_seqs="${PILOT_VLLM_MAX_NUM_SEQS:-${VLLM_MAX_NUM_SEQS:-128}}"
pilot_vllm_max_num_batched_tokens="${PILOT_VLLM_MAX_NUM_BATCHED_TOKENS:-${VLLM_MAX_NUM_BATCHED_TOKENS:-32768}}"
pilot_max_model_len="${PILOT_MAX_MODEL_LEN:-${MAX_MODEL_LEN:-65536}}"
pilot_gpu_memory_utilization="${PILOT_GPU_MEMORY_UTILIZATION:-${GPU_MEMORY_UTILIZATION:-0.85}}"

configure_single_l20
configure_deepseek_official

# One script owns one L20 and one local vLLM. Keep the optimizer provider,
# target model, seed, data order, and all non-ablation settings identical.
export DATASETS=searchqa
export MAX_PARALLEL=1
export WAIT_FOR_JOBS=1
export VLLM_TENSOR_PARALLEL_SIZE=1
export VLLM_MAX_NUM_SEQS="${pilot_vllm_max_num_seqs}"
export VLLM_MAX_NUM_BATCHED_TOKENS="${pilot_vllm_max_num_batched_tokens}"
export MAX_MODEL_LEN="${pilot_max_model_len}"
export GPU_MEMORY_UTILIZATION="${pilot_gpu_memory_utilization}"
export QWEN_CHAT_TIMEOUT_SECONDS="${QWEN_CHAT_TIMEOUT_SECONDS:-300}"
export TARGET_QWEN_CHAT_TIMEOUT_SECONDS="${TARGET_QWEN_CHAT_TIMEOUT_SECONDS:-300}"
export QWEN_CHAT_MAX_TOKENS="${QWEN_CHAT_MAX_TOKENS:-16384}"
export TARGET_QWEN_CHAT_MAX_TOKENS="${TARGET_QWEN_CHAT_MAX_TOKENS:-16384}"
export TARGET_MAX_COMPLETION_TOKENS="${TARGET_MAX_COMPLETION_TOKENS:-16384}"
export SEARCHQA_TARGET_MAX_COMPLETION_TOKENS="${SEARCHQA_TARGET_MAX_COMPLETION_TOKENS:-16384}"
export TARGET_QWEN_CHAT_TEMPERATURE=0.2
export TARGET_QWEN_CHAT_ENABLE_THINKING=false

# SearchQA pilot: train400 / val200, 10 optimization steps.
export NUM_EPOCHS=1
export TRAIN_SIZE=0
export LIMIT=0
export ACCUMULATION=1
export SEED="${PILOT_SEED:-42}"
export BATCH_SIZE=40
export SEARCHQA_BATCH_SIZE=40
export API_MAX_CONCURRENCY="${PILOT_API_MAX_CONCURRENCY:-128}"
export WORKERS="${PILOT_WORKERS:-128}"
export SEARCHQA_WORKERS="${PILOT_WORKERS:-128}"
export ANALYST_WORKERS="${PILOT_ANALYST_WORKERS:-48}"
export SEARCHQA_ANALYST_WORKERS="${PILOT_ANALYST_WORKERS:-48}"
export TYPE_GUIDED_PATCH_RECORD_WORKERS="${PILOT_PATCH_RECORD_WORKERS:-40}"

export LR_SCHEDULER="${PILOT_LR_SCHEDULER:-cosine}"
export LR_CONTROL_MODE=fixed
export EDIT_BUDGET="${PILOT_EDIT_BUDGET:-4}"
export MIN_EDIT_BUDGET="${PILOT_MIN_EDIT_BUDGET:-2}"
export USE_GATE=true
export GATE_METRIC=mixed
export GATE_MIXED_WEIGHT=0.5

# The root and any reconciled fallback combination always use the full val200.
# Experiments default to the full held-out test; only smoke/training-only runs
# should explicitly set PILOT_EVAL_TEST=false.
export SEARCHQA_SEL_ENV_NUM=200
export SEARCHQA_TEST_ENV_NUM=0
export EVAL_TEST="${PILOT_EVAL_TEST:-true}"

export TYPE_GUIDED_MIN_SUPPORT="${PILOT_TYPE_GUIDED_MIN_SUPPORT:-2}"
export TYPE_GUIDED_MAX_LEAF_GROUPS="${PILOT_TYPE_GUIDED_MAX_LEAF_GROUPS:-8}"
export TYPE_GUIDED_TREE_DEPTH="${PILOT_TREE_DEPTH}"
export TYPE_GUIDED_LEAF_FALLBACK="${PILOT_FALLBACK}"
export TYPE_GUIDED_ROLLOUT_REPEATS=3
export TYPE_GUIDED_TAU_SUCC=1.0
export TYPE_GUIDED_MAX_PATCH_RECORDS=40
export TYPE_GUIDED_FALLBACK_TOP_K=4
export TYPE_GUIDED_FALLBACK_TAU_CHILD=0.0
export TYPE_GUIDED_FALLBACK_SEL_ENV_NUM=40
export SEARCHQA_TYPE_GUIDED_FALLBACK_SEL_ENV_NUM=40
export TYPE_GUIDED_FALLBACK_RECONCILE="${PILOT_TYPE_GUIDED_FALLBACK_RECONCILE:-deterministic}"
export TYPE_GUIDED_FALLBACK_RECONCILE_MIN_CHILDREN="${PILOT_TYPE_GUIDED_FALLBACK_RECONCILE_MIN_CHILDREN:-2}"
export TYPE_GUIDED_CLUSTERING="${PILOT_TYPE_GUIDED_CLUSTERING:-false}"
export TYPE_GUIDED_CLUSTER_TARGET_SIZE="${PILOT_TYPE_GUIDED_CLUSTER_TARGET_SIZE:-4}"
export TYPE_GUIDED_CLUSTER_MAX_SIZE="${PILOT_TYPE_GUIDED_CLUSTER_MAX_SIZE:-8}"
export TYPE_GUIDED_LEAF_MERGE_WORKERS=8
export TYPE_GUIDED_MID_MERGE_WORKERS=4
export TYPE_GUIDED_TAIL_BANK="${PILOT_TYPE_GUIDED_TAIL_BANK:-false}"
export TYPE_GUIDED_TAIL_MIN_SUPPORT="${PILOT_TYPE_GUIDED_TAIL_MIN_SUPPORT:-2}"
export TYPE_GUIDED_TAIL_MAX_RECORDS="${PILOT_TYPE_GUIDED_TAIL_MAX_RECORDS:-40}"
export TYPE_GUIDED_TAIL_MAX_LEAF_GROUPS="${PILOT_TYPE_GUIDED_TAIL_MAX_LEAF_GROUPS:-4}"
export TYPE_GUIDED_TAIL_WINDOW_EPOCHS="${PILOT_TYPE_GUIDED_TAIL_WINDOW_EPOCHS:-3}"
export TYPE_GUIDED_TAIL_REQUIRE_CROSS_STEP="${PILOT_TYPE_GUIDED_TAIL_REQUIRE_CROSS_STEP:-true}"

export SEARCHQA_SPLIT_DIR="${SEARCHQA_SPLIT_DIR:-${PROJECT_ROOT}/data/searchqa_split}"
export SEARCHQA_MAX_TURNS=1
export SEARCHQA_EXEC_TIMEOUT="${SEARCHQA_EXEC_TIMEOUT:-300}"

RUN_STAMP="${PILOT_RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
export TS="${PILOT_SUITE_SLUG:-searchqa_fallback_pilot}_${PILOT_NAME}_seed${SEED}_${RUN_STAMP}"
export LOG_DIR="${PILOT_LOG_DIR:-${PROJECT_ROOT}/logs/${TS}}"
export OUT_BASE="${PILOT_OUT_BASE:-${PROJECT_ROOT}/outputs/${TS}}"

echo "============================================================"
echo "  SearchQA fallback pilot: ${PILOT_NAME}"
echo "============================================================"
echo "  depth/fallback:      ${TYPE_GUIDED_TREE_DEPTH}/${TYPE_GUIDED_LEAF_FALLBACK}"
echo "  train/batch/steps:   400/40/10"
echo "  repeats:             ${TYPE_GUIDED_ROLLOUT_REPEATS}"
echo "  leaf support/cap:    ${TYPE_GUIDED_MIN_SUPPORT}/${TYPE_GUIDED_MAX_LEAF_GROUPS}"
echo "  clustering:          ${TYPE_GUIDED_CLUSTERING} target/max=${TYPE_GUIDED_CLUSTER_TARGET_SIZE}/${TYPE_GUIDED_CLUSTER_MAX_SIZE}"
echo "  tail bank:           ${TYPE_GUIDED_TAIL_BANK} support=${TYPE_GUIDED_TAIL_MIN_SUPPORT}, records/leaves=${TYPE_GUIDED_TAIL_MAX_RECORDS}/${TYPE_GUIDED_TAIL_MAX_LEAF_GROUPS}"
echo "  tail window/cross:   ${TYPE_GUIDED_TAIL_WINDOW_EPOCHS}/${TYPE_GUIDED_TAIL_REQUIRE_CROSS_STEP}"
echo "  root/combination val:${SEARCHQA_SEL_ENV_NUM} (full)"
echo "  fallback:            top-k=${TYPE_GUIDED_FALLBACK_TOP_K}, random n=${SEARCHQA_TYPE_GUIDED_FALLBACK_SEL_ENV_NUM}"
echo "  fallback reconcile:  ${TYPE_GUIDED_FALLBACK_RECONCILE}"
echo "  edit budget:         ${LR_SCHEDULER} ${EDIT_BUDGET}/${MIN_EDIT_BUDGET}"
echo "  eval_test:           ${EVAL_TEST} (test_env_num=${SEARCHQA_TEST_ENV_NUM})"
echo "  seed:                ${SEED}"
echo "  optimizer:           ${OPTIMIZER_MODEL} via ${AZURE_OPENAI_ENDPOINT}"
echo "  local target:        ${TARGET_MODEL}, CUDA=${QWEN_CUDA_VISIBLE_DEVICES}"
echo "  vLLM seqs/tokens:    ${VLLM_MAX_NUM_SEQS}/${VLLM_MAX_NUM_BATCHED_TOKENS}"
echo "  vLLM memory/model:   ${GPU_MEMORY_UTILIZATION}/${MAX_MODEL_LEN}"
echo "  output:              ${OUT_BASE}"
echo "============================================================"

exec bash "${PROJECT_ROOT}/scripts/runs/multi/run_v3_deepseek_local_qwen_parallel.sh" "$@"

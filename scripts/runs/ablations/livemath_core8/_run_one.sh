#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../../resource_pool_4x_l20/_common.sh
source "${SCRIPT_DIR}/../../resource_pool_4x_l20/_common.sh"

: "${ABLATION_NAME:?ABLATION_NAME is required}"
: "${ABLATION_BATCH_SIZE:?ABLATION_BATCH_SIZE is required}"
: "${ABLATION_ROLLOUT_REPEATS:?ABLATION_ROLLOUT_REPEATS is required}"
: "${ABLATION_TREE_DEPTH:?ABLATION_TREE_DEPTH is required}"

configure_single_l20
configure_deepseek_official

# This helper runs one experiment. Pair launchers start the first copy in
# detached mode, then attach the second copy to the same local vLLM endpoint.
export DATASETS=livemath
export MAX_PARALLEL=1
export WAIT_FOR_JOBS="${ABLATION_WAIT_FOR_JOBS:-1}"
export VLLM_TENSOR_PARALLEL_SIZE=1
export VLLM_MAX_NUM_SEQS=128
export VLLM_MAX_NUM_BATCHED_TOKENS=65536
export MAX_MODEL_LEN=65536
export GPU_MEMORY_UTILIZATION=0.90
export QWEN_CHAT_TIMEOUT_SECONDS=300
export TARGET_QWEN_CHAT_TIMEOUT_SECONDS=300
export QWEN_CHAT_MAX_TOKENS=16384
export TARGET_QWEN_CHAT_MAX_TOKENS=16384
export TARGET_MAX_COMPLETION_TOKENS=16384
export LIVEMATH_TARGET_MAX_COMPLETION_TOKENS=16384
export TARGET_QWEN_CHAT_TEMPERATURE=0.2
export TARGET_QWEN_CHAT_ENABLE_THINKING=false

# Strictly fixed training controls. Use ABLATION_SEED for later replications.
export NUM_EPOCHS=4
export TRAIN_SIZE=0
export LIMIT=0
export ACCUMULATION=1
export SEED="${ABLATION_SEED:-42}"
export BATCH_SIZE="${ABLATION_BATCH_SIZE}"
export LIVEMATH_BATCH_SIZE="${ABLATION_BATCH_SIZE}"
export API_MAX_CONCURRENCY="${ABLATION_API_MAX_CONCURRENCY:-96}"
export WORKERS="${ABLATION_WORKERS:-96}"
export LIVEMATH_WORKERS="${ABLATION_WORKERS:-96}"
export ANALYST_WORKERS="${ABLATION_ANALYST_WORKERS:-32}"
export LIVEMATH_ANALYST_WORKERS="${ABLATION_ANALYST_WORKERS:-32}"
export TYPE_GUIDED_PATCH_RECORD_WORKERS="${ABLATION_PATCH_RECORD_WORKERS:-32}"

export LR_SCHEDULER=cosine
export LR_CONTROL_MODE=fixed
export EDIT_BUDGET=4
export MIN_EDIT_BUDGET=2
export USE_GATE=true
export GATE_METRIC=mixed
export GATE_MIXED_WEIGHT=0.5

# Select only on the 18-item validation split. Test evaluation is intentionally
# deferred until the eight runs have completed.
export LIVEMATH_SEL_ENV_NUM=18
export LIVEMATH_TEST_ENV_NUM=0
export EVAL_TEST="${ABLATION_EVAL_TEST:-false}"

# PatchTree-v4 controls. Only the three ABLATION_* values vary across scripts.
export TYPE_GUIDED_MIN_SUPPORT=2
export TYPE_GUIDED_MAX_LEAF_GROUPS=8
export TYPE_GUIDED_TREE_DEPTH="${ABLATION_TREE_DEPTH}"
export TYPE_GUIDED_LEAF_FALLBACK=true
export TYPE_GUIDED_ROLLOUT_REPEATS="${ABLATION_ROLLOUT_REPEATS}"
export TYPE_GUIDED_TAU_SUCC=1.0
export TYPE_GUIDED_MAX_PATCH_RECORDS=32
export TYPE_GUIDED_FALLBACK_TOP_K=4
export TYPE_GUIDED_FALLBACK_TAU_CHILD=0.0
export TYPE_GUIDED_FALLBACK_SEL_ENV_NUM=0
export TYPE_GUIDED_FALLBACK_RECONCILE=deterministic
export TYPE_GUIDED_FALLBACK_RECONCILE_MIN_CHILDREN=2
export TYPE_GUIDED_CLUSTERING=false
export TYPE_GUIDED_CLUSTER_TARGET_SIZE=4
export TYPE_GUIDED_CLUSTER_MAX_SIZE=8
export TYPE_GUIDED_LEAF_MERGE_WORKERS=8
export TYPE_GUIDED_MID_MERGE_WORKERS=4
export TYPE_GUIDED_TAIL_BANK=false
export TYPE_GUIDED_TAIL_MIN_SUPPORT=2
export TYPE_GUIDED_TAIL_MAX_RECORDS=32
export TYPE_GUIDED_TAIL_MAX_LEAF_GROUPS=4
export TYPE_GUIDED_TAIL_WINDOW_EPOCHS=3
export TYPE_GUIDED_TAIL_REQUIRE_CROSS_STEP=true

export LIVEMATH_SPLIT_DIR="${LIVEMATH_SPLIT_DIR:-${PROJECT_ROOT}/data/livemathematicianbench_split}"
export LIVEMATH_SHUFFLE_CHOICES=true
export LIVEMATH_USE_THEOREM=false
export LIVEMATH_USE_SKETCH=false
export LIVEMATH_MAX_TURNS=1

RUN_STAMP="${ABLATION_RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
export TS="${ABLATION_SUITE_SLUG:-livemath_core8}_${ABLATION_NAME}_seed${SEED}_${RUN_STAMP}"
export LOG_DIR="${ABLATION_LOG_DIR:-${PROJECT_ROOT}/logs/${TS}}"
export OUT_BASE="${ABLATION_OUT_BASE:-${PROJECT_ROOT}/outputs/${TS}}"

echo "============================================================"
echo "  ${ABLATION_SUITE_LABEL:-LiveMath core-8} ablation: ${ABLATION_NAME}"
echo "============================================================"
echo "  batch/repeats/depth: ${BATCH_SIZE}/${TYPE_GUIDED_ROLLOUT_REPEATS}/${TYPE_GUIDED_TREE_DEPTH}"
echo "  epochs/seed:         ${NUM_EPOCHS}/${SEED}"
echo "  target workers:      ${LIVEMATH_WORKERS}"
echo "  analyst workers:     ${LIVEMATH_ANALYST_WORKERS}"
echo "  vLLM max seqs:       ${VLLM_MAX_NUM_SEQS}"
echo "  optimizer:           ${OPTIMIZER_MODEL} via ${AZURE_OPENAI_ENDPOINT}"
echo "  fallback/tail:       top-k=${TYPE_GUIDED_FALLBACK_TOP_K} / ${TYPE_GUIDED_TAIL_BANK}"
echo "  eval_test:           ${EVAL_TEST}"
echo "  output:              ${OUT_BASE}"
echo "============================================================"

exec bash "${PROJECT_ROOT}/scripts/runs/multi/run_v3_deepseek_local_qwen_parallel.sh" "$@"

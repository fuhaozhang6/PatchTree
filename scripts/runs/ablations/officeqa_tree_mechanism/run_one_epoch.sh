#!/usr/bin/env bash
set -euo pipefail

# OfficeQA one-epoch / one-update PatchTree mechanism experiment.
#
# Run the same launcher with one variant per resource-pool GPU:
#   OFFICEQA_TREE_VARIANT=flat      bash run_one_epoch.sh
#   OFFICEQA_TREE_VARIANT=bottom_up bash run_one_epoch.sh
#   OFFICEQA_TREE_VARIANT=full      bash run_one_epoch.sh
#
# Variants:
#   flat      : PatchRecords -> Root (depth=1, no Leaf/Mid tree)
#   bottom_up : PatchRecords -> Cluster -> Leaf -> Mid -> Root, no fallback
#   full      : same bottom-up tree, then evaluate/reconcile Root children when
#               the Root gate rejects

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../../resource_pool_4x_l20/_common.sh
source "${SCRIPT_DIR}/../../resource_pool_4x_l20/_common.sh"

VARIANT="${OFFICEQA_TREE_VARIANT:-full}"
case "${VARIANT}" in
  flat)
    experiment_id=p0_flat_records_to_root
    tree_depth=1
    clustering=false
    fallback=false
    ;;
  bottom_up)
    experiment_id=p1_bottom_up_tree
    tree_depth=3
    clustering=true
    fallback=false
    ;;
  full)
    experiment_id=p2_full_tree_top_down
    tree_depth=3
    clustering=true
    fallback=true
    ;;
  *)
    echo "ERROR: OFFICEQA_TREE_VARIANT must be flat, bottom_up, or full; got: ${VARIANT}" >&2
    exit 2
    ;;
esac

# Preserve explicit caller overrides before the resource helper supplies its
# general defaults.
pilot_vllm_max_num_seqs="${OFFICEQA_VLLM_MAX_NUM_SEQS:-${VLLM_MAX_NUM_SEQS:-64}}"
pilot_vllm_max_num_batched_tokens="${OFFICEQA_VLLM_MAX_NUM_BATCHED_TOKENS:-${VLLM_MAX_NUM_BATCHED_TOKENS:-65536}}"
pilot_max_model_len="${OFFICEQA_MAX_MODEL_LEN:-${MAX_MODEL_LEN:-65536}}"
pilot_gpu_memory_utilization="${OFFICEQA_GPU_MEMORY_UTILIZATION:-${GPU_MEMORY_UTILIZATION:-0.85}}"

configure_single_l20
configure_deepseek_official

export DATASETS=officeqa
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
export OFFICEQA_TARGET_MAX_COMPLETION_TOKENS="${OFFICEQA_TARGET_MAX_COMPLETION_TOKENS:-16384}"
export TARGET_QWEN_CHAT_TEMPERATURE="${TARGET_QWEN_CHAT_TEMPERATURE:-0.2}"
export TARGET_QWEN_CHAT_ENABLE_THINKING=false

# Full OfficeQA train50 in one batch: one epoch is exactly one optimization
# step. All validation and final test items are used.
export NUM_EPOCHS=1
export TRAIN_SIZE=0
export LIMIT=0
export ACCUMULATION=1
export SEED="${OFFICEQA_TREE_SEED:-42}"
export BATCH_SIZE=50
export OFFICEQA_BATCH_SIZE=50
export API_MAX_CONCURRENCY="${OFFICEQA_API_MAX_CONCURRENCY:-64}"
export WORKERS="${OFFICEQA_TREE_WORKERS:-48}"
export OFFICEQA_WORKERS="${OFFICEQA_TREE_WORKERS:-48}"
export ANALYST_WORKERS="${OFFICEQA_TREE_ANALYST_WORKERS:-24}"
export OFFICEQA_ANALYST_WORKERS="${OFFICEQA_TREE_ANALYST_WORKERS:-24}"
export TYPE_GUIDED_PATCH_RECORD_WORKERS="${OFFICEQA_TREE_PATCH_RECORD_WORKERS:-24}"

export LR_SCHEDULER=constant
export LR_CONTROL_MODE=fixed
export EDIT_BUDGET="${OFFICEQA_TREE_EDIT_BUDGET:-4}"
export MIN_EDIT_BUDGET="${OFFICEQA_TREE_EDIT_BUDGET:-4}"
export USE_GATE=true
export GATE_METRIC=mixed
export GATE_MIXED_WEIGHT=0.5
export EVAL_TEST=true
export OFFICEQA_SEL_ENV_NUM=24
export OFFICEQA_TEST_ENV_NUM=0

export TYPE_GUIDED_TREE_DEPTH="${tree_depth}"
export TYPE_GUIDED_LEAF_FALLBACK="${fallback}"
export TYPE_GUIDED_MIN_SUPPORT=1
export TYPE_GUIDED_MAX_LEAF_GROUPS=50
export TYPE_GUIDED_ROLLOUT_REPEATS=3
export TYPE_GUIDED_MAX_PATCH_RECORDS=50
export TYPE_GUIDED_TAU_SUCC=1.0
export TYPE_GUIDED_CLUSTERING="${clustering}"
export TYPE_GUIDED_CLUSTER_TARGET_SIZE=2
export TYPE_GUIDED_CLUSTER_MAX_SIZE=4
export TYPE_GUIDED_LEAF_MERGE_WORKERS="${OFFICEQA_TREE_LEAF_MERGE_WORKERS:-12}"
export TYPE_GUIDED_MID_MERGE_WORKERS="${OFFICEQA_TREE_MID_MERGE_WORKERS:-8}"

# OfficeQA val contains only 24 items, so evaluate every direct Root child on
# the full val split when the full-tree Root is rejected.
export TYPE_GUIDED_FALLBACK_EVAL_ALL_LEAVES=true
export TYPE_GUIDED_FALLBACK_TOP_K=0
export TYPE_GUIDED_FALLBACK_TAU_CHILD=0.0
export TYPE_GUIDED_FALLBACK_SEL_ENV_NUM=24
export TYPE_GUIDED_FALLBACK_RECONCILE=deterministic
export TYPE_GUIDED_FALLBACK_RECONCILE_MIN_CHILDREN=2

# Tail merging is a separate experiment and would confound this one-step test.
export TYPE_GUIDED_TAIL_BANK=false
export TYPE_GUIDED_TAIL_MIN_SUPPORT=2
export TYPE_GUIDED_TAIL_MAX_RECORDS=50
export TYPE_GUIDED_TAIL_MAX_LEAF_GROUPS=12
export TYPE_GUIDED_TAIL_WINDOW_EPOCHS=1
export TYPE_GUIDED_TAIL_REQUIRE_CROSS_STEP=true

export OFFICEQA_SPLIT_DIR="${OFFICEQA_SPLIT_DIR:-${PROJECT_ROOT}/data/officeqa_split}"
export OFFICEQA_DOCS_DIR="${OFFICEQA_DOCS_DIR:-${PROJECT_ROOT}/data/officeqa_docs_official}"
export OFFICEQA_USE_LOCAL_TOOLS=true
export OFFICEQA_SEARCH_MODE=offline
export OFFICEQA_EXEC_TIMEOUT="${OFFICEQA_EXEC_TIMEOUT:-300}"

RUN_STAMP="${OFFICEQA_TREE_RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
export TS="officeqa_tree_mechanism_${experiment_id}_seed${SEED}_${RUN_STAMP}"
export LOG_DIR="${OFFICEQA_TREE_LOG_DIR:-${PROJECT_ROOT}/logs/${TS}}"
export OUT_BASE="${OFFICEQA_TREE_OUT_BASE:-${PROJECT_ROOT}/outputs/${TS}}"

echo "============================================================"
echo "  OfficeQA PatchTree mechanism: ${experiment_id}"
echo "============================================================"
echo "  variant:             ${VARIANT}"
echo "  structure:           depth=${TYPE_GUIDED_TREE_DEPTH}, clustering=${TYPE_GUIDED_CLUSTERING}"
echo "  fallback:            ${TYPE_GUIDED_LEAF_FALLBACK}, all children/full val24"
echo "  train/epoch/step:    50/1/1"
echo "  batch/repeats:       50/3 (150 trajectories)"
echo "  records/support/cap: 50/1/50"
echo "  cluster target/max:  ${TYPE_GUIDED_CLUSTER_TARGET_SIZE}/${TYPE_GUIDED_CLUSTER_MAX_SIZE}"
echo "  val/test:            24/172 (full)"
echo "  seed:                ${SEED}"
echo "  optimizer:           ${OPTIMIZER_MODEL} via ${AZURE_OPENAI_ENDPOINT}"
echo "  target:              ${TARGET_MODEL}, CUDA=${QWEN_CUDA_VISIBLE_DEVICES}"
echo "  vLLM seqs/tokens:    ${VLLM_MAX_NUM_SEQS}/${VLLM_MAX_NUM_BATCHED_TOKENS}"
echo "  output:              ${OUT_BASE}"
echo "============================================================"

exec bash "${PROJECT_ROOT}/scripts/runs/multi/run_v3_deepseek_local_qwen_parallel.sh" "$@"

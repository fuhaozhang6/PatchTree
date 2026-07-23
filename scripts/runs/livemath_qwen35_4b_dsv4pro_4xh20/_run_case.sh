#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "${SCRIPT_DIR}/_common.sh"

[[ $# -eq 3 ]] || suite_fail "usage: _run_case.sh CASE_NAME QWEN_BASE_URL QUEUE_ID"
CASE_NAME="$1"
QWEN_BASE_URL="$2"
QUEUE_ID="$3"

case_epochs="${NUM_EPOCHS}"
case_batch_size="${BATCH_SIZE}"
case_repeats="${ROLLOUT_REPEATS}"
case_skill_init="${INITIAL_SKILL_PATH}"
case_eval_test="${EVAL_TEST}"
case_cfg=()

# Full-method defaults. Every ablation below changes only its named factor.
case_cfg+=(
  "optimizer.type_guided_min_support=1"
  "optimizer.type_guided_max_leaf_groups=8"
  "optimizer.type_guided_tree_builder=recursive"
  "optimizer.type_guided_max_tree_depth=4"
  "optimizer.type_guided_merge_target_children=3"
  "optimizer.type_guided_merge_max_children=4"
  "optimizer.type_guided_merge_strategy=hierarchical"
  "optimizer.type_guided_grouping_mode=type"
  "optimizer.type_guided_grouping_seed=${SEED}"
  "optimizer.type_guided_top_mode=conservative_root"
  "optimizer.type_guided_fallback_enabled=true"
  "optimizer.type_guided_fallback_max_hops=-1"
  "optimizer.type_guided_fallback_allow_leaf=true"
  "optimizer.type_guided_fallback_min_leaf_coverage=1"
  "optimizer.type_guided_validation_budget=16"
  "optimizer.type_guided_fallback_sel_env_num=12"
  "optimizer.type_guided_fallback_reconcile=llm_fuse"
  "optimizer.type_guided_rollout_repeats=${case_repeats}"
  "optimizer.type_guided_tau_succ=0.5"
  "optimizer.type_guided_max_patch_records=24"
  "optimizer.type_guided_patch_record_workers=${PATCH_RECORD_WORKERS}"
  "optimizer.type_guided_leaf_merge_workers=${LEAF_MERGE_WORKERS}"
  "optimizer.type_guided_mid_merge_workers=${MID_MERGE_WORKERS}"
  "optimizer.type_guided_clustering=false"
  "optimizer.type_guided_tail_bank=false"
  "optimizer.learning_rate=999"
  "optimizer.min_learning_rate=999"
  "evaluation.gate_metric=mixed"
  "evaluation.gate_mixed_weight=0.5"
  "env.max_completion_tokens=${TARGET_MAX_COMPLETION_TOKENS}"
  "env.exec_timeout=300"
)

case "${CASE_NAME}" in
  full)
    ;;
  rollout_r2)
    case_repeats=2
    ;;
  rollout_r8)
    case_repeats=8
    ;;
  batch_12)
    case_batch_size=12
    ;;
  batch_35)
    case_batch_size=35
    ;;
  merge_concat)
    case_cfg+=("optimizer.type_guided_merge_strategy=concat")
    ;;
  flat_fuse_fixed_real_root|merge_flat_fuse|fixed_real_root)
    # These labels describe the same executable mechanism: typed leaves are
    # sent to one Root LLM call, without recursive intermediate nodes.
    case_cfg+=(
      "optimizer.type_guided_merge_strategy=flat_fuse"
      "optimizer.type_guided_tree_builder=fixed"
      "optimizer.type_guided_tree_depth=2"
      "optimizer.type_guided_top_mode=real_root"
    )
    ;;
  dynamic_virtual_root)
    case_cfg+=("optimizer.type_guided_top_mode=virtual_root")
    ;;
  fallback_none)
    case_cfg+=("optimizer.type_guided_fallback_enabled=false")
    ;;
  fallback_children)
    case_cfg+=(
      "optimizer.type_guided_fallback_max_hops=1"
      "optimizer.type_guided_fallback_allow_leaf=true"
    )
    ;;
  fallback_internal)
    case_cfg+=(
      "optimizer.type_guided_fallback_max_hops=-1"
      "optimizer.type_guided_fallback_allow_leaf=false"
    )
    ;;
  cluster_random)
    case_cfg+=("optimizer.type_guided_grouping_mode=random")
    ;;
  cluster_success_aware)
    case_cfg+=("optimizer.type_guided_grouping_mode=success_then_type")
    ;;
  system_prompt_only)
    case_epochs=0
    case_skill_init=/dev/null
    case_eval_test=true
    ;;
  init_skill_only)
    case_epochs=0
    case_eval_test=true
    ;;
  *)
    suite_fail "unknown CASE_NAME: ${CASE_NAME}"
    ;;
esac

# Replace the base repeat setting after the case-specific choice is known.
case_cfg+=("optimizer.type_guided_rollout_repeats=${case_repeats}")

case_out="${SEED_ROOT}/${CASE_NAME}"
case_log="${LOG_ROOT}/queue_${QUEUE_ID}/${CASE_NAME}.log"
done_marker="${case_out}/.suite_complete"

if [[ -f "${done_marker}" ]] && ! suite_truthy "${FORCE_RERUN}"; then
  echo "[skip] seed=${SEED} queue=${QUEUE_ID} case=${CASE_NAME} already complete"
  exit 0
fi

cmd=(
  "${PYTHON_BIN}" -u "${PROJECT_ROOT}/scripts/cli/train.py"
  --config "${PROJECT_ROOT}/configs/livemathematicianbench/default.yaml"
  --optimizer_backend openai_chat
  --target_backend qwen_chat
  --optimizer_model "${OPTIMIZER_MODEL}"
  --target_model "${TARGET_MODEL}"
  --target_qwen_chat_base_url "${QWEN_BASE_URL}"
  --target_qwen_chat_api_key dummy
  --target_qwen_chat_temperature "${TARGET_QWEN_CHAT_TEMPERATURE}"
  --target_qwen_chat_timeout_seconds "${TARGET_QWEN_CHAT_TIMEOUT_SECONDS}"
  --target_qwen_chat_max_tokens "${TARGET_MAX_COMPLETION_TOKENS}"
  --target_qwen_chat_enable_thinking "${TARGET_QWEN_CHAT_ENABLE_THINKING}"
  --reasoning_effort "${REASONING_EFFORT}"
  --skill_init "${case_skill_init}"
  --num_epochs "${case_epochs}"
  --train_size "${TRAIN_SIZE}"
  --batch_size "${case_batch_size}"
  --accumulation 1
  --seed "${SEED}"
  --workers "${TARGET_WORKERS}"
  --analyst_workers "${ANALYST_WORKERS}"
  --sel_env_num "${SEL_ENV_NUM}"
  --test_env_num "${TEST_ENV_NUM}"
  --eval_test "${case_eval_test}"
  --use_gate true
  --split_mode split_dir
  --split_dir "${LIVEMATH_SPLIT_DIR}"
  --limit "${LIMIT}"
  --max_turns 1
  --shuffle_choices true
  --use_theorem false
  --use_sketch false
  --out_root "${case_out}"
  --type_guided_cache_dir "${case_out}/type_guided_cache"
  --cfg-options "${case_cfg[@]}"
)

echo "[case] seed=${SEED} queue=${QUEUE_ID} name=${CASE_NAME}"
echo "       endpoint=${QWEN_BASE_URL}"
echo "       out=${case_out}"
printf '[cmd]'
suite_quote_cmd "${cmd[@]}"

if suite_truthy "${DRY_RUN}"; then
  echo "[dry-run] command not executed"
  exit 0
fi

mkdir -p "${case_out}" "$(dirname "${case_log}")"
set +e
"${cmd[@]}" 2>&1 | tee "${case_log}"
status=${PIPESTATUS[0]}
set -e
if [[ "${status}" -eq 0 ]]; then
  touch "${done_marker}"
  echo "[done] seed=${SEED} queue=${QUEUE_ID} case=${CASE_NAME}"
else
  echo "[failed] seed=${SEED} queue=${QUEUE_ID} case=${CASE_NAME} status=${status}" >&2
fi
exit "${status}"

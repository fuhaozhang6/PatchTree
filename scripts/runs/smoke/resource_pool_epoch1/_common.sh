#!/usr/bin/env bash

SMOKE_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SMOKE_PROJECT_ROOT="$(cd "${SMOKE_SCRIPT_DIR}/../../../.." && pwd)"

smoke_truthy() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

configure_epoch1_smoke() {
  local group="$1"
  local stamp
  stamp="${SMOKE_RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"

  # Exactly one complete epoch over a two-item limited split: one training step.
  export NUM_EPOCHS=1
  export TRAIN_SIZE=0
  export LIMIT=2
  export BATCH_SIZE=2
  export ACCUMULATION=1
  export SEED="${SMOKE_SEED:-42}"
  export EVAL_TEST=false

  # Keep the complete PatchTree-v4 path, but minimize its work.
  export EDIT_BUDGET=2
  export MIN_EDIT_BUDGET=1
  export TYPE_GUIDED_TREE_DEPTH=2
  export TYPE_GUIDED_MIN_SUPPORT=2
  export TYPE_GUIDED_MAX_LEAF_GROUPS=2
  export TYPE_GUIDED_ROLLOUT_REPEATS=2
  export TYPE_GUIDED_MAX_PATCH_RECORDS=2
  export TYPE_GUIDED_PATCH_RECORD_WORKERS=4
  export TYPE_GUIDED_LEAF_MERGE_WORKERS=2
  export TYPE_GUIDED_MID_MERGE_WORKERS=1
  export TYPE_GUIDED_LEAF_FALLBACK=true
  export TYPE_GUIDED_FALLBACK_TOP_K=1
  export TYPE_GUIDED_FALLBACK_SEL_ENV_NUM=1
  export TYPE_GUIDED_FALLBACK_RECONCILE=deterministic
  export TYPE_GUIDED_CLUSTERING=false
  export TYPE_GUIDED_TAIL_BANK=false

  # Smoke concurrency is intentionally small; startup/runtime settings match
  # the validated formal-training profile.
  export API_MAX_CONCURRENCY=8
  export WORKERS=8
  export ANALYST_WORKERS=4
  export VLLM_MAX_NUM_SEQS=128
  export VLLM_MAX_NUM_BATCHED_TOKENS=65536
  export MAX_MODEL_LEN=65536
  export GPU_MEMORY_UTILIZATION=0.90
  export VLLM_ENABLE_CHUNKED_PREFILL=1
  export VLLM_WAIT_SECONDS="${VLLM_WAIT_SECONDS:-1200}"

  export DEEPSEEK_THINKING=disabled
  export REASONING_EFFORT=''
  export QWEN_CHAT_ENABLE_THINKING=false
  export TARGET_QWEN_CHAT_ENABLE_THINKING=false
  export TARGET_QWEN_CHAT_TEMPERATURE=0.2
  export QWEN_CHAT_TIMEOUT_SECONDS=300
  export TARGET_QWEN_CHAT_TIMEOUT_SECONDS=300
  export QWEN_CHAT_MAX_TOKENS=16384
  export TARGET_QWEN_CHAT_MAX_TOKENS=16384
  export TARGET_MAX_COMPLETION_TOKENS=16384

  export TS="${TS:-smoke_epoch1_${group}_${stamp}}"
  export LOG_DIR="${LOG_DIR:-${SMOKE_PROJECT_ROOT}/logs/${TS}}"
  export OUT_BASE="${OUT_BASE:-${SMOKE_PROJECT_ROOT}/outputs/${TS}}"
}

verify_epoch1_smoke() {
  if smoke_truthy "${DRY_RUN:-0}"; then
    echo "[dry-run] Skip artifact verification."
    return 0
  fi
  python "${SMOKE_SCRIPT_DIR}/verify_smoke.py" \
    --out-base "${OUT_BASE}" \
    --expected-limit "${LIMIT}" \
    --datasets "$@"
}

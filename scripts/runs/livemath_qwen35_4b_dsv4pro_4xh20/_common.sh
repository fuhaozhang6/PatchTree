#!/usr/bin/env bash

# Shared settings for the four-H20 LiveMath ablation suite.
# Source this file; do not execute it directly.

SUITE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SUITE_DIR}/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

suite_fail() {
  echo "ERROR: $*" >&2
  exit 1
}

suite_truthy() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

suite_quote_cmd() {
  local arg
  for arg in "$@"; do
    printf ' %q' "${arg}"
  done
  printf '\n'
}

suite_require_layout() {
  command -v "${PYTHON_BIN}" >/dev/null 2>&1 \
    || suite_fail "Python not found: ${PYTHON_BIN}"
  [[ -f "${PROJECT_ROOT}/scripts/cli/train.py" ]] \
    || suite_fail "train.py not found under ${PROJECT_ROOT}"
  [[ -f "${PROJECT_ROOT}/configs/livemathematicianbench/default.yaml" ]] \
    || suite_fail "LiveMath config not found"
  [[ -d "${LIVEMATH_SPLIT_DIR}" ]] \
    || suite_fail "LiveMath split not found: ${LIVEMATH_SPLIT_DIR}"
  [[ -f "${INITIAL_SKILL_PATH}" ]] \
    || suite_fail "Initial skill not found: ${INITIAL_SKILL_PATH}"
  if ! suite_truthy "${DRY_RUN}"; then
    command -v vllm >/dev/null 2>&1 || suite_fail "vllm command not found"
    [[ -d "${MODEL_PATH}" ]] || suite_fail "MODEL_PATH not found: ${MODEL_PATH}"
    [[ -n "${DEEPSEEK_API_KEY}" ]] \
      || suite_fail "Set DEEPSEEK_API_KEY (DS_API_KEY is also accepted)"
  fi
}

export MODEL_PATH="${MODEL_PATH:-/ai-app-vepfs/modelscope/models/Qwen/Qwen3.5-4B}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen/Qwen3.5-4B}"
export TARGET_MODEL="${TARGET_MODEL:-${SERVED_MODEL_NAME}}"
export OPTIMIZER_MODEL="${OPTIMIZER_MODEL:-deepseek-v4-pro}"

export DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY:-${DS_API_KEY:-${DEEPSEEK_OFFICIAL_API_KEY:-}}}"
export DEEPSEEK_BASE_URL="${DEEPSEEK_BASE_URL:-https://api.deepseek.com}"
export AZURE_OPENAI_ENDPOINT="${DEEPSEEK_BASE_URL}"
export AZURE_OPENAI_API_KEY="${DEEPSEEK_API_KEY}"
export AZURE_OPENAI_AUTH_MODE=openai_compatible
export AZURE_OPENAI_API_VERSION=openai-compat
export OPTIMIZER_AZURE_OPENAI_ENDPOINT="${DEEPSEEK_BASE_URL}"
export OPTIMIZER_AZURE_OPENAI_API_KEY="${DEEPSEEK_API_KEY}"
export OPTIMIZER_AZURE_OPENAI_AUTH_MODE=openai_compatible
export OPTIMIZER_AZURE_OPENAI_API_VERSION=openai-compat
export DEEPSEEK_THINKING="${DEEPSEEK_THINKING:-enabled}"
export DEEPSEEK_OFFICIAL_THINKING="${DEEPSEEK_OFFICIAL_THINKING:-${DEEPSEEK_THINKING}}"
export REASONING_EFFORT="${REASONING_EFFORT:-high}"

export LIVEMATH_SPLIT_DIR="${LIVEMATH_SPLIT_DIR:-${PROJECT_ROOT}/data/livemathematicianbench_split}"
export INITIAL_SKILL_PATH="${INITIAL_SKILL_PATH:-${PROJECT_ROOT}/skillopt/envs/livemathematicianbench/skills/initial.md}"

export SEED="${SEED:-42}"
export NUM_EPOCHS="${NUM_EPOCHS:-4}"
export TRAIN_SIZE="${TRAIN_SIZE:-0}"
export BATCH_SIZE="${BATCH_SIZE:-18}"
export ROLLOUT_REPEATS="${ROLLOUT_REPEATS:-4}"
export SEL_ENV_NUM="${SEL_ENV_NUM:-0}"
export TEST_ENV_NUM="${TEST_ENV_NUM:-0}"
export EVAL_TEST="${EVAL_TEST:-true}"
export LIMIT="${LIMIT:-0}"

# One process per GPU. Four simultaneous cases use at most 4*48=192 optimizer
# analyst calls, below the official account-wide limit of 256.
export TARGET_WORKERS="${TARGET_WORKERS:-96}"
export ANALYST_WORKERS="${ANALYST_WORKERS:-48}"
export PATCH_RECORD_WORKERS="${PATCH_RECORD_WORKERS:-24}"
export LEAF_MERGE_WORKERS="${LEAF_MERGE_WORKERS:-4}"
export MID_MERGE_WORKERS="${MID_MERGE_WORKERS:-4}"
export API_MAX_CONCURRENCY="${API_MAX_CONCURRENCY:-64}"

export TARGET_MAX_COMPLETION_TOKENS="${TARGET_MAX_COMPLETION_TOKENS:-16384}"
export TARGET_QWEN_CHAT_TIMEOUT_SECONDS="${TARGET_QWEN_CHAT_TIMEOUT_SECONDS:-300}"
export TARGET_QWEN_CHAT_TEMPERATURE="${TARGET_QWEN_CHAT_TEMPERATURE:-0.2}"
export TARGET_QWEN_CHAT_ENABLE_THINKING="${TARGET_QWEN_CHAT_ENABLE_THINKING:-false}"

export VLLM_BASE_PORT="${VLLM_BASE_PORT:-8100}"
export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-128}"
export VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-65536}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
export VLLM_WAIT_SECONDS="${VLLM_WAIT_SECONDS:-600}"
export STOP_VLLM_ON_EXIT="${STOP_VLLM_ON_EXIT:-1}"

export DRY_RUN="${DRY_RUN:-0}"
export FORCE_RERUN="${FORCE_RERUN:-0}"
export KEEP_GOING="${KEEP_GOING:-1}"
export SMOKE="${SMOKE:-0}"

export RUN_TAG="${RUN_TAG:-livemath_qwen35_4b_dsv4pro_4xh20}"
export SUITE_ROOT="${SUITE_ROOT:-${PROJECT_ROOT}/outputs/${RUN_TAG}}"
export SEED_ROOT="${SEED_ROOT:-${SUITE_ROOT}/seed_${SEED}}"
export LOG_ROOT="${LOG_ROOT:-${PROJECT_ROOT}/logs/${RUN_TAG}/seed_${SEED}}"

if suite_truthy "${SMOKE}"; then
  export NUM_EPOCHS="${SMOKE_EPOCHS:-1}"
  export TRAIN_SIZE="${SMOKE_TRAIN_SIZE:-8}"
  export SEL_ENV_NUM="${SMOKE_SEL_ENV_NUM:-8}"
  export TEST_ENV_NUM="${SMOKE_TEST_ENV_NUM:-8}"
fi

[[ "${SEED}" =~ ^[0-9]+$ ]] || suite_fail "SEED must be an integer"
[[ "${API_MAX_CONCURRENCY}" =~ ^[0-9]+$ ]] \
  || suite_fail "API_MAX_CONCURRENCY must be an integer"
[[ "${TARGET_WORKERS}" =~ ^[0-9]+$ ]] || suite_fail "TARGET_WORKERS must be an integer"
[[ "${ANALYST_WORKERS}" =~ ^[0-9]+$ ]] || suite_fail "ANALYST_WORKERS must be an integer"
(( TARGET_WORKERS <= API_MAX_CONCURRENCY || TARGET_WORKERS <= VLLM_MAX_NUM_SEQS )) \
  || suite_fail "TARGET_WORKERS is larger than both configured concurrency ceilings"
(( ANALYST_WORKERS <= API_MAX_CONCURRENCY )) \
  || suite_fail "ANALYST_WORKERS exceeds per-process API_MAX_CONCURRENCY"

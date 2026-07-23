#!/usr/bin/env bash
set -euo pipefail

# SearchQA V3 run that mirrors scripts/runs/multi/run_v3_deepseek_local_qwen_parallel.sh
# (its searchqa branch) exactly, but swaps the backends to the Seed/Ark API:
#   optimizer / teacher = Doubao Seed 2.0 Pro   (openai_chat via Ark)
#   target / student    = Doubao Seed 1.6 Flash (openai_chat via Ark)
#
# Every V3 type-guided hyperparameter and every SearchQA data setting below is
# copied from run_v3_deepseek_local_qwen_parallel.sh so the two runs are
# directly comparable; the only intentional differences are the two model names
# and that the target is served through the API instead of a local vLLM Qwen.
#
# Examples:
#   DRY_RUN=1 bash scripts/runs/searchqa/run_searchqa_v3_seed_api_parallel.sh
#   bash scripts/runs/searchqa/run_searchqa_v3_seed_api_parallel.sh
#   NUM_EPOCHS=4 bash scripts/runs/searchqa/run_searchqa_v3_seed_api_parallel.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

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

quote_cmd() {
  local arg
  for arg in "$@"; do
    printf ' %q' "${arg}"
  done
  printf '\n'
}

require_dir() {
  local label="$1"
  local path="$2"
  [[ -d "${path}" ]] || fail "${label} not found: ${path}"
}

command -v "${PYTHON_BIN}" >/dev/null 2>&1 || fail "Python interpreter not found: ${PYTHON_BIN}"

# ---------------------------------------------------------------------------
# Ark OpenAI-compatible API for BOTH optimizer and target.
# ---------------------------------------------------------------------------
export AZURE_OPENAI_ENDPOINT="${AZURE_OPENAI_ENDPOINT:-https://ark.cn-beijing.volces.com/api/v3}"
export AZURE_OPENAI_API_KEY="${AZURE_OPENAI_API_KEY:-${ARK_API_KEY:-}}"
export AZURE_OPENAI_AUTH_MODE="${AZURE_OPENAI_AUTH_MODE:-openai_compatible}"
export AZURE_OPENAI_API_VERSION="${AZURE_OPENAI_API_VERSION:-2024-12-01-preview}"

export OPTIMIZER_AZURE_OPENAI_ENDPOINT="${OPTIMIZER_AZURE_OPENAI_ENDPOINT:-${AZURE_OPENAI_ENDPOINT}}"
export OPTIMIZER_AZURE_OPENAI_API_KEY="${OPTIMIZER_AZURE_OPENAI_API_KEY:-${AZURE_OPENAI_API_KEY}}"
export OPTIMIZER_AZURE_OPENAI_AUTH_MODE="${OPTIMIZER_AZURE_OPENAI_AUTH_MODE:-${AZURE_OPENAI_AUTH_MODE}}"
export OPTIMIZER_AZURE_OPENAI_API_VERSION="${OPTIMIZER_AZURE_OPENAI_API_VERSION:-${AZURE_OPENAI_API_VERSION}}"
export TARGET_AZURE_OPENAI_ENDPOINT="${TARGET_AZURE_OPENAI_ENDPOINT:-${AZURE_OPENAI_ENDPOINT}}"
export TARGET_AZURE_OPENAI_API_KEY="${TARGET_AZURE_OPENAI_API_KEY:-${AZURE_OPENAI_API_KEY}}"
export TARGET_AZURE_OPENAI_AUTH_MODE="${TARGET_AZURE_OPENAI_AUTH_MODE:-${AZURE_OPENAI_AUTH_MODE}}"
export TARGET_AZURE_OPENAI_API_VERSION="${TARGET_AZURE_OPENAI_API_VERSION:-${AZURE_OPENAI_API_VERSION}}"

# Seed models (this is the only intended difference vs the parallel launcher).
OPTIMIZER_MODEL="${OPTIMIZER_MODEL:-doubao-seed-2-0-pro-260215}"
TARGET_MODEL="${TARGET_MODEL:-doubao-seed-1-6-flash-250828}"

DRY_RUN="${DRY_RUN:-0}"
WAIT_FOR_JOB="${WAIT_FOR_JOB:-1}"
TS="${TS:-searchqa_v3_seed_api_parallel_$(date +%Y%m%d_%H%M%S)}"

# ---------------------------------------------------------------------------
# Trainer defaults copied from run_v3_deepseek_local_qwen_parallel.sh.
# ---------------------------------------------------------------------------
NUM_EPOCHS="${NUM_EPOCHS:-1}"
TRAIN_SIZE="${TRAIN_SIZE:-0}"
ACCUMULATION="${ACCUMULATION:-1}"
SEED="${SEED:-42}"
LIMIT="${LIMIT:-0}"
API_MAX_CONCURRENCY="${API_MAX_CONCURRENCY:-96}"
WORKERS="${WORKERS:-48}"
ANALYST_WORKERS="${ANALYST_WORKERS:-24}"

LR_SCHEDULER="${LR_SCHEDULER:-constant}"
LR_CONTROL_MODE="${LR_CONTROL_MODE:-fixed}"
EDIT_BUDGET="${EDIT_BUDGET:-999}"
MIN_EDIT_BUDGET="${MIN_EDIT_BUDGET:-999}"
USE_GATE="${USE_GATE:-true}"
GATE_METRIC="${GATE_METRIC:-mixed}"
GATE_MIXED_WEIGHT="${GATE_MIXED_WEIGHT:-0.5}"
EVAL_TEST="${EVAL_TEST:-true}"
REASONING_EFFORT="${REASONING_EFFORT:-}"

# V3 type-guided settings (identical to the parallel launcher).
TYPE_GUIDED_TREE_DEPTH="${TYPE_GUIDED_TREE_DEPTH:-3}"
TYPE_GUIDED_LEAF_FALLBACK="${TYPE_GUIDED_LEAF_FALLBACK:-true}"
TYPE_GUIDED_MIN_SUPPORT="${TYPE_GUIDED_MIN_SUPPORT:-2}"
TYPE_GUIDED_MAX_LEAF_GROUPS="${TYPE_GUIDED_MAX_LEAF_GROUPS:-8}"
TYPE_GUIDED_ROLLOUT_REPEATS="${TYPE_GUIDED_ROLLOUT_REPEATS:-3}"
TYPE_GUIDED_MAX_PATCH_RECORDS="${TYPE_GUIDED_MAX_PATCH_RECORDS:-24}"
TYPE_GUIDED_TAU_SUCC="${TYPE_GUIDED_TAU_SUCC:-1.0}"
TYPE_GUIDED_PATCH_RECORD_WORKERS="${TYPE_GUIDED_PATCH_RECORD_WORKERS:-${ANALYST_WORKERS}}"
TYPE_GUIDED_CLUSTERING="${TYPE_GUIDED_CLUSTERING:-false}"
TYPE_GUIDED_CLUSTER_TARGET_SIZE="${TYPE_GUIDED_CLUSTER_TARGET_SIZE:-4}"
TYPE_GUIDED_CLUSTER_MAX_SIZE="${TYPE_GUIDED_CLUSTER_MAX_SIZE:-8}"
TYPE_GUIDED_FALLBACK_TOP_K="${TYPE_GUIDED_FALLBACK_TOP_K:-0}"
TYPE_GUIDED_FALLBACK_TAU_CHILD="${TYPE_GUIDED_FALLBACK_TAU_CHILD:-0.0}"
TYPE_GUIDED_FALLBACK_RECONCILE="${TYPE_GUIDED_FALLBACK_RECONCILE:-llm_fuse}"
TYPE_GUIDED_FALLBACK_RECONCILE_MIN_CHILDREN="${TYPE_GUIDED_FALLBACK_RECONCILE_MIN_CHILDREN:-2}"
TYPE_GUIDED_TAIL_BANK="${TYPE_GUIDED_TAIL_BANK:-true}"
TYPE_GUIDED_TAIL_MIN_SUPPORT="${TYPE_GUIDED_TAIL_MIN_SUPPORT:-2}"
TYPE_GUIDED_TAIL_MAX_RECORDS="${TYPE_GUIDED_TAIL_MAX_RECORDS:-32}"
TYPE_GUIDED_TAIL_MAX_LEAF_GROUPS="${TYPE_GUIDED_TAIL_MAX_LEAF_GROUPS:-4}"
TYPE_GUIDED_TAIL_WINDOW_EPOCHS="${TYPE_GUIDED_TAIL_WINDOW_EPOCHS:-3}"
TYPE_GUIDED_TAIL_REQUIRE_CROSS_STEP="${TYPE_GUIDED_TAIL_REQUIRE_CROSS_STEP:-true}"

# SearchQA-specific settings (copied from the parallel launcher's searchqa branch).
SEARCHQA_CONFIG="${SEARCHQA_CONFIG:-${PROJECT_ROOT}/configs/searchqa/default.yaml}"
SEARCHQA_SPLIT_DIR="${SEARCHQA_SPLIT_DIR:-${PROJECT_ROOT}/data/searchqa_split}"
BATCH_SIZE="${SEARCHQA_BATCH_SIZE:-${BATCH_SIZE:-40}}"
# 0 means the full val/test split (val=200, test=1400).
SEL_ENV_NUM="${SEARCHQA_SEL_ENV_NUM:-0}"
TEST_ENV_NUM="${SEARCHQA_TEST_ENV_NUM:-0}"
MAX_TURNS="${SEARCHQA_MAX_TURNS:-1}"
TARGET_MAX_COMPLETION_TOKENS="${SEARCHQA_TARGET_MAX_COMPLETION_TOKENS:-8192}"
FALLBACK_SEL_ENV_NUM="${SEARCHQA_TYPE_GUIDED_FALLBACK_SEL_ENV_NUM:-80}"

LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs/skillopt_searchqa_v3_seed_api_parallel_${TS}}"
OUT_ROOT="${OUT_ROOT:-${PROJECT_ROOT}/outputs/skillopt_searchqa_v3_seed_api_parallel_${TS}}"
TYPE_GUIDED_CACHE_DIR="${TYPE_GUIDED_CACHE_DIR:-${OUT_ROOT}/type_guided_cache}"

# ---------------------------------------------------------------------------
# Validation (mirrors the parallel launcher).
# ---------------------------------------------------------------------------
[[ "${GATE_METRIC}" == "hard" || "${GATE_METRIC}" == "soft" || "${GATE_METRIC}" == "mixed" ]] || fail "GATE_METRIC must be hard, soft, or mixed."
[[ "${TYPE_GUIDED_FALLBACK_RECONCILE}" == "off" || "${TYPE_GUIDED_FALLBACK_RECONCILE}" == "deterministic" || "${TYPE_GUIDED_FALLBACK_RECONCILE}" == "llm_select" || "${TYPE_GUIDED_FALLBACK_RECONCILE}" == "llm_fuse" ]] || fail "TYPE_GUIDED_FALLBACK_RECONCILE must be off, deterministic, llm_select, or llm_fuse."
[[ "${API_MAX_CONCURRENCY}" =~ ^[0-9]+$ ]] || fail "API_MAX_CONCURRENCY must be an integer."
[[ "${WORKERS}" =~ ^[0-9]+$ ]] || fail "WORKERS must be an integer."
[[ "${ANALYST_WORKERS}" =~ ^[0-9]+$ ]] || fail "ANALYST_WORKERS must be an integer."
[[ "${WORKERS}" -ge 1 ]] || fail "WORKERS must be >= 1."
[[ "${ANALYST_WORKERS}" -ge 1 ]] || fail "ANALYST_WORKERS must be >= 1."
[[ "${WORKERS}" -le "${API_MAX_CONCURRENCY}" ]] || fail "WORKERS=${WORKERS} exceeds API_MAX_CONCURRENCY=${API_MAX_CONCURRENCY}."
[[ "${ANALYST_WORKERS}" -le "${API_MAX_CONCURRENCY}" ]] || fail "ANALYST_WORKERS=${ANALYST_WORKERS} exceeds API_MAX_CONCURRENCY=${API_MAX_CONCURRENCY}."
[[ "${BATCH_SIZE}" =~ ^[0-9]+$ ]] || fail "BATCH_SIZE must be an integer."
[[ "${SEL_ENV_NUM}" =~ ^[0-9]+$ ]] || fail "SEL_ENV_NUM must be an integer."
[[ "${TEST_ENV_NUM}" =~ ^[0-9]+$ ]] || fail "TEST_ENV_NUM must be an integer."
[[ -f "${SEARCHQA_CONFIG}" ]] || fail "Missing config: ${SEARCHQA_CONFIG}"
require_dir "searchqa split dir" "${SEARCHQA_SPLIT_DIR}"
require_dir "searchqa train split" "${SEARCHQA_SPLIT_DIR}/train"
require_dir "searchqa val split" "${SEARCHQA_SPLIT_DIR}/val"
require_dir "searchqa test split" "${SEARCHQA_SPLIT_DIR}/test"

mkdir -p "${LOG_DIR}" "${OUT_ROOT}" "${TYPE_GUIDED_CACHE_DIR}"
RUN_LOG="${LOG_DIR}/launcher.log"
TRAIN_LOG="${LOG_DIR}/searchqa.log"
STATUS_FILE="${LOG_DIR}/searchqa.status"
DONE_FILE="${LOG_DIR}/completed.tsv"
: > "${DONE_FILE}"

echo "============================================================"
echo "  SkillOpt-Tree SearchQA V3: Seed API (parallel-aligned)"
echo "============================================================"
echo "  project:             ${PROJECT_ROOT}"
echo "  optimizer:           ${OPTIMIZER_MODEL}"
echo "  target:              ${TARGET_MODEL}"
echo "  api_endpoint:        ${AZURE_OPENAI_ENDPOINT}"
echo "  split_dir:           ${SEARCHQA_SPLIT_DIR}"
echo "  num_epochs:          ${NUM_EPOCHS}"
echo "  train_size:          ${TRAIN_SIZE} (0 means split train size)"
echo "  batch_size:          ${BATCH_SIZE}"
echo "  workers:             ${WORKERS}"
echo "  analyst_workers:     ${ANALYST_WORKERS}"
echo "  sel_env_num:         ${SEL_ENV_NUM} (0 means full val)"
echo "  test_env_num:        ${TEST_ENV_NUM} (0 means full test)"
echo "  max_turns:           ${MAX_TURNS}"
echo "  target_max_tokens:   ${TARGET_MAX_COMPLETION_TOKENS}"
echo "  method:              PatchTree"
echo "  tree_depth:          ${TYPE_GUIDED_TREE_DEPTH}"
echo "  rollout_repeats:     ${TYPE_GUIDED_ROLLOUT_REPEATS}"
echo "  clustering:          ${TYPE_GUIDED_CLUSTERING} (target=${TYPE_GUIDED_CLUSTER_TARGET_SIZE} max=${TYPE_GUIDED_CLUSTER_MAX_SIZE})"
echo "  fallback_reconcile:  ${TYPE_GUIDED_FALLBACK_RECONCILE} (sel_env_num=${FALLBACK_SEL_ENV_NUM})"
echo "  tail_bank:           ${TYPE_GUIDED_TAIL_BANK} (window=${TYPE_GUIDED_TAIL_WINDOW_EPOCHS})"
echo "  gate:                ${GATE_METRIC} (mixed_weight=${GATE_MIXED_WEIGHT})"
echo "  dry_run:             ${DRY_RUN}"
echo "  out_root:            ${OUT_ROOT}"
echo "  log_dir:             ${LOG_DIR}"
echo "============================================================"
{
  echo "SkillOpt-Tree SearchQA V3 Seed API (parallel-aligned) started at $(date)"
  echo "optimizer=${OPTIMIZER_MODEL} target=${TARGET_MODEL}"
  echo "split_dir=${SEARCHQA_SPLIT_DIR} num_epochs=${NUM_EPOCHS} batch_size=${BATCH_SIZE}"
} > "${RUN_LOG}"

if ! truthy "${DRY_RUN}"; then
  [[ -n "${AZURE_OPENAI_API_KEY}" ]] || fail "AZURE_OPENAI_API_KEY is empty. Export ARK_API_KEY or AZURE_OPENAI_API_KEY first."
else
  echo "[dry-run] Skip API key check."
fi

CFG_OPTIONS=(
  "evaluation.gate_metric=${GATE_METRIC}"
  "evaluation.gate_mixed_weight=${GATE_MIXED_WEIGHT}"
  "env.max_completion_tokens=${TARGET_MAX_COMPLETION_TOKENS}"
)

CMD=(
  "${PYTHON_BIN}" -u "${PROJECT_ROOT}/scripts/cli/train.py"
  --config "${SEARCHQA_CONFIG}"
  --optimizer_backend openai_chat
  --target_backend openai_chat
  --optimizer_model "${OPTIMIZER_MODEL}"
  --target_model "${TARGET_MODEL}"
  --reasoning_effort "${REASONING_EFFORT}"
  --num_epochs "${NUM_EPOCHS}"
  --train_size "${TRAIN_SIZE}"
  --batch_size "${BATCH_SIZE}"
  --accumulation "${ACCUMULATION}"
  --workers "${WORKERS}"
  --analyst_workers "${ANALYST_WORKERS}"
  --seed "${SEED}"
  --edit_budget "${EDIT_BUDGET}"
  --min_edit_budget "${MIN_EDIT_BUDGET}"
  --lr_scheduler "${LR_SCHEDULER}"
  --lr_control_mode "${LR_CONTROL_MODE}"
  --use_gate "${USE_GATE}"
  --sel_env_num "${SEL_ENV_NUM}"
  --test_env_num "${TEST_ENV_NUM}"
  --eval_test "${EVAL_TEST}"
  --type_guided_min_support "${TYPE_GUIDED_MIN_SUPPORT}"
  --type_guided_max_leaf_groups "${TYPE_GUIDED_MAX_LEAF_GROUPS}"
  --type_guided_tree_depth "${TYPE_GUIDED_TREE_DEPTH}"
  --type_guided_leaf_fallback "${TYPE_GUIDED_LEAF_FALLBACK}"
  --type_guided_rollout_repeats "${TYPE_GUIDED_ROLLOUT_REPEATS}"
  --type_guided_tau_succ "${TYPE_GUIDED_TAU_SUCC}"
  --type_guided_max_patch_records "${TYPE_GUIDED_MAX_PATCH_RECORDS}"
  --type_guided_fallback_top_k "${TYPE_GUIDED_FALLBACK_TOP_K}"
  --type_guided_fallback_tau_child "${TYPE_GUIDED_FALLBACK_TAU_CHILD}"
  --type_guided_fallback_sel_env_num "${FALLBACK_SEL_ENV_NUM}"
  --type_guided_fallback_reconcile "${TYPE_GUIDED_FALLBACK_RECONCILE}"
  --type_guided_fallback_reconcile_min_children "${TYPE_GUIDED_FALLBACK_RECONCILE_MIN_CHILDREN}"
  --type_guided_clustering "${TYPE_GUIDED_CLUSTERING}"
  --type_guided_cluster_target_size "${TYPE_GUIDED_CLUSTER_TARGET_SIZE}"
  --type_guided_cluster_max_size "${TYPE_GUIDED_CLUSTER_MAX_SIZE}"
  --type_guided_tail_bank "${TYPE_GUIDED_TAIL_BANK}"
  --type_guided_tail_min_support "${TYPE_GUIDED_TAIL_MIN_SUPPORT}"
  --type_guided_tail_max_records "${TYPE_GUIDED_TAIL_MAX_RECORDS}"
  --type_guided_tail_max_leaf_groups "${TYPE_GUIDED_TAIL_MAX_LEAF_GROUPS}"
  --type_guided_tail_window_epochs "${TYPE_GUIDED_TAIL_WINDOW_EPOCHS}"
  --type_guided_tail_require_cross_step "${TYPE_GUIDED_TAIL_REQUIRE_CROSS_STEP}"
  --type_guided_patch_record_workers "${TYPE_GUIDED_PATCH_RECORD_WORKERS}"
  --type_guided_cache_dir "${TYPE_GUIDED_CACHE_DIR}"
  --split_mode split_dir
  --split_dir "${SEARCHQA_SPLIT_DIR}"
  --max_turns "${MAX_TURNS}"
  --out_root "${OUT_ROOT}"
  --limit "${LIMIT}"
  --cfg-options "${CFG_OPTIONS[@]}"
)

{
  printf '[cmd]'
  quote_cmd "${CMD[@]}"
} | tee -a "${RUN_LOG}"

if truthy "${DRY_RUN}"; then
  printf '[dry-run]'
  quote_cmd "${CMD[@]}"
  exit 0
fi

rm -f "${STATUS_FILE}"

if truthy "${WAIT_FOR_JOB}"; then
  set +e
  {
    echo "[start] $(date)"
    printf '[cmd]'
    quote_cmd "${CMD[@]}"
    "${CMD[@]}"
    status=$?
    echo "[exit] ${status} $(date)"
    exit "${status}"
  } 2>&1 | tee "${TRAIN_LOG}"
  status=${PIPESTATUS[0]}
  set -e
  echo "${status}" > "${STATUS_FILE}"
  printf 'searchqa\t%s\t%s\n' "${status}" "${TRAIN_LOG}" > "${DONE_FILE}"
  if [[ "${status}" != "0" ]]; then
    echo "[failed] SearchQA run failed. See ${TRAIN_LOG}" | tee -a "${RUN_LOG}" >&2
    exit "${status}"
  fi
  echo "[done] SearchQA run completed successfully. Log: ${TRAIN_LOG}" | tee -a "${RUN_LOG}"
else
  {
    echo "[start] $(date)"
    printf '[cmd]'
    quote_cmd "${CMD[@]}"
    set +e
    "${CMD[@]}"
    status=$?
    set -e
    echo "[exit] ${status} $(date)"
    echo "${status}" > "${STATUS_FILE}"
    printf 'searchqa\t%s\t%s\n' "${status}" "${TRAIN_LOG}" > "${DONE_FILE}"
    exit "${status}"
  } > "${TRAIN_LOG}" 2>&1 &
  pid=$!
  echo "${pid}" > "${LOG_DIR}/searchqa.pid"
  echo "[launched] pid=${pid} log=${TRAIN_LOG}" | tee -a "${RUN_LOG}"
fi

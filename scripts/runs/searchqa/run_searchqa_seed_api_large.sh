#!/usr/bin/env bash
set -euo pipefail

# Launch a larger SkillOpt-Tree SearchQA training run with OpenAI-compatible
# Seed/Ark API defaults.
#
# Defaults are intentionally larger than scripts/runs/searchqa/run_searchqa.sh:
#   train_size=400, batch_size=40, epochs=4, selection=200, test=400,
#   type-guided V2 enabled with 3 repeated rollouts.
#
# Usage:
#   DRY_RUN=1 bash scripts/runs/searchqa/run_searchqa_seed_api_large.sh
#   bash scripts/runs/searchqa/run_searchqa_seed_api_large.sh
#   TEST_ENV_NUM=1400 bash scripts/runs/searchqa/run_searchqa_seed_api_large.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
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

command -v "${PYTHON_BIN}" >/dev/null 2>&1 || fail "Python interpreter not found: ${PYTHON_BIN}"

# Ark OpenAI-compatible API. The trainer reads these through the OpenAI/Azure
# compatibility layer.
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

OPTIMIZER_MODEL="${OPTIMIZER_MODEL:-doubao-seed-2-0-pro-260215}"
TARGET_MODEL="${TARGET_MODEL:-doubao-seed-1-6-flash-250828}"

API_MAX_CONCURRENCY="${API_MAX_CONCURRENCY:-48}"
WORKERS="${WORKERS:-48}"
ANALYST_WORKERS="${ANALYST_WORKERS:-32}"
NUM_EPOCHS="${NUM_EPOCHS:-4}"
TRAIN_SIZE="${TRAIN_SIZE:-400}"
BATCH_SIZE="${BATCH_SIZE:-40}"
ACCUMULATION="${ACCUMULATION:-1}"
SEED="${SEED:-42}"
DRY_RUN="${DRY_RUN:-0}"
WAIT_FOR_JOB="${WAIT_FOR_JOB:-1}"

LR_SCHEDULER="${LR_SCHEDULER:-constant}"
LR_CONTROL_MODE="${LR_CONTROL_MODE:-fixed}"
EDIT_BUDGET="${EDIT_BUDGET:-999}"
MIN_EDIT_BUDGET="${MIN_EDIT_BUDGET:-999}"
USE_GATE="${USE_GATE:-true}"
GATE_METRIC="${GATE_METRIC:-mixed}"
GATE_MIXED_WEIGHT="${GATE_MIXED_WEIGHT:-0.5}"
SEL_ENV_NUM="${SEL_ENV_NUM:-200}"
TEST_ENV_NUM="${TEST_ENV_NUM:-400}"
EVAL_TEST="${EVAL_TEST:-true}"
REASONING_EFFORT="${REASONING_EFFORT:-}"
TARGET_MAX_COMPLETION_TOKENS="${TARGET_MAX_COMPLETION_TOKENS:-16384}"
MAX_TURNS="${MAX_TURNS:-1}"
LIMIT="${LIMIT:-0}"

TYPE_GUIDED_LEAF_FALLBACK="${TYPE_GUIDED_LEAF_FALLBACK:-true}"
TYPE_GUIDED_MIN_SUPPORT="${TYPE_GUIDED_MIN_SUPPORT:-2}"
TYPE_GUIDED_MAX_LEAF_GROUPS="${TYPE_GUIDED_MAX_LEAF_GROUPS:-10}"
TYPE_GUIDED_TREE_DEPTH="${TYPE_GUIDED_TREE_DEPTH:-2}"
TYPE_GUIDED_ROLLOUT_REPEATS="${TYPE_GUIDED_ROLLOUT_REPEATS:-3}"
TYPE_GUIDED_MAX_PATCH_RECORDS="${TYPE_GUIDED_MAX_PATCH_RECORDS:-32}"
TYPE_GUIDED_TAU_SUCC="${TYPE_GUIDED_TAU_SUCC:-1.0}"
TYPE_GUIDED_FALLBACK_TOP_K="${TYPE_GUIDED_FALLBACK_TOP_K:-0}"
TYPE_GUIDED_FALLBACK_TAU_CHILD="${TYPE_GUIDED_FALLBACK_TAU_CHILD:-0.0}"
TYPE_GUIDED_FALLBACK_RECONCILE="${TYPE_GUIDED_FALLBACK_RECONCILE:-llm_fuse}"
TYPE_GUIDED_FALLBACK_RECONCILE_MIN_CHILDREN="${TYPE_GUIDED_FALLBACK_RECONCILE_MIN_CHILDREN:-2}"
TYPE_GUIDED_PATCH_RECORD_WORKERS="${TYPE_GUIDED_PATCH_RECORD_WORKERS:-${ANALYST_WORKERS}}"
TYPE_GUIDED_CLUSTERING="${TYPE_GUIDED_CLUSTERING:-false}"
TYPE_GUIDED_CLUSTER_TARGET_SIZE="${TYPE_GUIDED_CLUSTER_TARGET_SIZE:-6}"
TYPE_GUIDED_CLUSTER_MAX_SIZE="${TYPE_GUIDED_CLUSTER_MAX_SIZE:-10}"
TYPE_GUIDED_TAIL_BANK="${TYPE_GUIDED_TAIL_BANK:-true}"
TYPE_GUIDED_TAIL_MIN_SUPPORT="${TYPE_GUIDED_TAIL_MIN_SUPPORT:-2}"
TYPE_GUIDED_TAIL_MAX_RECORDS="${TYPE_GUIDED_TAIL_MAX_RECORDS:-32}"
TYPE_GUIDED_TAIL_MAX_LEAF_GROUPS="${TYPE_GUIDED_TAIL_MAX_LEAF_GROUPS:-4}"
TYPE_GUIDED_TAIL_WINDOW_EPOCHS="${TYPE_GUIDED_TAIL_WINDOW_EPOCHS:-1}"
TYPE_GUIDED_TAIL_REQUIRE_CROSS_STEP="${TYPE_GUIDED_TAIL_REQUIRE_CROSS_STEP:-true}"

SEARCHQA_CONFIG="${SEARCHQA_CONFIG:-${PROJECT_ROOT}/configs/searchqa/default.yaml}"
SEARCHQA_SKILL_INIT="${SEARCHQA_SKILL_INIT:-${PROJECT_ROOT}/skillopt/envs/searchqa/skills/initial.md}"
SEARCHQA_SPLIT_MODE="${SEARCHQA_SPLIT_MODE:-split_dir}"
SEARCHQA_SPLIT_DIR="${SEARCHQA_SPLIT_DIR:-${PROJECT_ROOT}/data/searchqa_split}"
SEARCHQA_DATA_PATH="${SEARCHQA_DATA_PATH:-}"
SPLIT_RATIO="${SPLIT_RATIO:-2:1:7}"
SPLIT_SEED="${SPLIT_SEED:-42}"

TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs/skillopt_searchqa_seed_api_large_${TS}}"
OUT_ROOT="${OUT_ROOT:-${PROJECT_ROOT}/outputs/skillopt_searchqa_seed_api_large_${TS}}"
TYPE_GUIDED_CACHE_DIR="${TYPE_GUIDED_CACHE_DIR:-${OUT_ROOT}/type_guided_cache}"

[[ "${GATE_METRIC}" == "hard" || "${GATE_METRIC}" == "soft" || "${GATE_METRIC}" == "mixed" ]] || fail "GATE_METRIC must be hard, soft, or mixed."
[[ "${TYPE_GUIDED_TREE_DEPTH}" =~ ^[0-9]+$ ]] || fail "TYPE_GUIDED_TREE_DEPTH must be an integer."
[[ "${SEARCHQA_SPLIT_MODE}" == "ratio" || "${SEARCHQA_SPLIT_MODE}" == "split_dir" ]] || fail "SEARCHQA_SPLIT_MODE must be ratio or split_dir."
[[ "${API_MAX_CONCURRENCY}" =~ ^[0-9]+$ ]] || fail "API_MAX_CONCURRENCY must be an integer."
[[ "${WORKERS}" =~ ^[0-9]+$ ]] || fail "WORKERS must be an integer."
[[ "${ANALYST_WORKERS}" =~ ^[0-9]+$ ]] || fail "ANALYST_WORKERS must be an integer."
[[ "${WORKERS}" -ge 1 ]] || fail "WORKERS must be >= 1."
[[ "${ANALYST_WORKERS}" -ge 1 ]] || fail "ANALYST_WORKERS must be >= 1."
[[ "${WORKERS}" -le "${API_MAX_CONCURRENCY}" ]] || fail "WORKERS=${WORKERS} exceeds API_MAX_CONCURRENCY=${API_MAX_CONCURRENCY}."
[[ "${ANALYST_WORKERS}" -le "${API_MAX_CONCURRENCY}" ]] || fail "ANALYST_WORKERS=${ANALYST_WORKERS} exceeds API_MAX_CONCURRENCY=${API_MAX_CONCURRENCY}."
[[ -f "${SEARCHQA_CONFIG}" ]] || fail "Missing config: ${SEARCHQA_CONFIG}"
[[ -f "${SEARCHQA_SKILL_INIT}" ]] || fail "Missing initial skill: ${SEARCHQA_SKILL_INIT}"

check_searchqa_split() {
  "${PYTHON_BIN}" - "${SEARCHQA_SPLIT_DIR}" <<'PY'
import json
import os
import sys

split_dir = sys.argv[1]
required = {"train": 1, "val": 1, "test": 1}
for split, min_count in required.items():
    path = os.path.join(split_dir, split, "items.json")
    if not os.path.isfile(path):
        raise SystemExit(f"missing {path}")
    with open(path, encoding="utf-8") as f:
        items = json.load(f)
    if not isinstance(items, list) or len(items) < min_count:
        raise SystemExit(f"{path} has no usable items")
    sample = items[0]
    missing = [key for key in ("question", "context", "answers") if key not in sample]
    if missing:
        raise SystemExit(f"{path} missing fields {missing}")
print("searchqa split ok")
PY
}

check_searchqa_raw() {
  [[ -n "${SEARCHQA_DATA_PATH}" ]] || fail "SEARCHQA_DATA_PATH is required when SEARCHQA_SPLIT_MODE=ratio"
  [[ -e "${SEARCHQA_DATA_PATH}" ]] || fail "Missing SEARCHQA_DATA_PATH: ${SEARCHQA_DATA_PATH}"
}

if [[ "${SEARCHQA_SPLIT_MODE}" == "split_dir" ]]; then
  if ! check_searchqa_split >/dev/null; then
    if truthy "${DRY_RUN}"; then
      echo "[dry-run][warn] SearchQA split check failed: ${SEARCHQA_SPLIT_DIR}" >&2
    else
      fail "SearchQA split check failed: ${SEARCHQA_SPLIT_DIR}"
    fi
  fi
else
  check_searchqa_raw
fi

mkdir -p "${LOG_DIR}" "${OUT_ROOT}" "${TYPE_GUIDED_CACHE_DIR}"

RUN_LOG="${LOG_DIR}/launcher.log"
TRAIN_LOG="${LOG_DIR}/searchqa.log"
STATUS_FILE="${LOG_DIR}/searchqa.status"
DONE_FILE="${LOG_DIR}/completed.tsv"
: > "${DONE_FILE}"

echo "============================================================"
echo "  SkillOpt-Tree SearchQA: Seed API Large"
echo "============================================================"
echo "  project:             ${PROJECT_ROOT}"
echo "  optimizer:           ${OPTIMIZER_MODEL}"
echo "  target:              ${TARGET_MODEL}"
echo "  api_endpoint:        ${AZURE_OPENAI_ENDPOINT}"
echo "  split_mode:          ${SEARCHQA_SPLIT_MODE}"
echo "  split_dir:           ${SEARCHQA_SPLIT_DIR}"
echo "  train_size:          ${TRAIN_SIZE}"
echo "  num_epochs:          ${NUM_EPOCHS}"
echo "  batch_size:          ${BATCH_SIZE}"
echo "  workers:             ${WORKERS}"
echo "  analyst_workers:     ${ANALYST_WORKERS}"
echo "  sel_env_num:         ${SEL_ENV_NUM}"
echo "  test_env_num:        ${TEST_ENV_NUM}"
echo "  method:              PatchTree"
echo "  tree_depth:          ${TYPE_GUIDED_TREE_DEPTH}"
echo "  rollout_repeats:     ${TYPE_GUIDED_ROLLOUT_REPEATS}"
echo "  max_patch_records:   ${TYPE_GUIDED_MAX_PATCH_RECORDS}"
echo "  dry_run:             ${DRY_RUN}"
echo "  out_root:            ${OUT_ROOT}"
echo "  log_dir:             ${LOG_DIR}"
echo "============================================================"
{
  echo "============================================================"
  echo "  SkillOpt-Tree SearchQA launcher started at $(date)"
  echo "  project=${PROJECT_ROOT}"
  echo "  optimizer=${OPTIMIZER_MODEL}"
  echo "  target=${TARGET_MODEL}"
  echo "  split_mode=${SEARCHQA_SPLIT_MODE}"
  echo "  split_dir=${SEARCHQA_SPLIT_DIR}"
  echo "  train_size=${TRAIN_SIZE}"
  echo "  dry_run=${DRY_RUN}"
  echo "============================================================"
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
  --type_guided_fallback_reconcile "${TYPE_GUIDED_FALLBACK_RECONCILE}"
  --type_guided_fallback_reconcile_min_children "${TYPE_GUIDED_FALLBACK_RECONCILE_MIN_CHILDREN}"
  --type_guided_cache_dir "${TYPE_GUIDED_CACHE_DIR}"
  --type_guided_patch_record_workers "${TYPE_GUIDED_PATCH_RECORD_WORKERS}"
  --type_guided_clustering "${TYPE_GUIDED_CLUSTERING}"
  --type_guided_cluster_target_size "${TYPE_GUIDED_CLUSTER_TARGET_SIZE}"
  --type_guided_cluster_max_size "${TYPE_GUIDED_CLUSTER_MAX_SIZE}"
  --type_guided_tail_bank "${TYPE_GUIDED_TAIL_BANK}"
  --type_guided_tail_min_support "${TYPE_GUIDED_TAIL_MIN_SUPPORT}"
  --type_guided_tail_max_records "${TYPE_GUIDED_TAIL_MAX_RECORDS}"
  --type_guided_tail_max_leaf_groups "${TYPE_GUIDED_TAIL_MAX_LEAF_GROUPS}"
  --type_guided_tail_window_epochs "${TYPE_GUIDED_TAIL_WINDOW_EPOCHS}"
  --type_guided_tail_require_cross_step "${TYPE_GUIDED_TAIL_REQUIRE_CROSS_STEP}"
  --split_mode "${SEARCHQA_SPLIT_MODE}"
  --split_ratio "${SPLIT_RATIO}"
  --split_seed "${SPLIT_SEED}"
  --max_turns "${MAX_TURNS}"
  --skill_init "${SEARCHQA_SKILL_INIT}"
  --out_root "${OUT_ROOT}"
)


if [[ "${SEARCHQA_SPLIT_MODE}" == "split_dir" ]]; then
  CMD+=(--split_dir "${SEARCHQA_SPLIT_DIR}")
else
  CMD+=(--data_path "${SEARCHQA_DATA_PATH}")
fi

if [[ "${LIMIT}" != "0" ]]; then
  CMD+=(--limit "${LIMIT}")
fi

CMD+=(
  --cfg-options
  "${CFG_OPTIONS[@]}"
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

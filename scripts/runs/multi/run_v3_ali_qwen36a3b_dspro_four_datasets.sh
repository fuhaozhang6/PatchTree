#!/usr/bin/env bash
set -euo pipefail

# Parallel V3 launcher, ALL Aliyun DashScope OpenAI-compatible API:
#   optimizer / teacher = DeepSeek V4 Pro   (deepseek-v4-pro)
#   target / student    = Qwen3.6-35B-A3B   (qwen3.6-35b-a3b, vision + tools capable)
#   datasets            = livemath officeqa spreadsheetbench docvqa
#   epochs              = 4
#
# This is the API analogue of run_v3_deepseek_local_qwen_parallel.sh: it keeps
# the same V3 type-guided hyperparameters, the same per-dataset plumbing, and
# the same job scheduler, but BOTH roles are served through Aliyun DashScope
# instead of a local vLLM Qwen. There is therefore no vLLM startup here.
#
# Concurrency picked from the provider benchmark (Ds-test/BENCHMARK_REPORT.md):
# Aliyun scaled smoothly with zero failures through 48 concurrency with no
# throttling knee, so target rollout defaults to 48. Optimizer (ds-pro) calls
# are heavier, so ANALYST_WORKERS defaults to a more moderate 24.
#
# Examples:
#   DRY_RUN=1 bash scripts/runs/multi/run_v3_ali_qwen36a3b_dspro_four_datasets.sh
#   LIMIT=2 NUM_EPOCHS=1 MAX_PARALLEL=1 bash scripts/runs/multi/run_v3_ali_qwen36a3b_dspro_four_datasets.sh
#   DATASETS="livemath" bash scripts/runs/multi/run_v3_ali_qwen36a3b_dspro_four_datasets.sh

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

split_count() {
  local split_path="$1"
  "${PYTHON_BIN}" - "${split_path}" <<'PY'
import csv
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
files = sorted(path.glob("*.csv")) + sorted(path.glob("*.json"))
if not files:
    print("0")
    raise SystemExit(0)
file_path = files[0]
if file_path.suffix == ".csv":
    with file_path.open(encoding="utf-8", newline="") as f:
        print(sum(1 for _ in csv.DictReader(f)))
else:
    with file_path.open(encoding="utf-8") as f:
        data = json.load(f)
    print(len(data) if isinstance(data, list) else 0)
PY
}

effective_eval_count() {
  local requested="$1"
  local available="$2"
  if [[ "${requested}" == "0" || "${requested}" -ge "${available}" ]]; then
    printf '%s\n' "${available}"
  else
    printf '%s\n' "${requested}"
  fi
}

command -v "${PYTHON_BIN}" >/dev/null 2>&1 || fail "Python interpreter not found: ${PYTHON_BIN}"

# ---------------------------------------------------------------------------
# Aliyun DashScope OpenAI-compatible API for BOTH optimizer and target.
# Export DASHSCOPE_API_KEY (or ALI_API_KEY) before running.
# ---------------------------------------------------------------------------
export AZURE_OPENAI_ENDPOINT="${AZURE_OPENAI_ENDPOINT:-https://dashscope.aliyuncs.com/compatible-mode/v1}"
export AZURE_OPENAI_API_KEY="${AZURE_OPENAI_API_KEY:-${DASHSCOPE_API_KEY:-${ALI_API_KEY:-}}}"
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

# Models.
OPTIMIZER_MODEL="${OPTIMIZER_MODEL:-deepseek-v4-pro}"
TARGET_MODEL="${TARGET_MODEL:-qwen3.6-35b-a3b}"

# Run scope.
DATASETS="${DATASETS:-livemath officeqa spreadsheetbench docvqa}"
MAX_PARALLEL="${MAX_PARALLEL:-2}"
WAIT_FOR_JOBS="${WAIT_FOR_JOBS:-1}"
DRY_RUN="${DRY_RUN:-0}"
TS="${TS:-v3_ali_qwen36a3b_dspro_$(date +%Y%m%d_%H%M%S)}"

NUM_EPOCHS="${NUM_EPOCHS:-4}"
TRAIN_SIZE="${TRAIN_SIZE:-0}"
BATCH_SIZE="${BATCH_SIZE:-}"
ACCUMULATION="${ACCUMULATION:-1}"
SEED="${SEED:-42}"
LIMIT="${LIMIT:-0}"

# Concurrency (benchmark-informed). API_MAX_CONCURRENCY is per training process;
# with MAX_PARALLEL datasets active, total client concurrency can reach
# MAX_PARALLEL * WORKERS against DashScope.
API_MAX_CONCURRENCY="${API_MAX_CONCURRENCY:-48}"
WORKERS="${WORKERS:-32}"
ANALYST_WORKERS="${ANALYST_WORKERS:-24}"

LR_SCHEDULER="${LR_SCHEDULER:-cosine}"
LR_CONTROL_MODE="${LR_CONTROL_MODE:-fixed}"
EDIT_BUDGET="${EDIT_BUDGET:-4}"
MIN_EDIT_BUDGET="${MIN_EDIT_BUDGET:-2}"
USE_GATE="${USE_GATE:-true}"
GATE_METRIC="${GATE_METRIC:-mixed}"
GATE_MIXED_WEIGHT="${GATE_MIXED_WEIGHT:-0.5}"
EVAL_TEST="${EVAL_TEST:-true}"
REASONING_EFFORT="${REASONING_EFFORT:-}"
TARGET_MAX_COMPLETION_TOKENS="${TARGET_MAX_COMPLETION_TOKENS:-16384}"
LIVEMATH_TARGET_MAX_COMPLETION_TOKENS="${LIVEMATH_TARGET_MAX_COMPLETION_TOKENS:-16384}"
OFFICEQA_TARGET_MAX_COMPLETION_TOKENS="${OFFICEQA_TARGET_MAX_COMPLETION_TOKENS:-16384}"
SPREADSHEETBENCH_TARGET_MAX_COMPLETION_TOKENS="${SPREADSHEETBENCH_TARGET_MAX_COMPLETION_TOKENS:-16384}"
DOCVQA_TARGET_MAX_COMPLETION_TOKENS="${DOCVQA_TARGET_MAX_COMPLETION_TOKENS:-16384}"

# V3 type-guided settings (identical to run_v3_deepseek_local_qwen_parallel.sh).
TYPE_GUIDED_TREE_DEPTH="${TYPE_GUIDED_TREE_DEPTH:-3}"
TYPE_GUIDED_LEAF_FALLBACK="${TYPE_GUIDED_LEAF_FALLBACK:-true}"
TYPE_GUIDED_MIN_SUPPORT="${TYPE_GUIDED_MIN_SUPPORT:-2}"
TYPE_GUIDED_MAX_LEAF_GROUPS="${TYPE_GUIDED_MAX_LEAF_GROUPS:-8}"
TYPE_GUIDED_ROLLOUT_REPEATS="${TYPE_GUIDED_ROLLOUT_REPEATS:-3}"
TYPE_GUIDED_MAX_PATCH_RECORDS="${TYPE_GUIDED_MAX_PATCH_RECORDS:-24}"
TYPE_GUIDED_TAU_SUCC="${TYPE_GUIDED_TAU_SUCC:-1.0}"
TYPE_GUIDED_PATCH_RECORD_WORKERS="${TYPE_GUIDED_PATCH_RECORD_WORKERS:-}"
TYPE_GUIDED_CLUSTERING="${TYPE_GUIDED_CLUSTERING:-true}"
TYPE_GUIDED_CLUSTER_TARGET_SIZE="${TYPE_GUIDED_CLUSTER_TARGET_SIZE:-4}"
TYPE_GUIDED_CLUSTER_MAX_SIZE="${TYPE_GUIDED_CLUSTER_MAX_SIZE:-8}"
TYPE_GUIDED_FALLBACK_TOP_K="${TYPE_GUIDED_FALLBACK_TOP_K:-0}"
TYPE_GUIDED_FALLBACK_TAU_CHILD="${TYPE_GUIDED_FALLBACK_TAU_CHILD:-0.0}"
TYPE_GUIDED_FALLBACK_SEL_ENV_NUM="${TYPE_GUIDED_FALLBACK_SEL_ENV_NUM:-0}"
TYPE_GUIDED_FALLBACK_RECONCILE="${TYPE_GUIDED_FALLBACK_RECONCILE:-deterministic}"
TYPE_GUIDED_FALLBACK_RECONCILE_MIN_CHILDREN="${TYPE_GUIDED_FALLBACK_RECONCILE_MIN_CHILDREN:-2}"
TYPE_GUIDED_TAIL_BANK="${TYPE_GUIDED_TAIL_BANK:-true}"
TYPE_GUIDED_TAIL_MIN_SUPPORT="${TYPE_GUIDED_TAIL_MIN_SUPPORT:-2}"
TYPE_GUIDED_TAIL_MAX_RECORDS="${TYPE_GUIDED_TAIL_MAX_RECORDS:-32}"
TYPE_GUIDED_TAIL_MAX_LEAF_GROUPS="${TYPE_GUIDED_TAIL_MAX_LEAF_GROUPS:-4}"
TYPE_GUIDED_TAIL_WINDOW_EPOCHS="${TYPE_GUIDED_TAIL_WINDOW_EPOCHS:-3}"
TYPE_GUIDED_TAIL_REQUIRE_CROSS_STEP="${TYPE_GUIDED_TAIL_REQUIRE_CROSS_STEP:-true}"

# Split paths.
LIVEMATH_SPLIT_DIR="${LIVEMATH_SPLIT_DIR:-${PROJECT_ROOT}/data/livemathematicianbench_split}"
DOCVQA_SPLIT_DIR="${DOCVQA_SPLIT_DIR:-${PROJECT_ROOT}/data/docvqa/splits}"
OFFICEQA_SPLIT_DIR="${OFFICEQA_SPLIT_DIR:-${PROJECT_ROOT}/data/officeqa_split}"
OFFICEQA_DOCS_DIR="${OFFICEQA_DOCS_DIR:-${PROJECT_ROOT}/data/officeqa_docs_official}"
SPREADSHEETBENCH_SPLIT_DIR="${SPREADSHEETBENCH_SPLIT_DIR:-${PROJECT_ROOT}/data/spreadsheetbench_split}"
SPREADSHEETBENCH_DATA_ROOT="${SPREADSHEETBENCH_DATA_ROOT:-${PROJECT_ROOT}/data/spreadsheetbench_verified_400}"

# OfficeQA needs document-lookup tool calls. Qwen3.6-35B-A3B exposes native
# function calling through DashScope, so no server-side tool flags are needed.
OFFICEQA_USE_LOCAL_TOOLS="${OFFICEQA_USE_LOCAL_TOOLS:-true}"
OFFICEQA_SEARCH_MODE="${OFFICEQA_SEARCH_MODE:-offline}"

LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs/skillopt_v3_ali_qwen36a3b_dspro_${TS}}"
OUT_BASE="${OUT_BASE:-${PROJECT_ROOT}/outputs/skillopt_v3_ali_qwen36a3b_dspro_${TS}}"
mkdir -p "${LOG_DIR}" "${OUT_BASE}"

RUN_LOG="${LOG_DIR}/launcher.log"
PIDS_FILE="${LOG_DIR}/pids.txt"
DONE_FILE="${LOG_DIR}/completed.tsv"
: > "${RUN_LOG}"
: > "${PIDS_FILE}"
: > "${DONE_FILE}"

[[ "${MAX_PARALLEL}" =~ ^[0-9]+$ ]] || fail "MAX_PARALLEL must be an integer."
[[ "${MAX_PARALLEL}" -ge 1 ]] || fail "MAX_PARALLEL must be >= 1."
[[ "${API_MAX_CONCURRENCY}" =~ ^[0-9]+$ ]] || fail "API_MAX_CONCURRENCY must be an integer."
[[ "${WORKERS}" =~ ^[0-9]+$ ]] || fail "WORKERS must be an integer."
[[ "${ANALYST_WORKERS}" =~ ^[0-9]+$ ]] || fail "ANALYST_WORKERS must be an integer."
[[ "${WORKERS}" -ge 1 ]] || fail "WORKERS must be >= 1."
[[ "${ANALYST_WORKERS}" -ge 1 ]] || fail "ANALYST_WORKERS must be >= 1."
[[ "${WORKERS}" -le "${API_MAX_CONCURRENCY}" ]] || fail "WORKERS=${WORKERS} exceeds API_MAX_CONCURRENCY=${API_MAX_CONCURRENCY}."
[[ "${ANALYST_WORKERS}" -le "${API_MAX_CONCURRENCY}" ]] || fail "ANALYST_WORKERS=${ANALYST_WORKERS} exceeds API_MAX_CONCURRENCY=${API_MAX_CONCURRENCY}."
[[ "${TYPE_GUIDED_FALLBACK_RECONCILE}" == "off" || "${TYPE_GUIDED_FALLBACK_RECONCILE}" == "deterministic" || "${TYPE_GUIDED_FALLBACK_RECONCILE}" == "llm_select" || "${TYPE_GUIDED_FALLBACK_RECONCILE}" == "llm_fuse" ]] || fail "TYPE_GUIDED_FALLBACK_RECONCILE must be off, deterministic, llm_select, or llm_fuse."
[[ "${GATE_METRIC}" == "hard" || "${GATE_METRIC}" == "soft" || "${GATE_METRIC}" == "mixed" ]] || fail "GATE_METRIC must be hard, soft, or mixed."

# ---------------------------------------------------------------------------
# Optimizer + target smoke against the API (skipped on DRY_RUN).
# ---------------------------------------------------------------------------
api_smoke() {
  "${PYTHON_BIN}" - "${AZURE_OPENAI_ENDPOINT}" "${OPTIMIZER_AZURE_OPENAI_API_KEY}" "${OPTIMIZER_MODEL}" "${TARGET_AZURE_OPENAI_API_KEY}" "${TARGET_MODEL}" <<'PY'
import json
import sys
import urllib.error
import urllib.request

endpoint, opt_key, opt_model, tgt_key, tgt_model = sys.argv[1:6]
base = endpoint.rstrip("/")

def call(key, model, messages, timeout=120):
    payload = {"model": model, "messages": messages, "max_tokens": 32, "temperature": 0.0}
    req = urllib.request.Request(
        f"{base}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return (data.get("choices") or [{}])[0].get("message", {}).get("content", "")

failed = False
try:
    c = call(opt_key, opt_model, [{"role": "user", "content": "Return exactly OK."}])
    print(f"[smoke/optimizer] {opt_model} -> OK content={c[:40]!r}")
except urllib.error.HTTPError as e:
    failed = True
    print(f"[smoke/optimizer] {opt_model} -> FAIL HTTP {e.code}: {e.read().decode('utf-8','replace')[:200]}", file=sys.stderr)
except Exception as e:
    failed = True
    print(f"[smoke/optimizer] {opt_model} -> FAIL {type(e).__name__}: {str(e)[:160]}", file=sys.stderr)

try:
    c = call(tgt_key, tgt_model, [{"role": "user", "content": "Solve 2+2. Reply only <answer>4</answer>."}])
    print(f"[smoke/target]    {tgt_model} -> OK content={c[:40]!r}")
except urllib.error.HTTPError as e:
    failed = True
    print(f"[smoke/target]    {tgt_model} -> FAIL HTTP {e.code}: {e.read().decode('utf-8','replace')[:200]}", file=sys.stderr)
except Exception as e:
    failed = True
    print(f"[smoke/target]    {tgt_model} -> FAIL {type(e).__name__}: {str(e)[:160]}", file=sys.stderr)

raise SystemExit(1 if failed else 0)
PY
}

echo "============================================================"
echo "  SkillOpt-Tree V3 Parallel: Aliyun API (ds-pro + qwen3.6-35b-a3b)"
echo "============================================================"
echo "  project:          ${PROJECT_ROOT}"
echo "  optimizer:        ${OPTIMIZER_MODEL}"
echo "  target:           ${TARGET_MODEL}"
echo "  api_endpoint:     ${AZURE_OPENAI_ENDPOINT}"
echo "  datasets:         ${DATASETS}"
echo "  max_parallel:     ${MAX_PARALLEL}"
echo "  epochs:           ${NUM_EPOCHS}"
echo "  api_limit:        ${API_MAX_CONCURRENCY}"
echo "  workers:          ${WORKERS}"
echo "  analyst_workers:  ${ANALYST_WORKERS}"
echo "  train_size:       ${TRAIN_SIZE} (0 means split train size)"
echo "  limit:            ${LIMIT} (0 means full split)"
echo "  officeqa_tools:   ${OFFICEQA_USE_LOCAL_TOOLS}"
echo "  tree_depth:       ${TYPE_GUIDED_TREE_DEPTH}"
echo "  rollout_repeats:  ${TYPE_GUIDED_ROLLOUT_REPEATS}"
echo "  gate:             ${GATE_METRIC} (mixed_weight=${GATE_MIXED_WEIGHT})"
echo "  dry_run:          ${DRY_RUN}"
echo "  log_dir:          ${LOG_DIR}"
echo "  out_base:         ${OUT_BASE}"
echo "============================================================"

if truthy "${DRY_RUN}"; then
  echo "[dry-run] Skip API key check and API smoke test."
else
  [[ -n "${AZURE_OPENAI_API_KEY}" ]] || fail "API key is empty. Export DASHSCOPE_API_KEY (or ALI_API_KEY) first."
  api_smoke || fail "API smoke test failed; see errors above."
fi

COMMON_ARGS=(
  "${PYTHON_BIN}" -u "${PROJECT_ROOT}/scripts/cli/train.py"
  --optimizer_backend openai_chat
  --target_backend openai_chat
  --optimizer_model "${OPTIMIZER_MODEL}"
  --target_model "${TARGET_MODEL}"
  --reasoning_effort "${REASONING_EFFORT}"
  --num_epochs "${NUM_EPOCHS}"
  --train_size "${TRAIN_SIZE}"
  --accumulation "${ACCUMULATION}"
  --seed "${SEED}"
  --edit_budget "${EDIT_BUDGET}"
  --min_edit_budget "${MIN_EDIT_BUDGET}"
  --lr_scheduler "${LR_SCHEDULER}"
  --lr_control_mode "${LR_CONTROL_MODE}"
  --use_gate "${USE_GATE}"
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
  --type_guided_clustering "${TYPE_GUIDED_CLUSTERING}"
  --type_guided_cluster_target_size "${TYPE_GUIDED_CLUSTER_TARGET_SIZE}"
  --type_guided_cluster_max_size "${TYPE_GUIDED_CLUSTER_MAX_SIZE}"
  --type_guided_tail_bank "${TYPE_GUIDED_TAIL_BANK}"
  --type_guided_tail_min_support "${TYPE_GUIDED_TAIL_MIN_SUPPORT}"
  --type_guided_tail_max_records "${TYPE_GUIDED_TAIL_MAX_RECORDS}"
  --type_guided_tail_max_leaf_groups "${TYPE_GUIDED_TAIL_MAX_LEAF_GROUPS}"
  --type_guided_tail_window_epochs "${TYPE_GUIDED_TAIL_WINDOW_EPOCHS}"
  --type_guided_tail_require_cross_step "${TYPE_GUIDED_TAIL_REQUIRE_CROSS_STEP}"
  --split_mode split_dir
  --limit "${LIMIT}"
)

job_pids=()
job_names=()

reap_finished_jobs() {
  local new_pids=()
  local new_names=()
  local idx pid name status
  for idx in "${!job_pids[@]}"; do
    pid="${job_pids[$idx]}"
    name="${job_names[$idx]}"
    if kill -0 "${pid}" 2>/dev/null; then
      new_pids+=("${pid}")
      new_names+=("${name}")
    else
      if wait "${pid}"; then
        status=0
      else
        status=$?
      fi
      printf '%s\t%s\t%s\n' "${name}" "${status}" "${LOG_DIR}/${name}.log" | tee -a "${DONE_FILE}"
    fi
  done
  if [[ "${#new_pids[@]}" -gt 0 ]]; then
    job_pids=("${new_pids[@]}")
    job_names=("${new_names[@]}")
  else
    job_pids=()
    job_names=()
  fi
}

wait_for_slot() {
  while [[ "${#job_pids[@]}" -ge "${MAX_PARALLEL}" ]]; do
    reap_finished_jobs
    if [[ "${#job_pids[@]}" -ge "${MAX_PARALLEL}" ]]; then
      sleep 5
    fi
  done
}

launch_job() {
  local name="$1"
  shift
  local log_file="${LOG_DIR}/${name}.log"
  echo "[launch] ${name}"
  echo "         log: ${log_file}"
  {
    echo "[start] $(date)"
    printf '[cmd]'
    quote_cmd "$@"
    if truthy "${DRY_RUN}"; then
      echo "[dry-run] command not executed"
      status=0
    else
      set +e
      "$@"
      status=$?
      set -e
    fi
    echo "[exit] ${status} $(date)"
    exit "${status}"
  } > "${log_file}" 2>&1 &
  local pid=$!
  job_pids+=("${pid}")
  job_names+=("${name}")
  echo "${name} ${pid} ${log_file}" | tee -a "${PIDS_FILE}"
}

dataset_bool_default() {
  local value="$1"
  local default="$2"
  if [[ -n "${value}" ]]; then
    printf '%s\n' "${value}"
  else
    printf '%s\n' "${default}"
  fi
}

launch_dataset() {
  local dataset="$1"
  local config=""
  local split_dir=""
  local out_root=""
  local batch_size=""
  local workers=""
  local analyst_workers=""
  local sel_env_num=""
  local test_env_num=""
  local max_turns=""
  local max_steps=""
  local target_max_completion_tokens="${TARGET_MAX_COMPLETION_TOKENS}"
  local fallback_sel_env_num="${TYPE_GUIDED_FALLBACK_SEL_ENV_NUM}"
  local extra_args=()
  local cfg_options=(
    "evaluation.gate_metric=${GATE_METRIC}"
    "evaluation.gate_mixed_weight=${GATE_MIXED_WEIGHT}"
  )

  case "${dataset}" in
    livemath|livemathematicianbench)
      dataset="livemath"
      config="${PROJECT_ROOT}/configs/livemathematicianbench/default.yaml"
      split_dir="${LIVEMATH_SPLIT_DIR}"
      batch_size="${LIVEMATH_BATCH_SIZE:-${BATCH_SIZE:-16}}"
      workers="${LIVEMATH_WORKERS:-${WORKERS}}"
      analyst_workers="${LIVEMATH_ANALYST_WORKERS:-${ANALYST_WORKERS}}"
      sel_env_num="${LIVEMATH_SEL_ENV_NUM:-0}"
      test_env_num="${LIVEMATH_TEST_ENV_NUM:-0}"
      max_turns="${LIVEMATH_MAX_TURNS:-1}"
      target_max_completion_tokens="${LIVEMATH_TARGET_MAX_COMPLETION_TOKENS}"
      extra_args+=(
        --shuffle_choices "$(dataset_bool_default "${LIVEMATH_SHUFFLE_CHOICES:-}" true)"
        --use_theorem "$(dataset_bool_default "${LIVEMATH_USE_THEOREM:-}" false)"
        --use_sketch "$(dataset_bool_default "${LIVEMATH_USE_SKETCH:-}" false)"
      )
      ;;
    docvqa)
      config="${PROJECT_ROOT}/configs/docvqa/default.yaml"
      split_dir="${DOCVQA_SPLIT_DIR}"
      batch_size="${DOCVQA_BATCH_SIZE:-${BATCH_SIZE:-32}}"
      workers="${DOCVQA_WORKERS:-${WORKERS}}"
      analyst_workers="${DOCVQA_ANALYST_WORKERS:-${ANALYST_WORKERS}}"
      sel_env_num="${DOCVQA_SEL_ENV_NUM:-0}"
      test_env_num="${DOCVQA_TEST_ENV_NUM:-0}"
      max_turns="${DOCVQA_MAX_TURNS:-1}"
      target_max_completion_tokens="${DOCVQA_TARGET_MAX_COMPLETION_TOKENS}"
      extra_args+=(--image_detail "${DOCVQA_IMAGE_DETAIL:-auto}")
      ;;
    officeqa)
      config="${PROJECT_ROOT}/configs/officeqa/default.yaml"
      split_dir="${OFFICEQA_SPLIT_DIR}"
      batch_size="${OFFICEQA_BATCH_SIZE:-${BATCH_SIZE:-16}}"
      workers="${OFFICEQA_WORKERS:-${WORKERS}}"
      analyst_workers="${OFFICEQA_ANALYST_WORKERS:-${ANALYST_WORKERS}}"
      sel_env_num="${OFFICEQA_SEL_ENV_NUM:-0}"
      test_env_num="${OFFICEQA_TEST_ENV_NUM:-0}"
      target_max_completion_tokens="${OFFICEQA_TARGET_MAX_COMPLETION_TOKENS}"
      cfg_options+=("env.data_dirs=${OFFICEQA_DOCS_DIR}")
      cfg_options+=("env.use_local_tools=${OFFICEQA_USE_LOCAL_TOOLS}")
      cfg_options+=("env.search_mode=${OFFICEQA_SEARCH_MODE}")
      require_dir "OfficeQA docs dir" "${OFFICEQA_DOCS_DIR}"
      ;;
    spreadsheetbench)
      config="${PROJECT_ROOT}/configs/spreadsheetbench/default.yaml"
      split_dir="${SPREADSHEETBENCH_SPLIT_DIR}"
      batch_size="${SPREADSHEETBENCH_BATCH_SIZE:-${BATCH_SIZE:-16}}"
      workers="${SPREADSHEETBENCH_WORKERS:-${WORKERS}}"
      analyst_workers="${SPREADSHEETBENCH_ANALYST_WORKERS:-${ANALYST_WORKERS}}"
      sel_env_num="${SPREADSHEETBENCH_SEL_ENV_NUM:-0}"
      test_env_num="${SPREADSHEETBENCH_TEST_ENV_NUM:-0}"
      max_turns="${SPREADSHEETBENCH_MAX_TURNS:-30}"
      target_max_completion_tokens="${SPREADSHEETBENCH_TARGET_MAX_COMPLETION_TOKENS}"
      extra_args+=(--data_root "${SPREADSHEETBENCH_DATA_ROOT}")
      require_dir "SpreadsheetBench data root" "${SPREADSHEETBENCH_DATA_ROOT}"
      ;;
    *)
      fail "Unknown dataset '${dataset}'. Supported: livemath docvqa officeqa spreadsheetbench"
      ;;
  esac

  [[ -f "${config}" ]] || fail "Missing config: ${config}"
  require_dir "${dataset} split dir" "${split_dir}"
  require_dir "${dataset} train split" "${split_dir}/train"
  require_dir "${dataset} val split" "${split_dir}/val"
  require_dir "${dataset} test split" "${split_dir}/test"

  [[ "${batch_size}" =~ ^[0-9]+$ ]] || fail "${dataset} batch_size must be an integer: ${batch_size}"
  [[ "${workers}" =~ ^[0-9]+$ ]] || fail "${dataset} workers must be an integer: ${workers}"
  [[ "${analyst_workers}" =~ ^[0-9]+$ ]] || fail "${dataset} analyst_workers must be an integer: ${analyst_workers}"
  [[ "${sel_env_num}" =~ ^[0-9]+$ ]] || fail "${dataset} sel_env_num must be an integer: ${sel_env_num}"
  [[ "${test_env_num}" =~ ^[0-9]+$ ]] || fail "${dataset} test_env_num must be an integer: ${test_env_num}"
  [[ "${workers}" -ge 1 ]] || fail "${dataset} workers must be >= 1."
  [[ "${analyst_workers}" -ge 1 ]] || fail "${dataset} analyst_workers must be >= 1."
  [[ "${workers}" -le "${API_MAX_CONCURRENCY}" ]] || fail "${dataset} workers=${workers} exceeds API_MAX_CONCURRENCY=${API_MAX_CONCURRENCY}."
  [[ "${analyst_workers}" -le "${API_MAX_CONCURRENCY}" ]] || fail "${dataset} analyst_workers=${analyst_workers} exceeds API_MAX_CONCURRENCY=${API_MAX_CONCURRENCY}."

  local train_count val_count test_count effective_sel effective_test
  train_count="$(split_count "${split_dir}/train")"
  val_count="$(split_count "${split_dir}/val")"
  test_count="$(split_count "${split_dir}/test")"
  effective_sel="$(effective_eval_count "${sel_env_num}" "${val_count}")"
  effective_test="$(effective_eval_count "${test_env_num}" "${test_count}")"
  echo "[dataset] ${dataset}: split_dir=${split_dir}"
  echo "          split_counts train=${train_count} val=${val_count} test=${test_count}"
  echo "          train_size=${TRAIN_SIZE} limit=${LIMIT} sel_env_num=${sel_env_num} -> ${effective_sel} test_env_num=${test_env_num} -> ${effective_test}"
  cfg_options+=("env.max_completion_tokens=${target_max_completion_tokens}")

  out_root="${OUT_BASE}/${dataset}"
  mkdir -p "${out_root}" "${out_root}/type_guided_cache"

  local cmd=(
    "${COMMON_ARGS[@]}"
    --config "${config}"
    --batch_size "${batch_size}"
    --workers "${workers}"
    --analyst_workers "${analyst_workers}"
    --type_guided_patch_record_workers "${TYPE_GUIDED_PATCH_RECORD_WORKERS:-${analyst_workers}}"
    --type_guided_fallback_sel_env_num "${fallback_sel_env_num}"
    --type_guided_fallback_reconcile "${TYPE_GUIDED_FALLBACK_RECONCILE}"
    --type_guided_fallback_reconcile_min_children "${TYPE_GUIDED_FALLBACK_RECONCILE_MIN_CHILDREN}"
    --sel_env_num "${sel_env_num}"
    --test_env_num "${test_env_num}"
    --split_dir "${split_dir}"
    --out_root "${out_root}"
    --type_guided_cache_dir "${out_root}/type_guided_cache"
  )
  if [[ "${#extra_args[@]}" -gt 0 ]]; then
    cmd+=("${extra_args[@]}")
  fi
  if [[ -n "${max_turns}" ]]; then
    cmd+=(--max_turns "${max_turns}")
  fi
  if [[ -n "${max_steps}" ]]; then
    cmd+=(--max_steps "${max_steps}")
  fi
  cmd+=(--cfg-options "${cfg_options[@]}")

  wait_for_slot
  launch_job "${dataset}" "${cmd[@]}"
}

read -r -a DATASET_LIST <<< "${DATASETS}"
[[ "${#DATASET_LIST[@]}" -gt 0 ]] || fail "DATASETS is empty."

for dataset in "${DATASET_LIST[@]}"; do
  launch_dataset "${dataset}"
done

echo
echo "Launched ${#DATASET_LIST[@]} dataset jobs."
echo "Log dir: ${LOG_DIR}"
echo "Output base: ${OUT_BASE}"
echo "Training PIDs: ${PIDS_FILE}"
echo
echo "Monitor:"
echo "  tail -f ${LOG_DIR}/*.log"
echo
echo "Stop trainings:"
echo "  awk '{print \$2}' ${PIDS_FILE} | xargs -r kill"

if truthy "${WAIT_FOR_JOBS}"; then
  echo
  echo "Waiting for training jobs to finish..."
  overall_status=0
  while [[ "${#job_pids[@]}" -gt 0 ]]; do
    before="${#job_pids[@]}"
    reap_finished_jobs
    after="${#job_pids[@]}"
    if [[ "${after}" -eq "${before}" ]]; then
      sleep 5
    fi
  done

  while IFS=$'\t' read -r name status log_file; do
    [[ -n "${name}" ]] || continue
    if [[ "${status}" != "0" ]]; then
      overall_status=1
      echo "[failed] ${name} exit_code=${status} log=${log_file}" >&2
    fi
  done < "${DONE_FILE}"

  echo "All dataset jobs finished. Completed file: ${DONE_FILE}"
  exit "${overall_status}"
fi

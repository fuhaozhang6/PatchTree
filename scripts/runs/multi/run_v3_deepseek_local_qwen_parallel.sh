#!/usr/bin/env bash
set -euo pipefail

# Parallel V3 launcher:
#   optimizer / teacher = DeepSeek V4 through Ark OpenAI-compatible API
#   target / student    = local Qwen served by vLLM
#   data                = full existing split_dir train/val/test splits by default
#
# Examples:
#   DRY_RUN=1 bash scripts/runs/multi/run_v3_deepseek_local_qwen_parallel.sh
#   DATASETS="docvqa officeqa spreadsheetbench searchqa livemath" MAX_PARALLEL=3 bash scripts/runs/multi/run_v3_deepseek_local_qwen_parallel.sh

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

# DeepSeek optimizer through Ark OpenAI-compatible API.
export AZURE_OPENAI_ENDPOINT="${AZURE_OPENAI_ENDPOINT:-https://ark.cn-beijing.volces.com/api/v3}"
export AZURE_OPENAI_API_KEY="${AZURE_OPENAI_API_KEY:-${ARK_API_KEY:-}}"
export AZURE_OPENAI_AUTH_MODE="${AZURE_OPENAI_AUTH_MODE:-openai_compatible}"
export AZURE_OPENAI_API_VERSION="${AZURE_OPENAI_API_VERSION:-2024-12-01-preview}"

export OPTIMIZER_AZURE_OPENAI_ENDPOINT="${OPTIMIZER_AZURE_OPENAI_ENDPOINT:-${AZURE_OPENAI_ENDPOINT}}"
export OPTIMIZER_AZURE_OPENAI_API_KEY="${OPTIMIZER_AZURE_OPENAI_API_KEY:-${AZURE_OPENAI_API_KEY}}"
export OPTIMIZER_AZURE_OPENAI_AUTH_MODE="${OPTIMIZER_AZURE_OPENAI_AUTH_MODE:-${AZURE_OPENAI_AUTH_MODE}}"
export OPTIMIZER_AZURE_OPENAI_API_VERSION="${OPTIMIZER_AZURE_OPENAI_API_VERSION:-${AZURE_OPENAI_API_VERSION}}"

# Local Qwen target served by vLLM. The default uses four GPUs, intended for four H20s.
MODEL_PATH="${MODEL_PATH:-/ai-app-vepfs/modelscope/models/Qwen/Qwen3.5-4B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen/Qwen3.5-4B}"
QWEN_CUDA_VISIBLE_DEVICES="${QWEN_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-0,1,2,3}}"
VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-4}"
VLLM_HOST="${VLLM_HOST:-0.0.0.0}"
VLLM_PORT="${VLLM_PORT:-8000}"
QWEN_CHAT_BASE_URL="${QWEN_CHAT_BASE_URL:-http://127.0.0.1:${VLLM_PORT}/v1}"
QWEN_CHAT_API_KEY="${QWEN_CHAT_API_KEY:-dummy}"
QWEN_CHAT_MODEL="${QWEN_CHAT_MODEL:-${SERVED_MODEL_NAME}}"
QWEN_CHAT_MAX_TOKENS="${QWEN_CHAT_MAX_TOKENS:-16384}"
QWEN_CHAT_TIMEOUT_SECONDS="${QWEN_CHAT_TIMEOUT_SECONDS:-300}"
TARGET_QWEN_CHAT_TEMPERATURE="${TARGET_QWEN_CHAT_TEMPERATURE:-0.2}"
TARGET_QWEN_CHAT_ENABLE_THINKING="${TARGET_QWEN_CHAT_ENABLE_THINKING:-false}"
REQUIRE_QWEN_CONTENT="${REQUIRE_QWEN_CONTENT:-0}"
START_VLLM="${START_VLLM:-1}"
STOP_VLLM_ON_EXIT="${STOP_VLLM_ON_EXIT:-1}"
VLLM_ENABLE_AUTO_TOOL_CHOICE="${VLLM_ENABLE_AUTO_TOOL_CHOICE:-auto}"
VLLM_TOOL_CALL_PARSER="${VLLM_TOOL_CALL_PARSER:-qwen3_coder}"
VLLM_REASONING_PARSER="${VLLM_REASONING_PARSER:-qwen3}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-128}"
VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-}"
VLLM_ENABLE_CHUNKED_PREFILL="${VLLM_ENABLE_CHUNKED_PREFILL:-1}"
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"

export QWEN_CHAT_BASE_URL QWEN_CHAT_API_KEY QWEN_CHAT_MODEL QWEN_CHAT_MAX_TOKENS QWEN_CHAT_TIMEOUT_SECONDS
export TARGET_QWEN_CHAT_BASE_URL="${TARGET_QWEN_CHAT_BASE_URL:-${QWEN_CHAT_BASE_URL}}"
export TARGET_QWEN_CHAT_API_KEY="${TARGET_QWEN_CHAT_API_KEY:-${QWEN_CHAT_API_KEY}}"
export TARGET_QWEN_CHAT_TEMPERATURE
export TARGET_QWEN_CHAT_TIMEOUT_SECONDS="${TARGET_QWEN_CHAT_TIMEOUT_SECONDS:-${QWEN_CHAT_TIMEOUT_SECONDS}}"
export TARGET_QWEN_CHAT_MAX_TOKENS="${TARGET_QWEN_CHAT_MAX_TOKENS:-${QWEN_CHAT_MAX_TOKENS}}"
export TARGET_QWEN_CHAT_ENABLE_THINKING
export TARGET_QWEN_CHAT_ROLLOUT_RETRIES="${TARGET_QWEN_CHAT_ROLLOUT_RETRIES:-2}"

# Experiment defaults. DocVQA requires a vision-capable local Qwen model; if
# MODEL_PATH still points to a text-only model, override DATASETS or swap to Qwen-VL.
OPTIMIZER_MODEL="${OPTIMIZER_MODEL:-deepseek-v4-pro-260425}"
TARGET_MODEL="${TARGET_MODEL:-${SERVED_MODEL_NAME}}"
DATASETS="${DATASETS:-officeqa livemath searchqa}"
MAX_PARALLEL="${MAX_PARALLEL:-4}"
WAIT_FOR_JOBS="${WAIT_FOR_JOBS:-1}"
DRY_RUN="${DRY_RUN:-0}"
TS="${TS:-v3_deepseek_local_qwen_$(date +%Y%m%d_%H%M%S)}"

NUM_EPOCHS="${NUM_EPOCHS:-1}"
TRAIN_SIZE="${TRAIN_SIZE:-0}"
BATCH_SIZE="${BATCH_SIZE:-}"
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
TARGET_MAX_COMPLETION_TOKENS="${TARGET_MAX_COMPLETION_TOKENS:-16384}"
SEARCHQA_TARGET_MAX_COMPLETION_TOKENS="${SEARCHQA_TARGET_MAX_COMPLETION_TOKENS:-16384}"
LIVEMATH_TARGET_MAX_COMPLETION_TOKENS="${LIVEMATH_TARGET_MAX_COMPLETION_TOKENS:-16384}"

# V3 type-guided settings. soft is the default here to test semantic
# normalization without forcing all labels back to closed anchors.
TYPE_GUIDED_TREE_DEPTH="${TYPE_GUIDED_TREE_DEPTH:-3}"
TYPE_GUIDED_LEAF_FALLBACK="${TYPE_GUIDED_LEAF_FALLBACK:-true}"
TYPE_GUIDED_MIN_SUPPORT="${TYPE_GUIDED_MIN_SUPPORT:-2}"
TYPE_GUIDED_MAX_LEAF_GROUPS="${TYPE_GUIDED_MAX_LEAF_GROUPS:-8}"
TYPE_GUIDED_ROLLOUT_REPEATS="${TYPE_GUIDED_ROLLOUT_REPEATS:-3}"
TYPE_GUIDED_MAX_PATCH_RECORDS="${TYPE_GUIDED_MAX_PATCH_RECORDS:-24}"
TYPE_GUIDED_TAU_SUCC="${TYPE_GUIDED_TAU_SUCC:-1.0}"
TYPE_GUIDED_PATCH_RECORD_WORKERS="${TYPE_GUIDED_PATCH_RECORD_WORKERS:-}"
TYPE_GUIDED_CLUSTERING="${TYPE_GUIDED_CLUSTERING:-false}"
TYPE_GUIDED_CLUSTER_TARGET_SIZE="${TYPE_GUIDED_CLUSTER_TARGET_SIZE:-4}"
TYPE_GUIDED_CLUSTER_MAX_SIZE="${TYPE_GUIDED_CLUSTER_MAX_SIZE:-8}"
TYPE_GUIDED_LEAF_MERGE_WORKERS="${TYPE_GUIDED_LEAF_MERGE_WORKERS:-4}"
TYPE_GUIDED_MID_MERGE_WORKERS="${TYPE_GUIDED_MID_MERGE_WORKERS:-4}"
TYPE_GUIDED_FALLBACK_EVAL_ALL_LEAVES="${TYPE_GUIDED_FALLBACK_EVAL_ALL_LEAVES:-true}"
TYPE_GUIDED_FALLBACK_TOP_K="${TYPE_GUIDED_FALLBACK_TOP_K:-0}"
TYPE_GUIDED_FALLBACK_TAU_CHILD="${TYPE_GUIDED_FALLBACK_TAU_CHILD:-0.0}"
TYPE_GUIDED_FALLBACK_SEL_ENV_NUM="${TYPE_GUIDED_FALLBACK_SEL_ENV_NUM:-0}"
TYPE_GUIDED_FALLBACK_RECONCILE="${TYPE_GUIDED_FALLBACK_RECONCILE:-llm_fuse}"
TYPE_GUIDED_FALLBACK_RECONCILE_MIN_CHILDREN="${TYPE_GUIDED_FALLBACK_RECONCILE_MIN_CHILDREN:-2}"
TYPE_GUIDED_TAIL_BANK="${TYPE_GUIDED_TAIL_BANK:-true}"
TYPE_GUIDED_TAIL_MIN_SUPPORT="${TYPE_GUIDED_TAIL_MIN_SUPPORT:-2}"
TYPE_GUIDED_TAIL_MAX_RECORDS="${TYPE_GUIDED_TAIL_MAX_RECORDS:-32}"
TYPE_GUIDED_TAIL_MAX_LEAF_GROUPS="${TYPE_GUIDED_TAIL_MAX_LEAF_GROUPS:-4}"
TYPE_GUIDED_TAIL_WINDOW_EPOCHS="${TYPE_GUIDED_TAIL_WINDOW_EPOCHS:-3}"
TYPE_GUIDED_TAIL_REQUIRE_CROSS_STEP="${TYPE_GUIDED_TAIL_REQUIRE_CROSS_STEP:-true}"

# Split paths. DocVQA uses the materialized CSV/image split under
# data/docvqa/splits; data/docvqa_id_split is only the ID manifest.
SEARCHQA_SPLIT_DIR="${SEARCHQA_SPLIT_DIR:-${PROJECT_ROOT}/data/searchqa_split}"
ALFWORLD_SPLIT_DIR="${ALFWORLD_SPLIT_DIR:-${PROJECT_ROOT}/data/alfworld_path_split}"
LIVEMATH_SPLIT_DIR="${LIVEMATH_SPLIT_DIR:-${PROJECT_ROOT}/data/livemathematicianbench_split}"
DOCVQA_SPLIT_DIR="${DOCVQA_SPLIT_DIR:-${PROJECT_ROOT}/data/docvqa/splits}"
OFFICEQA_SPLIT_DIR="${OFFICEQA_SPLIT_DIR:-${PROJECT_ROOT}/data/officeqa_split}"
OFFICEQA_DOCS_DIR="${OFFICEQA_DOCS_DIR:-${PROJECT_ROOT}/data/officeqa_docs_official}"
SPREADSHEETBENCH_SPLIT_DIR="${SPREADSHEETBENCH_SPLIT_DIR:-${PROJECT_ROOT}/data/spreadsheetbench_split}"
SPREADSHEETBENCH_DATA_ROOT="${SPREADSHEETBENCH_DATA_ROOT:-${PROJECT_ROOT}/data/spreadsheetbench_verified_400}"

LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs/patchtree_deepseek_local_qwen_parallel_${TS}}"
OUT_BASE="${OUT_BASE:-${PROJECT_ROOT}/outputs/patchtree_deepseek_local_qwen_parallel_${TS}}"
mkdir -p "${LOG_DIR}" "${OUT_BASE}"

RUN_LOG="${LOG_DIR}/launcher.log"
VLLM_LOG="${LOG_DIR}/vllm_qwen.log"
VLLM_PID_FILE="${LOG_DIR}/vllm.pid"
PIDS_FILE="${LOG_DIR}/pids.txt"
DONE_FILE="${LOG_DIR}/completed.tsv"
DETACHED_JOBS=0
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
[[ "${VLLM_MAX_NUM_SEQS}" =~ ^[0-9]+$ ]] || fail "VLLM_MAX_NUM_SEQS must be an integer."
[[ "${VLLM_MAX_NUM_SEQS}" -ge 1 ]] || fail "VLLM_MAX_NUM_SEQS must be >= 1."
if [[ -n "${VLLM_MAX_NUM_BATCHED_TOKENS}" ]]; then
  [[ "${VLLM_MAX_NUM_BATCHED_TOKENS}" =~ ^[0-9]+$ ]] || fail "VLLM_MAX_NUM_BATCHED_TOKENS must be an integer when set."
fi
[[ "${TYPE_GUIDED_FALLBACK_RECONCILE}" == "off" || "${TYPE_GUIDED_FALLBACK_RECONCILE}" == "deterministic" || "${TYPE_GUIDED_FALLBACK_RECONCILE}" == "llm_select" || "${TYPE_GUIDED_FALLBACK_RECONCILE}" == "llm_fuse" ]] || fail "TYPE_GUIDED_FALLBACK_RECONCILE must be off, deterministic, llm_select, or llm_fuse."

officeqa_uses_local_tools() {
  case " ${DATASETS} " in
    *" officeqa "*) truthy "${OFFICEQA_USE_LOCAL_TOOLS:-true}" ;;
    *) return 1 ;;
  esac
}

vllm_auto_tool_choice_enabled() {
  case "${VLLM_ENABLE_AUTO_TOOL_CHOICE}" in
    auto) officeqa_uses_local_tools ;;
    *) truthy "${VLLM_ENABLE_AUTO_TOOL_CHOICE}" ;;
  esac
}

endpoint_ready() {
  "${PYTHON_BIN}" - "${QWEN_CHAT_BASE_URL}" "${QWEN_CHAT_API_KEY}" <<'PY'
import sys
import urllib.request

base = sys.argv[1].rstrip("/")
api_key = sys.argv[2]
req = urllib.request.Request(
    f"{base}/models",
    headers={"Authorization": f"Bearer {api_key}"},
    method="GET",
)
try:
    with urllib.request.urlopen(req, timeout=5) as resp:
        raise SystemExit(0 if resp.status == 200 else 1)
except Exception:
    raise SystemExit(1)
PY
}

cleanup() {
  if truthy "${DETACHED_JOBS}"; then
    return
  fi
  if truthy "${STOP_VLLM_ON_EXIT}" && [[ -f "${VLLM_PID_FILE}" ]]; then
    local pid
    pid="$(cat "${VLLM_PID_FILE}")"
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      echo "[cleanup] stopping vLLM pid=${pid}"
      kill "${pid}" 2>/dev/null || true
    fi
  fi
}
trap cleanup EXIT

qwen_smoke() {
  "${PYTHON_BIN}" - "${QWEN_CHAT_BASE_URL}" "${QWEN_CHAT_API_KEY}" "${SERVED_MODEL_NAME}" "${REQUIRE_QWEN_CONTENT}" <<'PY'
import json
import sys
import urllib.request

base, api_key, model = sys.argv[1].rstrip("/"), sys.argv[2], sys.argv[3]
require_content = str(sys.argv[4]).strip().lower() in {"1", "true", "yes", "on"}
payload = {
    "model": model,
    "messages": [{"role": "user", "content": "Say hello in one short sentence."}],
    "max_tokens": 128,
    "temperature": 0.2,
    "chat_template_kwargs": {"enable_thinking": False},
}
req = urllib.request.Request(
    f"{base}/chat/completions",
    data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(req, timeout=120) as resp:
    data = json.loads(resp.read().decode("utf-8"))
choice = (data.get("choices") or [{}])[0]
message = choice.get("message") or {}
content = message.get("content") or ""
reasoning = message.get("reasoning_content") or ""
print(f"[smoke/qwen] finish_reason={choice.get('finish_reason')} content_len={len(content)} reasoning_len={len(reasoning)}")
print(f"[smoke/qwen] content_preview={content[:120]!r}")
if not content.strip():
    warning = "[smoke/qwen] Empty message.content from Qwen target."
    print(warning, file=sys.stderr)
    if require_content:
        raise SystemExit(warning)
PY
}

qwen_tool_smoke() {
  "${PYTHON_BIN}" - "${QWEN_CHAT_BASE_URL}" "${QWEN_CHAT_API_KEY}" "${SERVED_MODEL_NAME}" <<'PY'
import json
import sys
import urllib.error
import urllib.request

base, api_key, model = sys.argv[1].rstrip("/"), sys.argv[2], sys.argv[3]
payload = {
    "model": model,
    "messages": [{"role": "user", "content": "Call the available lookup tool with query hello."}],
    "tools": [{
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "A tiny smoke-test lookup tool.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    }],
    "tool_choice": "auto",
    "max_tokens": 64,
    "temperature": 0.0,
    "chat_template_kwargs": {"enable_thinking": False},
}
req = urllib.request.Request(
    f"{base}/chat/completions",
    data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    method="POST",
)
try:
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode("utf-8"))
except urllib.error.HTTPError as e:
    body = e.read().decode("utf-8", errors="replace")
    raise SystemExit(f"[smoke/qwen-tools] failed HTTP {e.code}: {body}")
choice = (data.get("choices") or [{}])[0]
message = choice.get("message") or {}
tool_calls = message.get("tool_calls") or []
content = message.get("content") or ""
text_tool_call = "<tool_call" in content.lower()
print(
    "[smoke/qwen-tools] "
    f"finish_reason={choice.get('finish_reason')} "
    f"tool_calls={len(tool_calls)} "
    f"text_tool_call={int(text_tool_call)} "
    f"content_len={len(content)}"
)
if not tool_calls:
    raise SystemExit(
        "[smoke/qwen-tools] expected a structured tool call from the configured "
        "vLLM tool parser. "
        "Disable OfficeQA local tools with OFFICEQA_USE_LOCAL_TOOLS=false "
        "or fix vLLM tool-call parsing before training OfficeQA."
    )
PY
}

echo "============================================================"
echo "  PatchTree Parallel: DeepSeek optimizer + local Qwen/vLLM"
echo "============================================================"
echo "  project:          ${PROJECT_ROOT}"
echo "  optimizer:        ${OPTIMIZER_MODEL}"
echo "  target:           ${TARGET_MODEL}"
echo "  qwen_url:         ${QWEN_CHAT_BASE_URL}"
echo "  model_path:       ${MODEL_PATH}"
echo "  cuda_devices:     ${QWEN_CUDA_VISIBLE_DEVICES}"
echo "  tensor_parallel:  ${VLLM_TENSOR_PARALLEL_SIZE}"
echo "  vllm_max_seqs:    ${VLLM_MAX_NUM_SEQS}"
echo "  chunked_prefill:  ${VLLM_ENABLE_CHUNKED_PREFILL}"
echo "  vllm_tool_choice: ${VLLM_ENABLE_AUTO_TOOL_CHOICE} parser=${VLLM_TOOL_CALL_PARSER} reasoning=${VLLM_REASONING_PARSER:-off}"
echo "  datasets:         ${DATASETS}"
echo "  max_parallel:     ${MAX_PARALLEL}"
echo "  train_size:       ${TRAIN_SIZE} (0 means split train size)"
echo "  limit:            ${LIMIT} (0 means full split)"
echo "  clustering:       ${TYPE_GUIDED_CLUSTERING}"
echo "  fallback_reconcile:${TYPE_GUIDED_FALLBACK_RECONCILE}"
echo "  dry_run:          ${DRY_RUN}"
echo "  log_dir:          ${LOG_DIR}"
echo "  out_base:         ${OUT_BASE}"
echo "============================================================"

case " ${DATASETS} " in
  *" docvqa "*)
    case "${MODEL_PATH} ${SERVED_MODEL_NAME} ${TARGET_MODEL}" in
      *VL*|*vl*|*Vision*|*vision*|*Qwen3.5*|*qwen3.5*|*Qwen3.6*|*qwen3.6*) ;;
      *)
        echo "[warn] DATASETS includes docvqa, but the served Qwen name does not look vision-capable."
        echo "       For meaningful DocVQA runs, set MODEL_PATH/SERVED_MODEL_NAME/TARGET_MODEL to a Qwen-VL model."
        ;;
    esac
    ;;
esac

if truthy "${DRY_RUN}"; then
  echo "[dry-run] Skip Ark key check, vLLM startup, and Qwen smoke test."
else
  [[ -n "${AZURE_OPENAI_API_KEY}" ]] || fail "AZURE_OPENAI_API_KEY is empty. Export ARK_API_KEY or AZURE_OPENAI_API_KEY first."
  if endpoint_ready; then
    echo "[check] Existing Qwen endpoint is ready: ${QWEN_CHAT_BASE_URL}"
  else
    if ! truthy "${START_VLLM}"; then
      fail "Qwen endpoint is not ready and START_VLLM=${START_VLLM}."
    fi
    [[ -d "${MODEL_PATH}" ]] || fail "MODEL_PATH not found: ${MODEL_PATH}"
    echo "[vllm] Starting local Qwen service on CUDA_VISIBLE_DEVICES=${QWEN_CUDA_VISIBLE_DEVICES}..."
    vllm_args=(
      serve "${MODEL_PATH}"
      --served-model-name "${SERVED_MODEL_NAME}"
      --host "${VLLM_HOST}"
      --port "${VLLM_PORT}"
      --trust-remote-code
      --dtype "${VLLM_DTYPE:-bfloat16}"
      --tensor-parallel-size "${VLLM_TENSOR_PARALLEL_SIZE}"
      --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.90}"
      --max-model-len "${MAX_MODEL_LEN:-32768}"
      --enable-prefix-caching
      --max-num-seqs "${VLLM_MAX_NUM_SEQS}"
    )
    if [[ -n "${VLLM_MAX_NUM_BATCHED_TOKENS}" ]]; then
      vllm_args+=(--max-num-batched-tokens "${VLLM_MAX_NUM_BATCHED_TOKENS}")
    fi
    if truthy "${VLLM_ENABLE_CHUNKED_PREFILL}"; then
      vllm_args+=(--enable-chunked-prefill)
    fi
    # Detailed request logging is disabled by default in current vLLM releases.
    # Do not pass the removed legacy request-logging flag here.
    if vllm_auto_tool_choice_enabled; then
      vllm_args+=(--enable-auto-tool-choice --tool-call-parser "${VLLM_TOOL_CALL_PARSER}")
    fi
    if [[ -n "${VLLM_REASONING_PARSER}" ]]; then
      vllm_args+=(--reasoning-parser "${VLLM_REASONING_PARSER}")
    fi
    if [[ -n "${VLLM_EXTRA_ARGS}" ]]; then
      # shellcheck disable=SC2206
      extra_vllm_args=(${VLLM_EXTRA_ARGS})
      vllm_args+=("${extra_vllm_args[@]}")
    fi
    env CUDA_VISIBLE_DEVICES="${QWEN_CUDA_VISIBLE_DEVICES}" \
      nohup vllm "${vllm_args[@]}" \
        > "${VLLM_LOG}" 2>&1 &
    VLLM_PID=$!
    echo "${VLLM_PID}" > "${VLLM_PID_FILE}"
    echo "[vllm] pid=${VLLM_PID}"
    echo "[vllm] log=${VLLM_LOG}"

    echo "[vllm] Waiting for endpoint..."
    for i in $(seq 1 "${VLLM_WAIT_SECONDS:-600}"); do
      if endpoint_ready; then
        echo "[check] Qwen endpoint OK after ${i}s: ${QWEN_CHAT_BASE_URL}"
        break
      fi
      if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
        tail -n 80 "${VLLM_LOG}" || true
        fail "vLLM process exited before endpoint became ready."
      fi
      sleep 1
      if [[ "${i}" == "${VLLM_WAIT_SECONDS:-600}" ]]; then
        tail -n 80 "${VLLM_LOG}" || true
        fail "Timed out waiting for vLLM endpoint."
      fi
    done
  fi
  qwen_smoke
  if officeqa_uses_local_tools; then
    qwen_tool_smoke
  fi
fi

COMMON_ARGS=(
  "${PYTHON_BIN}" -u "${PROJECT_ROOT}/scripts/cli/train.py"
  --optimizer_backend openai_chat
  --target_backend qwen_chat
  --optimizer_model "${OPTIMIZER_MODEL}"
  --target_model "${TARGET_MODEL}"
  --target_qwen_chat_base_url "${TARGET_QWEN_CHAT_BASE_URL}"
  --target_qwen_chat_api_key "${TARGET_QWEN_CHAT_API_KEY}"
  --target_qwen_chat_temperature "${TARGET_QWEN_CHAT_TEMPERATURE}"
  --target_qwen_chat_timeout_seconds "${TARGET_QWEN_CHAT_TIMEOUT_SECONDS}"
  --target_qwen_chat_max_tokens "${TARGET_QWEN_CHAT_MAX_TOKENS}"
  --target_qwen_chat_enable_thinking "${TARGET_QWEN_CHAT_ENABLE_THINKING}"
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
  --type_guided_fallback_eval_all_leaves "${TYPE_GUIDED_FALLBACK_EVAL_ALL_LEAVES}"
  --type_guided_fallback_top_k "${TYPE_GUIDED_FALLBACK_TOP_K}"
  --type_guided_fallback_tau_child "${TYPE_GUIDED_FALLBACK_TAU_CHILD}"
  --type_guided_clustering "${TYPE_GUIDED_CLUSTERING}"
  --type_guided_cluster_target_size "${TYPE_GUIDED_CLUSTER_TARGET_SIZE}"
  --type_guided_cluster_max_size "${TYPE_GUIDED_CLUSTER_MAX_SIZE}"
  --type_guided_leaf_merge_workers "${TYPE_GUIDED_LEAF_MERGE_WORKERS}"
  --type_guided_mid_merge_workers "${TYPE_GUIDED_MID_MERGE_WORKERS}"
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
  local exec_timeout=""
  local llm_timeout=""
  local target_max_completion_tokens="${TARGET_MAX_COMPLETION_TOKENS}"
  local fallback_sel_env_num="${TYPE_GUIDED_FALLBACK_SEL_ENV_NUM}"
  local extra_args=()
  local cfg_options=(
    "evaluation.gate_metric=${GATE_METRIC}"
    "evaluation.gate_mixed_weight=${GATE_MIXED_WEIGHT}"
  )

  case "${dataset}" in
    searchqa)
      config="${PROJECT_ROOT}/configs/searchqa/default.yaml"
      split_dir="${SEARCHQA_SPLIT_DIR}"
      batch_size="${SEARCHQA_BATCH_SIZE:-${BATCH_SIZE:-40}}"
      workers="${SEARCHQA_WORKERS:-${WORKERS}}"
      analyst_workers="${SEARCHQA_ANALYST_WORKERS:-${ANALYST_WORKERS}}"
      # 0 means use the full val/test split.
      sel_env_num="${SEARCHQA_SEL_ENV_NUM:-0}"
      test_env_num="${SEARCHQA_TEST_ENV_NUM:-0}"
      max_turns="${SEARCHQA_MAX_TURNS:-1}"
      exec_timeout="${SEARCHQA_EXEC_TIMEOUT:-300}"
      target_max_completion_tokens="${SEARCHQA_TARGET_MAX_COMPLETION_TOKENS}"
      fallback_sel_env_num="${SEARCHQA_TYPE_GUIDED_FALLBACK_SEL_ENV_NUM:-80}"
      ;;
    alfworld)
      config="${PROJECT_ROOT}/configs/alfworld/default.yaml"
      split_dir="${ALFWORLD_SPLIT_DIR}"
      batch_size="${ALFWORLD_BATCH_SIZE:-${BATCH_SIZE:-8}}"
      workers="${ALFWORLD_WORKERS:-4}"
      analyst_workers="${ALFWORLD_ANALYST_WORKERS:-8}"
      sel_env_num="${ALFWORLD_SEL_ENV_NUM:-0}"
      test_env_num="${ALFWORLD_TEST_ENV_NUM:-0}"
      max_steps="${ALFWORLD_MAX_STEPS:-50}"
      extra_args+=(--max_api_workers "${ALFWORLD_MAX_API_WORKERS:-4}")
      cfg_options+=("optimizer.type_guided_min_support=${ALFWORLD_TYPE_GUIDED_MIN_SUPPORT:-1}")
      fallback_sel_env_num="${ALFWORLD_TYPE_GUIDED_FALLBACK_SEL_ENV_NUM:-12}"
      ;;
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
      exec_timeout="${LIVEMATH_EXEC_TIMEOUT:-300}"
      target_max_completion_tokens="${LIVEMATH_TARGET_MAX_COMPLETION_TOKENS}"
      fallback_sel_env_num="${LIVEMATH_TYPE_GUIDED_FALLBACK_SEL_ENV_NUM:-12}"
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
      workers="${DOCVQA_WORKERS:-16}"
      analyst_workers="${DOCVQA_ANALYST_WORKERS:-16}"
      # 0 means use the full val split. The default DocVQA val split has 53 items;
      # override DOCVQA_SEL_ENV_NUM=32 only when intentionally saving budget.
      sel_env_num="${DOCVQA_SEL_ENV_NUM:-0}"
      test_env_num="${DOCVQA_TEST_ENV_NUM:-0}"
      max_turns="${DOCVQA_MAX_TURNS:-1}"
      exec_timeout="${DOCVQA_EXEC_TIMEOUT:-300}"
      fallback_sel_env_num="${DOCVQA_TYPE_GUIDED_FALLBACK_SEL_ENV_NUM:-32}"
      extra_args+=(--image_detail "${DOCVQA_IMAGE_DETAIL:-auto}")
      ;;
    officeqa)
      config="${PROJECT_ROOT}/configs/officeqa/default.yaml"
      split_dir="${OFFICEQA_SPLIT_DIR}"
      batch_size="${OFFICEQA_BATCH_SIZE:-${BATCH_SIZE:-16}}"
      workers="${OFFICEQA_WORKERS:-4}"
      analyst_workers="${OFFICEQA_ANALYST_WORKERS:-12}"
      sel_env_num="${OFFICEQA_SEL_ENV_NUM:-0}"
      test_env_num="${OFFICEQA_TEST_ENV_NUM:-0}"
      exec_timeout="${OFFICEQA_EXEC_TIMEOUT:-300}"
      fallback_sel_env_num="${OFFICEQA_TYPE_GUIDED_FALLBACK_SEL_ENV_NUM:-16}"
      cfg_options+=("env.data_dirs=${OFFICEQA_DOCS_DIR}")
      # OfficeQA requires document lookup over Treasury bulletin files; no-tools mode
      # often collapses to all-zero exact match on numeric/table questions.
      cfg_options+=("env.use_local_tools=${OFFICEQA_USE_LOCAL_TOOLS:-true}")
      cfg_options+=("env.search_mode=${OFFICEQA_SEARCH_MODE:-offline}")
      if ! truthy "${DRY_RUN}"; then
        require_dir "OfficeQA docs dir" "${OFFICEQA_DOCS_DIR}"
      fi
      ;;
    spreadsheetbench)
      config="${PROJECT_ROOT}/configs/spreadsheetbench/default.yaml"
      split_dir="${SPREADSHEETBENCH_SPLIT_DIR}"
      batch_size="${SPREADSHEETBENCH_BATCH_SIZE:-${BATCH_SIZE:-16}}"
      workers="${SPREADSHEETBENCH_WORKERS:-12}"
      analyst_workers="${SPREADSHEETBENCH_ANALYST_WORKERS:-16}"
      sel_env_num="${SPREADSHEETBENCH_SEL_ENV_NUM:-0}"
      test_env_num="${SPREADSHEETBENCH_TEST_ENV_NUM:-0}"
      max_turns="${SPREADSHEETBENCH_MAX_TURNS:-30}"
      exec_timeout="${SPREADSHEETBENCH_EXEC_TIMEOUT:-1200}"
      llm_timeout="${SPREADSHEETBENCH_LLM_TIMEOUT:-300}"
      fallback_sel_env_num="${SPREADSHEETBENCH_TYPE_GUIDED_FALLBACK_SEL_ENV_NUM:-24}"
      extra_args+=(--data_root "${SPREADSHEETBENCH_DATA_ROOT}")
      if ! truthy "${DRY_RUN}"; then
        require_dir "SpreadsheetBench data root" "${SPREADSHEETBENCH_DATA_ROOT}"
      fi
      ;;
    *)
      fail "Unknown dataset '${dataset}'. Supported: searchqa alfworld livemath docvqa officeqa spreadsheetbench"
      ;;
  esac

  [[ -f "${config}" ]] || fail "Missing config: ${config}"
  if ! truthy "${DRY_RUN}"; then
    require_dir "${dataset} split dir" "${split_dir}"
    require_dir "${dataset} train split" "${split_dir}/train"
    require_dir "${dataset} val split" "${split_dir}/val"
    require_dir "${dataset} test split" "${split_dir}/test"
  elif [[ ! -d "${split_dir}" ]]; then
    echo "[dry-run] split dir is not mounted here; command will still be rendered: ${split_dir}"
  fi

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
  if [[ -n "${exec_timeout}" ]]; then
    cfg_options+=("env.exec_timeout=${exec_timeout}")
  fi
  if [[ -n "${llm_timeout}" ]]; then
    cfg_options+=("env.llm_timeout=${llm_timeout}")
  fi

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
[[ -f "${VLLM_PID_FILE}" ]] && echo "vLLM PID: $(cat "${VLLM_PID_FILE}")"
echo
echo "Monitor:"
echo "  tail -f ${LOG_DIR}/*.log"
echo
echo "Stop trainings:"
echo "  awk '{print \$2}' ${PIDS_FILE} | xargs -r kill"
echo
echo "Stop vLLM:"
echo "  [[ -f ${VLLM_PID_FILE} ]] && kill \$(cat ${VLLM_PID_FILE})"

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

# The caller explicitly requested detached training. Keep a vLLM process
# started by this launcher alive instead of killing it from the EXIT trap as
# soon as the launcher shell returns.
DETACHED_JOBS=1
if truthy "${STOP_VLLM_ON_EXIT}"; then
  echo "[detach] WAIT_FOR_JOBS=0: leaving vLLM running for background jobs."
  echo "         Stop it manually with the command printed above."
fi

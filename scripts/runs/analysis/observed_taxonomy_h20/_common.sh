#!/usr/bin/env bash

# Shared runtime for the observed-taxonomy H20 launchers. Source only.

OBSERVED_RUN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${OBSERVED_RUN_DIR}/../../../.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
AUDIT_PY="${PROJECT_ROOT}/scripts/tools/audit_observed_type_taxonomy.py"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUT_BASE="${OUT_BASE:-${PROJECT_ROOT}/outputs/observed_taxonomy_h20_${RUN_ID}}"
LOG_BASE="${LOG_BASE:-${PROJECT_ROOT}/logs/observed_taxonomy_h20_${RUN_ID}}"
SPLITS="${SPLITS:-train val test}"
REPEATS="${REPEATS:-3}"
SEED="${SEED:-42}"
LIMIT_PER_SPLIT="${LIMIT_PER_SPLIT:-0}"
DRY_RUN="${DRY_RUN:-0}"

MODEL_PATH="${MODEL_PATH:-/ai-app-vepfs/modelscope/models/Qwen/Qwen3.5-4B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen/Qwen3.5-4B}"
TARGET_MODEL="${TARGET_MODEL:-${SERVED_MODEL_NAME}}"
VLLM_HOST="${VLLM_HOST:-0.0.0.0}"
VLLM_PORT="${VLLM_PORT:-59317}"
QWEN_CHAT_BASE_URL="${QWEN_CHAT_BASE_URL:-http://127.0.0.1:${VLLM_PORT}/v1}"
QWEN_CHAT_API_KEY="${QWEN_CHAT_API_KEY:-dummy}"
QWEN_CUDA_VISIBLE_DEVICES="${QWEN_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-0}}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-128}"
VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-65536}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
VLLM_TOOL_CALL_PARSER="${VLLM_TOOL_CALL_PARSER:-qwen3_coder}"
VLLM_REASONING_PARSER="${VLLM_REASONING_PARSER:-qwen3}"
VLLM_ENABLE_CHUNKED_PREFILL="${VLLM_ENABLE_CHUNKED_PREFILL:-1}"
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"
START_VLLM="${START_VLLM:-1}"
STOP_VLLM_ON_EXIT="${STOP_VLLM_ON_EXIT:-1}"
TARGET_TEMPERATURE="${TARGET_TEMPERATURE:-0.2}"
TARGET_TIMEOUT_SECONDS="${TARGET_TIMEOUT_SECONDS:-300}"
TARGET_ENABLE_THINKING="${TARGET_ENABLE_THINKING:-false}"
OBSERVED_JOB_NAME="${OBSERVED_JOB_NAME:-$(hostname -s)_gpu${QWEN_CUDA_VISIBLE_DEVICES}}"
VLLM_LOG="${VLLM_LOG:-${LOG_BASE}/vllm_${OBSERVED_JOB_NAME}.log}"

VLLM_PID=""
JOB_PIDS=()

observed_fail() {
  echo "ERROR: $*" >&2
  exit 1
}

observed_truthy() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

configure_optimizer_source() {
  local source="${OPTIMIZER_SOURCE:-deepseek}"
  local key=""
  case "${source}" in
    deepseek)
      key="${DEEPSEEK_API_KEY:-${DS_API_KEY:-${DEEPSEEK_OFFICIAL_API_KEY:-}}}"
      export AZURE_OPENAI_ENDPOINT="${DEEPSEEK_BASE_URL:-https://api.deepseek.com}"
      export AZURE_OPENAI_API_VERSION=openai-compat
      export OPTIMIZER_MODEL="${OPTIMIZER_MODEL:-${DEEPSEEK_OFFICIAL_MODEL:-deepseek-v4-pro}}"
      ;;
    ark)
      key="${ARK_API_KEY:-}"
      export AZURE_OPENAI_ENDPOINT="${ARK_BASE_URL:-https://ark.cn-beijing.volces.com/api/v3}"
      export AZURE_OPENAI_API_VERSION="${ARK_API_VERSION:-2024-12-01-preview}"
      export OPTIMIZER_MODEL="${OPTIMIZER_MODEL:-${ARK_OPTIMIZER_MODEL:-deepseek-v4-pro-260425}}"
      ;;
    *)
      observed_fail "OPTIMIZER_SOURCE=${source}; expected deepseek or ark"
      ;;
  esac
  if [[ -z "${key}" ]] && ! observed_truthy "${DRY_RUN}"; then
    observed_fail "API key is empty for OPTIMIZER_SOURCE=${source}"
  fi
  export AZURE_OPENAI_API_KEY="${key}"
  export AZURE_OPENAI_AUTH_MODE=openai_compatible
  export OPTIMIZER_AZURE_OPENAI_ENDPOINT="${AZURE_OPENAI_ENDPOINT}"
  export OPTIMIZER_AZURE_OPENAI_API_VERSION="${AZURE_OPENAI_API_VERSION}"
  export OPTIMIZER_AZURE_OPENAI_API_KEY="${AZURE_OPENAI_API_KEY}"
  export OPTIMIZER_AZURE_OPENAI_AUTH_MODE="${AZURE_OPENAI_AUTH_MODE}"
  export DEEPSEEK_THINKING="${DEEPSEEK_THINKING:-disabled}"
}

endpoint_ready() {
  "${PYTHON_BIN}" - "${QWEN_CHAT_BASE_URL}" "${QWEN_CHAT_API_KEY}" <<'PY'
import sys
import urllib.request

base, key = sys.argv[1].rstrip("/"), sys.argv[2]
request = urllib.request.Request(
    f"{base}/models",
    headers={"Authorization": f"Bearer {key}"},
)
try:
    with urllib.request.urlopen(request, timeout=5) as response:
        raise SystemExit(0 if response.status == 200 else 1)
except Exception:
    raise SystemExit(1)
PY
}

cleanup_observed_runtime() {
  local pid
  for pid in "${JOB_PIDS[@]:-}"; do
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
    fi
  done
  if observed_truthy "${STOP_VLLM_ON_EXIT}" && [[ -n "${VLLM_PID}" ]] \
    && kill -0 "${VLLM_PID}" 2>/dev/null; then
    echo "[cleanup] stopping vLLM pid=${VLLM_PID}"
    kill "${VLLM_PID}" 2>/dev/null || true
  fi
}

start_vllm() {
  mkdir -p "${OUT_BASE}" "${LOG_BASE}"
  if observed_truthy "${DRY_RUN}"; then
    echo "[dry-run] skip vLLM startup"
    return
  fi
  if endpoint_ready; then
    echo "[vllm] reuse ready endpoint ${QWEN_CHAT_BASE_URL}"
    return
  fi
  observed_truthy "${START_VLLM}" || observed_fail "vLLM endpoint is not ready"
  [[ -d "${MODEL_PATH}" ]] || observed_fail "MODEL_PATH not found: ${MODEL_PATH}"
  case "${QWEN_CUDA_VISIBLE_DEVICES// /}" in
    *,*) observed_fail "Each launcher must receive exactly one H20 GPU" ;;
  esac

  local args=(
    serve "${MODEL_PATH}"
    --served-model-name "${SERVED_MODEL_NAME}"
    --host "${VLLM_HOST}"
    --port "${VLLM_PORT}"
    --trust-remote-code
    --dtype bfloat16
    --tensor-parallel-size 1
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
    --max-model-len "${MAX_MODEL_LEN}"
    --max-num-seqs "${VLLM_MAX_NUM_SEQS}"
    --max-num-batched-tokens "${VLLM_MAX_NUM_BATCHED_TOKENS}"
    --enable-prefix-caching
    --enable-auto-tool-choice
    --tool-call-parser "${VLLM_TOOL_CALL_PARSER}"
  )
  if observed_truthy "${VLLM_ENABLE_CHUNKED_PREFILL}"; then
    args+=(--enable-chunked-prefill)
  fi
  if [[ -n "${VLLM_REASONING_PARSER}" ]]; then
    args+=(--reasoning-parser "${VLLM_REASONING_PARSER}")
  fi
  if [[ -n "${VLLM_EXTRA_ARGS}" ]]; then
    # shellcheck disable=SC2206
    local extra_args=(${VLLM_EXTRA_ARGS})
    args+=("${extra_args[@]}")
  fi
  echo "[vllm] starting on CUDA_VISIBLE_DEVICES=${QWEN_CUDA_VISIBLE_DEVICES}"
  env CUDA_VISIBLE_DEVICES="${QWEN_CUDA_VISIBLE_DEVICES}" \
    nohup vllm "${args[@]}" > "${VLLM_LOG}" 2>&1 &
  VLLM_PID=$!
  echo "[vllm] pid=${VLLM_PID} log=${VLLM_LOG}"
  local second
  for second in $(seq 1 "${VLLM_WAIT_SECONDS:-900}"); do
    if endpoint_ready; then
      echo "[vllm] endpoint ready after ${second}s"
      return
    fi
    if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
      tail -n 100 "${VLLM_LOG}" || true
      observed_fail "vLLM exited before becoming ready"
    fi
    sleep 1
  done
  tail -n 100 "${VLLM_LOG}" || true
  observed_fail "timed out waiting for vLLM"
}

print_observed_header() {
  local job_name="$1"
  local datasets="$2"
  echo "============================================================"
  echo "  Observed taxonomy on fixed target trajectories"
  echo "============================================================"
  echo "  job:              ${job_name}"
  echo "  datasets:         ${datasets}"
  echo "  splits/repeats:   ${SPLITS} / ${REPEATS}"
  echo "  optimizer:        ${OPTIMIZER_SOURCE:-deepseek}/${OPTIMIZER_MODEL}"
  echo "  target:           ${TARGET_MODEL}"
  echo "  H20 device:       ${QWEN_CUDA_VISIBLE_DEVICES}"
  echo "  vLLM max seqs:    ${VLLM_MAX_NUM_SEQS}"
  echo "  vLLM log:         ${VLLM_LOG}"
  echo "  output:           ${OUT_BASE}"
  echo "  logs:             ${LOG_BASE}"
  echo "  dry_run:          ${DRY_RUN}"
  echo "============================================================"
}

run_observed_dataset() {
  local name="$1"
  local config="$2"
  local batch_size="$3"
  local target_workers="$4"
  local analyst_workers="$5"
  local target_max_tokens="$6"
  local shard_count="${7:-1}"
  local shard_index="${8:-0}"
  local out_dir="${OUT_BASE}/${name}"
  local log_file="${LOG_BASE}/${name}.log"
  local args=(
    "${PYTHON_BIN}" -u "${AUDIT_PY}"
    --config "${PROJECT_ROOT}/${config}"
    --output-dir "${out_dir}"
    --splits "${SPLITS}"
    --repeats "${REPEATS}"
    --batch-size "${batch_size}"
    --target-workers "${target_workers}"
    --analyst-workers "${analyst_workers}"
    --shard-count "${shard_count}"
    --shard-index "${shard_index}"
    --limit-per-split "${LIMIT_PER_SPLIT}"
    --seed "${SEED}"
    --optimizer-source "${OPTIMIZER_SOURCE:-deepseek}"
    --optimizer-model "${OPTIMIZER_MODEL}"
    --target-model "${TARGET_MODEL}"
    --target-base-url "${QWEN_CHAT_BASE_URL}"
    --target-api-key "${QWEN_CHAT_API_KEY}"
    --target-temperature "${TARGET_TEMPERATURE}"
    --target-timeout-seconds "${TARGET_TIMEOUT_SECONDS}"
    --target-max-tokens "${target_max_tokens}"
    --target-enable-thinking "${TARGET_ENABLE_THINKING}"
  )
  if observed_truthy "${DRY_RUN}"; then
    args+=(--dry-run)
  fi
  mkdir -p "${out_dir}" "${LOG_BASE}"
  echo "[launch] ${name} -> ${log_file}"
  "${args[@]}" 2>&1 | tee "${log_file}"
}

trap cleanup_observed_runtime EXIT INT TERM
configure_optimizer_source
command -v "${PYTHON_BIN}" >/dev/null 2>&1 || observed_fail "Python not found: ${PYTHON_BIN}"
[[ -f "${AUDIT_PY}" ]] || observed_fail "Audit program missing: ${AUDIT_PY}"

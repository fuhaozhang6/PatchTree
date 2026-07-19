#!/usr/bin/env bash
set -euo pipefail

# Isolated launcher for SearchQA prompt/concurrency probing.
# It starts a local Qwen vLLM endpoint when needed, then runs the Python probe.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_PATH="${MODEL_PATH:-/ai-app-vepfs/modelscope/models/Qwen/Qwen3.5-4B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen/Qwen3.5-4B}"
QWEN_CUDA_VISIBLE_DEVICES="${QWEN_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-0,1,2,3}}"
VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-4}"
VLLM_HOST="${VLLM_HOST:-0.0.0.0}"
VLLM_PORT="${VLLM_PORT:-8000}"
QWEN_CHAT_BASE_URL="${QWEN_CHAT_BASE_URL:-http://127.0.0.1:${VLLM_PORT}/v1}"
QWEN_CHAT_API_KEY="${QWEN_CHAT_API_KEY:-dummy}"
QWEN_CHAT_MODEL="${QWEN_CHAT_MODEL:-${SERVED_MODEL_NAME}}"
QWEN_CHAT_TIMEOUT_SECONDS="${QWEN_CHAT_TIMEOUT_SECONDS:-240}"
QWEN_SMOKE_TIMEOUT_SECONDS="${QWEN_SMOKE_TIMEOUT_SECONDS:-180}"
TARGET_QWEN_CHAT_TEMPERATURE="${TARGET_QWEN_CHAT_TEMPERATURE:-0.2}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
VLLM_DTYPE="${VLLM_DTYPE:-bfloat16}"
VLLM_ENABLE_AUTO_TOOL_CHOICE="${VLLM_ENABLE_AUTO_TOOL_CHOICE:-0}"
VLLM_TOOL_CALL_PARSER="${VLLM_TOOL_CALL_PARSER:-hermes}"
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"
START_VLLM="${START_VLLM:-1}"
STOP_VLLM_ON_EXIT="${STOP_VLLM_ON_EXIT:-1}"
VLLM_STARTUP_TIMEOUT_SECONDS="${VLLM_STARTUP_TIMEOUT_SECONDS:-900}"
VLLM_STARTUP_POLL_SECONDS="${VLLM_STARTUP_POLL_SECONDS:-2}"

SAMPLE_SIZE="${SAMPLE_SIZE:-256}"
WORKERS_LIST="${WORKERS_LIST:-48 96 128}"
MAX_TOKENS="${MAX_TOKENS:-16384}"
SPLIT_PATH="${SPLIT_PATH:-data/searchqa_split/val/items.json}"
SKILL_PATH="${SKILL_PATH:-skillopt/envs/searchqa/skills/initial.md}"
DATASET_LABEL="${DATASET_LABEL:-searchqa}"
START_ONLY="${START_ONLY:-0}"
PROBE_EXTRA_ARGS="${PROBE_EXTRA_ARGS:-}"
OUT_DIR="${OUT_DIR:-outputs/searchqa_prompt_latency_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-${OUT_DIR}/logs}"
VLLM_LOG="${VLLM_LOG:-${LOG_DIR}/vllm_qwen.log}"
VLLM_PID_FILE="${VLLM_PID_FILE:-${LOG_DIR}/vllm.pid}"

mkdir -p "${LOG_DIR}" "${OUT_DIR}"

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

endpoint_ready() {
  "${PYTHON_BIN}" - "${QWEN_CHAT_BASE_URL}" "${QWEN_CHAT_API_KEY}" <<'PY'
import sys
import urllib.request

base = sys.argv[1].rstrip("/")
api_key = sys.argv[2]
req = urllib.request.Request(base + "/models", headers={"Authorization": f"Bearer {api_key}"})
try:
    with urllib.request.urlopen(req, timeout=3) as resp:
        raise SystemExit(0 if resp.status < 500 else 1)
except Exception:
    raise SystemExit(1)
PY
}

qwen_smoke() {
  "${PYTHON_BIN}" - "${QWEN_CHAT_BASE_URL}" "${QWEN_CHAT_API_KEY}" "${QWEN_CHAT_MODEL}" "${QWEN_SMOKE_TIMEOUT_SECONDS}" <<'PY'
import json
import sys
import urllib.request

base, api_key, model, timeout_s = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
payload = {
    "model": model,
    "messages": [
        {"role": "system", "content": "You are a concise QA model."},
        {"role": "user", "content": "Answer in <answer> tags: What is 2+2?"},
    ],
    "max_tokens": 64,
    "temperature": 0.0,
    "chat_template_kwargs": {"enable_thinking": False},
}
req = urllib.request.Request(
    base.rstrip("/") + "/chat/completions",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
    method="POST",
)
with urllib.request.urlopen(req, timeout=timeout_s) as resp:
    data = json.loads(resp.read().decode("utf-8"))
choice = (data.get("choices") or [{}])[0]
message = choice.get("message") or {}
content = message.get("content") or ""
usage = data.get("usage") or {}
print(
    f"[smoke] finish_reason={choice.get('finish_reason')} "
    f"content_len={len(content)} usage={usage} preview={content[:120]!r}",
    flush=True,
)
PY
}

cleanup() {
  if truthy "${STOP_VLLM_ON_EXIT}" && [[ -f "${VLLM_PID_FILE}" ]]; then
    local pid
    pid="$(cat "${VLLM_PID_FILE}" 2>/dev/null || true)"
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      echo "[vllm] stopping pid=${pid}"
      kill "${pid}" || true
    fi
  fi
}
trap cleanup EXIT

echo "[config] project=${PROJECT_ROOT}"
echo "[config] model_path=${MODEL_PATH}"
echo "[config] served_model=${SERVED_MODEL_NAME}"
echo "[config] qwen_url=${QWEN_CHAT_BASE_URL}"
echo "[config] cuda_devices=${QWEN_CUDA_VISIBLE_DEVICES}"
echo "[config] tensor_parallel=${VLLM_TENSOR_PARALLEL_SIZE}"
echo "[config] sample_size=${SAMPLE_SIZE} workers=${WORKERS_LIST}"
echo "[config] out_dir=${OUT_DIR}"
echo "[config] startup_timeout=${VLLM_STARTUP_TIMEOUT_SECONDS}s smoke_timeout=${QWEN_SMOKE_TIMEOUT_SECONDS}s"

if endpoint_ready; then
  echo "[check] existing Qwen endpoint is ready: ${QWEN_CHAT_BASE_URL}"
else
  if ! truthy "${START_VLLM}"; then
    fail "Qwen endpoint is not ready and START_VLLM=${START_VLLM}"
  fi
  [[ -d "${MODEL_PATH}" ]] || fail "MODEL_PATH not found: ${MODEL_PATH}"
  command -v vllm >/dev/null 2>&1 || fail "vllm command not found"

  echo "[vllm] starting local Qwen service..."
  vllm_args=(
    serve "${MODEL_PATH}"
    --served-model-name "${SERVED_MODEL_NAME}"
    --host "${VLLM_HOST}"
    --port "${VLLM_PORT}"
    --trust-remote-code
    --dtype "${VLLM_DTYPE}"
    --tensor-parallel-size "${VLLM_TENSOR_PARALLEL_SIZE}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
    --max-model-len "${MAX_MODEL_LEN}"
    --enable-prefix-caching
  )
  if truthy "${VLLM_ENABLE_AUTO_TOOL_CHOICE}"; then
    vllm_args+=(--enable-auto-tool-choice --tool-call-parser "${VLLM_TOOL_CALL_PARSER}")
  fi
  if [[ -n "${VLLM_EXTRA_ARGS}" ]]; then
    # shellcheck disable=SC2206
    extra_vllm_args=(${VLLM_EXTRA_ARGS})
    vllm_args+=("${extra_vllm_args[@]}")
  fi

  env CUDA_VISIBLE_DEVICES="${QWEN_CUDA_VISIBLE_DEVICES}" \
    nohup vllm "${vllm_args[@]}" > "${VLLM_LOG}" 2>&1 &
  VLLM_PID=$!
  echo "${VLLM_PID}" > "${VLLM_PID_FILE}"
  echo "[vllm] pid=${VLLM_PID}"
  echo "[vllm] log=${VLLM_LOG}"

  echo "[vllm] waiting for endpoint..."
  deadline=$((SECONDS + VLLM_STARTUP_TIMEOUT_SECONDS))
  last_notice=0
  while [[ "${SECONDS}" -lt "${deadline}" ]]; do
    if endpoint_ready; then
      echo "[check] Qwen endpoint OK after ${SECONDS}s: ${QWEN_CHAT_BASE_URL}"
      break
    fi

    if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
      tail -120 "${VLLM_LOG}" || true
      fail "vLLM process exited before endpoint became ready"
    fi

    if (( SECONDS - last_notice >= 30 )); then
      last_notice="${SECONDS}"
      echo "[vllm] still waiting (${SECONDS}s elapsed). Recent log:"
      tail -20 "${VLLM_LOG}" || true
    fi
    sleep "${VLLM_STARTUP_POLL_SECONDS}"
  done

  if ! endpoint_ready; then
    tail -160 "${VLLM_LOG}" || true
    fail "Qwen endpoint did not become ready within ${VLLM_STARTUP_TIMEOUT_SECONDS}s"
  fi
fi

qwen_smoke

if truthy "${START_ONLY}"; then
  echo "[done] START_ONLY=1; vLLM is ready at ${QWEN_CHAT_BASE_URL}"
  exit 0
fi

# shellcheck disable=SC2206
probe_extra_args=(${PROBE_EXTRA_ARGS})

"${PYTHON_BIN}" experiments/searchqa_prompt_latency/run_searchqa_prompt_latency.py \
  --base-url "${QWEN_CHAT_BASE_URL}" \
  --api-key "${QWEN_CHAT_API_KEY}" \
  --model "${QWEN_CHAT_MODEL}" \
  --split "${SPLIT_PATH}" \
  --skill "${SKILL_PATH}" \
  --sample-size "${SAMPLE_SIZE}" \
  --workers ${WORKERS_LIST} \
  --max-tokens "${MAX_TOKENS}" \
  --temperature "${TARGET_QWEN_CHAT_TEMPERATURE}" \
  --timeout "${QWEN_CHAT_TIMEOUT_SECONDS}" \
  --dataset-label "${DATASET_LABEL}" \
  --out-dir "${OUT_DIR}" \
  "${probe_extra_args[@]}"

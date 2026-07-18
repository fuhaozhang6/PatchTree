#!/usr/bin/env bash
set -euo pipefail

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

command -v "${PYTHON_BIN}" >/dev/null 2>&1 || fail "Python interpreter not found: ${PYTHON_BIN}"

# DeepSeek optimizer through Ark OpenAI-compatible API.
export AZURE_OPENAI_ENDPOINT="${AZURE_OPENAI_ENDPOINT:-https://ark.cn-beijing.volces.com/api/v3}"
export AZURE_OPENAI_API_KEY="${AZURE_OPENAI_API_KEY:-${ARK_API_KEY:-}}"
export AZURE_OPENAI_AUTH_MODE="${AZURE_OPENAI_AUTH_MODE:-openai_compatible}"
export AZURE_OPENAI_API_VERSION="${AZURE_OPENAI_API_VERSION:-2024-12-01-preview}"

# Local Qwen target served by vLLM on two H20 GPUs by default.
MODEL_PATH="${MODEL_PATH:-/ai-app-vepfs/modelscope/models/Qwen/Qwen3.5-4B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen/Qwen3.5-4B}"
QWEN_CUDA_VISIBLE_DEVICES="${QWEN_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-0,1}}"
VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-2}"
VLLM_HOST="${VLLM_HOST:-0.0.0.0}"
VLLM_PORT="${VLLM_PORT:-8000}"
QWEN_CHAT_BASE_URL="${QWEN_CHAT_BASE_URL:-http://127.0.0.1:${VLLM_PORT}/v1}"
QWEN_CHAT_API_KEY="${QWEN_CHAT_API_KEY:-dummy}"
QWEN_CHAT_MODEL="${QWEN_CHAT_MODEL:-${SERVED_MODEL_NAME}}"
QWEN_CHAT_MAX_TOKENS="${QWEN_CHAT_MAX_TOKENS:-16384}"
QWEN_CHAT_TIMEOUT_SECONDS="${QWEN_CHAT_TIMEOUT_SECONDS:-120}"
TARGET_QWEN_CHAT_TEMPERATURE="${TARGET_QWEN_CHAT_TEMPERATURE:-0.2}"
TARGET_QWEN_CHAT_ENABLE_THINKING="${TARGET_QWEN_CHAT_ENABLE_THINKING:-false}"
REQUIRE_QWEN_CONTENT="${REQUIRE_QWEN_CONTENT:-0}"
START_VLLM="${START_VLLM:-1}"
STOP_VLLM_ON_EXIT="${STOP_VLLM_ON_EXIT:-1}"
DRY_RUN="${DRY_RUN:-0}"

export QWEN_CHAT_BASE_URL QWEN_CHAT_API_KEY QWEN_CHAT_MODEL QWEN_CHAT_MAX_TOKENS
export TARGET_QWEN_CHAT_BASE_URL="${TARGET_QWEN_CHAT_BASE_URL:-${QWEN_CHAT_BASE_URL}}"
export TARGET_QWEN_CHAT_API_KEY="${TARGET_QWEN_CHAT_API_KEY:-${QWEN_CHAT_API_KEY}}"
export TARGET_QWEN_CHAT_TEMPERATURE
export TARGET_QWEN_CHAT_TIMEOUT_SECONDS="${TARGET_QWEN_CHAT_TIMEOUT_SECONDS:-${QWEN_CHAT_TIMEOUT_SECONDS}}"
export TARGET_QWEN_CHAT_MAX_TOKENS="${TARGET_QWEN_CHAT_MAX_TOKENS:-${QWEN_CHAT_MAX_TOKENS}}"
export TARGET_QWEN_CHAT_ENABLE_THINKING

export OPTIMIZER_BACKEND="${OPTIMIZER_BACKEND:-openai_chat}"
export TARGET_BACKEND="${TARGET_BACKEND:-qwen_chat}"
export OPTIMIZER_MODEL="${OPTIMIZER_MODEL:-deepseek-v4-pro-260425}"
export TARGET_MODEL="${TARGET_MODEL:-${SERVED_MODEL_NAME}}"
export TARGET_MAX_COMPLETION_TOKENS="${TARGET_MAX_COMPLETION_TOKENS:-${QWEN_CHAT_MAX_TOKENS}}"

export RUN_NAME="${RUN_NAME:-DeepSeek + local Qwen full V3}"
export RUN_SLUG="${RUN_SLUG:-skillopt_alfworld_v3_deepseek_local_qwen_full}"
export TS="${TS:-alfworld_v3_deepseek_qwen_full_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs/${RUN_SLUG}_${TS}}"
mkdir -p "${LOG_DIR}"
VLLM_LOG="${VLLM_LOG:-${LOG_DIR}/vllm_qwen.log}"
VLLM_PID_FILE="${VLLM_PID_FILE:-${LOG_DIR}/vllm.pid}"

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
    headers={
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    },
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
    warning = (
        "[smoke/qwen] Empty message.content from Qwen target. SkillOpt qwen_chat "
        "uses message.content as the rollout answer; check chat template or "
        "enable_thinking if training outputs are empty."
    )
    print(warning, file=sys.stderr)
    if require_content:
        raise SystemExit(warning)
PY
}

echo "============================================================"
echo "  ALFWorld V3: DeepSeek optimizer + local Qwen target"
echo "============================================================"
echo "  project:        ${PROJECT_ROOT}"
echo "  optimizer:      ${OPTIMIZER_MODEL}"
echo "  target:         ${TARGET_MODEL}"
echo "  qwen_url:       ${QWEN_CHAT_BASE_URL}"
echo "  model_path:     ${MODEL_PATH}"
echo "  cuda_devices:   ${QWEN_CUDA_VISIBLE_DEVICES}"
echo "  tensor_parallel:${VLLM_TENSOR_PARALLEL_SIZE}"
echo "  dry_run:        ${DRY_RUN}"
echo "  log_dir:        ${LOG_DIR}"
echo "============================================================"

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
    env CUDA_VISIBLE_DEVICES="${QWEN_CUDA_VISIBLE_DEVICES}" \
      nohup vllm serve "${MODEL_PATH}" \
        --served-model-name "${SERVED_MODEL_NAME}" \
        --host "${VLLM_HOST}" \
        --port "${VLLM_PORT}" \
        --trust-remote-code \
        --dtype "${VLLM_DTYPE:-bfloat16}" \
        --tensor-parallel-size "${VLLM_TENSOR_PARALLEL_SIZE}" \
        --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.90}" \
        --max-model-len "${MAX_MODEL_LEN:-32768}" \
        --enable-prefix-caching \
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
fi

bash "${PROJECT_ROOT}/scripts/runs/alfworld/run_alfworld_v3_seed_api_full.sh" "$@"

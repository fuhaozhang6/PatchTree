#!/usr/bin/env bash
set -euo pipefail

# Formal ALFWorld V3 training launcher:
#   optimizer / teacher = DeepSeek official API, deepseek-v4-pro by default
#   target / student    = local Qwen3.5-4B served by four single-GPU vLLM endpoints
#   load balancing      = local round-robin OpenAI-compatible proxy
#   data                = existing project data/alfworld and data/alfworld_path_split
#
# Typical usage on the 4xA100 node:
#   cd /ai-car-vepfs1/ai_car/zhangfuhao/data/01/SkillOpt-Tree
#   export DEEPSEEK_API_KEY='...'
#   bash scripts/runs/alfworld/run_alfworld_qwen35_4b_4xa100_dsv4pro_train.sh
#
# Quick smoke:
#   LIMIT=8 SEL_ENV_NUM=8 TEST_ENV_NUM=0 EVAL_TEST=false NUM_EPOCHS=1 \
#     bash scripts/runs/alfworld/run_alfworld_qwen35_4b_4xa100_dsv4pro_train.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTHON_BIN="${PYTHON_BIN:-python}"
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

endpoint_ready() {
  local base_url="$1"
  "${PYTHON_BIN}" - "${base_url}" "${QWEN_CHAT_API_KEY}" <<'PY'
import sys
import urllib.request

base = sys.argv[1].rstrip("/")
api_key = sys.argv[2]
headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
try:
    with urllib.request.urlopen(urllib.request.Request(base + "/models", headers=headers), timeout=3) as resp:
        raise SystemExit(0 if resp.status < 500 else 1)
except Exception:
    raise SystemExit(1)
PY
}

wait_for_endpoint() {
  local name="$1"
  local base_url="$2"
  local pid="$3"
  local log_file="$4"
  local deadline=$((SECONDS + VLLM_STARTUP_TIMEOUT_SECONDS))
  local last_notice=0

  while [[ "${SECONDS}" -lt "${deadline}" ]]; do
    if endpoint_ready "${base_url}"; then
      echo "[check] ${name} ready: ${base_url}"
      return 0
    fi
    if [[ -n "${pid}" ]] && ! kill -0 "${pid}" 2>/dev/null; then
      tail -120 "${log_file}" || true
      fail "${name} exited before becoming ready"
    fi
    if (( SECONDS - last_notice >= 30 )); then
      last_notice="${SECONDS}"
      echo "[wait] ${name} not ready yet (${SECONDS}s elapsed). Recent log:"
      tail -20 "${log_file}" || true
    fi
    sleep "${VLLM_STARTUP_POLL_SECONDS}"
  done

  tail -160 "${log_file}" || true
  fail "${name} did not become ready within ${VLLM_STARTUP_TIMEOUT_SECONDS}s"
}

cleanup() {
  if ! truthy "${STOP_VLLM_ON_EXIT}"; then
    return 0
  fi
  if [[ -f "${MULTI_VLLM_DIR}/pids.txt" ]]; then
    tac "${MULTI_VLLM_DIR}/pids.txt" 2>/dev/null | while read -r pid; do
      if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
        echo "[cleanup] stopping pid=${pid}"
        kill "${pid}" || true
      fi
    done
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
    "messages": [
        {
            "role": "user",
            "content": "This is an ALFWorld target smoke test. Reply with one short action: look",
        }
    ],
    "max_tokens": 64,
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
with urllib.request.urlopen(req, timeout=180) as resp:
    data = json.loads(resp.read().decode("utf-8"))
choice = (data.get("choices") or [{}])[0]
message = choice.get("message") or {}
content = message.get("content") or ""
reasoning = message.get("reasoning_content") or ""
print(f"[smoke/qwen] finish_reason={choice.get('finish_reason')} content_len={len(content)} reasoning_len={len(reasoning)}")
print(f"[smoke/qwen] content_preview={content[:120]!r}")
if require_content and not str(content).strip():
    raise SystemExit("[smoke/qwen] empty message.content from Qwen target")
PY
}

command -v "${PYTHON_BIN}" >/dev/null 2>&1 || fail "Python interpreter not found: ${PYTHON_BIN}"

# DeepSeek official optimizer. This matches the tested practice in Opt/Ds-test.
export DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY:-${DS_API_KEY:-${DEEPSEEK_OFFICIAL_API_KEY:-}}}"
export AZURE_OPENAI_ENDPOINT="${AZURE_OPENAI_ENDPOINT:-https://api.deepseek.com}"
export AZURE_OPENAI_API_KEY="${AZURE_OPENAI_API_KEY:-${DEEPSEEK_API_KEY}}"
export AZURE_OPENAI_AUTH_MODE="${AZURE_OPENAI_AUTH_MODE:-openai_compatible}"
export AZURE_OPENAI_API_VERSION="${AZURE_OPENAI_API_VERSION:-openai-compat}"
export OPTIMIZER_AZURE_OPENAI_ENDPOINT="${OPTIMIZER_AZURE_OPENAI_ENDPOINT:-${AZURE_OPENAI_ENDPOINT}}"
export OPTIMIZER_AZURE_OPENAI_API_KEY="${OPTIMIZER_AZURE_OPENAI_API_KEY:-${AZURE_OPENAI_API_KEY}}"
export OPTIMIZER_AZURE_OPENAI_AUTH_MODE="${OPTIMIZER_AZURE_OPENAI_AUTH_MODE:-${AZURE_OPENAI_AUTH_MODE}}"
export OPTIMIZER_AZURE_OPENAI_API_VERSION="${OPTIMIZER_AZURE_OPENAI_API_VERSION:-${AZURE_OPENAI_API_VERSION}}"
export OPTIMIZER_BACKEND="${OPTIMIZER_BACKEND:-openai_chat}"
export OPTIMIZER_MODEL="${OPTIMIZER_MODEL:-${DEEPSEEK_MODEL:-deepseek-v4-pro}}"
export OPTIMIZER_SMOKE_REASONING_EFFORT="${OPTIMIZER_SMOKE_REASONING_EFFORT:-high}"
export DEEPSEEK_THINKING="${DEEPSEEK_THINKING:-enabled}"
export REASONING_EFFORT="${REASONING_EFFORT:-high}"
export REWRITE_REASONING_EFFORT="${REWRITE_REASONING_EFFORT:-medium}"

# Local Qwen target served by independent single-GPU vLLM endpoints.
export MODEL_PATH="${MODEL_PATH:-/ai-car-vepfs1/ai_car/share/model/Qwen/Qwen3.5-4B}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen/Qwen3.5-4B}"
export TARGET_MODEL="${TARGET_MODEL:-${SERVED_MODEL_NAME}}"
export TARGET_BACKEND="${TARGET_BACKEND:-qwen_chat}"
export QWEN_CHAT_MODEL="${QWEN_CHAT_MODEL:-${SERVED_MODEL_NAME}}"
export QWEN_CUDA_VISIBLE_DEVICES="${QWEN_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-0,1,2,3}}"
export VLLM_BASE_PORT="${VLLM_BASE_PORT:-49317}"
export VLLM_PROXY_PORT="${VLLM_PROXY_PORT:-49417}"
export QWEN_CHAT_BASE_URL="${QWEN_CHAT_BASE_URL:-http://127.0.0.1:${VLLM_PROXY_PORT}/v1}"
export QWEN_CHAT_API_KEY="${QWEN_CHAT_API_KEY:-dummy}"
export QWEN_CHAT_TIMEOUT_SECONDS="${QWEN_CHAT_TIMEOUT_SECONDS:-300}"
export QWEN_CHAT_MAX_TOKENS="${QWEN_CHAT_MAX_TOKENS:-2048}"
export QWEN_CHAT_ROLLOUT_RETRIES="${QWEN_CHAT_ROLLOUT_RETRIES:-2}"
export TARGET_QWEN_CHAT_BASE_URL="${TARGET_QWEN_CHAT_BASE_URL:-${QWEN_CHAT_BASE_URL}}"
export TARGET_QWEN_CHAT_API_KEY="${TARGET_QWEN_CHAT_API_KEY:-${QWEN_CHAT_API_KEY}}"
export TARGET_QWEN_CHAT_TEMPERATURE="${TARGET_QWEN_CHAT_TEMPERATURE:-0.2}"
export TARGET_QWEN_CHAT_TIMEOUT_SECONDS="${TARGET_QWEN_CHAT_TIMEOUT_SECONDS:-${QWEN_CHAT_TIMEOUT_SECONDS}}"
export TARGET_QWEN_CHAT_MAX_TOKENS="${TARGET_QWEN_CHAT_MAX_TOKENS:-${QWEN_CHAT_MAX_TOKENS}}"
export TARGET_QWEN_CHAT_ENABLE_THINKING="${TARGET_QWEN_CHAT_ENABLE_THINKING:-false}"

# vLLM defaults inherited from the validated speed probe, with longer context
# for formal training. Override MAX_MODEL_LEN=8192 if startup/memory is tight.
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.88}"
export VLLM_DTYPE="${VLLM_DTYPE:-bfloat16}"
export VLLM_STARTUP_TIMEOUT_SECONDS="${VLLM_STARTUP_TIMEOUT_SECONDS:-1200}"
export VLLM_STARTUP_POLL_SECONDS="${VLLM_STARTUP_POLL_SECONDS:-2}"
export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-128}"
export VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-32768}"
export VLLM_ENABLE_PREFIX_CACHING="${VLLM_ENABLE_PREFIX_CACHING:-0}"
export VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"
export START_VLLM="${START_VLLM:-1}"
export STOP_VLLM_ON_EXIT="${STOP_VLLM_ON_EXIT:-1}"
export REQUIRE_QWEN_CONTENT="${REQUIRE_QWEN_CONTENT:-1}"

# Formal ALFWorld full-run defaults.
export ALFWORLD_DATA="${ALFWORLD_DATA:-${PROJECT_ROOT}/data/alfworld}"
export ALFWORLD_SPLIT_DIR="${ALFWORLD_SPLIT_DIR:-${PROJECT_ROOT}/data/alfworld_path_split}"
export NUM_EPOCHS="${NUM_EPOCHS:-4}"
export BATCH_SIZE="${BATCH_SIZE:-8}"
export LIMIT="${LIMIT:-0}"
export SEL_ENV_NUM="${SEL_ENV_NUM:-0}"
export TEST_ENV_NUM="${TEST_ENV_NUM:-0}"
export EVAL_TEST="${EVAL_TEST:-true}"
export MAX_STEPS="${MAX_STEPS:-50}"
export TARGET_MAX_COMPLETION_TOKENS="${TARGET_MAX_COMPLETION_TOKENS:-2048}"
export API_MAX_CONCURRENCY="${API_MAX_CONCURRENCY:-192}"
export WORKERS="${WORKERS:-192}"
export MAX_API_WORKERS="${MAX_API_WORKERS:-192}"
export ANALYST_WORKERS="${ANALYST_WORKERS:-16}"
export DRY_RUN="${DRY_RUN:-0}"
export TYPE_GUIDED_TREE_DEPTH="${TYPE_GUIDED_TREE_DEPTH:-3}"
export TYPE_GUIDED_LEAF_FALLBACK="${TYPE_GUIDED_LEAF_FALLBACK:-true}"
export TYPE_GUIDED_PATCH_RECORD_WORKERS="${TYPE_GUIDED_PATCH_RECORD_WORKERS:-${ANALYST_WORKERS}}"
export WAIT_FOR_JOB="${WAIT_FOR_JOB:-1}"

export TS="${TS:-alfworld_qwen35_4b_4xa100_dsv4pro_train_$(date +%Y%m%d_%H%M%S)}"
export RUN_NAME="${RUN_NAME:-ALFWorld Qwen3.5-4B 4xA100 + DeepSeek V4 Pro}"
export RUN_SLUG="${RUN_SLUG:-skillopt_alfworld_qwen35_4b_4xa100_dsv4pro_train}"
export LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs/${RUN_SLUG}_${TS}}"
export OUT_ROOT="${OUT_ROOT:-${PROJECT_ROOT}/outputs/${RUN_SLUG}_${TS}}"
export MULTI_VLLM_DIR="${MULTI_VLLM_DIR:-${OUT_ROOT}/multi_vllm}"
export MULTI_VLLM_LOG_DIR="${MULTI_VLLM_LOG_DIR:-${MULTI_VLLM_DIR}/logs}"

mkdir -p "${MULTI_VLLM_LOG_DIR}" "${OUT_ROOT}" "${LOG_DIR}"

IFS=',' read -r -a GPU_IDS <<< "${QWEN_CUDA_VISIBLE_DEVICES// /}"
GPU_COUNT="${#GPU_IDS[@]}"
[[ "${GPU_COUNT}" -ge 1 ]] || fail "No GPU ids found in QWEN_CUDA_VISIBLE_DEVICES=${QWEN_CUDA_VISIBLE_DEVICES}"

echo "============================================================"
echo "  SkillOpt ALFWorld Formal Train: 4xA100 Qwen + DeepSeek"
echo "============================================================"
echo "  project_root:      ${PROJECT_ROOT}"
echo "  optimizer:         ${OPTIMIZER_MODEL}"
echo "  target:            ${TARGET_MODEL}"
echo "  model_path:        ${MODEL_PATH}"
echo "  gpu_ids:           ${QWEN_CUDA_VISIBLE_DEVICES}"
echo "  endpoint_ports:    ${VLLM_BASE_PORT}..$((VLLM_BASE_PORT + GPU_COUNT - 1))"
echo "  proxy_url:         ${QWEN_CHAT_BASE_URL}"
echo "  epochs:            ${NUM_EPOCHS}"
echo "  workers:           ${WORKERS}"
echo "  max_api_workers:   ${MAX_API_WORKERS}"
echo "  analyst_workers:   ${ANALYST_WORKERS}"
echo "  max_steps:         ${MAX_STEPS}"
echo "  target_max_tokens: ${TARGET_MAX_COMPLETION_TOKENS}"
echo "  max_model_len:     ${MAX_MODEL_LEN}"
echo "  split_dir:         ${ALFWORLD_SPLIT_DIR}"
echo "  out_root:          ${OUT_ROOT}"
echo "  log_dir:           ${LOG_DIR}"
echo "============================================================"

if ! truthy "${DRY_RUN}"; then
  [[ -n "${AZURE_OPENAI_API_KEY}" ]] || fail "DEEPSEEK_API_KEY is required for DeepSeek official optimizer."
  [[ -d "${MODEL_PATH}" ]] || fail "MODEL_PATH not found: ${MODEL_PATH}"
  [[ -d "${ALFWORLD_DATA}/json_2.1.1" ]] || fail "ALFWorld data not found under ${ALFWORLD_DATA}/json_2.1.1"
  [[ -f "${ALFWORLD_SPLIT_DIR}/train/items.json" ]] || fail "ALFWorld train split not found: ${ALFWORLD_SPLIT_DIR}/train/items.json"
fi

if truthy "${DRY_RUN}"; then
  echo "[dry-run] Skip vLLM startup and Qwen smoke test."
elif truthy "${START_VLLM}"; then
  command -v vllm >/dev/null 2>&1 || fail "vllm command not found"

  ports=("${VLLM_PROXY_PORT}")
  for idx in "${!GPU_IDS[@]}"; do
    ports+=("$((VLLM_BASE_PORT + idx))")
  done
  "${PYTHON_BIN}" - "${ports[@]}" <<'PY'
import socket
import sys

busy = []
for raw in sys.argv[1:]:
    port = int(raw)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind(("0.0.0.0", port))
    except OSError:
        busy.append(port)
    finally:
        sock.close()
if busy:
    raise SystemExit("ERROR: port(s) already in use: " + ", ".join(map(str, busy)))
PY

  : > "${MULTI_VLLM_DIR}/pids.txt"
  upstreams=()
  for idx in "${!GPU_IDS[@]}"; do
    gpu="${GPU_IDS[$idx]}"
    port="$((VLLM_BASE_PORT + idx))"
    base_url="http://127.0.0.1:${port}/v1"
    log_file="${MULTI_VLLM_LOG_DIR}/vllm_gpu${gpu}_port${port}.log"
    upstreams+=("${base_url}")

    vllm_args=(
      serve "${MODEL_PATH}"
      --served-model-name "${SERVED_MODEL_NAME}"
      --host "127.0.0.1"
      --port "${port}"
      --trust-remote-code
      --dtype "${VLLM_DTYPE}"
      --tensor-parallel-size 1
      --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
      --max-model-len "${MAX_MODEL_LEN}"
      --max-num-seqs "${VLLM_MAX_NUM_SEQS}"
      --max-num-batched-tokens "${VLLM_MAX_NUM_BATCHED_TOKENS}"
    )
    if truthy "${VLLM_ENABLE_PREFIX_CACHING}"; then
      vllm_args+=(--enable-prefix-caching)
    fi
    if [[ -n "${VLLM_EXTRA_ARGS}" ]]; then
      # shellcheck disable=SC2206
      extra_args=(${VLLM_EXTRA_ARGS})
      vllm_args+=("${extra_args[@]}")
    fi

    echo "[vllm] starting gpu=${gpu} port=${port} log=${log_file}"
    env CUDA_VISIBLE_DEVICES="${gpu}" \
      nohup vllm "${vllm_args[@]}" > "${log_file}" 2>&1 &
    pid=$!
    echo "${pid}" >> "${MULTI_VLLM_DIR}/pids.txt"
    wait_for_endpoint "vllm_gpu${gpu}" "${base_url}" "${pid}" "${log_file}"
  done

  proxy_script="${MULTI_VLLM_DIR}/round_robin_proxy.py"
  cat > "${proxy_script}" <<'PY'
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import sys
import threading
import urllib.error
import urllib.request


PORT = int(sys.argv[1])
UPSTREAMS = [u.rstrip("/") for u in sys.argv[2].split(",") if u.strip()]
API_KEY = sys.argv[3]
lock = threading.Lock()
counter = 0


def pick_upstream() -> str:
    global counter
    with lock:
        upstream = UPSTREAMS[counter % len(UPSTREAMS)]
        counter += 1
        return upstream


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("[proxy] " + fmt % args + "\n")

    def _target_url(self, upstream: str) -> str:
        path = self.path
        if path.startswith("/v1/"):
            path = path[3:]
        return upstream + path

    def _send(self, status: int, body: bytes, content_type: str = "application/json") -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path.rstrip("/") in {"/models", "/v1/models"}:
            upstream = UPSTREAMS[0]
            req = urllib.request.Request(self._target_url(upstream), headers=self._headers())
            try:
                with urllib.request.urlopen(req, timeout=10) as resp:
                    self._send(resp.status, resp.read(), resp.headers.get("Content-Type", "application/json"))
            except Exception as exc:
                self._send(502, json.dumps({"error": str(exc)}).encode())
            return
        self._send(404, b'{"error":"not found"}')

    def do_POST(self) -> None:
        if not self.path.rstrip("/").endswith("/chat/completions"):
            self._send(404, b'{"error":"not found"}')
            return
        body = self.rfile.read(int(self.headers.get("Content-Length", "0") or 0))
        upstream = pick_upstream()
        req = urllib.request.Request(
            self._target_url(upstream),
            data=body,
            headers=self._headers(content_type=self.headers.get("Content-Type", "application/json")),
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=3600) as resp:
                self._send(resp.status, resp.read(), resp.headers.get("Content-Type", "application/json"))
        except urllib.error.HTTPError as exc:
            self._send(exc.code, exc.read(), exc.headers.get("Content-Type", "application/json"))
        except Exception as exc:
            self._send(502, json.dumps({"error": str(exc), "upstream": upstream}).encode())

    def _headers(self, content_type: str | None = None) -> dict[str, str]:
        headers: dict[str, str] = {}
        if content_type:
            headers["Content-Type"] = content_type
        auth = self.headers.get("Authorization")
        if auth:
            headers["Authorization"] = auth
        elif API_KEY:
            headers["Authorization"] = f"Bearer {API_KEY}"
        return headers


server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
print(f"[proxy] listening on http://127.0.0.1:{PORT}/v1 -> {UPSTREAMS}", flush=True)
server.serve_forever()
PY

  old_ifs="${IFS}"
  IFS=','
  upstreams_csv="${upstreams[*]}"
  IFS="${old_ifs}"
  proxy_log="${MULTI_VLLM_LOG_DIR}/round_robin_proxy.log"
  echo "[proxy] starting port=${VLLM_PROXY_PORT} upstreams=${upstreams_csv}"
  nohup "${PYTHON_BIN}" "${proxy_script}" "${VLLM_PROXY_PORT}" "${upstreams_csv}" "${QWEN_CHAT_API_KEY}" > "${proxy_log}" 2>&1 &
  proxy_pid=$!
  echo "${proxy_pid}" >> "${MULTI_VLLM_DIR}/pids.txt"
  wait_for_endpoint "round_robin_proxy" "${QWEN_CHAT_BASE_URL}" "${proxy_pid}" "${proxy_log}"
fi

if ! truthy "${DRY_RUN}"; then
  qwen_smoke
fi

bash "${PROJECT_ROOT}/scripts/runs/alfworld/run_alfworld_v3_seed_api_full.sh" "$@"

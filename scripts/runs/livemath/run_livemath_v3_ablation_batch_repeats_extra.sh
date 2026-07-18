#!/usr/bin/env bash
set -euo pipefail

# LiveMath V3 ablation runner for the second two-point pass.
# It keeps the Qwen+DeepSeek/V3 defaults from the 2x128 launcher and changes
# only the two intended variables:
#   1) batch=8, repeats=3
#   2) batch=16, repeats=2
#
# The two ablations run concurrently. The first one starts or reuses vLLM; the
# second waits for the endpoint and then reuses the same service.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_ROOT}"

BASE_LAUNCHER="${SCRIPT_DIR}/run_v3_deepseek_local_qwen_2x128_four_datasets.sh"
[[ -f "${BASE_LAUNCHER}" ]] || {
  echo "[error] Missing base launcher: ${BASE_LAUNCHER}" >&2
  exit 1
}

RUN_GROUP="${RUN_GROUP:-livemath_v3_ablation_extra_$(date +%Y%m%d_%H%M%S)}"
WORKERS_FOR_ABLATION="${WORKERS_FOR_ABLATION:-128}"
API_MAX_CONCURRENCY_FOR_ABLATION="${API_MAX_CONCURRENCY_FOR_ABLATION:-${WORKERS_FOR_ABLATION}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
QWEN_CHAT_BASE_URL="${QWEN_CHAT_BASE_URL:-http://127.0.0.1:${VLLM_PORT:-8000}/v1}"
VLLM_WAIT_SECONDS="${VLLM_WAIT_SECONDS:-900}"

export DATASETS="livemath"
export MAX_PARALLEL="${MAX_PARALLEL:-1}"
export NUM_EPOCHS="${NUM_EPOCHS:-2}"
export WORKERS="${WORKERS:-${WORKERS_FOR_ABLATION}}"
export LIVEMATH_WORKERS="${LIVEMATH_WORKERS:-${WORKERS_FOR_ABLATION}}"
export API_MAX_CONCURRENCY="${API_MAX_CONCURRENCY:-${API_MAX_CONCURRENCY_FOR_ABLATION}}"
export WAIT_FOR_JOBS="${WAIT_FOR_JOBS:-1}"

export TYPE_GUIDED_CLUSTERING="${TYPE_GUIDED_CLUSTERING:-true}"
export TYPE_GUIDED_TREE_DEPTH="${TYPE_GUIDED_TREE_DEPTH:-3}"

export TARGET_MAX_COMPLETION_TOKENS="${TARGET_MAX_COMPLETION_TOKENS:-16384}"
export LIVEMATH_TARGET_MAX_COMPLETION_TOKENS="${LIVEMATH_TARGET_MAX_COMPLETION_TOKENS:-16384}"
export TARGET_QWEN_CHAT_ENABLE_THINKING="${TARGET_QWEN_CHAT_ENABLE_THINKING:-false}"
export QWEN_CHAT_ROLLOUT_RETRIES="${QWEN_CHAT_ROLLOUT_RETRIES:-1}"
export TARGET_QWEN_CHAT_ROLLOUT_RETRIES="${TARGET_QWEN_CHAT_ROLLOUT_RETRIES:-${QWEN_CHAT_ROLLOUT_RETRIES}}"

run_one() {
  local label="$1"
  local batch_size="$2"
  local repeats="$3"
  local start_vllm="$4"
  local stop_vllm="$5"

  echo "============================================================"
  echo "  LiveMath V3 Ablation: ${label}"
  echo "============================================================"
  echo "  group:       ${RUN_GROUP}"
  echo "  epoch:       ${NUM_EPOCHS}"
  echo "  batch:       ${batch_size}"
  echo "  repeats:     ${repeats}"
  echo "  workers:     ${LIVEMATH_WORKERS}"
  echo "  concurrency: ${API_MAX_CONCURRENCY}"
  echo "  start_vllm:  ${start_vllm}"
  echo "  stop_vllm:   ${stop_vllm}"
  echo "============================================================"

  TS="${RUN_GROUP}_${label}" \
  LIVEMATH_BATCH_SIZE="${batch_size}" \
  TYPE_GUIDED_ROLLOUT_REPEATS="${repeats}" \
  START_VLLM="${start_vllm}" \
  STOP_VLLM_ON_EXIT="${stop_vllm}" \
    bash "${BASE_LAUNCHER}"
}

truthy() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

endpoint_ready() {
  "${PYTHON_BIN}" - "${QWEN_CHAT_BASE_URL}" <<'PY' >/dev/null 2>&1
import json
import sys
import urllib.request

base = sys.argv[1].rstrip("/")
try:
    with urllib.request.urlopen(f"{base}/models", timeout=5) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    if data.get("data"):
        raise SystemExit(0)
except Exception:
    raise SystemExit(1)
raise SystemExit(1)
PY
}

wait_for_endpoint_or_first_exit() {
  local first_pid="$1"
  local i
  if truthy "${DRY_RUN:-0}"; then
    return 0
  fi
  for i in $(seq 1 "${VLLM_WAIT_SECONDS}"); do
    if endpoint_ready; then
      echo "[check] Qwen endpoint ready for second ablation after ${i}s: ${QWEN_CHAT_BASE_URL}"
      return 0
    fi
    if ! kill -0 "${first_pid}" 2>/dev/null; then
      wait "${first_pid}" || true
      echo "[error] First ablation launcher exited before Qwen endpoint became ready." >&2
      return 1
    fi
    sleep 1
  done
  echo "[error] Timed out waiting for Qwen endpoint: ${QWEN_CHAT_BASE_URL}" >&2
  return 1
}

stop_vllm_from_first_run() {
  local pid_file="${PROJECT_ROOT}/logs/skillopt_v3_deepseek_local_qwen_parallel_${RUN_GROUP}_e1_b8_r3/vllm.pid"
  if ! truthy "${STOP_VLLM_AFTER_LAST:-1}"; then
    return 0
  fi
  if [[ -f "${pid_file}" ]]; then
    local pid
    pid="$(cat "${pid_file}")"
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      echo "[vllm] stopping shared vLLM pid=${pid}"
      kill "${pid}" 2>/dev/null || true
    fi
  else
    echo "[vllm] no first-run vllm.pid found; assuming endpoint was external or already stopped"
  fi
}

first_pid=""
second_pid=""
cleanup_launchers() {
  local pid
  for pid in "${first_pid}" "${second_pid}"; do
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
    fi
  done
}
trap cleanup_launchers INT TERM

run_one "e1_b8_r3" 8 3 "${START_VLLM_FIRST:-1}" 0 &
first_pid=$!

wait_for_endpoint_or_first_exit "${first_pid}"

run_one "e1_b16_r2" 16 2 0 0 &
second_pid=$!

status_first=0
status_second=0
wait "${first_pid}" || status_first=$?
wait "${second_pid}" || status_second=$?

stop_vllm_from_first_run

if [[ "${status_first}" -ne 0 || "${status_second}" -ne 0 ]]; then
  echo "[error] ablation failures: e1_b8_r3=${status_first}, e1_b16_r2=${status_second}" >&2
  exit 1
fi

echo "============================================================"
echo "  LiveMath V3 extra ablation group finished: ${RUN_GROUP}"
echo "============================================================"

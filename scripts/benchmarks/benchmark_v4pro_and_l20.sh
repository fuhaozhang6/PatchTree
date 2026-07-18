#!/usr/bin/env bash
set -euo pipefail

# Sweep concurrency for:
#   1. DeepSeek V4 Pro on Volcano Ark
#   2. DeepSeek V4 Pro on the official DeepSeek API
#   3. Qwen3.5-4B served by local vLLM on one L20
#
# The API keys are read from environment variables and are never put on the
# Python command line or written to result files.
#
# Quick start:
#   export ARK_API_KEY='...'
#   export DEEPSEEK_API_KEY='...'
#   bash scripts/benchmarks/benchmark_v4pro_and_l20.sh
#
# Test one target only:
#   TARGETS=ark bash scripts/benchmarks/benchmark_v4pro_and_l20.sh
#   TARGETS=deepseek bash scripts/benchmarks/benchmark_v4pro_and_l20.sh
#   TARGETS=local bash scripts/benchmarks/benchmark_v4pro_and_l20.sh
#
# Cheap smoke test before a full sweep:
#   CONCURRENCY_LEVELS='1 2' MIN_REQUESTS=2 MAX_TOKEN_OPTIONS='64 128' \
#     bash scripts/benchmarks/benchmark_v4pro_and_l20.sh
#
# Reuse an already-running vLLM endpoint:
#   TARGETS=local START_VLLM=0 LOCAL_BASE_URL=http://127.0.0.1:8000/v1 \
#     bash scripts/benchmarks/benchmark_v4pro_and_l20.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
BENCHMARK_PY="${SCRIPT_DIR}/benchmark_openai_compatible.py"
TARGETS="${TARGETS:-ark deepseek local}"
DRY_RUN="${DRY_RUN:-0}"

# Common workload. The prompt uses mostly unique tokens so prefix caching does
# not make the local result artificially optimistic.
PROMPT_TOKEN_OPTIONS="${PROMPT_TOKEN_OPTIONS:-512 1024 2048 4096}"
MAX_TOKEN_OPTIONS="${MAX_TOKEN_OPTIONS:-256 512 1024 2048}"
MIN_REQUESTS="${MIN_REQUESTS:-16}"
REQUEST_MULTIPLIER="${REQUEST_MULTIPLIER:-2}"
WARMUP_REQUESTS="${WARMUP_REQUESTS:-2}"
REQUEST_TIMEOUT_SECONDS="${REQUEST_TIMEOUT_SECONDS:-600}"
ROUND_PAUSE_SECONDS="${ROUND_PAUSE_SECONDS:-2}"
MIN_SUCCESS_RATE="${MIN_SUCCESS_RATE:-0.99}"
STOP_ON_UNSTABLE="${STOP_ON_UNSTABLE:-1}"
THINKING="${THINKING:-disabled}"

# Cloud and local defaults differ because a local queue can be pushed farther
# without consuming paid tokens or being capped by an account RPM quota.
CLOUD_CONCURRENCY_LEVELS="${CLOUD_CONCURRENCY_LEVELS:-8 16 32 48 64 128}"
LOCAL_CONCURRENCY_LEVELS="${LOCAL_CONCURRENCY_LEVELS:-32 64 96 128 192 256}"
CONCURRENCY_LEVELS="${CONCURRENCY_LEVELS:-}"

ARK_BASE_URL="${ARK_BASE_URL:-https://ark.cn-beijing.volces.com/api/v3}"
ARK_MODEL="${ARK_MODEL:-deepseek-v4-pro-260425}"
ARK_KEY="${ARK_API_KEY:-${AZURE_OPENAI_API_KEY:-}}"

DEEPSEEK_BASE_URL="${DEEPSEEK_BASE_URL:-https://api.deepseek.com}"
DEEPSEEK_MODEL="${DEEPSEEK_MODEL:-deepseek-v4-pro}"
DEEPSEEK_KEY="${DEEPSEEK_API_KEY:-${DS_API_KEY:-${DEEPSEEK_OFFICIAL_API_KEY:-}}}"

MODEL_PATH="${MODEL_PATH:-/ai-app-vepfs/modelscope/models/Qwen/Qwen3.5-4B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen/Qwen3.5-4B}"
L20_GPU="${L20_GPU:-0}"
VLLM_HOST="${VLLM_HOST:-0.0.0.0}"
VLLM_PORT="${VLLM_PORT:-8010}"
LOCAL_BASE_URL="${LOCAL_BASE_URL:-http://127.0.0.1:${VLLM_PORT}/v1}"
LOCAL_API_KEY="${LOCAL_API_KEY:-dummy}"
START_VLLM="${START_VLLM:-1}"
STOP_VLLM_ON_EXIT="${STOP_VLLM_ON_EXIT:-1}"
VLLM_STARTUP_TIMEOUT_SECONDS="${VLLM_STARTUP_TIMEOUT_SECONDS:-1200}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-256}"
VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-65536}"
VLLM_DTYPE="${VLLM_DTYPE:-bfloat16}"
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"

TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/outputs/throughput_v4pro_l20_${TS}}"
VLLM_LOG="${OUT_DIR}/local/vllm.log"
VLLM_PID_FILE="${OUT_DIR}/local/vllm.pid"

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

has_target() {
  case " ${TARGETS} " in
    *" $1 "*) return 0 ;;
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

command -v "${PYTHON_BIN}" >/dev/null 2>&1 || fail "Python not found: ${PYTHON_BIN}"
[[ -f "${BENCHMARK_PY}" ]] || fail "Benchmark driver not found: ${BENCHMARK_PY}"
mkdir -p "${OUT_DIR}"

for target in ${TARGETS}; do
  case "${target}" in
    ark|deepseek|local) ;;
    *) fail "Unknown target '${target}'; expected ark, deepseek, or local" ;;
  esac
done

VLLM_PID=""
STARTED_VLLM=0

cleanup() {
  if [[ "${STARTED_VLLM}" == "1" ]] && truthy "${STOP_VLLM_ON_EXIT}"; then
    if [[ -n "${VLLM_PID}" ]] && kill -0 "${VLLM_PID}" 2>/dev/null; then
      echo "[vllm] stopping pid=${VLLM_PID}"
      kill "${VLLM_PID}" 2>/dev/null || true
      wait "${VLLM_PID}" 2>/dev/null || true
    fi
  fi
}
trap cleanup EXIT INT TERM

endpoint_ready() {
  BENCH_BASE_URL="${LOCAL_BASE_URL}" BENCH_API_KEY="${LOCAL_API_KEY}" \
    "${PYTHON_BIN}" - <<'PY'
import os
import urllib.request

request = urllib.request.Request(
    os.environ["BENCH_BASE_URL"].rstrip("/") + "/models",
    headers={"Authorization": f"Bearer {os.environ['BENCH_API_KEY']}"},
)
try:
    with urllib.request.urlopen(request, timeout=3) as response:
        raise SystemExit(0 if response.status == 200 else 1)
except Exception:
    raise SystemExit(1)
PY
}

start_local_vllm() {
  if endpoint_ready; then
    echo "[vllm] reusing ready endpoint: ${LOCAL_BASE_URL}"
    return
  fi
  truthy "${START_VLLM}" || fail "Local endpoint is not ready and START_VLLM=${START_VLLM}"
  command -v vllm >/dev/null 2>&1 || fail "vllm command not found"
  [[ -d "${MODEL_PATH}" ]] || fail "MODEL_PATH not found: ${MODEL_PATH}"
  mkdir -p "$(dirname "${VLLM_LOG}")"

  local vllm_args=(
    serve "${MODEL_PATH}"
    --served-model-name "${SERVED_MODEL_NAME}"
    --host "${VLLM_HOST}"
    --port "${VLLM_PORT}"
    --trust-remote-code
    --dtype "${VLLM_DTYPE}"
    --tensor-parallel-size 1
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
    --max-model-len "${MAX_MODEL_LEN}"
    --max-num-seqs "${VLLM_MAX_NUM_SEQS}"
    --max-num-batched-tokens "${VLLM_MAX_NUM_BATCHED_TOKENS}"
    --enable-prefix-caching
    --enable-chunked-prefill
  )
  if [[ -n "${VLLM_EXTRA_ARGS}" ]]; then
    # shellcheck disable=SC2206
    local extra_args=(${VLLM_EXTRA_ARGS})
    vllm_args+=("${extra_args[@]}")
  fi

  echo "[vllm] starting on CUDA_VISIBLE_DEVICES=${L20_GPU}"
  env CUDA_VISIBLE_DEVICES="${L20_GPU}" \
    nohup vllm "${vllm_args[@]}" > "${VLLM_LOG}" 2>&1 &
  VLLM_PID=$!
  STARTED_VLLM=1
  echo "${VLLM_PID}" > "${VLLM_PID_FILE}"

  local deadline=$((SECONDS + VLLM_STARTUP_TIMEOUT_SECONDS))
  until endpoint_ready; do
    if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
      tail -n 100 "${VLLM_LOG}" >&2 || true
      fail "vLLM exited during startup"
    fi
    if (( SECONDS >= deadline )); then
      tail -n 100 "${VLLM_LOG}" >&2 || true
      fail "Timed out waiting for vLLM"
    fi
    sleep 3
  done
  echo "[vllm] ready pid=${VLLM_PID}"
}

run_benchmark() {
  local name="$1"
  local provider="$2"
  local base_url="$3"
  local model="$4"
  local api_key="$5"
  local levels="$6"
  local output_dir="${OUT_DIR}/${name}"
  local args=(
    "${PYTHON_BIN}" "${BENCHMARK_PY}"
    --name "${name}"
    --provider "${provider}"
    --base-url "${base_url}"
    --model "${model}"
    --output-dir "${output_dir}"
    --concurrency-levels "${levels}"
    --min-requests "${MIN_REQUESTS}"
    --request-multiplier "${REQUEST_MULTIPLIER}"
    --warmup-requests "${WARMUP_REQUESTS}"
    --prompt-token-options "${PROMPT_TOKEN_OPTIONS}"
    --max-token-options "${MAX_TOKEN_OPTIONS}"
    --timeout-seconds "${REQUEST_TIMEOUT_SECONDS}"
    --round-pause-seconds "${ROUND_PAUSE_SECONDS}"
    --min-success-rate "${MIN_SUCCESS_RATE}"
    --thinking "${THINKING}"
  )
  if truthy "${STOP_ON_UNSTABLE}"; then
    args+=(--stop-on-unstable)
  fi
  if truthy "${DRY_RUN}"; then
    args+=(--dry-run)
  elif [[ -z "${api_key}" ]]; then
    echo "[skip] ${name}: API key is empty" >&2
    return 2
  fi

  echo "------------------------------------------------------------"
  echo "[benchmark] ${name}: ${base_url} model=${model}"
  echo "[cmd] BENCH_API_KEY=<redacted>$(quote_cmd "${args[@]}")"
  BENCH_API_KEY="${api_key}" "${args[@]}"
}

echo "============================================================"
echo "  V4 Pro APIs + local L20/vLLM throughput benchmark"
echo "============================================================"
echo "  targets:             ${TARGETS}"
echo "  input token options: ${PROMPT_TOKEN_OPTIONS} (approximate)"
echo "  output token limits: ${MAX_TOKEN_OPTIONS}"
echo "  cloud concurrency:   ${CONCURRENCY_LEVELS:-${CLOUD_CONCURRENCY_LEVELS}}"
echo "  local concurrency:   ${CONCURRENCY_LEVELS:-${LOCAL_CONCURRENCY_LEVELS}}"
echo "  requests/level:      max(${MIN_REQUESTS}, concurrency * ${REQUEST_MULTIPLIER})"
echo "  thinking:            ${THINKING}"
echo "  stable threshold:    ${MIN_SUCCESS_RATE}"
echo "  output:              ${OUT_DIR}"
echo "============================================================"

status=0
if has_target ark; then
  run_benchmark \
    ark deepseek "${ARK_BASE_URL}" "${ARK_MODEL}" "${ARK_KEY}" \
    "${CONCURRENCY_LEVELS:-${CLOUD_CONCURRENCY_LEVELS}}" || status=$?
fi

if has_target deepseek; then
  run_benchmark \
    deepseek_official deepseek "${DEEPSEEK_BASE_URL}" "${DEEPSEEK_MODEL}" \
    "${DEEPSEEK_KEY}" "${CONCURRENCY_LEVELS:-${CLOUD_CONCURRENCY_LEVELS}}" || status=$?
fi

if has_target local; then
  if ! truthy "${DRY_RUN}"; then
    start_local_vllm
  fi
  run_benchmark \
    local_l20_vllm vllm "${LOCAL_BASE_URL}" "${SERVED_MODEL_NAME}" \
    "${LOCAL_API_KEY}" "${CONCURRENCY_LEVELS:-${LOCAL_CONCURRENCY_LEVELS}}" || status=$?
fi

if ! truthy "${DRY_RUN}"; then
  "${PYTHON_BIN}" - "${OUT_DIR}" <<'PY'
import csv
import sys
from pathlib import Path

root = Path(sys.argv[1])
best_rows = []
for csv_path in sorted(root.glob("*/results.csv")):
    with csv_path.open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    stable = [row for row in rows if row.get("stable", "").lower() == "true"]
    if not stable:
        continue
    best = max(stable, key=lambda row: float(row["completion_tokens_per_s"]))
    best_rows.append(best)

lines = [
    "# Throughput comparison",
    "",
    "The values below are each backend's best stable point in this sweep.",
    "",
    "| backend | model | concurrency | success | req/s | input tok/s | output tok/s | total tok/s | TTFT p95 | latency p95 |",
    "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for row in best_rows:
    lines.append(
        f"| {row['name']} | {row['model']} | {row['concurrency']} "
        f"| {float(row['success_rate']):.1%} | {float(row['requests_per_s']):.2f} "
        f"| {float(row['prompt_tokens_per_s']):.1f} "
        f"| {float(row['completion_tokens_per_s']):.1f} "
        f"| {float(row['total_tokens_per_s']):.1f} "
        f"| {float(row['ttft_p95_s'] or 0):.3f}s "
        f"| {float(row['latency_p95_s'] or 0):.3f}s |"
    )
(root / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"[result] comparison={root / 'comparison.md'}")
PY
fi

if [[ "${status}" -ne 0 ]]; then
  echo "[warn] at least one target did not complete; inspect the per-target output above" >&2
fi
echo "[done] output=${OUT_DIR}"
exit "${status}"

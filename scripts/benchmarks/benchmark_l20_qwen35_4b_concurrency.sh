#!/usr/bin/env bash
set -euo pipefail

# Measure practical client concurrency for Qwen3.5-4B on one L20.
#
# Default:
#   - starts one local vLLM server
#   - disables Qwen thinking
#   - tests concurrency: 1 8 16 32 48 64 96 128 160 192 256
#   - forces 512 output tokens per request for comparable load
#   - writes JSONL, CSV, Markdown, and the vLLM log under outputs/
#
# Usage:
#   bash scripts/benchmarks/benchmark_l20_qwen35_4b_concurrency.sh
#
# Useful overrides:
#   L20_GPU=1 bash scripts/benchmarks/benchmark_l20_qwen35_4b_concurrency.sh
#   CONCURRENCY_LEVELS="32 64 96 128" bash scripts/benchmarks/benchmark_l20_qwen35_4b_concurrency.sh
#   MAX_TOKENS=16384 REQUEST_MULTIPLIER=1 MIN_REQUESTS=32 \
#     bash scripts/benchmarks/benchmark_l20_qwen35_4b_concurrency.sh
#   START_VLLM=0 BASE_URL=http://127.0.0.1:8000/v1 \
#     bash scripts/benchmarks/benchmark_l20_qwen35_4b_concurrency.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_PATH="${MODEL_PATH:-/ai-app-vepfs/modelscope/models/Qwen/Qwen3.5-4B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen/Qwen3.5-4B}"
L20_GPU="${L20_GPU:-0}"
VLLM_HOST="${VLLM_HOST:-0.0.0.0}"
VLLM_PORT="${VLLM_PORT:-8010}"
BASE_URL="${BASE_URL:-http://127.0.0.1:${VLLM_PORT}/v1}"
API_KEY="${API_KEY:-dummy}"

CONCURRENCY_LEVELS="${CONCURRENCY_LEVELS:-1 8 16 32 48 64 96 128 160 192 256}"
MAX_TOKENS="${MAX_TOKENS:-512}"
MIN_REQUESTS="${MIN_REQUESTS:-64}"
REQUEST_MULTIPLIER="${REQUEST_MULTIPLIER:-2}"
REQUEST_TIMEOUT_SECONDS="${REQUEST_TIMEOUT_SECONDS:-600}"
WARMUP_REQUESTS="${WARMUP_REQUESTS:-8}"
ROUND_PAUSE_SECONDS="${ROUND_PAUSE_SECONDS:-3}"
STOP_ON_FAILURE="${STOP_ON_FAILURE:-1}"
MIN_SUCCESS_RATE="${MIN_SUCCESS_RATE:-0.99}"

MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-256}"
VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-65536}"
VLLM_DTYPE="${VLLM_DTYPE:-bfloat16}"
VLLM_STARTUP_TIMEOUT_SECONDS="${VLLM_STARTUP_TIMEOUT_SECONDS:-1200}"
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"
START_VLLM="${START_VLLM:-1}"
STOP_VLLM_ON_EXIT="${STOP_VLLM_ON_EXIT:-1}"

TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/outputs/l20_qwen35_4b_concurrency_${TS}}"
VLLM_LOG="${OUT_DIR}/vllm.log"
VLLM_PID_FILE="${OUT_DIR}/vllm.pid"
RESULT_JSONL="${OUT_DIR}/results.jsonl"
RESULT_CSV="${OUT_DIR}/results.csv"
RESULT_MD="${OUT_DIR}/report.md"

mkdir -p "${OUT_DIR}"

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

command -v "${PYTHON_BIN}" >/dev/null 2>&1 || fail "Python not found: ${PYTHON_BIN}"
if truthy "${START_VLLM}"; then
  command -v vllm >/dev/null 2>&1 || fail "vllm command not found"
  [[ -d "${MODEL_PATH}" ]] || fail "MODEL_PATH not found: ${MODEL_PATH}"
fi

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
  "${PYTHON_BIN}" - "${BASE_URL}" "${API_KEY}" <<'PY'
import sys
import urllib.request

base_url, api_key = sys.argv[1:3]
req = urllib.request.Request(
    base_url.rstrip("/") + "/models",
    headers={"Authorization": f"Bearer {api_key}"},
)
try:
    with urllib.request.urlopen(req, timeout=3) as response:
        raise SystemExit(0 if response.status == 200 else 1)
except Exception:
    raise SystemExit(1)
PY
}

echo "============================================================"
echo "  L20 Qwen3.5-4B concurrency benchmark"
echo "============================================================"
echo "  model_path:          ${MODEL_PATH}"
echo "  gpu:                 ${L20_GPU}"
echo "  base_url:            ${BASE_URL}"
echo "  concurrency:         ${CONCURRENCY_LEVELS}"
echo "  max_tokens:          ${MAX_TOKENS}"
echo "  min_requests:        ${MIN_REQUESTS}"
echo "  request_multiplier:  ${REQUEST_MULTIPLIER}"
echo "  vllm_max_num_seqs:   ${VLLM_MAX_NUM_SEQS}"
echo "  max_model_len:       ${MAX_MODEL_LEN}"
echo "  output:              ${OUT_DIR}"
echo "============================================================"

if endpoint_ready; then
  echo "[vllm] reusing ready endpoint: ${BASE_URL}"
elif ! truthy "${START_VLLM}"; then
  fail "Endpoint is not ready and START_VLLM=${START_VLLM}"
else
  vllm_args=(
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
    extra_args=(${VLLM_EXTRA_ARGS})
    vllm_args+=("${extra_args[@]}")
  fi

  echo "[vllm] starting on GPU ${L20_GPU}"
  env CUDA_VISIBLE_DEVICES="${L20_GPU}" \
    nohup vllm "${vllm_args[@]}" > "${VLLM_LOG}" 2>&1 &
  VLLM_PID=$!
  STARTED_VLLM=1
  echo "${VLLM_PID}" > "${VLLM_PID_FILE}"

  deadline=$((SECONDS + VLLM_STARTUP_TIMEOUT_SECONDS))
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
fi

export BASE_URL API_KEY SERVED_MODEL_NAME CONCURRENCY_LEVELS MAX_TOKENS
export MIN_REQUESTS REQUEST_MULTIPLIER REQUEST_TIMEOUT_SECONDS WARMUP_REQUESTS
export ROUND_PAUSE_SECONDS STOP_ON_FAILURE MIN_SUCCESS_RATE
export RESULT_JSONL RESULT_CSV RESULT_MD

"${PYTHON_BIN}" - <<'PY'
from __future__ import annotations

import concurrent.futures
import csv
import json
import math
import os
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path


BASE_URL = os.environ["BASE_URL"].rstrip("/")
API_KEY = os.environ["API_KEY"]
MODEL = os.environ["SERVED_MODEL_NAME"]
LEVELS = [int(value) for value in os.environ["CONCURRENCY_LEVELS"].split()]
MAX_TOKENS = int(os.environ["MAX_TOKENS"])
MIN_REQUESTS = int(os.environ["MIN_REQUESTS"])
REQUEST_MULTIPLIER = int(os.environ["REQUEST_MULTIPLIER"])
TIMEOUT = float(os.environ["REQUEST_TIMEOUT_SECONDS"])
WARMUP_REQUESTS = int(os.environ["WARMUP_REQUESTS"])
ROUND_PAUSE = float(os.environ["ROUND_PAUSE_SECONDS"])
STOP_ON_FAILURE = os.environ["STOP_ON_FAILURE"].lower() in {"1", "true", "yes", "on"}
MIN_SUCCESS_RATE = float(os.environ["MIN_SUCCESS_RATE"])
JSONL_PATH = Path(os.environ["RESULT_JSONL"])
CSV_PATH = Path(os.environ["RESULT_CSV"])
MD_PATH = Path(os.environ["RESULT_MD"])


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1))
    return ordered[index]


def request_once(request_id: int) -> dict:
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a benchmark assistant. Follow the user request "
                    "directly and do not use a hidden reasoning process."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Request {request_id}: write a long numbered list of short "
                    "facts about mathematics. Keep producing items until the "
                    "response limit is reached."
                ),
            },
        ],
        "temperature": 0.0,
        "max_tokens": MAX_TOKENS,
        "stream": True,
        "stream_options": {"include_usage": True},
        "ignore_eos": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        BASE_URL + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    started = time.perf_counter()
    first_token_at = None
    completion_tokens = 0
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data or data == "[DONE]":
                    continue
                chunk = json.loads(data)
                usage = chunk.get("usage") or {}
                completion_tokens = int(
                    usage.get("completion_tokens") or completion_tokens
                )
                for choice in chunk.get("choices") or []:
                    delta = choice.get("delta") or {}
                    if (
                        first_token_at is None
                        and (delta.get("content") or delta.get("reasoning_content"))
                    ):
                        first_token_at = time.perf_counter()
        finished = time.perf_counter()
        return {
            "ok": True,
            "latency_s": finished - started,
            "ttft_s": (
                first_token_at - started if first_token_at is not None else None
            ),
            "completion_tokens": completion_tokens,
            "error": "",
        }
    except Exception as exc:
        finished = time.perf_counter()
        if isinstance(exc, urllib.error.HTTPError):
            try:
                detail = exc.read().decode("utf-8", errors="replace")[:500]
            except Exception:
                detail = ""
            error = f"HTTP {exc.code}: {detail}"
        else:
            error = f"{type(exc).__name__}: {exc}"
        return {
            "ok": False,
            "latency_s": finished - started,
            "ttft_s": None,
            "completion_tokens": 0,
            "error": error,
        }


def run_round(concurrency: int, total_requests: int) -> dict:
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        rows = list(pool.map(request_once, range(total_requests)))
    wall_s = time.perf_counter() - started

    ok_rows = [row for row in rows if row["ok"]]
    latencies = [float(row["latency_s"]) for row in ok_rows]
    ttfts = [
        float(row["ttft_s"]) for row in ok_rows if row["ttft_s"] is not None
    ]
    completion_tokens = sum(int(row["completion_tokens"]) for row in ok_rows)
    errors: dict[str, int] = {}
    for row in rows:
        if row["ok"]:
            continue
        key = str(row["error"])[:200]
        errors[key] = errors.get(key, 0) + 1

    success_rate = len(ok_rows) / len(rows) if rows else 0.0
    return {
        "concurrency": concurrency,
        "requests": len(rows),
        "ok": len(ok_rows),
        "failed": len(rows) - len(ok_rows),
        "success_rate": success_rate,
        "wall_s": wall_s,
        "requests_per_s": len(ok_rows) / wall_s if wall_s else 0.0,
        "completion_tokens": completion_tokens,
        "completion_tokens_per_s": (
            completion_tokens / wall_s if wall_s else 0.0
        ),
        "ttft_p50_s": percentile(ttfts, 0.50),
        "ttft_p95_s": percentile(ttfts, 0.95),
        "latency_p50_s": percentile(latencies, 0.50),
        "latency_p95_s": percentile(latencies, 0.95),
        "stable": success_rate >= MIN_SUCCESS_RATE,
        "errors": errors,
    }


print(f"[warmup] requests={WARMUP_REQUESTS}", flush=True)
warmup = run_round(max(1, min(WARMUP_REQUESTS, 8)), WARMUP_REQUESTS)
if warmup["ok"] != WARMUP_REQUESTS:
    raise SystemExit(f"Warmup failed: {warmup['errors']}")

JSONL_PATH.write_text("", encoding="utf-8")
results = []
for concurrency in LEVELS:
    total_requests = max(MIN_REQUESTS, concurrency * REQUEST_MULTIPLIER)
    print(
        f"[round] concurrency={concurrency} requests={total_requests}",
        flush=True,
    )
    row = run_round(concurrency, total_requests)
    results.append(row)
    with JSONL_PATH.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(
        "  "
        f"ok={row['ok']}/{row['requests']} "
        f"wall={row['wall_s']:.1f}s "
        f"tok/s={row['completion_tokens_per_s']:.1f} "
        f"ttft_p95={row['ttft_p95_s'] or 0:.3f}s "
        f"lat_p95={row['latency_p95_s'] or 0:.3f}s",
        flush=True,
    )
    if not row["stable"] and STOP_ON_FAILURE:
        print("[stop] success rate fell below threshold", flush=True)
        break
    time.sleep(ROUND_PAUSE)

fields = [
    "concurrency",
    "requests",
    "ok",
    "failed",
    "success_rate",
    "wall_s",
    "requests_per_s",
    "completion_tokens_per_s",
    "ttft_p50_s",
    "ttft_p95_s",
    "latency_p50_s",
    "latency_p95_s",
    "stable",
]
with CSV_PATH.open("w", encoding="utf-8", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for row in results:
        writer.writerow({key: row.get(key) for key in fields})

stable_rows = [row for row in results if row["stable"]]
stable_max = (
    max(stable_rows, key=lambda row: row["concurrency"])
    if stable_rows
    else None
)
throughput_best = (
    max(stable_rows, key=lambda row: row["completion_tokens_per_s"])
    if stable_rows
    else None
)

lines = [
    "# L20 Qwen3.5-4B Concurrency Benchmark",
    "",
    f"- Model: `{MODEL}`",
    f"- Max output tokens/request: `{MAX_TOKENS}`",
    f"- Stable threshold: `{MIN_SUCCESS_RATE:.1%}`",
    (
        f"- Highest stable tested concurrency: `{stable_max['concurrency']}`"
        if stable_max
        else "- Highest stable tested concurrency: none"
    ),
    (
        "- Best throughput: "
        f"`{throughput_best['concurrency']}` concurrency, "
        f"`{throughput_best['completion_tokens_per_s']:.1f}` completion tok/s"
        if throughput_best
        else "- Best throughput: none"
    ),
    "",
    "| concurrency | success | req/s | completion tok/s | TTFT p50 | TTFT p95 | latency p50 | latency p95 | stable |",
    "|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
]
for row in results:
    lines.append(
        f"| {row['concurrency']} "
        f"| {row['ok']}/{row['requests']} "
        f"| {row['requests_per_s']:.2f} "
        f"| {row['completion_tokens_per_s']:.1f} "
        f"| {(row['ttft_p50_s'] or 0):.3f}s "
        f"| {(row['ttft_p95_s'] or 0):.3f}s "
        f"| {(row['latency_p50_s'] or 0):.3f}s "
        f"| {(row['latency_p95_s'] or 0):.3f}s "
        f"| {'yes' if row['stable'] else 'no'} |"
    )
MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

print("", flush=True)
print(f"[result] JSONL: {JSONL_PATH}", flush=True)
print(f"[result] CSV:   {CSV_PATH}", flush=True)
print(f"[result] report:{MD_PATH}", flush=True)
if stable_max:
    print(
        f"[result] highest stable tested concurrency={stable_max['concurrency']}",
        flush=True,
    )
if throughput_best:
    print(
        "[result] best throughput concurrency="
        f"{throughput_best['concurrency']} "
        f"tokens/s={throughput_best['completion_tokens_per_s']:.1f}",
        flush=True,
    )
PY

if [[ -f "${VLLM_LOG}" ]] && grep -Eqi "out of memory|cuda error|engine.*dead" "${VLLM_LOG}"; then
  echo "[warn] vLLM log contains OOM/CUDA/engine errors: ${VLLM_LOG}" >&2
fi

echo "[done] report: ${RESULT_MD}"

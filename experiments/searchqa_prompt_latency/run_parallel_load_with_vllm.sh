#!/usr/bin/env bash
set -euo pipefail

# Start one local vLLM endpoint, then run several SearchQA prompt probes
# concurrently to simulate 2-3 datasets sharing the same target model service.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
JOBS="${JOBS:-3}"
WORKERS_PER_JOB="${WORKERS_PER_JOB:-128}"
SPLIT_PATH="${SPLIT_PATH:-data/searchqa_split/test/items.json}"
SAMPLE_SIZE="${SAMPLE_SIZE:-0}"
MAX_TOKENS="${MAX_TOKENS:-16384}"
PROMPT_LIST="${PROMPT_LIST:-baseline_current direct_with_evidence_check}"
OUT_BASE="${OUT_BASE:-outputs/searchqa_parallel_load_$(date +%Y%m%d_%H%M%S)}"
SERVER_OUT_DIR="${SERVER_OUT_DIR:-${OUT_BASE}/server}"
STOP_VLLM_ON_EXIT="${STOP_VLLM_ON_EXIT:-1}"

mkdir -p "${OUT_BASE}"

cleanup() {
  if [[ "${STOP_VLLM_ON_EXIT}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
    local pid_file="${SERVER_OUT_DIR}/logs/vllm.pid"
    if [[ -f "${pid_file}" ]]; then
      local pid
      pid="$(cat "${pid_file}" 2>/dev/null || true)"
      if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
        echo "[vllm] stopping pid=${pid}"
        kill "${pid}" || true
      fi
    fi
  fi
}
trap cleanup EXIT

echo "[server] starting/checking vLLM"
START_ONLY=1 STOP_VLLM_ON_EXIT=0 OUT_DIR="${SERVER_OUT_DIR}" \
  bash experiments/searchqa_prompt_latency/run_with_vllm.sh

prompt_args=()
for prompt in ${PROMPT_LIST}; do
  prompt_args+=(--prompt "${prompt}")
done

pids=()
for idx in $(seq 1 "${JOBS}"); do
  out_dir="${OUT_BASE}/job_${idx}"
  log_file="${OUT_BASE}/job_${idx}.log"
  echo "[start] job=${idx} workers=${WORKERS_PER_JOB} out=${out_dir}"
  "${PYTHON_BIN}" experiments/searchqa_prompt_latency/run_searchqa_prompt_latency.py \
    --split "${SPLIT_PATH}" \
    --sample-size "${SAMPLE_SIZE}" \
    --workers "${WORKERS_PER_JOB}" \
    --max-tokens "${MAX_TOKENS}" \
    --dataset-label "searchqa_sim_${idx}" \
    --out-dir "${out_dir}" \
    "${prompt_args[@]}" \
    > "${log_file}" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

echo "[done] out_base=${OUT_BASE}"
exit "${status}"

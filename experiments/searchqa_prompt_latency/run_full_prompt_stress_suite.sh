#!/usr/bin/env bash
set -euo pipefail

# Unified SearchQA prompt/effect/latency stress suite.
# Steps:
#   1. Start or reuse one local Qwen vLLM endpoint.
#   2. Run full SearchQA test split prompt comparisons with exact EM/F1.
#   3. Summarize the full-test results.
#   4. Optionally run several concurrent jobs to simulate multiple datasets.
#   5. Summarize every concurrent job.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
OUT_BASE="${OUT_BASE:-outputs/searchqa_prompt_stress_suite_${TS}}"
SERVER_OUT_DIR="${SERVER_OUT_DIR:-${OUT_BASE}/server}"

RUN_FULL_TEST="${RUN_FULL_TEST:-1}"
RUN_PARALLEL_LOAD="${RUN_PARALLEL_LOAD:-1}"
STOP_VLLM_ON_EXIT="${STOP_VLLM_ON_EXIT:-1}"

SPLIT_PATH="${SPLIT_PATH:-data/searchqa_split/test/items.json}"
SAMPLE_SIZE="${SAMPLE_SIZE:-0}"
MAX_TOKENS="${MAX_TOKENS:-16384}"
SKILL_PATH="${SKILL_PATH:-skillopt/envs/searchqa/skills/initial.md}"

FULL_WORKERS_LIST="${FULL_WORKERS_LIST:-96 128 192 256}"
FULL_PROMPT_LIST="${FULL_PROMPT_LIST:-baseline_current direct_when_identified direct_with_evidence_check}"
FULL_OUT_DIR="${FULL_OUT_DIR:-${OUT_BASE}/full_test}"
FULL_LOG="${FULL_LOG:-${OUT_BASE}/full_test.log}"
FULL_SUMMARY_MD="${FULL_SUMMARY_MD:-${OUT_BASE}/full_test_summary.md}"

PARALLEL_JOBS="${PARALLEL_JOBS:-3}"
PARALLEL_WORKERS_PER_JOB="${PARALLEL_WORKERS_PER_JOB:-128}"
PARALLEL_PROMPT_LIST="${PARALLEL_PROMPT_LIST:-baseline_current direct_with_evidence_check}"
PARALLEL_OUT_BASE="${PARALLEL_OUT_BASE:-${OUT_BASE}/parallel_load}"

QWEN_CHAT_BASE_URL="${QWEN_CHAT_BASE_URL:-http://127.0.0.1:${VLLM_PORT:-8000}/v1}"
QWEN_CHAT_API_KEY="${QWEN_CHAT_API_KEY:-dummy}"
QWEN_CHAT_MODEL="${QWEN_CHAT_MODEL:-${SERVED_MODEL_NAME:-Qwen/Qwen3.5-4B}}"
QWEN_CHAT_TIMEOUT_SECONDS="${QWEN_CHAT_TIMEOUT_SECONDS:-240}"
TARGET_QWEN_CHAT_TEMPERATURE="${TARGET_QWEN_CHAT_TEMPERATURE:-0.2}"

mkdir -p "${OUT_BASE}" "${FULL_OUT_DIR}" "${PARALLEL_OUT_BASE}"

truthy() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

cleanup() {
  if truthy "${STOP_VLLM_ON_EXIT}"; then
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

run_and_log() {
  local log_file="$1"
  shift
  mkdir -p "$(dirname "${log_file}")"
  "$@" 2>&1 | tee "${log_file}"
}

echo "[suite] project=${PROJECT_ROOT}"
echo "[suite] out_base=${OUT_BASE}"
echo "[suite] split=${SPLIT_PATH}"
echo "[suite] sample_size=${SAMPLE_SIZE} max_tokens=${MAX_TOKENS}"
echo "[suite] full_prompts=${FULL_PROMPT_LIST}"
echo "[suite] full_workers=${FULL_WORKERS_LIST}"
echo "[suite] parallel_jobs=${PARALLEL_JOBS} parallel_workers_per_job=${PARALLEL_WORKERS_PER_JOB}"
echo "[suite] parallel_prompts=${PARALLEL_PROMPT_LIST}"

echo "[step 1/5] start or reuse vLLM"
START_ONLY=1 STOP_VLLM_ON_EXIT=0 OUT_DIR="${SERVER_OUT_DIR}" \
  QWEN_CHAT_BASE_URL="${QWEN_CHAT_BASE_URL}" \
  QWEN_CHAT_API_KEY="${QWEN_CHAT_API_KEY}" \
  QWEN_CHAT_MODEL="${QWEN_CHAT_MODEL}" \
  bash experiments/searchqa_prompt_latency/run_with_vllm.sh

if truthy "${RUN_FULL_TEST}"; then
  echo "[step 2/5] run full SearchQA prompt test"
  full_prompt_args=()
  for prompt in ${FULL_PROMPT_LIST}; do
    full_prompt_args+=(--prompt "${prompt}")
  done
  run_and_log "${FULL_LOG}" \
    "${PYTHON_BIN}" experiments/searchqa_prompt_latency/run_searchqa_prompt_latency.py \
      --base-url "${QWEN_CHAT_BASE_URL}" \
      --api-key "${QWEN_CHAT_API_KEY}" \
      --model "${QWEN_CHAT_MODEL}" \
      --split "${SPLIT_PATH}" \
      --skill "${SKILL_PATH}" \
      --sample-size "${SAMPLE_SIZE}" \
      --workers ${FULL_WORKERS_LIST} \
      --max-tokens "${MAX_TOKENS}" \
      --temperature "${TARGET_QWEN_CHAT_TEMPERATURE}" \
      --timeout "${QWEN_CHAT_TIMEOUT_SECONDS}" \
      --dataset-label "searchqa_full_test" \
      --out-dir "${FULL_OUT_DIR}" \
      "${full_prompt_args[@]}"

  echo "[step 3/5] summarize full SearchQA prompt test"
  "${PYTHON_BIN}" experiments/searchqa_prompt_latency/summarize_prompt_latency.py \
    "${FULL_OUT_DIR}" > "${FULL_SUMMARY_MD}"
  cat "${FULL_SUMMARY_MD}"
else
  echo "[step 2/5] skipped full SearchQA prompt test"
  echo "[step 3/5] skipped full SearchQA summary"
fi

if truthy "${RUN_PARALLEL_LOAD}"; then
  echo "[step 4/5] run parallel load simulation"
  parallel_prompt_args=()
  for prompt in ${PARALLEL_PROMPT_LIST}; do
    parallel_prompt_args+=(--prompt "${prompt}")
  done
  pids=()
  for idx in $(seq 1 "${PARALLEL_JOBS}"); do
    job_out_dir="${PARALLEL_OUT_BASE}/job_${idx}"
    job_log="${PARALLEL_OUT_BASE}/job_${idx}.log"
    mkdir -p "${job_out_dir}"
    echo "[parallel-start] job=${idx} workers=${PARALLEL_WORKERS_PER_JOB} out=${job_out_dir}"
    "${PYTHON_BIN}" experiments/searchqa_prompt_latency/run_searchqa_prompt_latency.py \
      --base-url "${QWEN_CHAT_BASE_URL}" \
      --api-key "${QWEN_CHAT_API_KEY}" \
      --model "${QWEN_CHAT_MODEL}" \
      --split "${SPLIT_PATH}" \
      --skill "${SKILL_PATH}" \
      --sample-size "${SAMPLE_SIZE}" \
      --workers "${PARALLEL_WORKERS_PER_JOB}" \
      --max-tokens "${MAX_TOKENS}" \
      --temperature "${TARGET_QWEN_CHAT_TEMPERATURE}" \
      --timeout "${QWEN_CHAT_TIMEOUT_SECONDS}" \
      --dataset-label "searchqa_sim_${idx}" \
      --out-dir "${job_out_dir}" \
      "${parallel_prompt_args[@]}" \
      > "${job_log}" 2>&1 &
    pids+=("$!")
  done

  parallel_status=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      parallel_status=1
    fi
  done
  if [[ "${parallel_status}" -ne 0 ]]; then
    echo "[warn] at least one parallel load job failed; check ${PARALLEL_OUT_BASE}/job_*.log"
  fi

  echo "[step 5/5] summarize parallel load jobs"
  for idx in $(seq 1 "${PARALLEL_JOBS}"); do
    job_out_dir="${PARALLEL_OUT_BASE}/job_${idx}"
    job_summary="${PARALLEL_OUT_BASE}/job_${idx}_summary.md"
    if [[ -f "${job_out_dir}/summary.json" ]]; then
      "${PYTHON_BIN}" experiments/searchqa_prompt_latency/summarize_prompt_latency.py \
        "${job_out_dir}" > "${job_summary}"
      echo
      echo "## parallel job ${idx}"
      cat "${job_summary}"
    else
      echo "[warn] missing summary for parallel job ${idx}: ${job_out_dir}/summary.json"
    fi
  done
  exit "${parallel_status}"
else
  echo "[step 4/5] skipped parallel load simulation"
  echo "[step 5/5] skipped parallel load summaries"
fi

echo "[done] out_base=${OUT_BASE}"

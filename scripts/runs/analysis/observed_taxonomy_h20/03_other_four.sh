#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export OBSERVED_JOB_NAME=other_four
export OPTIMIZER_SOURCE=deepseek
export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-128}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
# shellcheck source=_common.sh
source "${HERE}/_common.sh"

SPREADSHEET_TARGET_WORKERS="${SPREADSHEET_TARGET_WORKERS:-24}"
OFFICEQA_TARGET_WORKERS="${OFFICEQA_TARGET_WORKERS:-16}"
DOCVQA_TARGET_WORKERS="${DOCVQA_TARGET_WORKERS:-28}"
LIVEMATH_TARGET_WORKERS="${LIVEMATH_TARGET_WORKERS:-28}"
TOTAL_TARGET_WORKERS=$((
  SPREADSHEET_TARGET_WORKERS
  + OFFICEQA_TARGET_WORKERS
  + DOCVQA_TARGET_WORKERS
  + LIVEMATH_TARGET_WORKERS
))
(( TOTAL_TARGET_WORKERS <= VLLM_MAX_NUM_SEQS )) || observed_fail \
  "combined target workers=${TOTAL_TARGET_WORKERS} exceeds VLLM_MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS}"

print_observed_header \
  "spreadsheet_office_doc_livemath" \
  "spreadsheetbench officeqa docvqa livemathematicianbench"
start_vllm

run_observed_dataset \
  spreadsheetbench configs/spreadsheetbench/default.yaml \
  "${SPREADSHEET_BATCH_SIZE:-40}" \
  "${SPREADSHEET_TARGET_WORKERS}" \
  "${SPREADSHEET_ANALYST_WORKERS:-16}" \
  "${SPREADSHEET_TARGET_MAX_TOKENS:-16384}" &
JOB_PIDS+=("$!")

run_observed_dataset \
  officeqa configs/officeqa/default.yaml \
  "${OFFICEQA_BATCH_SIZE:-24}" \
  "${OFFICEQA_TARGET_WORKERS}" \
  "${OFFICEQA_ANALYST_WORKERS:-16}" \
  "${OFFICEQA_TARGET_MAX_TOKENS:-16384}" &
JOB_PIDS+=("$!")

run_observed_dataset \
  docvqa configs/docvqa/default.yaml \
  "${DOCVQA_BATCH_SIZE:-64}" \
  "${DOCVQA_TARGET_WORKERS}" \
  "${DOCVQA_ANALYST_WORKERS:-24}" \
  "${DOCVQA_TARGET_MAX_TOKENS:-4096}" &
JOB_PIDS+=("$!")

run_observed_dataset \
  livemathematicianbench configs/livemathematicianbench/default.yaml \
  "${LIVEMATH_BATCH_SIZE:-64}" \
  "${LIVEMATH_TARGET_WORKERS}" \
  "${LIVEMATH_ANALYST_WORKERS:-24}" \
  "${LIVEMATH_TARGET_MAX_TOKENS:-8192}" &
JOB_PIDS+=("$!")

status=0
for pid in "${JOB_PIDS[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
JOB_PIDS=()
exit "${status}"

#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export OBSERVED_JOB_NAME=alfworld
export OPTIMIZER_SOURCE=deepseek
export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-96}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
# shellcheck source=_common.sh
source "${HERE}/_common.sh"

print_observed_header "alfworld" "alfworld"
start_vllm
run_observed_dataset \
  alfworld configs/alfworld/default.yaml \
  "${ALFWORLD_BATCH_SIZE:-8}" \
  "${ALFWORLD_TARGET_WORKERS:-24}" \
  "${ALFWORLD_ANALYST_WORKERS:-16}" \
  "${ALFWORLD_TARGET_MAX_TOKENS:-2048}"

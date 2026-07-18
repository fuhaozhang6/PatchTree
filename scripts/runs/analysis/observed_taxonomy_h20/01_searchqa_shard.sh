#!/usr/bin/env bash
set -euo pipefail

# Run this launcher twice on two independent H20 workers:
#   RUN_ID=taxonomy_v1 SHARD_INDEX=0 bash .../01_searchqa_shard.sh
#   RUN_ID=taxonomy_v1 SHARD_INDEX=1 bash .../01_searchqa_shard.sh

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHARD_COUNT="${SHARD_COUNT:-2}"
SHARD_INDEX="${SHARD_INDEX:-0}"
export OBSERVED_JOB_NAME="searchqa_shard${SHARD_INDEX}of${SHARD_COUNT}"
export OPTIMIZER_SOURCE=deepseek
export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-128}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
# shellcheck source=_common.sh
source "${HERE}/_common.sh"

[[ "${SHARD_INDEX}" =~ ^[0-9]+$ ]] || observed_fail "SHARD_INDEX must be an integer"
(( SHARD_INDEX >= 0 && SHARD_INDEX < SHARD_COUNT )) \
  || observed_fail "require 0 <= SHARD_INDEX < SHARD_COUNT"

name="${OBSERVED_JOB_NAME}"
SEARCHQA_TARGET_WORKERS="${SEARCHQA_TARGET_WORKERS:-128}"
SEARCHQA_ANALYST_WORKERS="${SEARCHQA_ANALYST_WORKERS:-64}"
(( SEARCHQA_TARGET_WORKERS <= VLLM_MAX_NUM_SEQS )) || observed_fail \
  "SEARCHQA_TARGET_WORKERS=${SEARCHQA_TARGET_WORKERS} exceeds VLLM_MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS}"
print_observed_header "${name}" "searchqa"
start_vllm
run_observed_dataset \
  "${name}" configs/searchqa/default.yaml \
  "${SEARCHQA_BATCH_SIZE:-100}" \
  "${SEARCHQA_TARGET_WORKERS}" \
  "${SEARCHQA_ANALYST_WORKERS}" \
  "${SEARCHQA_TARGET_MAX_TOKENS:-4096}" \
  "${SHARD_COUNT}" "${SHARD_INDEX}"

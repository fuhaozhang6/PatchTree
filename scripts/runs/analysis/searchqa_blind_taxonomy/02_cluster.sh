#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "${HERE}/_common.sh"

SHARD_COUNT="${SHARD_COUNT:-2}"
CLUSTER_DIR="${OUT_BASE}/clusters"
SIMILARITY_THRESHOLD="${SIMILARITY_THRESHOLD:-auto}"
RESIDUAL_SIMILARITY_THRESHOLD="${RESIDUAL_SIMILARITY_THRESHOLD:-auto}"
ASSIGNMENT_THRESHOLD="${ASSIGNMENT_THRESHOLD:-auto}"
MIN_SAMPLES="${MIN_SAMPLES:-3}"
MIN_CLUSTER_SIZE="${MIN_CLUSTER_SIZE:-6}"
FIT_FRACTION="${FIT_FRACTION:-0.60}"

args=(
  "${PYTHON_BIN}" -u scripts/tools/cluster_searchqa_blind_mechanisms.py
  --output-dir "${CLUSTER_DIR}"
  --similarity-threshold "${SIMILARITY_THRESHOLD}"
  --residual-similarity-threshold "${RESIDUAL_SIMILARITY_THRESHOLD}"
  --assignment-threshold "${ASSIGNMENT_THRESHOLD}"
  --min-samples "${MIN_SAMPLES}"
  --min-cluster-size "${MIN_CLUSTER_SIZE}"
  --fit-fraction "${FIT_FRACTION}"
)
for ((index=0; index<SHARD_COUNT; index++)); do
  card_dir="${OUT_BASE}/extract_shard${index}of${SHARD_COUNT}"
  [[ -f "${card_dir}/usable_mechanism_cards.jsonl" ]] \
    || blind_fail "missing extraction output: ${card_dir}"
  args+=(--input "${card_dir}")
done

echo "[run] blind cards -> candidate clusters"
"${args[@]}" 2>&1 | tee "${LOG_BASE}/cluster.log"

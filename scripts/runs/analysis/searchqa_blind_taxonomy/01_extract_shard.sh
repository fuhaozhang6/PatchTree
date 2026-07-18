#!/usr/bin/env bash
set -euo pipefail

# Run twice in parallel on ordinary CPU workers. No GPU/vLLM is used:
#   SHARD_INDEX=0 RUN_ID=blind_v1 bash .../01_extract_shard.sh
#   SHARD_INDEX=1 RUN_ID=blind_v1 bash .../01_extract_shard.sh

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "${HERE}/_common.sh"
configure_deepseek_official

SHARD_COUNT="${SHARD_COUNT:-2}"
SHARD_INDEX="${SHARD_INDEX:-0}"
ANALYST_WORKERS="${ANALYST_WORKERS:-64}"
SPLITS="${SPLITS:-train val test}"
SEARCHQA_SPLIT_DIR="${SEARCHQA_SPLIT_DIR:-${PROJECT_ROOT}/data/searchqa_split}"
LIMIT="${LIMIT:-0}"
OUT_DIR="${OUT_BASE}/extract_shard${SHARD_INDEX}of${SHARD_COUNT}"
LOG_FILE="${LOG_BASE}/extract_shard${SHARD_INDEX}of${SHARD_COUNT}.log"

args=(
  "${PYTHON_BIN}" -u scripts/tools/extract_searchqa_blind_mechanisms.py
  --input "${OBSERVED_ROOT}"
  --output-dir "${OUT_DIR}"
  --split-dir "${SEARCHQA_SPLIT_DIR}"
  --splits "${SPLITS}"
  --statuses "unstable failure"
  --optimizer-model "${OPTIMIZER_MODEL}"
  --analyst-workers "${ANALYST_WORKERS}"
  --shard-count "${SHARD_COUNT}"
  --shard-index "${SHARD_INDEX}"
  --limit "${LIMIT}"
)
if blind_truthy "${DRY_RUN}"; then
  args+=(--dry-run)
fi

echo "[run] saved trajectories -> blind mechanism cards"
echo "[run] shard=${SHARD_INDEX}/${SHARD_COUNT} workers=${ANALYST_WORKERS}"
echo "[run] input=${OBSERVED_ROOT}"
echo "[run] output=${OUT_DIR}"
"${args[@]}" 2>&1 | tee "${LOG_FILE}"

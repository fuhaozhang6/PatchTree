#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "${HERE}/_common.sh"
configure_deepseek_official

SHARD_COUNT="${SHARD_COUNT:-2}"
CLUSTER_DIR="${OUT_BASE}/clusters"
TAXONOMY_DIR="${OUT_BASE}/taxonomy"
DRAFTS="${DRAFTS:-2}"
ADJUDICATION_WORKERS="${ADJUDICATION_WORKERS:-12}"
MAX_FIT_CARDS="${MAX_FIT_CARDS:-36}"
SKIP_GLOBAL_MERGE="${SKIP_GLOBAL_MERGE:-0}"

args=(
  "${PYTHON_BIN}" -u scripts/tools/adjudicate_searchqa_blind_clusters.py
  --clusters "${CLUSTER_DIR}/candidate_clusters.json"
  --output-dir "${TAXONOMY_DIR}"
  --optimizer-model "${OPTIMIZER_MODEL}"
  --drafts "${DRAFTS}"
  --workers "${ADJUDICATION_WORKERS}"
  --max-fit-cards "${MAX_FIT_CARDS}"
)
for ((index=0; index<SHARD_COUNT; index++)); do
  args+=(--cards "${OUT_BASE}/extract_shard${index}of${SHARD_COUNT}")
  args+=(--seeded-labels \
    "${OUT_BASE}/extract_shard${index}of${SHARD_COUNT}/posthoc_seeded_labels.jsonl")
done
if blind_truthy "${SKIP_GLOBAL_MERGE}"; then
  args+=(--skip-global-merge)
fi
if blind_truthy "${DRY_RUN}"; then
  args+=(--dry-run)
fi

echo "[run] candidate clusters -> blind named taxonomy"
"${args[@]}" 2>&1 | tee "${LOG_BASE}/adjudicate.log"

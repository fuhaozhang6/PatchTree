#!/usr/bin/env bash
set -euo pipefail

# Read-only preflight. It checks both old SearchQA shards, restored contexts,
# and three-attempt coverage without making any API/model call.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "${HERE}/_common.sh"

"${PYTHON_BIN}" -u scripts/tools/extract_searchqa_blind_mechanisms.py \
  --input "${OBSERVED_ROOT}" \
  --output-dir "${OUT_BASE}/preflight_only" \
  --split-dir "${SEARCHQA_SPLIT_DIR:-${PROJECT_ROOT}/data/searchqa_split}" \
  --splits "${SPLITS:-train val test}" \
  --statuses "unstable failure" \
  --shard-count 1 \
  --shard-index 0 \
  --dry-run

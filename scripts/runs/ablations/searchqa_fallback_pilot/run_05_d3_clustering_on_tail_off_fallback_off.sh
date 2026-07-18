#!/usr/bin/env bash
set -euo pipefail

# Tree-shape pilot:
#   Record -> LLM mechanism clusters/Leaves -> semantic Mid nodes -> Root
# Child fallback validation and epoch-level tail generation are both disabled,
# isolating clustering plus the stricter Mid prompts.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PILOT_SUITE_SLUG=searchqa_tree_shape
export PILOT_NAME=d3_clustering_on_tail_off_fallback_off
export PILOT_TREE_DEPTH=3
export PILOT_FALLBACK=false
export PILOT_TYPE_GUIDED_CLUSTERING=true
export PILOT_TYPE_GUIDED_CLUSTER_TARGET_SIZE=2
export PILOT_TYPE_GUIDED_CLUSTER_MAX_SIZE=4
export PILOT_TYPE_GUIDED_TAIL_BANK=false
export PILOT_TYPE_GUIDED_MIN_SUPPORT=2
export PILOT_TYPE_GUIDED_MAX_LEAF_GROUPS=8
exec bash "${SCRIPT_DIR}/_run_one.sh" "$@"

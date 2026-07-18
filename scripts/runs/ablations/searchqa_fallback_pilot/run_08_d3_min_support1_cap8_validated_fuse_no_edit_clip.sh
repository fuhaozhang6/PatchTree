#!/usr/bin/env bash
set -euo pipefail

# First combined tree + validation-fusion pilot:
#   - strengthened Mid planner/merge prompts;
#   - min_support=1 retains low-support leaves, capped at 8 per step;
#   - rejected roots evaluate at most four direct children on one shared val40;
#   - positive children are semantically integrated by validated_frontier_fuse;
#   - a constant budget of 64 effectively disables post-root edit clipping for
#     this pilot without removing the edit-budget feature from other runs.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PILOT_SUITE_SLUG=searchqa_tree_fusion
export PILOT_NAME=d3_min_support1_cap8_validated_fuse_no_edit_clip
export PILOT_TREE_DEPTH=3
export PILOT_FALLBACK=true
export PILOT_TYPE_GUIDED_CLUSTERING=true
export PILOT_TYPE_GUIDED_CLUSTER_TARGET_SIZE=2
export PILOT_TYPE_GUIDED_CLUSTER_MAX_SIZE=4
export PILOT_TYPE_GUIDED_TAIL_BANK=false
export PILOT_TYPE_GUIDED_MIN_SUPPORT=1
export PILOT_TYPE_GUIDED_MAX_LEAF_GROUPS=8
export PILOT_TYPE_GUIDED_FALLBACK_RECONCILE=llm_fuse
export PILOT_TYPE_GUIDED_FALLBACK_RECONCILE_MIN_CHILDREN=2
export PILOT_LR_SCHEDULER=constant
export PILOT_EDIT_BUDGET=64
export PILOT_MIN_EDIT_BUDGET=64
exec bash "${SCRIPT_DIR}/_run_one.sh" "$@"

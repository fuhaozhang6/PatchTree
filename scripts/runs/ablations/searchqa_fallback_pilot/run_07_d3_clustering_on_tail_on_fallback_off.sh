#!/usr/bin/env bash
set -euo pipefail

# Epoch-tail pilot, based on run_05:
#   - the per-step tree keeps run_05's min_support=2 and max_leaf_groups=8;
#   - low-support records excluded from per-step trees are pooled across steps;
#   - at epoch end, only exact mechanism groups seen in at least two distinct
#     steps are eligible for one tail merge and full-val gate.
# SearchQA uses one epoch here, so this tail pass runs once after all 10 steps.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PILOT_SUITE_SLUG=searchqa_tree_shape
export PILOT_NAME=d3_clustering_on_tail_on_fallback_off
export PILOT_TREE_DEPTH=3
export PILOT_FALLBACK=false
export PILOT_TYPE_GUIDED_CLUSTERING=true
export PILOT_TYPE_GUIDED_CLUSTER_TARGET_SIZE=2
export PILOT_TYPE_GUIDED_CLUSTER_MAX_SIZE=4
export PILOT_TYPE_GUIDED_MIN_SUPPORT=2
export PILOT_TYPE_GUIDED_MAX_LEAF_GROUPS=8
export PILOT_TYPE_GUIDED_TAIL_BANK=true
export PILOT_TYPE_GUIDED_TAIL_MIN_SUPPORT=2
export PILOT_TYPE_GUIDED_TAIL_MAX_RECORDS=40
export PILOT_TYPE_GUIDED_TAIL_MAX_LEAF_GROUPS=4
export PILOT_TYPE_GUIDED_TAIL_WINDOW_EPOCHS=3
export PILOT_TYPE_GUIDED_TAIL_REQUIRE_CROSS_STEP=true
exec bash "${SCRIPT_DIR}/_run_one.sh" "$@"

#!/usr/bin/env bash
set -euo pipefail

# All-clusters pilot, based on run_05:
#   - min_support=1 keeps singleton mechanism clusters;
#   - max_leaf_groups=40 removes the earlier top-8 cap for this train batch,
#     so every retained PatchRecord cluster can enter the tree;
#   - fallback and the epoch-level tail bank remain off.
#
# This intentionally tests "put all clusters into the per-step tree", not only
# a strict one-variable min_support change. TYPE_GUIDED_MAX_PATCH_RECORDS=40 in
# _run_one.sh is the upstream ceiling, so a leaf cap of 40 does not drop any
# cluster produced from the step's PatchRecords.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PILOT_SUITE_SLUG=searchqa_tree_shape
export PILOT_NAME=d3_clustering_on_min_support1_all_clusters_tail_off_fallback_off
export PILOT_TREE_DEPTH=3
export PILOT_FALLBACK=false
export PILOT_TYPE_GUIDED_CLUSTERING=true
export PILOT_TYPE_GUIDED_CLUSTER_TARGET_SIZE=2
export PILOT_TYPE_GUIDED_CLUSTER_MAX_SIZE=4
export PILOT_TYPE_GUIDED_TAIL_BANK=false
export PILOT_TYPE_GUIDED_MIN_SUPPORT=1
export PILOT_TYPE_GUIDED_MAX_LEAF_GROUPS=40
exec bash "${SCRIPT_DIR}/_run_one.sh" "$@"

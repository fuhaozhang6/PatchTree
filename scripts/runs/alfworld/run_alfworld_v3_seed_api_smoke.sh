#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_ROOT}"

# V3: leaf -> LLM-planned middle nodes -> root. Root fallback uses middle-node
# children instead of jumping directly back to leaf clusters.
export TYPE_GUIDED_TREE_DEPTH="${TYPE_GUIDED_TREE_DEPTH:-3}"
export TYPE_GUIDED_LEAF_FALLBACK="${TYPE_GUIDED_LEAF_FALLBACK:-true}"

# Small ALFWorld smoke defaults. Override any of these from the environment for
# a larger run or ablation.
export NUM_EPOCHS="${NUM_EPOCHS:-1}"
export BATCH_SIZE="${BATCH_SIZE:-2}"
export LIMIT="${LIMIT:-2}"
export SEL_ENV_NUM="${SEL_ENV_NUM:-2}"
export TEST_ENV_NUM="${TEST_ENV_NUM:-0}"
export EVAL_TEST="${EVAL_TEST:-false}"
export MAX_STEPS="${MAX_STEPS:-4}"

export TYPE_GUIDED_MIN_SUPPORT="${TYPE_GUIDED_MIN_SUPPORT:-1}"
export TYPE_GUIDED_MAX_LEAF_GROUPS="${TYPE_GUIDED_MAX_LEAF_GROUPS:-4}"
export TYPE_GUIDED_ROLLOUT_REPEATS="${TYPE_GUIDED_ROLLOUT_REPEATS:-2}"
export TYPE_GUIDED_MAX_PATCH_RECORDS="${TYPE_GUIDED_MAX_PATCH_RECORDS:-4}"
export TYPE_GUIDED_CLUSTERING="${TYPE_GUIDED_CLUSTERING:-true}"
export TYPE_GUIDED_CLUSTER_TARGET_SIZE="${TYPE_GUIDED_CLUSTER_TARGET_SIZE:-2}"
export TYPE_GUIDED_CLUSTER_MAX_SIZE="${TYPE_GUIDED_CLUSTER_MAX_SIZE:-4}"

export OPTIMIZER_MODEL="${OPTIMIZER_MODEL:-doubao-seed-2-0-pro-260215}"
export TARGET_MODEL="${TARGET_MODEL:-doubao-seed-1-6-flash-250828}"
export TS="${TS:-alfworld_v3_smoke_$(date +%Y%m%d_%H%M%S)}"

echo "[info] Launching ALFWorld V3 smoke: optimizer=${OPTIMIZER_MODEL}, target=${TARGET_MODEL}, tree_depth=${TYPE_GUIDED_TREE_DEPTH}"
exec bash "${PROJECT_ROOT}/scripts/runs/alfworld/run_alfworld_seed_api_smoke.sh" "$@"

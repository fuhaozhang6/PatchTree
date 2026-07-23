#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_ROOT}"

# Full ALFWorld V3 run:
#   leaf patches -> LLM-planned middle nodes -> root patch.
# If the root fails, fallback evaluates middle-node children rather than
# jumping directly back to leaf clusters.
export TYPE_GUIDED_TREE_DEPTH="${TYPE_GUIDED_TREE_DEPTH:-3}"
export TYPE_GUIDED_LEAF_FALLBACK="${TYPE_GUIDED_LEAF_FALLBACK:-true}"

# Full split defaults. ALFWorld's local split manifest is small enough to use
# complete train/val/test by default; override from the environment for budgeted
# runs.
export NUM_EPOCHS="${NUM_EPOCHS:-4}"
export BATCH_SIZE="${BATCH_SIZE:-8}"
export LIMIT="${LIMIT:-0}"
export SEL_ENV_NUM="${SEL_ENV_NUM:-0}"
export TEST_ENV_NUM="${TEST_ENV_NUM:-0}"
export EVAL_TEST="${EVAL_TEST:-true}"
export MAX_STEPS="${MAX_STEPS:-50}"

# Moderate parallelism for ALFWorld target calls. Raise these only if the API
# quota and local ALFWorld runtime can keep up.
export API_MAX_CONCURRENCY="${API_MAX_CONCURRENCY:-8}"
export WORKERS="${WORKERS:-4}"
export MAX_API_WORKERS="${MAX_API_WORKERS:-4}"
export ANALYST_WORKERS="${ANALYST_WORKERS:-8}"

# V3/type-guided settings for a full run.
export TYPE_GUIDED_MIN_SUPPORT="${TYPE_GUIDED_MIN_SUPPORT:-1}"
export TYPE_GUIDED_MAX_LEAF_GROUPS="${TYPE_GUIDED_MAX_LEAF_GROUPS:-8}"
export TYPE_GUIDED_ROLLOUT_REPEATS="${TYPE_GUIDED_ROLLOUT_REPEATS:-3}"
export TYPE_GUIDED_MAX_PATCH_RECORDS="${TYPE_GUIDED_MAX_PATCH_RECORDS:-24}"
export TYPE_GUIDED_CLUSTERING="${TYPE_GUIDED_CLUSTERING:-false}"
export TYPE_GUIDED_CLUSTER_TARGET_SIZE="${TYPE_GUIDED_CLUSTER_TARGET_SIZE:-4}"
export TYPE_GUIDED_CLUSTER_MAX_SIZE="${TYPE_GUIDED_CLUSTER_MAX_SIZE:-8}"

export OPTIMIZER_MODEL="${OPTIMIZER_MODEL:-doubao-seed-2-0-pro-260215}"
export TARGET_MODEL="${TARGET_MODEL:-doubao-seed-1-6-flash-250828}"
export RUN_NAME="${RUN_NAME:-Seed API full V3}"
export RUN_SLUG="${RUN_SLUG:-skillopt_alfworld_v3_seed_api_full}"
export TS="${TS:-alfworld_v3_full_$(date +%Y%m%d_%H%M%S)}"

echo "[info] Launching ALFWorld V3 full run: optimizer=${OPTIMIZER_MODEL}, target=${TARGET_MODEL}, tree_depth=${TYPE_GUIDED_TREE_DEPTH}"
exec bash "${PROJECT_ROOT}/scripts/runs/alfworld/run_alfworld_seed_api_smoke.sh" "$@"

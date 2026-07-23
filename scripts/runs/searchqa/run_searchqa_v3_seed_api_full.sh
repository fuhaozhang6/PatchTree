#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_ROOT}"

# SearchQA V3/latest run:
#   optimizer / teacher = Doubao Seed 2.0 Pro
#   target / student    = Doubao Seed 1.6 Flash
#   merge path          = type-guided v2 with depth-3 V3 tree
#
# Recommended smoke before a larger run:
#   TRAIN_SIZE=8 NUM_EPOCHS=1 BATCH_SIZE=4 SEL_ENV_NUM=8 TEST_ENV_NUM=0 EVAL_TEST=false \
#     bash scripts/runs/searchqa/run_searchqa_v3_seed_api_full.sh
#
# For full SearchQA test evaluation, override:
#   TEST_ENV_NUM=1400 bash scripts/runs/searchqa/run_searchqa_v3_seed_api_full.sh

export OPTIMIZER_MODEL="${OPTIMIZER_MODEL:-doubao-seed-2-0-pro-260215}"
export TARGET_MODEL="${TARGET_MODEL:-doubao-seed-1-6-flash-250828}"

# Large SearchQA defaults from the base launcher. They are intentionally bounded
# for cost; the local split has train=400, val=200, test=1400.
export NUM_EPOCHS="${NUM_EPOCHS:-4}"
export TRAIN_SIZE="${TRAIN_SIZE:-400}"
export BATCH_SIZE="${BATCH_SIZE:-40}"
export ACCUMULATION="${ACCUMULATION:-1}"
export SEL_ENV_NUM="${SEL_ENV_NUM:-200}"
export TEST_ENV_NUM="${TEST_ENV_NUM:-400}"
export EVAL_TEST="${EVAL_TEST:-true}"
export LIMIT="${LIMIT:-0}"

# Ark API concurrency. Override down if the account quota is tighter.
export API_MAX_CONCURRENCY="${API_MAX_CONCURRENCY:-48}"
export WORKERS="${WORKERS:-48}"
export ANALYST_WORKERS="${ANALYST_WORKERS:-32}"

# V3/latest type-guided settings.
export TYPE_GUIDED_TREE_DEPTH="${TYPE_GUIDED_TREE_DEPTH:-3}"
export TYPE_GUIDED_LEAF_FALLBACK="${TYPE_GUIDED_LEAF_FALLBACK:-true}"
export TYPE_GUIDED_MIN_SUPPORT="${TYPE_GUIDED_MIN_SUPPORT:-2}"
export TYPE_GUIDED_MAX_LEAF_GROUPS="${TYPE_GUIDED_MAX_LEAF_GROUPS:-10}"
export TYPE_GUIDED_ROLLOUT_REPEATS="${TYPE_GUIDED_ROLLOUT_REPEATS:-3}"
export TYPE_GUIDED_MAX_PATCH_RECORDS="${TYPE_GUIDED_MAX_PATCH_RECORDS:-32}"
export TYPE_GUIDED_TAU_SUCC="${TYPE_GUIDED_TAU_SUCC:-1.0}"

# V2.3-light features.
export TYPE_GUIDED_CLUSTERING="${TYPE_GUIDED_CLUSTERING:-false}"
export TYPE_GUIDED_CLUSTER_TARGET_SIZE="${TYPE_GUIDED_CLUSTER_TARGET_SIZE:-6}"
export TYPE_GUIDED_CLUSTER_MAX_SIZE="${TYPE_GUIDED_CLUSTER_MAX_SIZE:-10}"
export TYPE_GUIDED_FALLBACK_RECONCILE="${TYPE_GUIDED_FALLBACK_RECONCILE:-llm_fuse}"
export TYPE_GUIDED_FALLBACK_RECONCILE_MIN_CHILDREN="${TYPE_GUIDED_FALLBACK_RECONCILE_MIN_CHILDREN:-2}"
export TYPE_GUIDED_TAIL_BANK="${TYPE_GUIDED_TAIL_BANK:-true}"
export TYPE_GUIDED_TAIL_WINDOW_EPOCHS="${TYPE_GUIDED_TAIL_WINDOW_EPOCHS:-3}"
export TYPE_GUIDED_TAIL_MIN_SUPPORT="${TYPE_GUIDED_TAIL_MIN_SUPPORT:-2}"
export TYPE_GUIDED_TAIL_MAX_RECORDS="${TYPE_GUIDED_TAIL_MAX_RECORDS:-32}"
export TYPE_GUIDED_TAIL_MAX_LEAF_GROUPS="${TYPE_GUIDED_TAIL_MAX_LEAF_GROUPS:-4}"
export TYPE_GUIDED_TAIL_REQUIRE_CROSS_STEP="${TYPE_GUIDED_TAIL_REQUIRE_CROSS_STEP:-true}"

export GATE_METRIC="${GATE_METRIC:-mixed}"
export GATE_MIXED_WEIGHT="${GATE_MIXED_WEIGHT:-0.5}"
export REASONING_EFFORT="${REASONING_EFFORT:-}"
export REWRITE_REASONING_EFFORT="${REWRITE_REASONING_EFFORT:-medium}"
export TARGET_MAX_COMPLETION_TOKENS="${TARGET_MAX_COMPLETION_TOKENS:-16384}"
export TS="${TS:-searchqa_v3_seed_api_full_$(date +%Y%m%d_%H%M%S)}"

echo "[info] Launching SearchQA V3 run: optimizer=${OPTIMIZER_MODEL}, target=${TARGET_MODEL}, tree_depth=${TYPE_GUIDED_TREE_DEPTH}"
exec bash "${PROJECT_ROOT}/scripts/runs/searchqa/run_searchqa_seed_api_large.sh" "$@"

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_livemath_ark_common.sh
source "${SCRIPT_DIR}/_livemath_ark_common.sh"

export RUN_ID="${RUN_ID:-livemath_ark_dynamic_main_$(date +%Y%m%d_%H%M%S)}"
export OUT_BASE="${OUT_BASE:-${PROJECT_ROOT}/outputs/${RUN_ID}}"
export LIVEMATH_ARK_SUITE_ROOT="${LIVEMATH_ARK_SUITE_ROOT:-${OUT_BASE}}"

MAIN_EPOCHS="${MAIN_EPOCHS:-4}"
MAIN_CFG="${MAIN_CFG:-optimizer.type_guided_tree_builder=recursive optimizer.type_guided_max_tree_depth=4 optimizer.type_guided_merge_target_children=3 optimizer.type_guided_merge_max_children=4 optimizer.type_guided_top_mode=auto optimizer.type_guided_leaf_fallback=true optimizer.type_guided_fallback_min_leaf_coverage=1 optimizer.type_guided_validation_budget=16 optimizer.type_guided_fallback_reconcile=llm_fuse}"

require_layout
configure_models
trap cleanup_datasets EXIT INT TERM

print_comparison_header "17-livemath-dynamic-main" "livemathematicianbench"
echo "  concurrency:        target=${LIVEMATH_ARK_WORKERS} analyst=${LIVEMATH_ARK_ANALYST_WORKERS}"
echo "  epochs:             ${MAIN_EPOCHS}"

run_livemath_ark_case "dynamic_auto_main" "${MAIN_EPOCHS}" initial "${MAIN_CFG}"
finish_livemath_ark_suite

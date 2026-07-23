#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_livemath_ark_common.sh
source "${SCRIPT_DIR}/_livemath_ark_common.sh"

export RUN_ID="${RUN_ID:-livemath_ark_dynamic_ablation_$(date +%Y%m%d_%H%M%S)}"
export OUT_BASE="${OUT_BASE:-${PROJECT_ROOT}/outputs/${RUN_ID}}"
export LIVEMATH_ARK_SUITE_ROOT="${LIVEMATH_ARK_SUITE_ROOT:-${OUT_BASE}}"

ABLATION_EPOCHS="${ABLATION_EPOCHS:-4}"
# Core rows answer the dynamic-tree design questions. Runs are sequential, so
# the suite never multiplies the requested concurrency of 48 across rows.
ABLATION_ROWS="${ABLATION_ROWS:-dynamic_auto fixed_real_root dynamic_real_root dynamic_virtual_root no_recursive_fallback min_support_2}"
if comparison_truthy "${DO_EXTENDED:-0}"; then
  ABLATION_ROWS+=" fanout_2 max_depth_2 validation_budget_4"
fi

DYNAMIC_BASE="optimizer.type_guided_tree_builder=recursive optimizer.type_guided_max_tree_depth=4 optimizer.type_guided_merge_target_children=3 optimizer.type_guided_merge_max_children=4 optimizer.type_guided_top_mode=auto optimizer.type_guided_leaf_fallback=true optimizer.type_guided_fallback_min_leaf_coverage=1 optimizer.type_guided_validation_budget=16 optimizer.type_guided_fallback_reconcile=llm_fuse"

require_layout
configure_models
trap cleanup_datasets EXIT INT TERM

print_comparison_header "18-livemath-dynamic-ablation" "livemathematicianbench"
echo "  concurrency:        target=${LIVEMATH_ARK_WORKERS} analyst=${LIVEMATH_ARK_ANALYST_WORKERS}"
echo "  epochs/row:         ${ABLATION_EPOCHS}"
echo "  rows:               ${ABLATION_ROWS}"

for row in ${ABLATION_ROWS}; do
  case "${row}" in
    dynamic_auto)
      cfg="${DYNAMIC_BASE}"
      ;;
    fixed_real_root)
      # Legacy path: leaf -> one genuinely fused root, with the old fixed builder.
      cfg="optimizer.type_guided_tree_builder=fixed optimizer.type_guided_tree_depth=2 optimizer.type_guided_top_mode=real_root optimizer.type_guided_leaf_fallback=true"
      ;;
    dynamic_real_root)
      cfg="${DYNAMIC_BASE} optimizer.type_guided_top_mode=real_root"
      ;;
    dynamic_virtual_root)
      cfg="${DYNAMIC_BASE} optimizer.type_guided_top_mode=virtual_root"
      ;;
    no_recursive_fallback)
      cfg="${DYNAMIC_BASE} optimizer.type_guided_leaf_fallback=false"
      ;;
    min_support_2)
      cfg="${DYNAMIC_BASE} optimizer.type_guided_min_support=2"
      ;;
    fanout_2)
      cfg="${DYNAMIC_BASE} optimizer.type_guided_merge_target_children=2 optimizer.type_guided_merge_max_children=2"
      ;;
    max_depth_2)
      cfg="${DYNAMIC_BASE} optimizer.type_guided_max_tree_depth=2"
      ;;
    validation_budget_4)
      cfg="${DYNAMIC_BASE} optimizer.type_guided_validation_budget=4"
      ;;
    *)
      comparison_fail "Unknown ABLATION_ROWS entry: ${row}"
      ;;
  esac
  run_livemath_ark_case "${row}" "${ABLATION_EPOCHS}" initial "${cfg}"
done

finish_livemath_ark_suite

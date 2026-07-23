#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_livemath_ark_common.sh
source "${SCRIPT_DIR}/_livemath_ark_common.sh"

export RUN_ID="${RUN_ID:-livemath_ark_system_prompt_only_$(date +%Y%m%d_%H%M%S)}"
export OUT_BASE="${OUT_BASE:-${PROJECT_ROOT}/outputs/${RUN_ID}}"
export LIVEMATH_ARK_SUITE_ROOT="${LIVEMATH_ARK_SUITE_ROOT:-${OUT_BASE}}"

require_layout
configure_models
trap cleanup_datasets EXIT INT TERM

print_comparison_header "15-livemath-system-prompt-only" "livemathematicianbench"
echo "  concurrency:        target=${LIVEMATH_ARK_WORKERS} analyst=${LIVEMATH_ARK_ANALYST_WORKERS}"
echo "  epochs:             0"

# Dataset rollout system prompt only: no injected Skill and no training.
run_livemath_ark_case "system_prompt_only" 0 system ""

finish_livemath_ark_suite

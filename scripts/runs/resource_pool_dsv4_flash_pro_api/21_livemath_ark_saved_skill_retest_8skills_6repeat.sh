#!/usr/bin/env bash
set -euo pipefail

# Extra pure TEST reevaluation for the eight LiveMath settings that were
# evaluated in the 5-repeat pass:
#   system_prompt_only, init_skill, dynamic_auto, fixed_real_root,
#   dynamic_real_root, dynamic_virtual_root, no_recursive_fallback,
#   min_support_2
#
# This wrapper intentionally reuses 20_livemath_ark_saved_skill_retest.sh so the
# model config, manifest paths, Ark options, and summary generation stay in one
# place. Override SOURCE_RUN_ROOT if you need to evaluate a non-latest suite.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export REPEATS="${REPEATS:-6}"
export WORKERS="${WORKERS:-96}"
export RUN_ID="${RUN_ID:-livemath_saved_skill_retest_8skills_extra6_$(date +%Y%m%d_%H%M%S)}"

exec bash "${SCRIPT_DIR}/20_livemath_ark_saved_skill_retest.sh" "$@"

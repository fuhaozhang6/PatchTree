#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "${SCRIPT_DIR}/_common.sh"

# Single-dataset lane: livemathematicianbench only, via the DeepSeek official API
# (target=deepseek-v4-flash, optimizer=deepseek-v4-pro). See _common.sh for wiring.

export RUN_ID="${RUN_ID:-skillopt_tree_livemath_dsv4_flash_pro_$(date +%Y%m%d_%H%M%S)}"
export OUT_BASE="${OUT_BASE:-${PROJECT_ROOT}/outputs/${RUN_ID}}"
LIVEMATH_WORKERS="${LIVEMATH_WORKERS:-16}"
LIVEMATH_ANALYST_WORKERS="${LIVEMATH_ANALYST_WORKERS:-16}"
LIVEMATH_EXEC_TIMEOUT="${LIVEMATH_EXEC_TIMEOUT:-1800}"

require_layout
configure_models
trap cleanup_datasets EXIT INT TERM
print_comparison_header "08-livemath" "livemathematicianbench"
echo "  workers:            ${LIVEMATH_WORKERS}"
echo "  analysts:           ${LIVEMATH_ANALYST_WORKERS}"
echo "  request timeout:    ${LIVEMATH_EXEC_TIMEOUT}s"
run_dataset livemathematicianbench \
  --workers "${LIVEMATH_WORKERS}" \
  --analyst_workers "${LIVEMATH_ANALYST_WORKERS}" \
  --exec_timeout "${LIVEMATH_EXEC_TIMEOUT}" \
  "$@"

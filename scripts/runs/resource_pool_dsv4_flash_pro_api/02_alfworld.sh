#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "${SCRIPT_DIR}/_common.sh"

export RUN_ID="${RUN_ID:-skillopt_tree_alfworld_dsv4_flash_pro_$(date +%Y%m%d_%H%M%S)}"
export OUT_BASE="${OUT_BASE:-${PROJECT_ROOT}/outputs/${RUN_ID}}"
# Silence the alfworld library's per-env tqdm scan bars in this process and in
# the forked env workers (they inherit the environment). Set TQDM_DISABLE=0 to
# show them again for debugging.
export TQDM_DISABLE="${TQDM_DISABLE:-1}"
# ALFWorld is a multi-turn agent task: rollout dominates wall-time. Keep env
# workers high enough to feed the batch scheduler, but keep model API workers
# conservative; each episode can make many sequential target calls.
ALFWORLD_WORKERS="${ALFWORLD_WORKERS:-32}"
ALFWORLD_API_WORKERS="${ALFWORLD_API_WORKERS:-8}"
ALFWORLD_MAX_STEPS="${ALFWORLD_MAX_STEPS:-30}"

require_layout
configure_models
require_alfworld_environment
trap cleanup_datasets EXIT INT TERM
print_comparison_header "02-alfworld" "alfworld"
echo "  ALFWORLD_DATA:      ${ALFWORLD_DATA}"
echo "  env workers:        ${ALFWORLD_WORKERS}"
echo "  API workers:        ${ALFWORLD_API_WORKERS}"
echo "  max steps/episode:  ${ALFWORLD_MAX_STEPS}"
run_dataset alfworld \
  --workers "${ALFWORLD_WORKERS}" \
  --max_api_workers "${ALFWORLD_API_WORKERS}" \
  --max_steps "${ALFWORLD_MAX_STEPS}" \
  "$@"

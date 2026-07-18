#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "${SCRIPT_DIR}/_common.sh"

configure_epoch1_smoke livemath_officeqa_ark

export LIVEMATH_BATCH_SIZE=2
export LIVEMATH_WORKERS=8
export LIVEMATH_ANALYST_WORKERS=4
export LIVEMATH_SEL_ENV_NUM=1
export LIVEMATH_TEST_ENV_NUM=0
export LIVEMATH_EXEC_TIMEOUT=300

export OFFICEQA_BATCH_SIZE=2
export OFFICEQA_WORKERS=8
export OFFICEQA_ANALYST_WORKERS=4
export OFFICEQA_SEL_ENV_NUM=1
export OFFICEQA_TEST_ENV_NUM=0
export OFFICEQA_USE_LOCAL_TOOLS=true
export OFFICEQA_SEARCH_MODE=offline
export VLLM_ENABLE_AUTO_TOOL_CHOICE=auto
export VLLM_TOOL_CALL_PARSER=qwen3_coder
export VLLM_REASONING_PARSER=qwen3

echo "[smoke] LiveMath + OfficeQA: each has 2 train items, 1 step, 1 complete epoch"
bash "${SMOKE_PROJECT_ROOT}/scripts/runs/resource_pool_4x_l20/04_livemath_officeqa_ark.sh" "$@"
verify_epoch1_smoke livemath officeqa

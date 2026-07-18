#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "${SCRIPT_DIR}/_common.sh"

configure_epoch1_smoke searchqa_official

export SEARCHQA_BATCH_SIZE=2
export SEARCHQA_WORKERS=8
export SEARCHQA_ANALYST_WORKERS=4
export SEARCHQA_SEL_ENV_NUM=1
export SEARCHQA_TEST_ENV_NUM=0
export SEARCHQA_TYPE_GUIDED_FALLBACK_SEL_ENV_NUM=1
export SEARCHQA_EXEC_TIMEOUT=300

echo "[smoke] SearchQA: 2 train items, 1 step, 1 complete epoch, val=1, test=off"
bash "${SMOKE_PROJECT_ROOT}/scripts/runs/resource_pool_4x_l20/01_searchqa_deepseek_official.sh" "$@"
verify_epoch1_smoke searchqa

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "${SCRIPT_DIR}/_common.sh"

configure_epoch1_smoke docvqa_spreadsheet_ark

export DOCVQA_BATCH_SIZE=2
export DOCVQA_WORKERS=8
export DOCVQA_ANALYST_WORKERS=4
export DOCVQA_SEL_ENV_NUM=1
export DOCVQA_TEST_ENV_NUM=0
export DOCVQA_EXEC_TIMEOUT=300
export DOCVQA_IMAGE_DETAIL=auto

export SPREADSHEETBENCH_BATCH_SIZE=2
export SPREADSHEETBENCH_WORKERS=8
export SPREADSHEETBENCH_ANALYST_WORKERS=4
export SPREADSHEETBENCH_SEL_ENV_NUM=1
export SPREADSHEETBENCH_TEST_ENV_NUM=0
export SPREADSHEETBENCH_EXEC_TIMEOUT=1200
export SPREADSHEETBENCH_LLM_TIMEOUT=300
export SPREADSHEETBENCH_MAX_TURNS=30

echo "[smoke] DocVQA + SpreadsheetBench: each has 2 train items, 1 step, 1 complete epoch"
bash "${SMOKE_PROJECT_ROOT}/scripts/runs/resource_pool_4x_l20/03_docvqa_spreadsheet_ark.sh" "$@"
verify_epoch1_smoke docvqa spreadsheetbench

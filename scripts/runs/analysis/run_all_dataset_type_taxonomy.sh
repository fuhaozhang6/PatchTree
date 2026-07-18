#!/usr/bin/env bash
set -euo pipefail

# Classify question_type and revision_type priors for every item in all six
# datasets and all train/val/test splits. This performs no Skill update, no
# training, and needs no GPU.
#
# DeepSeek official:
#   export DEEPSEEK_API_KEY='...'
#   bash scripts/runs/analysis/run_all_dataset_type_taxonomy.sh
#
# Volcano Ark:
#   export ARK_API_KEY='...'
#   TAXONOMY_API_SOURCE=ark \
#     bash scripts/runs/analysis/run_all_dataset_type_taxonomy.sh
#
# Cheap configuration/data smoke test:
#   DRY_RUN=1 LIMIT_PER_SPLIT=2 \
#     bash scripts/runs/analysis/run_all_dataset_type_taxonomy.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
AUDIT_PY="${PROJECT_ROOT}/scripts/tools/audit_dataset_type_taxonomy.py"
TAXONOMY_API_SOURCE="${TAXONOMY_API_SOURCE:-deepseek}"
DATASETS="${DATASETS:-searchqa spreadsheetbench officeqa docvqa livemath alfworld}"
SPLITS="${SPLITS:-train val test}"
TAXONOMY_WORKERS="${TAXONOMY_WORKERS:-32}"
TAXONOMY_TIMEOUT_SECONDS="${TAXONOMY_TIMEOUT_SECONDS:-300}"
TAXONOMY_MAX_TOKENS="${TAXONOMY_MAX_TOKENS:-700}"
TAXONOMY_MAX_ITEM_CHARS="${TAXONOMY_MAX_ITEM_CHARS:-7000}"
TAXONOMY_MAX_RETRIES="${TAXONOMY_MAX_RETRIES:-4}"
LIMIT_PER_SPLIT="${LIMIT_PER_SPLIT:-0}"
DRY_RUN="${DRY_RUN:-0}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/outputs/dataset_type_taxonomy_${TS}}"

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

truthy() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

case "${TAXONOMY_API_SOURCE}" in
  deepseek)
    export TAXONOMY_BASE_URL="${TAXONOMY_BASE_URL:-${DEEPSEEK_BASE_URL:-https://api.deepseek.com}}"
    export TAXONOMY_MODEL="${TAXONOMY_MODEL:-${DEEPSEEK_OFFICIAL_MODEL:-deepseek-v4-pro}}"
    export TAXONOMY_API_KEY="${TAXONOMY_API_KEY:-${DEEPSEEK_API_KEY:-${DS_API_KEY:-${DEEPSEEK_OFFICIAL_API_KEY:-}}}}"
    ;;
  ark)
    export TAXONOMY_BASE_URL="${TAXONOMY_BASE_URL:-${ARK_BASE_URL:-https://ark.cn-beijing.volces.com/api/v3}}"
    export TAXONOMY_MODEL="${TAXONOMY_MODEL:-${ARK_OPTIMIZER_MODEL:-deepseek-v4-pro-260425}}"
    export TAXONOMY_API_KEY="${TAXONOMY_API_KEY:-${ARK_API_KEY:-}}"
    ;;
  *)
    fail "Unknown TAXONOMY_API_SOURCE=${TAXONOMY_API_SOURCE}; expected deepseek or ark."
    ;;
esac

command -v "${PYTHON_BIN}" >/dev/null 2>&1 || fail "Python not found: ${PYTHON_BIN}"
[[ -f "${AUDIT_PY}" ]] || fail "Audit program not found: ${AUDIT_PY}"
if ! truthy "${DRY_RUN}" && [[ -z "${TAXONOMY_API_KEY}" ]]; then
  fail "API key is empty for source=${TAXONOMY_API_SOURCE}."
fi

args=(
  "${PYTHON_BIN}" -u "${AUDIT_PY}"
  --datasets "${DATASETS}"
  --splits "${SPLITS}"
  --output-dir "${OUT_DIR}"
  --base-url "${TAXONOMY_BASE_URL}"
  --model "${TAXONOMY_MODEL}"
  --api-key-env TAXONOMY_API_KEY
  --workers "${TAXONOMY_WORKERS}"
  --timeout-seconds "${TAXONOMY_TIMEOUT_SECONDS}"
  --max-tokens "${TAXONOMY_MAX_TOKENS}"
  --max-item-chars "${TAXONOMY_MAX_ITEM_CHARS}"
  --max-retries "${TAXONOMY_MAX_RETRIES}"
  --limit-per-split "${LIMIT_PER_SPLIT}"
)
if truthy "${DRY_RUN}"; then
  args+=(--dry-run)
fi

echo "============================================================"
echo "  All-dataset type taxonomy audit"
echo "============================================================"
echo "  source/model: ${TAXONOMY_API_SOURCE}/${TAXONOMY_MODEL}"
echo "  datasets:     ${DATASETS}"
echo "  splits:       ${SPLITS}"
echo "  workers:      ${TAXONOMY_WORKERS}"
echo "  limit/split:  ${LIMIT_PER_SPLIT} (0 means all)"
echo "  output:       ${OUT_DIR}"
echo "  dry_run:      ${DRY_RUN}"
echo "============================================================"

exec "${args[@]}"

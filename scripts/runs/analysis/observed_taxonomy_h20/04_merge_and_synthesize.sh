#!/usr/bin/env bash
set -euo pipefail

# CPU/API post-processing after all four H20 jobs finish. This does not launch
# vLLM and does not modify dataset prompts.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${HERE}/../../../.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_ID="${RUN_ID:-}"
INPUT_ROOTS="${INPUT_ROOTS:-}"
if [[ -z "${INPUT_ROOTS}" ]]; then
  [[ -n "${RUN_ID}" ]] || {
    echo "ERROR: set RUN_ID or INPUT_ROOTS" >&2
    exit 1
  }
  INPUT_ROOTS="${PROJECT_ROOT}/outputs/observed_taxonomy_h20_${RUN_ID}"
fi
MERGED_DIR="${MERGED_DIR:-${PROJECT_ROOT}/outputs/observed_taxonomy_merged_${RUN_ID:-manual}}"
FEW_SHOT_DIR="${FEW_SHOT_DIR:-${MERGED_DIR}/proposed_few_shots}"
MAX_FEW_SHOTS="${MAX_FEW_SHOTS:-8}"
MAX_EVIDENCE_PER_PAIR="${MAX_EVIDENCE_PER_PAIR:-3}"
EXPECTED_DATASETS="${EXPECTED_DATASETS:-searchqa spreadsheetbench officeqa docvqa livemathematicianbench alfworld}"
ADJUDICATION_DRAFTS="${ADJUDICATION_DRAFTS:-2}"
SYNTHESIZE="${SYNTHESIZE:-1}"
DRY_RUN="${DRY_RUN:-0}"
OPTIMIZER_SOURCE=deepseek

truthy() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

merge_args=(
  "${PYTHON_BIN}" -u scripts/tools/merge_observed_type_taxonomy.py
  --output-dir "${MERGED_DIR}"
  --max-few-shots "${MAX_FEW_SHOTS}"
  --max-evidence-per-pair "${MAX_EVIDENCE_PER_PAIR}"
  --expected-datasets "${EXPECTED_DATASETS}"
)
for input_root in ${INPUT_ROOTS}; do
  merge_args+=(--input "${input_root}")
done

echo "============================================================"
echo "  Merge and adjudicate observed taxonomy"
echo "============================================================"
echo "  inputs:        ${INPUT_ROOTS}"
echo "  merged:        ${MERGED_DIR}"
echo "  proposed:      ${FEW_SHOT_DIR}"
echo "  max few-shots: ${MAX_FEW_SHOTS}"
echo "  drafts:        ${ADJUDICATION_DRAFTS} + 1 reconciliation"
echo "  synthesize:    ${SYNTHESIZE}"
echo "============================================================"

if truthy "${DRY_RUN}"; then
  printf '[dry-run] merge command:'
  printf ' %q' "${merge_args[@]}"
  printf '\n'
  exit 0
fi

"${merge_args[@]}"
truthy "${SYNTHESIZE}" || exit 0

key="${DEEPSEEK_API_KEY:-${DS_API_KEY:-${DEEPSEEK_OFFICIAL_API_KEY:-}}}"
export AZURE_OPENAI_ENDPOINT="${DEEPSEEK_BASE_URL:-https://api.deepseek.com}"
export AZURE_OPENAI_API_VERSION=openai-compat
export OPTIMIZER_MODEL="${OPTIMIZER_MODEL:-${DEEPSEEK_OFFICIAL_MODEL:-deepseek-v4-pro}}"
[[ -n "${key}" ]] || {
  echo "ERROR: optimizer API key is empty" >&2
  exit 1
}
export AZURE_OPENAI_API_KEY="${key}"
export AZURE_OPENAI_AUTH_MODE=openai_compatible
export OPTIMIZER_AZURE_OPENAI_ENDPOINT="${AZURE_OPENAI_ENDPOINT}"
export OPTIMIZER_AZURE_OPENAI_API_VERSION="${AZURE_OPENAI_API_VERSION}"
export OPTIMIZER_AZURE_OPENAI_API_KEY="${AZURE_OPENAI_API_KEY}"
export OPTIMIZER_AZURE_OPENAI_AUTH_MODE="${AZURE_OPENAI_AUTH_MODE}"
export DEEPSEEK_THINKING="${DEEPSEEK_THINKING:-disabled}"

"${PYTHON_BIN}" -u scripts/tools/synthesize_observed_few_shots.py \
  --merged-dir "${MERGED_DIR}" \
  --output-dir "${FEW_SHOT_DIR}" \
  --optimizer-model "${OPTIMIZER_MODEL}" \
  --max-few-shots "${MAX_FEW_SHOTS}" \
  --drafts "${ADJUDICATION_DRAFTS}"

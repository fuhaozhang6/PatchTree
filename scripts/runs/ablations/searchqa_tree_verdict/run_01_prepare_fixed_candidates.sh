#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
# shellcheck source=../../resource_pool_4x_l20/_common.sh
source "${SCRIPT_DIR}/../../resource_pool_4x_l20/_common.sh"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DRY_RUN="${DRY_RUN:-0}"

SOURCE_RUN="${SOURCE_RUN:-${PROJECT_ROOT}/outputs/searchqa_tree_shape_d3_clustering_on_min_support1_all_clusters_tail_off_fallback_off_seed42_20260718_165434/searchqa}"
REPLAY_DIR="${REPLAY_DIR:-${PROJECT_ROOT}/outputs/searchqa_tree_verdict_p6_fixed}"
MAIN_STEP="${MAIN_STEP:-auto}"
FALLBACK_STEP="${FALLBACK_STEP:-auto}"
OPTIMIZER_PROVIDER="${OPTIMIZER_PROVIDER:-deepseek}"

case "${OPTIMIZER_PROVIDER}" in
  deepseek|official)
    configure_deepseek_official
    ;;
  ark|volcano)
    configure_volcano_ark
    ;;
  *)
    resource_fail "OPTIMIZER_PROVIDER must be deepseek or ark"
    ;;
esac

echo "============================================================"
echo "  SearchQA Tree Verdict: prepare fixed candidates"
echo "============================================================"
echo "  source_run:       ${SOURCE_RUN}"
echo "  replay_dir:       ${REPLAY_DIR}"
echo "  main_step:        ${MAIN_STEP}"
echo "  fallback_step:    ${FALLBACK_STEP}"
echo "  optimizer_source: ${OPTIMIZER_PROVIDER}"
echo "  optimizer_model:  ${OPTIMIZER_MODEL}"
echo "  new optimizer work: Flat Root + frozen-Leaf Root only"
echo "============================================================"

prepare_cmd=(
  "${PYTHON_BIN}" -u
  "${PROJECT_ROOT}/scripts/tools/prepare_searchqa_tree_verdict.py"
  --source-run "${SOURCE_RUN}"
  --out-dir "${REPLAY_DIR}"
  --main-step "${MAIN_STEP}"
  --fallback-step "${FALLBACK_STEP}"
  --optimizer-model "${OPTIMIZER_MODEL}"
)

if resource_truthy "${DRY_RUN}"; then
  printf '[dry-run]'
  printf ' %q' "${prepare_cmd[@]}"
  printf '\n'
  exit 0
fi

[[ -d "${SOURCE_RUN}/steps" ]] || \
  resource_fail "P6 source output is missing: ${SOURCE_RUN}"
if [[ -f "${REPLAY_DIR}/replay_manifest.json" ]]; then
  resource_fail \
    "Replay is already frozen at ${REPLAY_DIR}; do not regenerate it. Run the evaluation script."
fi
[[ -n "${OPTIMIZER_AZURE_OPENAI_API_KEY:-}" ]] || \
  resource_fail "Optimizer API key is empty."

mkdir -p "${REPLAY_DIR}"
"${prepare_cmd[@]}" 2>&1 | tee "${REPLAY_DIR}/prepare.log"

[[ -f "${REPLAY_DIR}/main/skill_manifest.tsv" ]] || \
  resource_fail "Main skill manifest was not generated."
[[ -f "${REPLAY_DIR}/topdown/phase1_val_manifest.tsv" ]] || \
  resource_fail "Top-down phase-1 manifest was not generated."

echo ""
echo "[done] fixed candidates: ${REPLAY_DIR}"
echo "[next] REPLAY_DIR=${REPLAY_DIR} bash ${SCRIPT_DIR}/run_02_eval_one_gpu.sh"

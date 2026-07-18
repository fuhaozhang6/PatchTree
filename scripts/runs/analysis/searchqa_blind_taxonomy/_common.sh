#!/usr/bin/env bash

# Shared settings for SearchQA blind-taxonomy CPU/API stages. Source only.

BLIND_RUN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${BLIND_RUN_DIR}/../../../.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_ID="${RUN_ID:-blind_v1}"
OBSERVED_ROOT="${OBSERVED_ROOT:-${PROJECT_ROOT}/outputs/observed_taxonomy_h20_taxonomy_v1}"
OUT_BASE="${OUT_BASE:-${PROJECT_ROOT}/outputs/searchqa_blind_taxonomy_${RUN_ID}}"
LOG_BASE="${LOG_BASE:-${PROJECT_ROOT}/logs/searchqa_blind_taxonomy_${RUN_ID}}"
OPTIMIZER_MODEL="${OPTIMIZER_MODEL:-${DEEPSEEK_OFFICIAL_MODEL:-deepseek-v4-pro}}"
DRY_RUN="${DRY_RUN:-0}"

blind_truthy() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

blind_fail() {
  echo "ERROR: $*" >&2
  exit 1
}

configure_deepseek_official() {
  local key="${DEEPSEEK_API_KEY:-${DS_API_KEY:-${DEEPSEEK_OFFICIAL_API_KEY:-}}}"
  if [[ -z "${key}" ]] && ! blind_truthy "${DRY_RUN}"; then
    blind_fail "DEEPSEEK_API_KEY is empty"
  fi
  export AZURE_OPENAI_ENDPOINT="${DEEPSEEK_BASE_URL:-https://api.deepseek.com}"
  export AZURE_OPENAI_API_VERSION=openai-compat
  export AZURE_OPENAI_API_KEY="${key}"
  export AZURE_OPENAI_AUTH_MODE=openai_compatible
  export OPTIMIZER_AZURE_OPENAI_ENDPOINT="${AZURE_OPENAI_ENDPOINT}"
  export OPTIMIZER_AZURE_OPENAI_API_VERSION="${AZURE_OPENAI_API_VERSION}"
  export OPTIMIZER_AZURE_OPENAI_API_KEY="${AZURE_OPENAI_API_KEY}"
  export OPTIMIZER_AZURE_OPENAI_AUTH_MODE="${AZURE_OPENAI_AUTH_MODE}"
  export OPTIMIZER_MODEL
  export DEEPSEEK_THINKING="${DEEPSEEK_THINKING:-disabled}"
}

mkdir -p "${OUT_BASE}" "${LOG_BASE}"
command -v "${PYTHON_BIN}" >/dev/null 2>&1 || blind_fail "Python not found: ${PYTHON_BIN}"

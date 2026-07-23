#!/usr/bin/env bash

# Shared helpers for the LiveMath DeepSeek-v4 Flash/Pro ablation suite.
# Reuses the resource-pool runner but replaces its DeepSeek-official model
# configuration with Volcano Ark defaults.

LIVEMATH_ARK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "${LIVEMATH_ARK_DIR}/_common.sh"

configure_models() {
  local endpoint key
  endpoint="${ARK_BASE_URL:-https://ark.cn-beijing.volces.com/api/v3}"
  key="${ARK_API_KEY:-${AZURE_OPENAI_API_KEY:-}}"
  if [[ -z "${key}" ]] && ! comparison_truthy "${DRY_RUN:-0}"; then
    comparison_fail "ARK_API_KEY is required for the Volcano Ark LiveMath suite."
  fi

  export OPTIMIZER_SOURCE="volcano_ark"
  export OPTIMIZER_PROVIDER_LABEL="Volcano Ark"
  export TARGET_PROVIDER_LABEL="Volcano Ark"
  export OPTIMIZER_MODEL="${OPTIMIZER_MODEL:-${ARK_OPTIMIZER_MODEL:-deepseek-v4-pro-260425}}"
  export TARGET_MODEL="${TARGET_MODEL:-${ARK_TARGET_MODEL:-deepseek-v4-flash-260425}}"

  export AZURE_OPENAI_ENDPOINT="${endpoint}"
  export AZURE_OPENAI_API_KEY="${key}"
  export AZURE_OPENAI_AUTH_MODE=openai_compatible
  export AZURE_OPENAI_API_VERSION="${ARK_API_VERSION:-2024-12-01-preview}"

  export OPTIMIZER_AZURE_OPENAI_ENDPOINT="${endpoint}"
  export OPTIMIZER_AZURE_OPENAI_API_KEY="${key}"
  export OPTIMIZER_AZURE_OPENAI_AUTH_MODE=openai_compatible
  export OPTIMIZER_AZURE_OPENAI_API_VERSION="${AZURE_OPENAI_API_VERSION}"

  export TARGET_AZURE_OPENAI_ENDPOINT="${TARGET_AZURE_OPENAI_ENDPOINT:-${endpoint}}"
  export TARGET_AZURE_OPENAI_API_KEY="${TARGET_AZURE_OPENAI_API_KEY:-${key}}"
  export TARGET_AZURE_OPENAI_AUTH_MODE="${TARGET_AZURE_OPENAI_AUTH_MODE:-openai_compatible}"
  export TARGET_AZURE_OPENAI_API_VERSION="${TARGET_AZURE_OPENAI_API_VERSION:-${AZURE_OPENAI_API_VERSION}}"

  export DEEPSEEK_THINKING="${DEEPSEEK_THINKING:-disabled}"
  export DEEPSEEK_OFFICIAL_THINKING="${DEEPSEEK_OFFICIAL_THINKING:-${DEEPSEEK_THINKING}}"
  export REASONING_EFFORT=""
  export REWRITE_REASONING_EFFORT=""
  export TARGET_MAX_COMPLETION_TOKENS="${TARGET_MAX_COMPLETION_TOKENS:-16384}"
}

LIVEMATH_ARK_WORKERS="${LIVEMATH_ARK_WORKERS:-48}"
LIVEMATH_ARK_ANALYST_WORKERS="${LIVEMATH_ARK_ANALYST_WORKERS:-48}"
LIVEMATH_ARK_EXEC_TIMEOUT="${LIVEMATH_ARK_EXEC_TIMEOUT:-1800}"
LIVEMATH_ARK_CONTINUE_ON_ERROR="${LIVEMATH_ARK_CONTINUE_ON_ERROR:-1}"

# Shared method settings. Each ablation row appends one focused override.
LIVEMATH_ARK_SHARED_CFG="${LIVEMATH_ARK_SHARED_CFG:-train.batch_size=18 optimizer.type_guided_rollout_repeats=4 optimizer.type_guided_tau_succ=0.5 optimizer.type_guided_min_support=1 optimizer.type_guided_clustering=false optimizer.type_guided_tail_bank=false optimizer.learning_rate=999 optimizer.min_learning_rate=999 optimizer.lr_scheduler=constant train.seed=42}"

LIVEMATH_ARK_RESULTS=()

run_livemath_ark_case() {
  local tag="$1"
  local epochs="$2"
  local skill_mode="$3"
  local case_cfg="${4:-}"
  local suite_root="${LIVEMATH_ARK_SUITE_ROOT:-${OUT_BASE}}"
  local case_root="${suite_root}/${tag}"
  local extra_cfg="${LIVEMATH_ARK_SHARED_CFG}"
  local -a extra_args=(--num_epochs "${epochs}")

  if [[ -n "${case_cfg}" ]]; then
    extra_cfg+=" ${case_cfg}"
  fi
  case "${skill_mode}" in
    system)
      # The rollout omits its ## Skill section when the loaded file is empty.
      # Passing /dev/null after _common.sh's default --skill_init makes argparse
      # keep the final value and yields the dataset system prompt only.
      extra_args+=(--skill_init /dev/null)
      ;;
    initial)
      ;;
    *)
      comparison_fail "Unknown skill mode '${skill_mode}' for case ${tag}"
      ;;
  esac

  echo
  echo "############################################################"
  echo "# [livemath-ark] ${tag}"
  echo "#   epochs=${epochs} skill_mode=${skill_mode}"
  echo "#   cfg=${extra_cfg}"
  echo "############################################################"

  local rc=0
  (
    export OUT_BASE="${case_root}"
    export EXTRA_CFG_OPTIONS="${extra_cfg}"
    export PYTHONUNBUFFERED=1
    run_dataset livemathematicianbench \
      --workers "${LIVEMATH_ARK_WORKERS}" \
      --analyst_workers "${LIVEMATH_ARK_ANALYST_WORKERS}" \
      --exec_timeout "${LIVEMATH_ARK_EXEC_TIMEOUT}" \
      "${extra_args[@]}"
  ) || rc=$?

  if (( rc != 0 )); then
    LIVEMATH_ARK_RESULTS+=("FAIL  ${tag}")
    echo "[livemath-ark] ${tag} FAILED rc=${rc}" >&2
    if ! comparison_truthy "${LIVEMATH_ARK_CONTINUE_ON_ERROR}"; then
      exit "${rc}"
    fi
  else
    LIVEMATH_ARK_RESULTS+=("OK    ${tag}")
  fi

  if ! comparison_truthy "${DRY_RUN:-0}"; then
    mkdir -p "${suite_root}"
    printf '%s\t%s\t%s\n' "${tag}" "${rc}" "${case_root}/livemathematicianbench" \
      >>"${suite_root}/completed.tsv"
  fi
}

finish_livemath_ark_suite() {
  local suite_root="${LIVEMATH_ARK_SUITE_ROOT:-${OUT_BASE}}"
  echo
  echo "============================================================"
  echo "  LiveMath Ark suite complete: ${suite_root}"
  local line
  for line in "${LIVEMATH_ARK_RESULTS[@]:-}"; do
    [[ -n "${line}" ]] && echo "    ${line}"
  done
  echo "============================================================"
  if printf '%s\n' "${LIVEMATH_ARK_RESULTS[@]:-}" | grep -q '^FAIL'; then
    return 1
  fi
}

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "${SCRIPT_DIR}/_common.sh"

# Combo lane: livemathematicianbench only, DeepSeek official API
# (target=deepseek-v4-flash, optimizer=deepseek-v4-pro).
#
# This stacks the knobs that showed a CLEAR benefit across the 09 (training
# knobs) and 10 (mechanism) sweeps into ONE configuration, to see whether the
# gains compose and can close the gap to SkillOpt-main (~0.4597).
#
# Beneficial knobs baked in (see 10_..._mechanism.md §4 for evidence):
#   train.batch_size                    = 35   (09: bigger = better & faster; 35 = full train set)
#   optimizer.type_guided_rollout_repeats = 4  (09: peak of the 1..5 sweep)
#   optimizer.type_guided_tau_succ      = 0.5  (10: biggest winner, best_test 0.4355)
#   optimizer.type_guided_min_support   = 1    (10: dropped=0, best_test 0.4274; monotone "drop less = better")
#   EDIT_BUDGET_OFF                     = 1    (10: mild positive, edit_budget_off > on)
#
# Deliberately NOT baked in (undecided under the 0.10 noise band):
#   evaluation.use_gate  — force-accept (main-alignment) was inconclusive; expose
#                          as COMBO_USE_GATE so you can A/B it.
#   info retention       — clustering+tail_bank activated but did NOT move test.
#
# NOTE on noise: the 10 sweep showed a base_test spread of ~0.10 across
# byte-identical initial skills. A single run is therefore NOT trustworthy on
# its own. SEED_GRID defaults to one run (honoring "跑一次"), but set e.g.
# SEED_GRID="42 1 7" to average out the noise — strongly recommended before
# drawing any conclusion.
#
# Overrides are routed through EXTRA_CFG_OPTIONS, which _common.sh merges into
# the SINGLE --cfg-options block (a second --cfg-options would be clobbered by
# argparse nargs="+").

export RUN_ID="${RUN_ID:-skillopt_tree_livemath_combo_dsv4_flash_pro_$(date +%Y%m%d_%H%M%S)}"
export OUT_BASE="${OUT_BASE:-${PROJECT_ROOT}/outputs/${RUN_ID}}"

COMBO_WORKERS="${COMBO_WORKERS:-128}"
COMBO_ANALYST_WORKERS="${COMBO_ANALYST_WORKERS:-64}"
COMBO_EXEC_TIMEOUT="${COMBO_EXEC_TIMEOUT:-1800}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"

# The beneficial config集合 (override any if you want to re-test a knob).
COMBO_BATCH_SIZE="${COMBO_BATCH_SIZE:-35}"
COMBO_ROLLOUT_REPEATS="${COMBO_ROLLOUT_REPEATS:-4}"
COMBO_TAU_SUCC="${COMBO_TAU_SUCC:-0.5}"
COMBO_MIN_SUPPORT="${COMBO_MIN_SUPPORT:-1}"
COMBO_EDIT_BUDGET_OFF="${COMBO_EDIT_BUDGET_OFF:-1}"   # 1 => lr_scheduler=autonomous
COMBO_USE_GATE="${COMBO_USE_GATE:-true}"              # true=keep gate; false=force-accept (mirror main)

# One run per seed. Default: single run. Set "42 1 7" to fight the ~0.10 noise.
SEED_GRID="${SEED_GRID:-42}"

require_layout
configure_models
trap cleanup_datasets EXIT INT TERM

print_comparison_header "11-livemath-combo" "livemathematicianbench (beneficial combo)"
echo "  workers:            ${COMBO_WORKERS} (target) / ${COMBO_ANALYST_WORKERS} (analyst)"
echo "  request timeout:    ${COMBO_EXEC_TIMEOUT}s"
echo "  combo cfg:          batch_size=${COMBO_BATCH_SIZE} rollout_repeats=${COMBO_ROLLOUT_REPEATS} tau_succ=${COMBO_TAU_SUCC} min_support=${COMBO_MIN_SUPPORT}"
echo "  edit_budget_off:    ${COMBO_EDIT_BUDGET_OFF}   use_gate=${COMBO_USE_GATE}"
echo "  seeds:              [${SEED_GRID}]"
echo "  continue_on_error:  ${CONTINUE_ON_ERROR}"

COMBO_RESULTS=()

# run_combo <tag> <seed>
run_combo() {
  local tag="$1" seed="$2"

  echo
  echo "############################################################"
  echo "# [combo] ${tag} (seed=${seed})"
  echo "############################################################"

  local row_out="${OUT_BASE}/${tag}"
  local extra="train.batch_size=${COMBO_BATCH_SIZE}"
  extra+=" optimizer.type_guided_rollout_repeats=${COMBO_ROLLOUT_REPEATS}"
  extra+=" optimizer.type_guided_tau_succ=${COMBO_TAU_SUCC}"
  extra+=" optimizer.type_guided_min_support=${COMBO_MIN_SUPPORT}"
  extra+=" evaluation.use_gate=${COMBO_USE_GATE}"
  extra+=" train.seed=${seed}"

  local rc=0
  # Subshell so per-run env vars (EDIT_BUDGET_OFF) do not leak into later rows;
  # do NOT use the external `env` command — run_dataset is a shell function.
  (
    export OUT_BASE="${row_out}" EXTRA_CFG_OPTIONS="${extra}"
    export EDIT_BUDGET_OFF="${COMBO_EDIT_BUDGET_OFF}"
    run_dataset livemathematicianbench \
      --workers "${COMBO_WORKERS}" \
      --analyst_workers "${COMBO_ANALYST_WORKERS}" \
      --exec_timeout "${COMBO_EXEC_TIMEOUT}"
  ) || rc=$?

  if (( rc != 0 )); then
    echo "[combo] ${tag} FAILED rc=${rc}" >&2
    COMBO_RESULTS+=("FAIL  ${tag}")
    if ! comparison_truthy "${CONTINUE_ON_ERROR}"; then
      exit "${rc}"
    fi
  else
    COMBO_RESULTS+=("OK    ${tag}")
  fi
}

for s in ${SEED_GRID}; do
  run_combo "combo_seed_${s}" "${s}"
done

echo
echo "============================================================"
echo "  [combo] complete — results under ${OUT_BASE}"
for line in "${COMBO_RESULTS[@]}"; do
  echo "    ${line}"
done
echo "============================================================"
if printf '%s\n' "${COMBO_RESULTS[@]}" | grep -q '^FAIL'; then
  exit 1
fi

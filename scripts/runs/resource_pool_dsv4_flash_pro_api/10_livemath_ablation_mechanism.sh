#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "${SCRIPT_DIR}/_common.sh"

# Mechanism ablation lane: livemathematicianbench only, DeepSeek official API
# (target=deepseek-v4-flash, optimizer=deepseek-v4-pro). Companion to
# 09_livemath_ablation.sh — that one sweeps the "training knobs"
# (batch_size / rollout_repeats / leaf_fallback); THIS one sweeps the
# "method / acceptance mechanism" knobs that are suspected to explain why Tree
# stalls while SkillOpt-main climbs.
#
# All rows run SEQUENTIALLY (one train.py at a time), each into its own
# OUT_BASE subdir + log, so results are directly comparable.
#
# Every row is one-factor-at-a-time (OFAT): it flips ONE mechanism off a shared
# baseline and holds everything else at PatchTree defaults. Each row also
# carries the "robust training knobs" found best in 09 (batch_size=35,
# rollout_repeats=4) so the mechanism signal is not drowned by the weak-knob
# noise; override via MECH_BASE_* if you want raw defaults.
#
# Dimensions (each an independent row unless a grid is given):
#   A. acceptance mechanism   evaluation.use_gate = true|false
#        false => force-accept every candidate (trainer.py:1949), i.e. mirror
#        SkillOpt-main's unconditional slow-update injection.
#   B. edit budget            EDIT_BUDGET_OFF = 0|1
#        1 => optimizer.lr_scheduler=autonomous => per-step edit cap lifted
#        (NO_LIMIT=999) instead of the cosine 4->2 decay.
#   C. information retention  TYPE_GUIDED_CLUSTERING + TYPE_GUIDED_TAIL_BANK
#        on => cross-type LLM semantic clustering (merges long-tail edits with
#        different type labels but the same repair mechanism, BEFORE the
#        min_support filter) + cross-epoch tail bank (recycles dropped
#        long-tail across rounds). Reduces the merge-stage `dropped=N` loss.
#   D. gate strictness        optimizer.type_guided_tau_succ (grid)
#        lower => a rollout counts as "success" more easily => more edits admitted.
#   E. long-tail threshold    optimizer.type_guided_min_support (grid)
#        1 => almost nothing is dropped as long-tail; higher => stricter.
#
# Non-env overrides are routed through EXTRA_CFG_OPTIONS, which _common.sh
# merges into the SINGLE --cfg-options block (a second --cfg-options on the CLI
# would be clobbered by argparse nargs="+").

export RUN_ID="${RUN_ID:-skillopt_tree_livemath_mech_dsv4_flash_pro_$(date +%Y%m%d_%H%M%S)}"
export OUT_BASE="${OUT_BASE:-${PROJECT_ROOT}/outputs/${RUN_ID}}"

# 128 target workers + 64 analyst workers (per the requested 128+64 level).
MECH_WORKERS="${MECH_WORKERS:-128}"
MECH_ANALYST_WORKERS="${MECH_ANALYST_WORKERS:-64}"
MECH_EXEC_TIMEOUT="${MECH_EXEC_TIMEOUT:-1800}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"

# Shared training knobs held fixed across all mechanism rows. Defaults reflect
# the robust optimum found in the 09 sweep; override to test on raw defaults.
MECH_BASE_BATCH_SIZE="${MECH_BASE_BATCH_SIZE:-35}"
MECH_BASE_ROLLOUT_REPEATS="${MECH_BASE_ROLLOUT_REPEATS:-4}"

# Which dimensions to run (1 = on). Turn any off to shorten the sweep.
DO_USE_GATE="${DO_USE_GATE:-1}"          # A
DO_EDIT_BUDGET="${DO_EDIT_BUDGET:-1}"    # B
DO_INFO_RETENTION="${DO_INFO_RETENTION:-1}"  # C
DO_TAU_SUCC="${DO_TAU_SUCC:-1}"          # D
DO_MIN_SUPPORT="${DO_MIN_SUPPORT:-1}"    # E

# Grids for the grid-style dimensions.
TAU_SUCC_GRID="${TAU_SUCC_GRID:-0.5 1.0}"
MIN_SUPPORT_GRID="${MIN_SUPPORT_GRID:-1 2 3}"

require_layout
configure_models
trap cleanup_datasets EXIT INT TERM

print_comparison_header "10-livemath-mechanism-ablation" "livemathematicianbench (mechanism sweep)"
echo "  workers:            ${MECH_WORKERS} (target) / ${MECH_ANALYST_WORKERS} (analyst)"
echo "  request timeout:    ${MECH_EXEC_TIMEOUT}s"
echo "  held fixed:         batch_size=${MECH_BASE_BATCH_SIZE} rollout_repeats=${MECH_BASE_ROLLOUT_REPEATS}"
echo "  dims:               use_gate=${DO_USE_GATE} edit_budget=${DO_EDIT_BUDGET} info_retention=${DO_INFO_RETENTION} tau_succ=${DO_TAU_SUCC} min_support=${DO_MIN_SUPPORT}"
echo "  grids:              tau_succ=[${TAU_SUCC_GRID}] min_support=[${MIN_SUPPORT_GRID}]"
echo "  continue_on_error:  ${CONTINUE_ON_ERROR}"

MECH_RESULTS=()

# run_mech <tag> <extra_env_kv...> -- <extra_cfg_kv...>
# Everything before "--" is exported as env (e.g. TYPE_GUIDED_CLUSTERING=1);
# everything after "--" is appended to EXTRA_CFG_OPTIONS. The shared training
# knobs (batch_size / rollout_repeats) are always injected.
run_mech() {
  local tag="$1"; shift
  local -a env_kv=() cfg_kv=()
  local seen_sep=0
  while (( $# )); do
    if [[ "$1" == "--" ]]; then seen_sep=1; shift; continue; fi
    if (( seen_sep )); then cfg_kv+=("$1"); else env_kv+=("$1"); fi
    shift
  done

  echo
  echo "############################################################"
  echo "# [mech] ${tag}"
  echo "#   env:  ${env_kv[*]:-<none>}"
  echo "#   cfg:  ${cfg_kv[*]:-<none>}"
  echo "############################################################"

  local row_out="${OUT_BASE}/${tag}"
  # Always-on shared training knobs + this row's extra cfg overrides.
  local extra="train.batch_size=${MECH_BASE_BATCH_SIZE} optimizer.type_guided_rollout_repeats=${MECH_BASE_ROLLOUT_REPEATS}"
  if (( ${#cfg_kv[@]} )); then
    extra+=" ${cfg_kv[*]}"
  fi

  local rc=0
  # env_kv entries are `KEY=VALUE`. We must NOT use the external `env` command:
  # run_dataset is a shell function and `env` can only exec real binaries
  # (that failed with rc=127). Instead export the row's env vars in a subshell
  # so they apply only to this invocation and do not leak into later rows.
  (
    export OUT_BASE="${row_out}" EXTRA_CFG_OPTIONS="${extra}"
    local kv
    for kv in "${env_kv[@]:-}"; do
      [[ -n "${kv}" ]] || continue
      export "${kv?}"
    done
    run_dataset livemathematicianbench \
      --workers "${MECH_WORKERS}" \
      --analyst_workers "${MECH_ANALYST_WORKERS}" \
      --exec_timeout "${MECH_EXEC_TIMEOUT}"
  ) || rc=$?

  if (( rc != 0 )); then
    echo "[mech] ${tag} FAILED rc=${rc}" >&2
    MECH_RESULTS+=("FAIL  ${tag}")
    if ! comparison_truthy "${CONTINUE_ON_ERROR}"; then
      exit "${rc}"
    fi
  else
    MECH_RESULTS+=("OK    ${tag}")
  fi
}

# ── A. acceptance mechanism: gate on/off (main-alignment test) ──────────────
if comparison_truthy "${DO_USE_GATE}"; then
  run_mech "use_gate_true"  -- "evaluation.use_gate=true"
  run_mech "use_gate_false" -- "evaluation.use_gate=false"   # force-accept, mirror main
fi

# ── B. edit budget: capped (default) vs unlimited ───────────────────────────
if comparison_truthy "${DO_EDIT_BUDGET}"; then
  run_mech "edit_budget_on"  EDIT_BUDGET_OFF=0 --
  run_mech "edit_budget_off" EDIT_BUDGET_OFF=1 --
fi

# ── C. information retention: default drop vs clustering + tail_bank ─────────
if comparison_truthy "${DO_INFO_RETENTION}"; then
  run_mech "info_default" TYPE_GUIDED_CLUSTERING=0 TYPE_GUIDED_TAIL_BANK=0 --
  run_mech "info_retain"  TYPE_GUIDED_CLUSTERING=1 TYPE_GUIDED_TAIL_BANK=1 --
fi

# ── D. gate strictness: type_guided_tau_succ ────────────────────────────────
if comparison_truthy "${DO_TAU_SUCC}"; then
  for ts in ${TAU_SUCC_GRID}; do
    run_mech "tau_succ_${ts}" -- "optimizer.type_guided_tau_succ=${ts}"
  done
fi

# ── E. long-tail threshold: type_guided_min_support ─────────────────────────
if comparison_truthy "${DO_MIN_SUPPORT}"; then
  for ms in ${MIN_SUPPORT_GRID}; do
    run_mech "min_support_${ms}" -- "optimizer.type_guided_min_support=${ms}"
  done
fi

echo
echo "============================================================"
echo "  [mech] sweep complete — results under ${OUT_BASE}"
for line in "${MECH_RESULTS[@]}"; do
  echo "    ${line}"
done
echo "============================================================"
if printf '%s\n' "${MECH_RESULTS[@]}" | grep -q '^FAIL'; then
  exit 1
fi

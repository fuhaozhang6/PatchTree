#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "${SCRIPT_DIR}/_common.sh"

# Mechanism / knob ablation lane: searchqa only, DeepSeek official API
# (target=deepseek-v4-flash, optimizer=deepseek-v4-pro). SearchQA analogue of
# 10_livemath_ablation_mechanism.sh, re-designed for SearchQA's scale:
#   train=400 / val=200 / test=1400  (vs livemath 35/18/124)
#
# Because train=400, batch_size is a GENUINE gradient-batching knob here (unlike
# livemath where 35 was the cap), and the tree-shape knobs (tree_depth,
# leaf_fallback) matter — those were the core axes of the earlier
# scripts/runs/ablations/searchqa_fallback_pilot study, so they default OFF here
# to avoid redoing that work (flip DO_TREE_DEPTH / DO_LEAF_FALLBACK to re-check).
#
# All rows run SEQUENTIALLY, each into its own OUT_BASE subdir + log. Every row
# is one-factor-at-a-time (OFAT): it changes ONE knob off a shared baseline and
# holds everything else at PatchTree defaults.
#
# No multi-seed here (per request "暂时不用多seed重复"); SEARCHQA is expensive on
# test1400, so add seeds later only for the knobs that show signal.
#
# Dimensions (each an independent row unless a grid is given):
#   H. batch size            train.batch_size (grid)    [searchqa-specific: real batching]
#   A. acceptance mechanism  evaluation.use_gate = true|false  (false => force-accept, mirror main)
#   B. edit budget           EDIT_BUDGET_OFF = 0|1  (1 => lr_scheduler=autonomous, cap lifted)
#   D. gate strictness       optimizer.type_guided_tau_succ (grid) — livemath's biggest winner
#   E. long-tail threshold   optimizer.type_guided_min_support (grid)
#   C. information retention  TYPE_GUIDED_CLUSTERING + TYPE_GUIDED_TAIL_BANK   [default off; heavy]
#   F. tree depth            optimizer.type_guided_tree_depth (grid)          [default off; pilot covered]
#   G. leaf fallback         optimizer.type_guided_leaf_fallback = true|false [default off; pilot covered]
#
# Non-env overrides are routed through EXTRA_CFG_OPTIONS, which _common.sh
# merges into the SINGLE --cfg-options block (a second --cfg-options on the CLI
# would be clobbered by argparse nargs="+").

export RUN_ID="${RUN_ID:-skillopt_tree_searchqa_ablation_dsv4_flash_pro_$(date +%Y%m%d_%H%M%S)}"
export OUT_BASE="${OUT_BASE:-${PROJECT_ROOT}/outputs/${RUN_ID}}"

SEARCHQA_WORKERS="${SEARCHQA_WORKERS:-128}"
SEARCHQA_ANALYST_WORKERS="${SEARCHQA_ANALYST_WORKERS:-64}"
SEARCHQA_EXEC_TIMEOUT="${SEARCHQA_EXEC_TIMEOUT:-600}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"

# Shared knobs held fixed across all rows (SearchQA config defaults).
SQA_BASE_BATCH_SIZE="${SQA_BASE_BATCH_SIZE:-40}"
SQA_BASE_ROLLOUT_REPEATS="${SQA_BASE_ROLLOUT_REPEATS:-3}"

# Which dimensions to run (1 = on). Trim to shorten this (expensive) sweep.
DO_BATCH_SIZE="${DO_BATCH_SIZE:-1}"          # H
DO_USE_GATE="${DO_USE_GATE:-1}"              # A
DO_EDIT_BUDGET="${DO_EDIT_BUDGET:-1}"        # B
DO_TAU_SUCC="${DO_TAU_SUCC:-1}"              # D
DO_MIN_SUPPORT="${DO_MIN_SUPPORT:-1}"        # E
DO_INFO_RETENTION="${DO_INFO_RETENTION:-0}"  # C — heavy; off by default
DO_TREE_DEPTH="${DO_TREE_DEPTH:-0}"          # F — fallback_pilot covered; off by default
DO_LEAF_FALLBACK="${DO_LEAF_FALLBACK:-0}"    # G — fallback_pilot covered; off by default

# Grids for the grid-style dimensions.
BATCH_SIZE_GRID="${BATCH_SIZE_GRID:-20 40 80}"
TAU_SUCC_GRID="${TAU_SUCC_GRID:-0.5 1.0}"
MIN_SUPPORT_GRID="${MIN_SUPPORT_GRID:-1 2 3}"
TREE_DEPTH_GRID="${TREE_DEPTH_GRID:-2 3}"

require_layout
configure_models
trap cleanup_datasets EXIT INT TERM

print_comparison_header "12-searchqa-ablation" "searchqa (knob + mechanism sweep)"
echo "  workers:            ${SEARCHQA_WORKERS} (target) / ${SEARCHQA_ANALYST_WORKERS} (analyst)"
echo "  request timeout:    ${SEARCHQA_EXEC_TIMEOUT}s"
echo "  held fixed:         batch_size=${SQA_BASE_BATCH_SIZE} rollout_repeats=${SQA_BASE_ROLLOUT_REPEATS}"
echo "  dims on:            batch_size=${DO_BATCH_SIZE} use_gate=${DO_USE_GATE} edit_budget=${DO_EDIT_BUDGET} tau_succ=${DO_TAU_SUCC} min_support=${DO_MIN_SUPPORT}"
echo "  dims off-by-def:    info_retention=${DO_INFO_RETENTION} tree_depth=${DO_TREE_DEPTH} leaf_fallback=${DO_LEAF_FALLBACK}"
echo "  grids:              batch=[${BATCH_SIZE_GRID}] tau_succ=[${TAU_SUCC_GRID}] min_support=[${MIN_SUPPORT_GRID}] tree_depth=[${TREE_DEPTH_GRID}]"
echo "  continue_on_error:  ${CONTINUE_ON_ERROR}"

SQA_RESULTS=()

# run_sqa <tag> <extra_env_kv...> -- <extra_cfg_kv...>
# Everything before "--" is exported as env (e.g. EDIT_BUDGET_OFF=1); everything
# after "--" is appended to EXTRA_CFG_OPTIONS. Shared knobs (batch_size /
# rollout_repeats) are always injected; a per-row batch override in cfg_kv wins
# because it comes later in the --cfg-options list.
run_sqa() {
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
  echo "# [sqa] ${tag}"
  echo "#   env:  ${env_kv[*]:-<none>}"
  echo "#   cfg:  ${cfg_kv[*]:-<none>}"
  echo "############################################################"

  local row_out="${OUT_BASE}/${tag}"
  local extra="train.batch_size=${SQA_BASE_BATCH_SIZE} optimizer.type_guided_rollout_repeats=${SQA_BASE_ROLLOUT_REPEATS}"
  if (( ${#cfg_kv[@]} )); then
    extra+=" ${cfg_kv[*]}"
  fi

  local rc=0
  # Subshell so per-row env vars don't leak to later rows; do NOT use the
  # external `env` command (run_dataset is a shell function, env only execs
  # real binaries and fails rc=127).
  (
    export OUT_BASE="${row_out}" EXTRA_CFG_OPTIONS="${extra}"
    local kv
    for kv in "${env_kv[@]:-}"; do
      [[ -n "${kv}" ]] || continue
      export "${kv?}"
    done
    run_dataset searchqa \
      --workers "${SEARCHQA_WORKERS}" \
      --analyst_workers "${SEARCHQA_ANALYST_WORKERS}" \
      --exec_timeout "${SEARCHQA_EXEC_TIMEOUT}"
  ) || rc=$?

  if (( rc != 0 )); then
    echo "[sqa] ${tag} FAILED rc=${rc}" >&2
    SQA_RESULTS+=("FAIL  ${tag}")
    if ! comparison_truthy "${CONTINUE_ON_ERROR}"; then
      exit "${rc}"
    fi
  else
    SQA_RESULTS+=("OK    ${tag}")
  fi
}

# ── H. batch size: real gradient-batching knob (train=400) ──────────────────
if comparison_truthy "${DO_BATCH_SIZE}"; then
  for bs in ${BATCH_SIZE_GRID}; do
    run_sqa "batch_size_${bs}" -- "train.batch_size=${bs}"
  done
fi

# ── A. acceptance mechanism: gate on/off (main-alignment test) ──────────────
if comparison_truthy "${DO_USE_GATE}"; then
  run_sqa "use_gate_true"  -- "evaluation.use_gate=true"
  run_sqa "use_gate_false" -- "evaluation.use_gate=false"   # force-accept, mirror main
fi

# ── B. edit budget: capped (default) vs unlimited ───────────────────────────
if comparison_truthy "${DO_EDIT_BUDGET}"; then
  run_sqa "edit_budget_on"  EDIT_BUDGET_OFF=0 --
  run_sqa "edit_budget_off" EDIT_BUDGET_OFF=1 --
fi

# ── D. gate strictness: type_guided_tau_succ ────────────────────────────────
if comparison_truthy "${DO_TAU_SUCC}"; then
  for ts in ${TAU_SUCC_GRID}; do
    run_sqa "tau_succ_${ts}" -- "optimizer.type_guided_tau_succ=${ts}"
  done
fi

# ── E. long-tail threshold: type_guided_min_support ─────────────────────────
if comparison_truthy "${DO_MIN_SUPPORT}"; then
  for ms in ${MIN_SUPPORT_GRID}; do
    run_sqa "min_support_${ms}" -- "optimizer.type_guided_min_support=${ms}"
  done
fi

# ── C. information retention: default drop vs clustering + tail_bank ─────────
if comparison_truthy "${DO_INFO_RETENTION}"; then
  run_sqa "info_default" TYPE_GUIDED_CLUSTERING=0 TYPE_GUIDED_TAIL_BANK=0 --
  run_sqa "info_retain"  TYPE_GUIDED_CLUSTERING=1 TYPE_GUIDED_TAIL_BANK=1 --
fi

# ── F. tree depth: leaf->root (2) vs leaf->mid->root (3) ─────────────────────
if comparison_truthy "${DO_TREE_DEPTH}"; then
  for td in ${TREE_DEPTH_GRID}; do
    run_sqa "tree_depth_${td}" -- "optimizer.type_guided_tree_depth=${td}"
  done
fi

# ── G. leaf fallback: probe children when the root candidate is rejected ─────
if comparison_truthy "${DO_LEAF_FALLBACK}"; then
  run_sqa "leaf_fallback_on"  -- "optimizer.type_guided_leaf_fallback=true"
  run_sqa "leaf_fallback_off" -- "optimizer.type_guided_leaf_fallback=false"
fi

echo
echo "============================================================"
echo "  [sqa] sweep complete — results under ${OUT_BASE}"
for line in "${SQA_RESULTS[@]}"; do
  echo "    ${line}"
done
echo "============================================================"
if printf '%s\n' "${SQA_RESULTS[@]}" | grep -q '^FAIL'; then
  exit 1
fi

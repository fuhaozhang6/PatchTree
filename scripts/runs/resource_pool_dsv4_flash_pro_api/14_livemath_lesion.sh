#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "${SCRIPT_DIR}/_common.sh"

# Lesion study (ablation-by-removal): livemath, DeepSeek official API
# (target=deepseek-v4-flash, optimizer=deepseek-v4-pro).
#
# Unlike the knob sweeps (09/10/13), this is a focused LESION table: one "Full"
# configuration with every PatchTree selling-point ON, then one row per removed
# component. Run over 3 seeds because livemath's single-seed base_test noise
# (~0.10) otherwise drowns the effect sizes.
#
#   livemath: train=35 / val(selection)=18 / test=124
#   fixed:    train.batch_size=35, optimizer.type_guided_rollout_repeats=4
#
# Components under test (chosen selling-points):
#   C1 tree hierarchy   tree_depth 2 (Full) -> 1 (flat: records->root, no leaf grouping)
#   C2 leaf fallback    leaf_fallback on (Full) -> off (rejected root is NOT rescued)
#   C3 long-tail keep   tail_bank on + min_support=1 (Full) -> tail_bank off + min_support=2
#
# Lesion rows (each run over every seed in SEED_GRID):
#   full           depth=2, fallback on, tail_bank on,  min_support=1
#   no_hierarchy   depth=1, fallback on, tail_bank on,  min_support=1   (removes C1)
#   no_fallback    depth=2, fallback OFF, tail_bank on, min_support=1   (removes C2)
#   no_tail        depth=2, fallback on, tail_bank OFF, min_support=2   (removes C3)
#
# Non-env overrides route through EXTRA_CFG_OPTIONS, merged by _common.sh into
# the SINGLE --cfg-options block. tail_bank / leaf_fallback use the _common.sh
# env switches (TYPE_GUIDED_TAIL_BANK / TYPE_GUIDED_LEAF_FALLBACK).

export RUN_ID="${RUN_ID:-skillopt_tree_livemath_lesion_dsv4_flash_pro_$(date +%Y%m%d_%H%M%S)}"
export OUT_BASE="${OUT_BASE:-${PROJECT_ROOT}/outputs/${RUN_ID}}"

LESION_WORKERS="${LESION_WORKERS:-128}"
LESION_ANALYST_WORKERS="${LESION_ANALYST_WORKERS:-64}"
LESION_EXEC_TIMEOUT="${LESION_EXEC_TIMEOUT:-1800}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"

# Fixed training knobs shared by every row (robust 09/10 values).
LESION_BATCH_SIZE="${LESION_BATCH_SIZE:-35}"
LESION_ROLLOUT_REPEATS="${LESION_ROLLOUT_REPEATS:-4}"

# 3 seeds to beat the ~0.10 single-seed noise.
SEED_GRID="${SEED_GRID:-42 1 7}"

# Which lesion rows to run (1 = on).
DO_FULL="${DO_FULL:-1}"
DO_NO_HIERARCHY="${DO_NO_HIERARCHY:-1}"   # removes C1
DO_NO_FALLBACK="${DO_NO_FALLBACK:-1}"     # removes C2
DO_NO_TAIL="${DO_NO_TAIL:-1}"             # removes C3

require_layout
configure_models
trap cleanup_datasets EXIT INT TERM

print_comparison_header "14-livemath-lesion" "livemath (PatchTree lesion study, 3 seed)"
echo "  workers:            ${LESION_WORKERS} (target) / ${LESION_ANALYST_WORKERS} (analyst)"
echo "  request timeout:    ${LESION_EXEC_TIMEOUT}s"
echo "  fixed:              batch_size=${LESION_BATCH_SIZE} rollout_repeats=${LESION_ROLLOUT_REPEATS}"
echo "  seeds:              [${SEED_GRID}]"
echo "  rows on:            full=${DO_FULL} no_hierarchy=${DO_NO_HIERARCHY} no_fallback=${DO_NO_FALLBACK} no_tail=${DO_NO_TAIL}"
echo "  continue_on_error:  ${CONTINUE_ON_ERROR}"

LESION_RESULTS=()

# run_lesion <row_tag> <extra_env_kv...> -- <extra_cfg_kv...>
# Loops over SEED_GRID. Everything before "--" is exported as env for the row;
# everything after "--" is appended to EXTRA_CFG_OPTIONS (train.seed is added per
# seed). Shared knobs (batch_size / rollout_repeats) are always injected.
run_lesion() {
  local row_tag="$1"; shift
  local -a env_kv=() cfg_kv=()
  local seen_sep=0
  while (( $# )); do
    if [[ "$1" == "--" ]]; then seen_sep=1; shift; continue; fi
    if (( seen_sep )); then cfg_kv+=("$1"); else env_kv+=("$1"); fi
    shift
  done

  local seed
  for seed in ${SEED_GRID}; do
    local tag="${row_tag}_seed_${seed}"
    echo
    echo "############################################################"
    echo "# [lesion] ${tag}"
    echo "#   env:  ${env_kv[*]:-<none>}"
    echo "#   cfg:  ${cfg_kv[*]:-<none>}  train.seed=${seed}"
    echo "############################################################"

    local row_out="${OUT_BASE}/${tag}"
    local extra="train.batch_size=${LESION_BATCH_SIZE} optimizer.type_guided_rollout_repeats=${LESION_ROLLOUT_REPEATS}"
    if (( ${#cfg_kv[@]} )); then
      extra+=" ${cfg_kv[*]}"
    fi
    extra+=" train.seed=${seed}"

    local rc=0
    # Subshell isolates per-row/seed env; do NOT use the external `env` command
    # (run_dataset is a shell function; env only execs real binaries).
    (
      export OUT_BASE="${row_out}" EXTRA_CFG_OPTIONS="${extra}"
      local kv
      for kv in "${env_kv[@]:-}"; do
        [[ -n "${kv}" ]] || continue
        export "${kv?}"
      done
      run_dataset livemathematicianbench \
        --workers "${LESION_WORKERS}" \
        --analyst_workers "${LESION_ANALYST_WORKERS}" \
        --exec_timeout "${LESION_EXEC_TIMEOUT}"
    ) || rc=$?

    if (( rc != 0 )); then
      echo "[lesion] ${tag} FAILED rc=${rc}" >&2
      LESION_RESULTS+=("FAIL  ${tag}")
      if ! comparison_truthy "${CONTINUE_ON_ERROR}"; then
        exit "${rc}"
      fi
    else
      LESION_RESULTS+=("OK    ${tag}")
    fi
  done
}

# ── Full: every selling-point ON ────────────────────────────────────────────
if comparison_truthy "${DO_FULL}"; then
  run_lesion "full" TYPE_GUIDED_TAIL_BANK=1 -- \
    "optimizer.type_guided_tree_depth=2" "optimizer.type_guided_min_support=1"
fi

# ── −C1: remove tree hierarchy (flat: records -> root) ──────────────────────
if comparison_truthy "${DO_NO_HIERARCHY}"; then
  run_lesion "no_hierarchy" TYPE_GUIDED_TAIL_BANK=1 -- \
    "optimizer.type_guided_tree_depth=1" "optimizer.type_guided_min_support=1"
fi

# ── −C2: remove leaf/child fallback ─────────────────────────────────────────
if comparison_truthy "${DO_NO_FALLBACK}"; then
  run_lesion "no_fallback" TYPE_GUIDED_TAIL_BANK=1 TYPE_GUIDED_LEAF_FALLBACK=0 -- \
    "optimizer.type_guided_tree_depth=2" "optimizer.type_guided_min_support=1"
fi

# ── −C3: remove long-tail retention (tail_bank off + default drop) ──────────
if comparison_truthy "${DO_NO_TAIL}"; then
  run_lesion "no_tail" -- \
    "optimizer.type_guided_tree_depth=2" "optimizer.type_guided_min_support=2"
fi

echo
echo "============================================================"
echo "  [lesion] sweep complete — results under ${OUT_BASE}"
for line in "${LESION_RESULTS[@]}"; do
  echo "    ${line}"
done
echo "============================================================"
if printf '%s\n' "${LESION_RESULTS[@]}" | grep -q '^FAIL'; then
  exit 1
fi

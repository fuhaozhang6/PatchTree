#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "${SCRIPT_DIR}/_common.sh"

# Ablation lane: livemathematicianbench only, DeepSeek official API
# (target=deepseek-v4-flash, optimizer=deepseek-v4-pro). Runs a matrix of
# ablation configurations SEQUENTIALLY (one train.py at a time), each into its
# own output subdirectory + log, so results are directly comparable.
#
# Ablated dimensions (each row overrides ONE knob off a shared baseline):
#   - train.batch_size                          (sampling batch per step)
#   - optimizer.type_guided_rollout_repeats     (rollout sampling count / tg_repeats)
#   - optimizer.type_guided_leaf_fallback       (root-reject -> child probing on/off)
#
# Each row's overrides go through EXTRA_CFG_OPTIONS, which _common.sh merges into
# the single --cfg-options block (a second --cfg-options on the CLI would be
# clobbered by argparse nargs="+"; that is exactly why we route through the env).
#
# Concurrency: API-only, so no vLLM ceiling. Defaults target the requested
# 128 workers (target rollouts) + 64 analyst workers (optimizer analysis).

export RUN_ID="${RUN_ID:-skillopt_tree_livemath_ablation_dsv4_flash_pro_$(date +%Y%m%d_%H%M%S)}"
export OUT_BASE="${OUT_BASE:-${PROJECT_ROOT}/outputs/${RUN_ID}}"

# 128 target workers + 64 analyst workers (per the requested 128+64 level).
ABL_WORKERS="${ABL_WORKERS:-128}"
ABL_ANALYST_WORKERS="${ABL_ANALYST_WORKERS:-64}"
ABL_EXEC_TIMEOUT="${ABL_EXEC_TIMEOUT:-1800}"
# When 1, a failed row is logged and the sweep continues; when 0 (default),
# `set -e` stops the whole sweep on the first failure (fail-fast).
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"

# Baseline knob values (used to hold the other two fixed while one is ablated).
# livemath train split has 35 items, so 35 == full batch; values >35 would be
# indistinguishable from full-batch, hence the baseline is pinned at 35.
BASE_BATCH_SIZE="${BASE_BATCH_SIZE:-35}"
BASE_ROLLOUT_REPEATS="${BASE_ROLLOUT_REPEATS:-3}"
BASE_LEAF_FALLBACK="${BASE_LEAF_FALLBACK:-true}"

# Ablation grids (space-separated). Override any of these to reshape the sweep.
# batch_size: train_size=35, so 8/16 are sub-batch and 35 is full-batch.
BATCH_SIZE_GRID="${BATCH_SIZE_GRID:-8 16 35}"
ROLLOUT_REPEATS_GRID="${ROLLOUT_REPEATS_GRID:-1 2 3 4 5}"
LEAF_FALLBACK_GRID="${LEAF_FALLBACK_GRID:-true false}"

require_layout
configure_models
trap cleanup_datasets EXIT INT TERM

print_comparison_header "09-livemath-ablation" "livemathematicianbench (ablation sweep)"
echo "  workers:            ${ABL_WORKERS} (target) / ${ABL_ANALYST_WORKERS} (analyst)"
echo "  request timeout:    ${ABL_EXEC_TIMEOUT}s"
echo "  baseline:           batch_size=${BASE_BATCH_SIZE} rollout_repeats=${BASE_ROLLOUT_REPEATS} leaf_fallback=${BASE_LEAF_FALLBACK}"
echo "  grids:              batch_size=[${BATCH_SIZE_GRID}] rollout_repeats=[${ROLLOUT_REPEATS_GRID}] leaf_fallback=[${LEAF_FALLBACK_GRID}]"
echo "  continue_on_error:  ${CONTINUE_ON_ERROR}"

ABL_RESULTS=()

# run_ablation <tag> <batch_size> <rollout_repeats> <leaf_fallback>
# Runs one livemath training with the given knobs into OUT_BASE/<tag>.
run_ablation() {
  local tag="$1" batch_size="$2" repeats="$3" fallback="$4"
  shift 4  # drop the 4 knob args; "$@" now carries only extra passthrough flags
  echo
  echo "############################################################"
  echo "# [ablation] ${tag}"
  echo "#   batch_size=${batch_size} rollout_repeats=${repeats} leaf_fallback=${fallback}"
  echo "############################################################"

  # Each row lands in its own OUT_BASE subdir via a per-row RUN sub-output.
  local row_out="${OUT_BASE}/${tag}"
  # Route the three ablated knobs through the escape hatch so they join the
  # single --cfg-options block instead of forming a clobbered second one.
  local extra="train.batch_size=${batch_size} optimizer.type_guided_rollout_repeats=${repeats} optimizer.type_guided_leaf_fallback=${fallback}"

  local rc=0
  OUT_BASE="${row_out}" EXTRA_CFG_OPTIONS="${extra}" \
    run_dataset livemathematicianbench \
      --workers "${ABL_WORKERS}" \
      --analyst_workers "${ABL_ANALYST_WORKERS}" \
      --exec_timeout "${ABL_EXEC_TIMEOUT}" \
      "$@" || rc=$?

  if (( rc != 0 )); then
    echo "[ablation] ${tag} FAILED rc=${rc}" >&2
    ABL_RESULTS+=("FAIL  ${tag}")
    if ! comparison_truthy "${CONTINUE_ON_ERROR}"; then
      exit "${rc}"
    fi
  else
    ABL_RESULTS+=("OK    ${tag}")
  fi
}

# NOTE: run_dataset appends guard_dataset_results per row, and each row is fully
# sequential (foreground), so one config finishes before the next starts.

# ── 1) batch_size ablation (hold repeats + fallback at baseline) ────────────
for bs in ${BATCH_SIZE_GRID}; do
  run_ablation "batch_size_${bs}" "${bs}" "${BASE_ROLLOUT_REPEATS}" "${BASE_LEAF_FALLBACK}"
done

# ── 2) rollout_repeats (sampling count) ablation ────────────────────────────
for rr in ${ROLLOUT_REPEATS_GRID}; do
  run_ablation "rollout_repeats_${rr}" "${BASE_BATCH_SIZE}" "${rr}" "${BASE_LEAF_FALLBACK}"
done

# ── 3) leaf_fallback on/off ablation ────────────────────────────────────────
for ff in ${LEAF_FALLBACK_GRID}; do
  run_ablation "leaf_fallback_${ff}" "${BASE_BATCH_SIZE}" "${BASE_ROLLOUT_REPEATS}" "${ff}"
done

echo
echo "============================================================"
echo "  [ablation] sweep complete — results under ${OUT_BASE}"
for line in "${ABL_RESULTS[@]}"; do
  echo "    ${line}"
done
echo "============================================================"
# Fail the whole script if any row failed (even under CONTINUE_ON_ERROR), so CI
# / callers can detect partial failures.
if printf '%s\n' "${ABL_RESULTS[@]}" | grep -q '^FAIL'; then
  exit 1
fi

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "${SCRIPT_DIR}/_common.sh"

# Tree-structure ablation lane: livemath only, DeepSeek official API
# (target=deepseek-v4-flash, optimizer=deepseek-v4-pro). Companion to
# 10_livemath_ablation_mechanism.sh, but focused on the SHAPE of the PatchTree-v2
# tree rather than the acceptance/edit mechanisms.
#
#   livemath: train=35 / val(selection)=18 / test=124
#
# Shared fixed knobs across every row (the robust values from 09/10):
#   train.batch_size=35, optimizer.type_guided_rollout_repeats=4
#
# No multi-seed (per request). Each row is one-factor-at-a-time (OFAT): change ONE
# tree-shape knob off the shared baseline, hold everything else at PatchTree
# defaults. All rows run SEQUENTIALLY into their own OUT_BASE subdir + log.
#
# Dimensions (grouped):
#   ① tree height / abstraction levels
#       G_TREE_DEPTH        optimizer.type_guided_tree_depth        1|2|3
#         1 = records -> root (flat, no leaf/mid tiers; "lowest abstraction")
#         2 = leaf -> root (default)
#         3 = leaf -> mid -> root (adds the intermediate abstraction tier)
#   ② tree width / node budget
#       G_MAX_LEAF_GROUPS   optimizer.type_guided_max_leaf_groups   4|8|16   (leaf fan-out cap)
#       G_MAX_PATCH_RECORDS optimizer.type_guided_max_patch_records 12|24|48 (records fed into tree)
#   ③ grouping granularity
#       G_CLUSTERING        optimizer.type_guided_clustering        off vs on
#                           when on, sweep optimizer.type_guided_cluster_target_size 4|6|10
#   ④ fallback fan-out / recombination
#       G_FALLBACK_TOPK     optimizer.type_guided_fallback_top_k    0|1|3   (0 = all children)
#       G_FALLBACK_RECONCILE optimizer.type_guided_fallback_reconcile  deterministic|llm_fuse|off
#
# Non-env overrides route through EXTRA_CFG_OPTIONS, which _common.sh merges into
# the SINGLE --cfg-options block (a second --cfg-options would be clobbered by
# argparse nargs="+").

export RUN_ID="${RUN_ID:-skillopt_tree_livemath_treeshape_dsv4_flash_pro_$(date +%Y%m%d_%H%M%S)}"
export OUT_BASE="${OUT_BASE:-${PROJECT_ROOT}/outputs/${RUN_ID}}"

TREE_WORKERS="${TREE_WORKERS:-128}"
TREE_ANALYST_WORKERS="${TREE_ANALYST_WORKERS:-64}"
TREE_EXEC_TIMEOUT="${TREE_EXEC_TIMEOUT:-1800}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"

# Shared knobs held fixed across all rows (robust 09/10 values).
TREE_BASE_BATCH_SIZE="${TREE_BASE_BATCH_SIZE:-35}"
TREE_BASE_ROLLOUT_REPEATS="${TREE_BASE_ROLLOUT_REPEATS:-4}"

# Which dimensions to run (1 = on). Width (②) and granularity (③) are off by
# default per request; flip their DO_* to 1 to include them.
DO_TREE_DEPTH="${DO_TREE_DEPTH:-1}"                  # ①
DO_MAX_LEAF_GROUPS="${DO_MAX_LEAF_GROUPS:-0}"        # ② off by default
DO_MAX_PATCH_RECORDS="${DO_MAX_PATCH_RECORDS:-0}"    # ② off by default
DO_CLUSTERING="${DO_CLUSTERING:-0}"                  # ③ off by default
DO_FALLBACK="${DO_FALLBACK:-1}"                      # ④

# Grids.
TREE_DEPTH_GRID="${TREE_DEPTH_GRID:-1 2 3}"
MAX_LEAF_GROUPS_GRID="${MAX_LEAF_GROUPS_GRID:-4 8 16}"
MAX_PATCH_RECORDS_GRID="${MAX_PATCH_RECORDS_GRID:-12 24 48}"
CLUSTER_TARGET_SIZE_GRID="${CLUSTER_TARGET_SIZE_GRID:-4 6 10}"
FALLBACK_TOPK_GRID="${FALLBACK_TOPK_GRID:-0 1 3}"
FALLBACK_RECONCILE_GRID="${FALLBACK_RECONCILE_GRID:-deterministic llm_fuse off}"

require_layout
configure_models
trap cleanup_datasets EXIT INT TERM

print_comparison_header "13-livemath-treeshape" "livemath (PatchTree structure sweep)"
echo "  workers:            ${TREE_WORKERS} (target) / ${TREE_ANALYST_WORKERS} (analyst)"
echo "  request timeout:    ${TREE_EXEC_TIMEOUT}s"
echo "  held fixed:         batch_size=${TREE_BASE_BATCH_SIZE} rollout_repeats=${TREE_BASE_ROLLOUT_REPEATS}"
echo "  dims on:            tree_depth=${DO_TREE_DEPTH} max_leaf_groups=${DO_MAX_LEAF_GROUPS} max_patch_records=${DO_MAX_PATCH_RECORDS} clustering=${DO_CLUSTERING} fallback=${DO_FALLBACK}"
echo "  grids:              depth=[${TREE_DEPTH_GRID}] leaf_groups=[${MAX_LEAF_GROUPS_GRID}] patch_records=[${MAX_PATCH_RECORDS_GRID}] cluster_size=[${CLUSTER_TARGET_SIZE_GRID}] fb_topk=[${FALLBACK_TOPK_GRID}] fb_reconcile=[${FALLBACK_RECONCILE_GRID}]"
echo "  continue_on_error:  ${CONTINUE_ON_ERROR}"

TREE_RESULTS=()

# run_tree <tag> <extra_env_kv...> -- <extra_cfg_kv...>
# Everything before "--" is exported as env; everything after "--" is appended to
# EXTRA_CFG_OPTIONS. Shared knobs (batch_size / rollout_repeats) are always
# injected first; per-row cfg overrides come later and win.
run_tree() {
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
  echo "# [tree] ${tag}"
  echo "#   env:  ${env_kv[*]:-<none>}"
  echo "#   cfg:  ${cfg_kv[*]:-<none>}"
  echo "############################################################"

  local row_out="${OUT_BASE}/${tag}"
  local extra="train.batch_size=${TREE_BASE_BATCH_SIZE} optimizer.type_guided_rollout_repeats=${TREE_BASE_ROLLOUT_REPEATS}"
  if (( ${#cfg_kv[@]} )); then
    extra+=" ${cfg_kv[*]}"
  fi

  local rc=0
  # Subshell so per-row env vars don't leak; do NOT use the external `env`
  # command (run_dataset is a shell function; env only execs real binaries).
  (
    export OUT_BASE="${row_out}" EXTRA_CFG_OPTIONS="${extra}"
    local kv
    for kv in "${env_kv[@]:-}"; do
      [[ -n "${kv}" ]] || continue
      export "${kv?}"
    done
    run_dataset livemathematicianbench \
      --workers "${TREE_WORKERS}" \
      --analyst_workers "${TREE_ANALYST_WORKERS}" \
      --exec_timeout "${TREE_EXEC_TIMEOUT}"
  ) || rc=$?

  if (( rc != 0 )); then
    echo "[tree] ${tag} FAILED rc=${rc}" >&2
    TREE_RESULTS+=("FAIL  ${tag}")
    if ! comparison_truthy "${CONTINUE_ON_ERROR}"; then
      exit "${rc}"
    fi
  else
    TREE_RESULTS+=("OK    ${tag}")
  fi
}

# ── ① tree height / abstraction levels ──────────────────────────────────────
if comparison_truthy "${DO_TREE_DEPTH}"; then
  for td in ${TREE_DEPTH_GRID}; do
    run_tree "tree_depth_${td}" -- "optimizer.type_guided_tree_depth=${td}"
  done
fi

# ── ② tree width: leaf fan-out cap ──────────────────────────────────────────
if comparison_truthy "${DO_MAX_LEAF_GROUPS}"; then
  for mg in ${MAX_LEAF_GROUPS_GRID}; do
    run_tree "max_leaf_groups_${mg}" -- "optimizer.type_guided_max_leaf_groups=${mg}"
  done
fi

# ── ② tree width: record budget fed into the tree ───────────────────────────
if comparison_truthy "${DO_MAX_PATCH_RECORDS}"; then
  for mr in ${MAX_PATCH_RECORDS_GRID}; do
    run_tree "max_patch_records_${mr}" -- "optimizer.type_guided_max_patch_records=${mr}"
  done
fi

# ── ③ grouping granularity: type-key (off) vs semantic clustering (on) ──────
if comparison_truthy "${DO_CLUSTERING}"; then
  run_tree "clustering_off" TYPE_GUIDED_CLUSTERING=0 --
  for cs in ${CLUSTER_TARGET_SIZE_GRID}; do
    run_tree "clustering_on_size_${cs}" TYPE_GUIDED_CLUSTERING=1 -- "optimizer.type_guided_cluster_target_size=${cs}"
  done
fi

# ── ④ fallback fan-out width + recombination strategy ───────────────────────
if comparison_truthy "${DO_FALLBACK}"; then
  for tk in ${FALLBACK_TOPK_GRID}; do
    run_tree "fallback_topk_${tk}" -- "optimizer.type_guided_fallback_top_k=${tk}"
  done
  for rc_mode in ${FALLBACK_RECONCILE_GRID}; do
    run_tree "fallback_reconcile_${rc_mode}" -- "optimizer.type_guided_fallback_reconcile=${rc_mode}"
  done
fi

echo
echo "============================================================"
echo "  [tree] sweep complete — results under ${OUT_BASE}"
for line in "${TREE_RESULTS[@]}"; do
  echo "    ${line}"
done
echo "============================================================"
if printf '%s\n' "${TREE_RESULTS[@]}" | grep -q '^FAIL'; then
  exit 1
fi

#!/usr/bin/env bash
set -euo pipefail

# Pure TEST reevaluation for all saved LiveMath skills from the Ark suite.
# Reuses Volcano Ark deepseek-v4-flash as the target model and runs repeated
# eval_only.py calls sequentially so total target concurrency stays at WORKERS.
#
# Default:
#   - skills: system_prompt_only, init_skill, dynamic_auto, fixed_real_root,
#             dynamic_real_root, dynamic_virtual_root, no_recursive_fallback,
#             min_support_2
#   - repeats: 5 (override with REPEATS=3 for a lighter 3-repeat pass)
#   - workers: 96
#
# Example:
#   ARK_API_KEY=... bash scripts/runs/resource_pool_dsv4_flash_pro_api/20_livemath_ark_saved_skill_retest.sh
#   ARK_API_KEY=... REPEATS=3 bash scripts/runs/resource_pool_dsv4_flash_pro_api/20_livemath_ark_saved_skill_retest.sh
#   ARK_API_KEY=... RESULT_ROOT=outputs/run SKIP_EXISTING=1 SKILLS="min_support_2" bash ...

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_livemath_ark_common.sh
source "${SCRIPT_DIR}/_livemath_ark_common.sh"

find_latest_suite_root() {
  find "${PROJECT_ROOT}/outputs" -maxdepth 1 -type d -name 'livemath_ark_all_*' | sort | tail -n 1
}

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1 && command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN=python3
fi

SOURCE_RUN_ROOT="${SOURCE_RUN_ROOT:-$(find_latest_suite_root)}"
[[ -n "${SOURCE_RUN_ROOT}" ]] || comparison_fail \
  "No outputs/livemath_ark_all_* directory found. Set SOURCE_RUN_ROOT explicitly."

REPEATS="${REPEATS:-5}"
REPEAT_START="${REPEAT_START:-1}"
REPEAT_END="${REPEAT_END:-${REPEATS}}"
BASE_SEED="${BASE_SEED:-42}"
SEED_MODE="${SEED_MODE:-fixed}"
WORKERS="${WORKERS:-96}"
SKILLS="${SKILLS:-}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
LIVEMATH_SPLIT_DIR="${LIVEMATH_SPLIT_DIR:-${DATA_ROOT}/livemathematicianbench_split}"
EVAL_SPLIT="${EVAL_SPLIT:-valid_unseen}"
TEST_ENV_NUM="${TEST_ENV_NUM:-0}"
MAX_TURNS="${MAX_TURNS:-1}"
TARGET_TIMEOUT_SECONDS="${TARGET_TIMEOUT_SECONDS:-1800}"
TARGET_MAX_TOKENS="${TARGET_MAX_TOKENS:-16384}"
EXEC_TIMEOUT="${EXEC_TIMEOUT:-1800}"

[[ "${REPEATS}" =~ ^[0-9]+$ ]] || comparison_fail "REPEATS must be an integer: ${REPEATS}"
[[ "${REPEATS}" -ge 1 ]] || comparison_fail "REPEATS must be >= 1"
[[ "${REPEAT_START}" =~ ^[0-9]+$ ]] || comparison_fail "REPEAT_START must be an integer: ${REPEAT_START}"
[[ "${REPEAT_END}" =~ ^[0-9]+$ ]] || comparison_fail "REPEAT_END must be an integer: ${REPEAT_END}"
[[ "${REPEAT_START}" -ge 1 ]] || comparison_fail "REPEAT_START must be >= 1"
[[ "${REPEAT_END}" -ge "${REPEAT_START}" ]] || comparison_fail "REPEAT_END must be >= REPEAT_START"
[[ "${REPEAT_END}" -le "${REPEATS}" ]] || comparison_fail "REPEAT_END must be <= REPEATS"
[[ "${WORKERS}" =~ ^[0-9]+$ ]] || comparison_fail "WORKERS must be an integer: ${WORKERS}"
[[ "${WORKERS}" -ge 1 ]] || comparison_fail "WORKERS must be >= 1"
[[ "${BASE_SEED}" =~ ^[0-9]+$ ]] || comparison_fail "BASE_SEED must be an integer: ${BASE_SEED}"
case "${SEED_MODE}" in
  fixed|increment) ;;
  *) comparison_fail "SEED_MODE must be fixed or increment (got: ${SEED_MODE})" ;;
esac

skill_selected() {
  local candidate="$1"
  local selected
  [[ -n "${SKILLS}" ]] || return 0
  for selected in ${SKILLS}; do
    [[ "${candidate}" == "${selected}" ]] && return 0
  done
  return 1
}

require_layout
configure_models

[[ -d "${SOURCE_RUN_ROOT}" ]] || comparison_fail "SOURCE_RUN_ROOT not found: ${SOURCE_RUN_ROOT}"
[[ -d "${LIVEMATH_SPLIT_DIR}" ]] || comparison_fail "LiveMath split dir not found: ${LIVEMATH_SPLIT_DIR}"

RUN_ID="${RUN_ID:-livemath_saved_skill_retest_$(date +%Y%m%d_%H%M%S)}"
RESULT_ROOT="${RESULT_ROOT:-${PROJECT_ROOT}/outputs/${RUN_ID}}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs/${RUN_ID}}"
MANIFEST_PATH="${RESULT_ROOT}/skill_manifest.tsv"
mkdir -p "${RESULT_ROOT}" "${LOG_DIR}"

SYSTEM_PROMPT_ONLY_SKILL_PATH="${SYSTEM_PROMPT_ONLY_SKILL_PATH:-${SOURCE_RUN_ROOT}/system_prompt_only/livemathematicianbench/best_skill.md}"
INIT_SKILL_PATH="${INIT_SKILL_PATH:-${SOURCE_RUN_ROOT}/init_skill_no_train/livemathematicianbench/best_skill.md}"
DYNAMIC_AUTO_SKILL_PATH="${DYNAMIC_AUTO_SKILL_PATH:-${SOURCE_RUN_ROOT}/ablations/dynamic_auto/livemathematicianbench/best_skill.md}"
FIXED_REAL_ROOT_SKILL_PATH="${FIXED_REAL_ROOT_SKILL_PATH:-${SOURCE_RUN_ROOT}/ablations/fixed_real_root/livemathematicianbench/best_skill.md}"
DYNAMIC_REAL_ROOT_SKILL_PATH="${DYNAMIC_REAL_ROOT_SKILL_PATH:-${SOURCE_RUN_ROOT}/ablations/dynamic_real_root/livemathematicianbench/best_skill.md}"
DYNAMIC_VIRTUAL_ROOT_SKILL_PATH="${DYNAMIC_VIRTUAL_ROOT_SKILL_PATH:-${SOURCE_RUN_ROOT}/ablations/dynamic_virtual_root/livemathematicianbench/best_skill.md}"
NO_RECURSIVE_FALLBACK_SKILL_PATH="${NO_RECURSIVE_FALLBACK_SKILL_PATH:-${SOURCE_RUN_ROOT}/ablations/no_recursive_fallback/livemathematicianbench/best_skill.md}"
MIN_SUPPORT_2_SKILL_PATH="${MIN_SUPPORT_2_SKILL_PATH:-${SOURCE_RUN_ROOT}/ablations/min_support_2/livemathematicianbench/best_skill.md}"

{
  printf 'skill_name\tskill_path\n'
  printf 'system_prompt_only\t%s\n' "${SYSTEM_PROMPT_ONLY_SKILL_PATH}"
  printf 'init_skill\t%s\n' "${INIT_SKILL_PATH}"
  printf 'dynamic_auto\t%s\n' "${DYNAMIC_AUTO_SKILL_PATH}"
  printf 'fixed_real_root\t%s\n' "${FIXED_REAL_ROOT_SKILL_PATH}"
  printf 'dynamic_real_root\t%s\n' "${DYNAMIC_REAL_ROOT_SKILL_PATH}"
  printf 'dynamic_virtual_root\t%s\n' "${DYNAMIC_VIRTUAL_ROOT_SKILL_PATH}"
  printf 'no_recursive_fallback\t%s\n' "${NO_RECURSIVE_FALLBACK_SKILL_PATH}"
  printf 'min_support_2\t%s\n' "${MIN_SUPPORT_2_SKILL_PATH}"
} > "${MANIFEST_PATH}"

if ! comparison_truthy "${DRY_RUN:-0}"; then
  while IFS=$'\t' read -r skill_name skill_path; do
    [[ "${skill_name}" == "skill_name" ]] && continue
    [[ -f "${skill_path}" ]] || comparison_fail "Skill not found for ${skill_name}: ${skill_path}"
  done < "${MANIFEST_PATH}"
fi

echo "============================================================"
echo "  LiveMath saved-skill pure TEST reevaluation"
echo "============================================================"
echo "  source_run_root:    ${SOURCE_RUN_ROOT}"
echo "  split_dir:          ${LIVEMATH_SPLIT_DIR}"
echo "  split:              ${EVAL_SPLIT}"
echo "  repeats:            ${REPEATS}"
echo "  repeat range:       ${REPEAT_START}-${REPEAT_END}"
echo "  seed mode/base:     ${SEED_MODE} / ${BASE_SEED}"
echo "  skill filter:       ${SKILLS:-<all>}"
echo "  skip existing:      ${SKIP_EXISTING}"
echo "  result_root:        ${RESULT_ROOT}"
echo "  logs:               ${LOG_DIR}"
echo "  target provider:    ${TARGET_PROVIDER_LABEL}"
echo "  target model:       ${TARGET_MODEL}"
echo "  target endpoint:    ${TARGET_AZURE_OPENAI_ENDPOINT}"
echo "  workers:            ${WORKERS}"
echo "  timeout / tokens:   ${TARGET_TIMEOUT_SECONDS}s / ${TARGET_MAX_TOKENS}"
echo "============================================================"

failed=0
while IFS=$'\t' read -r skill_name skill_path; do
  [[ "${skill_name}" == "skill_name" ]] && continue
  skill_selected "${skill_name}" || continue
  for (( repeat_idx = REPEAT_START; repeat_idx <= REPEAT_END; repeat_idx++ )); do
    if [[ "${SEED_MODE}" == "increment" ]]; then
      seed=$((BASE_SEED + repeat_idx - 1))
    else
      seed="${BASE_SEED}"
    fi
    run_name="${skill_name}_r$(printf '%02d' "${repeat_idx}")"
    run_out="${RESULT_ROOT}/${skill_name}/repeat_$(printf '%02d' "${repeat_idx}")"
    run_log="${LOG_DIR}/${run_name}.log"
    summary_path="${run_out}/eval_summary.json"

    if comparison_truthy "${SKIP_EXISTING}" && [[ -f "${summary_path}" ]]; then
      echo
      echo "[skip] ${run_name}"
      echo "       summary=${summary_path}"
      continue
    fi

    eval_cmd=(
      "${PYTHON_BIN}" -u "${PROJECT_ROOT}/scripts/cli/eval_only.py"
      --config "${PROJECT_ROOT}/configs/livemathematicianbench/default.yaml"
      --skill "${skill_path}"
      --split "${EVAL_SPLIT}"
      --out_root "${run_out}"
      --reasoning_effort ""
      --optimizer_backend openai_chat
      --target_backend openai_chat
      --optimizer_model "${OPTIMIZER_MODEL}"
      --target_model "${TARGET_MODEL}"
      --azure_openai_endpoint "${AZURE_OPENAI_ENDPOINT}"
      --azure_openai_api_version "${AZURE_OPENAI_API_VERSION}"
      --azure_openai_api_key "${AZURE_OPENAI_API_KEY}"
      --azure_openai_auth_mode "${AZURE_OPENAI_AUTH_MODE}"
      --optimizer_azure_openai_endpoint "${OPTIMIZER_AZURE_OPENAI_ENDPOINT}"
      --optimizer_azure_openai_api_version "${OPTIMIZER_AZURE_OPENAI_API_VERSION}"
      --optimizer_azure_openai_api_key "${OPTIMIZER_AZURE_OPENAI_API_KEY}"
      --optimizer_azure_openai_auth_mode "${OPTIMIZER_AZURE_OPENAI_AUTH_MODE}"
      --target_azure_openai_endpoint "${TARGET_AZURE_OPENAI_ENDPOINT}"
      --target_azure_openai_api_version "${TARGET_AZURE_OPENAI_API_VERSION}"
      --target_azure_openai_api_key "${TARGET_AZURE_OPENAI_API_KEY}"
      --target_azure_openai_auth_mode "${TARGET_AZURE_OPENAI_AUTH_MODE}"
      --split_mode split_dir
      --split_dir "${LIVEMATH_SPLIT_DIR}"
      --workers "${WORKERS}"
      --seed "${seed}"
      --test_env_num "${TEST_ENV_NUM}"
      --max_turns "${MAX_TURNS}"
      --cfg-options
        "env.exec_timeout=${EXEC_TIMEOUT}"
        "env.max_completion_tokens=${TARGET_MAX_TOKENS}"
        "env.limit=0"
        "env.shuffle_choices=true"
        "env.use_theorem=false"
        "env.use_sketch=false"
    )

    echo
    echo "[eval] ${run_name}"
    echo "       skill=${skill_path}"
    echo "       seed=${seed}"
    echo "       output=${run_out}"
    if comparison_truthy "${DRY_RUN:-0}"; then
      printf '[dry-run]'
      quote_cmd "${eval_cmd[@]}"
      continue
    fi

    set +e
    "${eval_cmd[@]}" 2>&1 | tee "${run_log}"
    status=${PIPESTATUS[0]}
    set -e
    if (( status != 0 )); then
      echo "ERROR: ${run_name} failed with exit=${status}" >&2
      failed=1
    fi
  done
done < "${MANIFEST_PATH}"

if comparison_truthy "${DRY_RUN:-0}"; then
  echo "[dry-run] manifest=${MANIFEST_PATH}"
  exit 0
fi

"${PYTHON_BIN}" - "${RESULT_ROOT}" "${MANIFEST_PATH}" "${REPEATS}" <<'PY'
import csv
import json
import math
import os
import statistics
import sys

result_root, manifest_path, repeats = sys.argv[1], sys.argv[2], int(sys.argv[3])

skills = []
with open(manifest_path, newline="") as f:
    for row in csv.DictReader(f, delimiter="\t"):
        skills.append(row)

rows = []
for skill in skills:
    skill_name = skill["skill_name"]
    for idx in range(1, repeats + 1):
        repeat_name = f"repeat_{idx:02d}"
        summary_path = os.path.join(result_root, skill_name, repeat_name, "eval_summary.json")
        if os.path.exists(summary_path):
            with open(summary_path) as sf:
                summary = json.load(sf)
            rows.append({
                "skill_name": skill_name,
                "repeat_idx": idx,
                "hard": summary.get("hard"),
                "soft": summary.get("soft"),
                "n_items": summary.get("n_items"),
                "status": "ok",
            })
        else:
            rows.append({
                "skill_name": skill_name,
                "repeat_idx": idx,
                "hard": None,
                "soft": None,
                "n_items": None,
                "status": "missing",
            })

repeat_json = os.path.join(result_root, "retest_repeat_results.json")
with open(repeat_json, "w") as f:
    json.dump(rows, f, indent=2, ensure_ascii=False)

repeat_csv = os.path.join(result_root, "retest_repeat_results.csv")
with open(repeat_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["skill_name", "repeat_idx", "hard", "soft", "n_items", "status"])
    writer.writeheader()
    writer.writerows(rows)

repeat_md = os.path.join(result_root, "retest_repeat_results.md")
with open(repeat_md, "w") as f:
    f.write("# LiveMath saved-skill pure TEST repeats\n\n")
    f.write("| skill | repeat | hard | soft | n | status |\n")
    f.write("|---|---:|---:|---:|---:|---|\n")
    for row in rows:
        hard = "-" if row["hard"] is None else f'{row["hard"]:.4f}'
        soft = "-" if row["soft"] is None else f'{row["soft"]:.4f}'
        n_items = "-" if row["n_items"] is None else str(row["n_items"])
        f.write(f'| {row["skill_name"]} | {row["repeat_idx"]} | {hard} | {soft} | {n_items} | {row["status"]} |\n')

agg_rows = []
for skill in skills:
    skill_name = skill["skill_name"]
    subset = [row for row in rows if row["skill_name"] == skill_name and row["hard"] is not None]
    hard_vals = [float(row["hard"]) for row in subset]
    soft_vals = [float(row["soft"]) for row in subset]
    n_items = subset[0]["n_items"] if subset else None
    agg_rows.append({
        "skill_name": skill_name,
        "repeats_ok": len(subset),
        "n_items": n_items,
        "hard_mean": statistics.mean(hard_vals) if hard_vals else None,
        "hard_std": statistics.stdev(hard_vals) if len(hard_vals) >= 2 else (0.0 if len(hard_vals) == 1 else None),
        "hard_min": min(hard_vals) if hard_vals else None,
        "hard_max": max(hard_vals) if hard_vals else None,
        "soft_mean": statistics.mean(soft_vals) if soft_vals else None,
        "soft_std": statistics.stdev(soft_vals) if len(soft_vals) >= 2 else (0.0 if len(soft_vals) == 1 else None),
        "status": "ok" if len(subset) == repeats else "partial",
    })

agg_rows.sort(key=lambda row: (-1.0 if row["hard_mean"] is None else -row["hard_mean"], row["skill_name"]))

agg_json = os.path.join(result_root, "retest_aggregate.json")
with open(agg_json, "w") as f:
    json.dump(agg_rows, f, indent=2, ensure_ascii=False)

agg_csv = os.path.join(result_root, "retest_aggregate.csv")
with open(agg_csv, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "skill_name",
            "repeats_ok",
            "n_items",
            "hard_mean",
            "hard_std",
            "hard_min",
            "hard_max",
            "soft_mean",
            "soft_std",
            "status",
        ],
    )
    writer.writeheader()
    writer.writerows(agg_rows)

agg_md = os.path.join(result_root, "retest_aggregate.md")
with open(agg_md, "w") as f:
    f.write("# LiveMath saved-skill pure TEST aggregate\n\n")
    f.write("| skill | repeats | n | hard mean | hard std | hard min | hard max | soft mean | soft std | status |\n")
    f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|\n")
    for row in agg_rows:
        def fmt(x):
            return "-" if x is None else f"{x:.4f}"
        n_items = "-" if row["n_items"] is None else str(row["n_items"])
        f.write(
            f'| {row["skill_name"]} | {row["repeats_ok"]} | {n_items} | '
            f'{fmt(row["hard_mean"])} | {fmt(row["hard_std"])} | {fmt(row["hard_min"])} | {fmt(row["hard_max"])} | '
            f'{fmt(row["soft_mean"])} | {fmt(row["soft_std"])} | {row["status"]} |\n'
        )

print(f"[summary] {repeat_md}")
print(f"[summary] {agg_md}")
PY

echo
echo "============================================================"
echo "  LiveMath saved-skill pure TEST reevaluation finished"
echo "  results: ${RESULT_ROOT}"
echo "  logs:    ${LOG_DIR}"
echo "============================================================"

(( failed == 0 )) || exit 1

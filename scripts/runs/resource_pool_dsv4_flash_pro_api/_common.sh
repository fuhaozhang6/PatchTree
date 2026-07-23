#!/usr/bin/env bash

# Shared runtime for the DeepSeek-v4 flash/pro API comparison launchers
# (SkillOpt-Tree). Source this file; do not execute it directly.
#
# Unlike the sibling resource_pool_qwen36_35b_a3b_4xl20 group, this group runs
# BOTH models through the DeepSeek official API:
#   - target / student  : deepseek-v4-flash  (--target_backend openai_chat)
#   - optimizer / teacher: deepseek-v4-pro    (--optimizer_backend openai_chat)
# There is no local vLLM service — no GPU, no tensor-parallel, no vLLM guard.
#
# SkillOpt method parameters remain inherited from each dataset's
# configs/<dataset>/default.yaml. As in the other Tree launchers, train.py has
# no --exec_timeout / --llm_timeout flags, so run_dataset translates the
# reference-style `--exec_timeout N` / `--llm_timeout N` arguments into
# `--cfg-options env.exec_timeout=N env.llm_timeout=N` (appended last, because
# --cfg-options is nargs="+").

RESOURCE_POOL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${RESOURCE_POOL_DIR}/../../.." && pwd)"
# Scripts live inside SkillOpt-Tree, so data and train.py are in the same repo.
DATA_PROJECT_ROOT="${DATA_PROJECT_ROOT:-${PROJECT_ROOT}}"
DATA_ROOT="${DATA_ROOT:-${DATA_PROJECT_ROOT}/data}"
PYTHON_BIN="${PYTHON_BIN:-python}"

comparison_fail() {
  echo "ERROR: $*" >&2
  exit 1
}

comparison_truthy() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

quote_cmd() {
  local arg
  for arg in "$@"; do
    printf ' %q' "${arg}"
  done
  printf '\n'
}

require_layout() {
  command -v "${PYTHON_BIN}" >/dev/null 2>&1 || comparison_fail "Python not found: ${PYTHON_BIN}"
  [[ -f "${PROJECT_ROOT}/scripts/train.py" ]] || comparison_fail "SkillOpt-Tree train.py not found"
  [[ -d "${DATA_ROOT}" ]] || comparison_fail "Shared data root not found: ${DATA_ROOT}"
}

# Probe whether the ALFWorld Python stack is importable in the run interpreter.
# Returns 0 when alfworld + gymnasium + omegaconf and our adapter all import.
alfworld_imports_ok() {
  PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}" "${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1
import importlib.util
for name in ("alfworld", "gymnasium", "omegaconf"):
    if importlib.util.find_spec(name) is None:
        raise SystemExit(1)
from skillopt.envs.alfworld.adapter import ALFWorldAdapter  # noqa: F401
PY
}

# Ensure ALFWorld data + Python deps are ready before launching a run. The
# gymnasium/omegaconf deps ship in the ".[alfworld]" extra but are easy to miss
# on a fresh node (the run then dies mid-training on `import gymnasium`). When
# ALFWORLD_AUTO_INSTALL=1 (default) we pip-install them on the fly and re-probe;
# set it to 0 to only self-check and fail with the exact install command.
require_alfworld_environment() {
  export ALFWORLD_DATA="${ALFWORLD_DATA:-${DATA_ROOT}/alfworld}"
  if comparison_truthy "${DRY_RUN:-0}"; then
    return
  fi
  [[ -d "${ALFWORLD_DATA}/json_2.1.1" ]] || comparison_fail \
    "ALFWorld data not found at ${ALFWORLD_DATA}/json_2.1.1 (unpack the split data bundle first)"

  if alfworld_imports_ok; then
    return
  fi

  if comparison_truthy "${ALFWORLD_AUTO_INSTALL:-1}"; then
    echo "[deps] ALFWorld Python deps missing — installing .[alfworld] + omegaconf json_repair ..."
    ( cd "${PROJECT_ROOT}" \
        && "${PYTHON_BIN}" -m pip install -e ".[alfworld]" \
        && "${PYTHON_BIN}" -m pip install omegaconf json_repair ) \
      || comparison_fail "ALFWorld dependency install failed (see pip output above)"
  fi

  alfworld_imports_ok || comparison_fail \
    "ALFWorld Python env still not ready. Run: cd ${PROJECT_ROOT} && ${PYTHON_BIN} -m pip install -e \".[alfworld]\" && ${PYTHON_BIN} -m pip install omegaconf json_repair"
}

configure_models() {
  # Both models come from the DeepSeek official API. Optimizer (teacher) and
  # target (student) share one endpoint + key but use different model names.
  local endpoint key
  endpoint="${DEEPSEEK_BASE_URL:-${DEEPSEEK_OFFICIAL_BASE_URL:-https://api.deepseek.com}}"
  key="${DEEPSEEK_API_KEY:-${DS_API_KEY:-${DEEPSEEK_OFFICIAL_API_KEY:-}}}"
  if [[ -z "${key}" ]] && ! comparison_truthy "${DRY_RUN:-0}"; then
    comparison_fail "DEEPSEEK_API_KEY is required. DS_API_KEY and DEEPSEEK_OFFICIAL_API_KEY are also accepted."
  fi
  export DEEPSEEK_API_KEY="${key}"

  export OPTIMIZER_SOURCE="deepseek_official"
  export OPTIMIZER_PROVIDER_LABEL="DeepSeek official"
  export TARGET_PROVIDER_LABEL="DeepSeek official"
  export OPTIMIZER_MODEL="${OPTIMIZER_MODEL:-${DEEPSEEK_OPTIMIZER_MODEL:-deepseek-v4-pro}}"
  export TARGET_MODEL="${TARGET_MODEL:-${DEEPSEEK_TARGET_MODEL:-deepseek-v4-flash}}"

  # Shared OpenAI-compatible backend defaults. Set these explicitly so stale
  # AZURE_OPENAI_* variables from a previous run cannot silently mix sources.
  export AZURE_OPENAI_ENDPOINT="${endpoint}"
  export AZURE_OPENAI_API_KEY="${key}"
  export AZURE_OPENAI_AUTH_MODE=openai_compatible
  export AZURE_OPENAI_API_VERSION=openai-compat

  # Optimizer (teacher) = deepseek-v4-pro on the DeepSeek official endpoint.
  export OPTIMIZER_AZURE_OPENAI_ENDPOINT="${endpoint}"
  export OPTIMIZER_AZURE_OPENAI_API_KEY="${key}"
  export OPTIMIZER_AZURE_OPENAI_AUTH_MODE=openai_compatible
  export OPTIMIZER_AZURE_OPENAI_API_VERSION=openai-compat

  # Target (student) = deepseek-v4-flash on the same endpoint. Keeping a
  # separate TARGET_* block lets you point the student at a different
  # endpoint/key later without touching the optimizer wiring.
  export TARGET_AZURE_OPENAI_ENDPOINT="${TARGET_AZURE_OPENAI_ENDPOINT:-${endpoint}}"
  export TARGET_AZURE_OPENAI_API_KEY="${TARGET_AZURE_OPENAI_API_KEY:-${key}}"
  export TARGET_AZURE_OPENAI_AUTH_MODE="${TARGET_AZURE_OPENAI_AUTH_MODE:-openai_compatible}"
  export TARGET_AZURE_OPENAI_API_VERSION="${TARGET_AZURE_OPENAI_API_VERSION:-openai-compat}"

  # DeepSeek thinking is disabled for both flash (student) and pro (teacher):
  # _deepseek_extra_body() applies DEEPSEEK_THINKING to every deepseek-* call.
  export DEEPSEEK_THINKING="${DEEPSEEK_THINKING:-disabled}"
  export DEEPSEEK_OFFICIAL_THINKING="${DEEPSEEK_OFFICIAL_THINKING:-${DEEPSEEK_THINKING}}"
  export REASONING_EFFORT=""
  export REWRITE_REASONING_EFFORT=""

  # Per-call ceiling for the student's generations. Override for long-form
  # datasets if needed.
  export TARGET_MAX_COMPLETION_TOKENS="${TARGET_MAX_COMPLETION_TOKENS:-16384}"
}

DATASET_PIDS=()
DATASET_NAMES=()
DATASET_OUTPUTS=()

cleanup_datasets() {
  local pid
  for pid in "${DATASET_PIDS[@]:-}"; do
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
    fi
  done
}

run_dataset() {
  local dataset="$1"
  shift
  local config_path="${PROJECT_ROOT}/configs/${dataset}/default.yaml"
  local skill_path="${PROJECT_ROOT}/skillopt/envs/${dataset}/skills/initial.md"
  local split_dir
  local output_dir="${OUT_BASE}/${dataset}"
  local -a dataset_args=()

  case "${dataset}" in
    searchqa)
      split_dir="${DATA_ROOT}/searchqa_split"
      ;;
    alfworld)
      split_dir="${DATA_ROOT}/alfworld_path_split"
      export ALFWORLD_DATA="${ALFWORLD_DATA:-${DATA_ROOT}/alfworld}"
      ;;
    docvqa)
      split_dir="${DATA_ROOT}/docvqa/splits"
      ;;
    spreadsheetbench)
      split_dir="${DATA_ROOT}/spreadsheetbench_split"
      dataset_args+=(--data_root "${DATA_ROOT}/spreadsheetbench_verified_400")
      ;;
    livemathematicianbench)
      split_dir="${DATA_ROOT}/livemathematicianbench_split"
      ;;
    officeqa)
      split_dir="${DATA_ROOT}/officeqa_split"
      export OFFICEQA_DOCS_DIR="${OFFICEQA_DOCS_DIR:-${DATA_ROOT}/officeqa_docs_official}"
      ;;
    *) comparison_fail "Unknown dataset: ${dataset}" ;;
  esac

  [[ -f "${config_path}" ]] || comparison_fail "Config not found: ${config_path}"
  [[ -f "${skill_path}" ]] || comparison_fail "Initial skill not found: ${skill_path}"
  [[ -d "${split_dir}" ]] || comparison_fail "Materialized split not found: ${split_dir}"

  # Tree's train.py has no --exec_timeout / --llm_timeout flags. Translate the
  # reference-style arguments into structured --cfg-options overrides so the
  # per-dataset launchers can stay identical to the vLLM-group launchers.
  local -a passthrough=()
  local -a cfg_opts=()
  while (( $# )); do
    case "$1" in
      --exec_timeout)
        [[ $# -ge 2 ]] || comparison_fail "--exec_timeout requires a value"
        cfg_opts+=("env.exec_timeout=$2")
        shift 2
        ;;
      --llm_timeout)
        [[ $# -ge 2 ]] || comparison_fail "--llm_timeout requires a value"
        cfg_opts+=("env.llm_timeout=$2")
        shift 2
        ;;
      *)
        passthrough+=("$1")
        shift
        ;;
    esac
  done

  # Invoke the CLI as a module (python -m scripts.cli.train) instead of running
  # scripts/train.py by path. With -m, Python puts the working directory (we cd
  # into DATA_PROJECT_ROOT below) on sys.path, so `import scripts.cli...` resolves
  # without relying on the wrapper's sys.path injection. This is more robust when
  # only part of the repo was synced to a remote box. Note: this still requires
  # scripts/cli/ to physically exist on the run machine — if it does not, sync
  # the full repo (see the group README) rather than editing this launcher.
  local -a cmd=(
    "${PYTHON_BIN}" -m scripts.cli.train
    --config "${config_path}"
    --optimizer_model "${OPTIMIZER_MODEL}"
    --target_model "${TARGET_MODEL}"
    --optimizer_backend openai_chat
    --target_backend openai_chat
    --reasoning_effort ""
    --skill_init "${skill_path}"
    --split_dir "${split_dir}"
    --out_root "${output_dir}"
  )
  if (( ${#dataset_args[@]} )); then
    cmd+=("${dataset_args[@]}")
  fi
  if (( ${#passthrough[@]} )); then
    cmd+=("${passthrough[@]}")
  fi
  # env.max_completion_tokens caps the student generations; it is always set, so
  # the trailing --cfg-options block is never empty. Build it fresh (rather than
  # prepending into cfg_opts) so an empty cfg_opts cannot trip `set -u`.
  local -a all_cfg=("env.max_completion_tokens=${TARGET_MAX_COMPLETION_TOKENS}")
  # Optional type-guided-v2 toggles. These live in the SAME --cfg-options block
  # as the env.* keys above, because train.py's argparse uses nargs="+" and a
  # second --cfg-options occurrence would clobber the first. Set the env vars to
  # 1/true to enable; leaving them unset keeps the config defaults (both false).
  if comparison_truthy "${TYPE_GUIDED_CLUSTERING:-0}"; then
    all_cfg+=("optimizer.type_guided_clustering=true")
  fi
  if comparison_truthy "${TYPE_GUIDED_TAIL_BANK:-0}"; then
    all_cfg+=("optimizer.type_guided_tail_bank=true")
  fi
  # Root-reject -> child fallback. type_guided_leaf_fallback defaults to true in
  # the base config: when the merged root patch is rejected by the gate, the
  # trainer falls back to evaluating individual leaf/child patches. Set
  # TYPE_GUIDED_LEAF_FALLBACK=0 (or false) to DISABLE that fallback, so a rejected
  # root simply ends the step with no child search. Leaving it unset keeps the
  # default (enabled); we only emit an override when explicitly turned off.
  if [[ -n "${TYPE_GUIDED_LEAF_FALLBACK:-}" ]] && ! comparison_truthy "${TYPE_GUIDED_LEAF_FALLBACK}"; then
    all_cfg+=("optimizer.type_guided_leaf_fallback=false")
  fi
  # Edit-budget (a.k.a. optimizer.learning_rate) control. The base config caps
  # each step at learning_rate=4 edits, decaying toward min_learning_rate=2 via
  # the cosine scheduler — so most merged patches get clipped hard before the
  # gate ever sees them. Set EDIT_BUDGET_OFF=1 to switch the scheduler to
  # "autonomous", whose LRScheduler returns NO_LIMIT=999 every step (no extra LLM
  # call, unlike lr_control_mode=autonomous), effectively removing the edit cap.
  # This is merged into the SAME --cfg-options block on purpose: passing
  # `--cfg-options optimizer.lr_scheduler=autonomous` on the CLI would be a
  # SECOND --cfg-options occurrence and argparse (nargs="+") would let the
  # built-in block clobber it, silently dropping the override.
  if comparison_truthy "${EDIT_BUDGET_OFF:-0}"; then
    all_cfg+=("optimizer.lr_scheduler=autonomous")
  fi
  if (( ${#cfg_opts[@]} )); then
    all_cfg+=("${cfg_opts[@]}")
  fi
  # General escape hatch for arbitrary `key=value` overrides (space-separated),
  # merged into the SAME --cfg-options block. Ablation launchers use this to
  # inject per-run knobs (train.batch_size, optimizer.type_guided_rollout_repeats,
  # optimizer.type_guided_leaf_fallback, ...) without adding a second
  # --cfg-options occurrence, which argparse (nargs="+") would let clobber the
  # built-in block. Placed last so an explicit ablation value wins over the
  # env-toggle defaults above.
  if [[ -n "${EXTRA_CFG_OPTIONS:-}" ]]; then
    # shellcheck disable=SC2206 - intentional word-splitting into cfg tokens.
    local -a extra_cfg=(${EXTRA_CFG_OPTIONS})
    all_cfg+=("${extra_cfg[@]}")
  fi
  # --cfg-options is nargs="+"; keep it last so it does not swallow later flags.
  cmd+=(--cfg-options "${all_cfg[@]}")

  echo "------------------------------------------------------------"
  echo "[train] dataset=${dataset} output=${output_dir}"
  echo "[cmd] cd ${DATA_PROJECT_ROOT} && PYTHONPATH=${DATA_PROJECT_ROOT}:\${PYTHONPATH}$(quote_cmd "${cmd[@]}")"
  if comparison_truthy "${DRY_RUN:-0}"; then
    return
  fi
  # Export the repo root on PYTHONPATH as a belt-and-suspenders fallback so
  # `import scripts.cli...` resolves even if `python -m` CWD injection is
  # disturbed by a wrapper or virtualenv.
  (cd "${DATA_PROJECT_ROOT}" \
     && PYTHONPATH="${DATA_PROJECT_ROOT}:${PYTHONPATH:-}" "${cmd[@]}")
  guard_dataset_results "${dataset}" "${output_dir}"
}

run_dataset_background() {
  local dataset="$1"
  shift
  if comparison_truthy "${DRY_RUN:-0}"; then
    run_dataset "${dataset}" "$@"
    return
  fi

  local log_dir="${OUT_BASE}/logs"
  local log_path="${log_dir}/${dataset}.log"
  mkdir -p "${log_dir}"
  run_dataset "${dataset}" "$@" >"${log_path}" 2>&1 &
  local pid=$!
  DATASET_PIDS+=("${pid}")
  DATASET_NAMES+=("${dataset}")
  DATASET_OUTPUTS+=("${OUT_BASE}/${dataset}")
  echo "[launch] dataset=${dataset} pid=${pid} log=${log_path}"
}

guard_dataset_results() {
  local dataset="$1"
  local output_dir="$2"
  if ! comparison_truthy "${RESULT_GUARD:-1}"; then
    return 0
  fi
  if comparison_truthy "${DRY_RUN:-0}"; then
    return 0
  fi
  [[ -d "${output_dir}" ]] || return 0
  RESULT_GUARD_MIN_RECORDS="${RESULT_GUARD_MIN_RECORDS:-8}" \
  "${PYTHON_BIN}" - "${dataset}" "${output_dir}" <<'PY'
from __future__ import annotations

import json
import os
import sys

dataset = sys.argv[1]
root = sys.argv[2]
min_records = int(os.environ.get("RESULT_GUARD_MIN_RECORDS", "8") or 8)

paths = []
for dirpath, _, filenames in os.walk(root):
    if "results.jsonl" in filenames:
        paths.append(os.path.join(dirpath, "results.jsonl"))

rows = []
for path in paths:
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                row["_result_path"] = path
                rows.append(row)
    except OSError:
        continue

if len(rows) < min_records:
    print(f"[guard] dataset={dataset} records={len(rows)} below threshold={min_records}; skip all-zero guard")
    raise SystemExit(0)

def positive(row: dict) -> bool:
    for key in ("hard", "soft", "score", "reward", "success"):
        value = row.get(key)
        if isinstance(value, bool) and value:
            return True
        if isinstance(value, (int, float)) and float(value) > 0:
            return True
    return False

failure_terms = (
    "call failed",
    "connection refused",
    "timed out",
    "timeout",
    "task-timeout",
    "api request failed",
    "http 401",
    "http 402",
    "http 429",
    "http 500",
    "http 502",
    "http 503",
    "http 504",
    "no choices",
    "non-json response",
    "no structured tool",
    "model neither produced",
    "empty message",
)

positives = sum(1 for row in rows if positive(row))
failish = []
for row in rows:
    text = " ".join(
        str(row.get(key, ""))
        for key in ("fail_reason", "error", "phase", "response", "last_finish_reason")
    ).lower()
    if any(term in text for term in failure_terms):
        failish.append(row)

if positives == 0 and failish and len(failish) / max(len(rows), 1) >= 0.5:
    print(
        f"[guard:fatal] dataset={dataset} all-zero suspicious results: "
        f"records={len(rows)} failish={len(failish)} paths={len(paths)}",
        file=sys.stderr,
    )
    sample = failish[0]
    print(
        "[guard:fatal] sample "
        + json.dumps(
            {
                "id": sample.get("id"),
                "fail_reason": sample.get("fail_reason"),
                "path": sample.get("_result_path"),
            },
            ensure_ascii=False,
        ),
        file=sys.stderr,
    )
    raise SystemExit(2)

print(f"[guard] dataset={dataset} records={len(rows)} positives={positives} failish={len(failish)}")
PY
}

wait_for_datasets() {
  local status=0
  local index pid name exit_code

  if [[ -z "${DATASET_PIDS[*]:-}" ]]; then
    return 0
  fi

  for index in "${!DATASET_PIDS[@]}"; do
    pid="${DATASET_PIDS[${index}]}"
    name="${DATASET_NAMES[${index}]}"
    if wait "${pid}"; then
      echo "[complete] dataset=${name} exit=0"
      guard_dataset_results "${name}" "${DATASET_OUTPUTS[${index}]}"
    else
      exit_code=$?
      echo "[failed] dataset=${name} exit=${exit_code}" >&2
      status=1
    fi
  done
  return "${status}"
}

print_comparison_header() {
  local group="$1"
  local datasets="$2"
  echo "============================================================"
  echo "  SkillOpt-Tree comparison: ${group}"
  echo "============================================================"
  echo "  datasets:           ${datasets}"
  echo "  optimizer source:   ${OPTIMIZER_SOURCE}"
  echo "  optimizer endpoint: ${OPTIMIZER_AZURE_OPENAI_ENDPOINT}"
  echo "  optimizer:          ${OPTIMIZER_MODEL} (${OPTIMIZER_PROVIDER_LABEL})"
  echo "  target endpoint:    ${TARGET_AZURE_OPENAI_ENDPOINT}"
  echo "  target:             ${TARGET_MODEL} (${TARGET_PROVIDER_LABEL})"
  echo "  target max tokens:  ${TARGET_MAX_COMPLETION_TOKENS}"
  echo "  deepseek thinking:  ${DEEPSEEK_THINKING}"
  echo "  method parameters:  inherited from configs/<dataset>/default.yaml"
  echo "  shared data:        ${DATA_ROOT}"
  echo "  output:             ${OUT_BASE}"
  echo "============================================================"
}

#!/usr/bin/env bash

# Shared runtime for the Qwen3.6-35B-A3B comparison launchers (SkillOpt-Tree).
# Source this file; do not execute it directly.
#
# This mirrors SkillOpt-main/scripts/runs/resource_pool_qwen36_35b_a3b_4xl20,
# adapted to the SkillOpt-Tree CLI. The one meaningful contract difference is
# that Tree's scripts/train.py (scripts/cli/train.py) does NOT define an
# --exec_timeout / --llm_timeout flag; argparse would reject them. It does
# accept --cfg-options and exposes env.exec_timeout / env.llm_timeout in the
# structured config, so run_dataset translates the reference-style
# `--exec_timeout N` / `--llm_timeout N` arguments into
# `--cfg-options env.exec_timeout=N env.llm_timeout=N` (appended last, because
# --cfg-options is nargs="+"). Per-dataset launchers stay identical to the
# SkillOpt-main reference.

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
  local optimizer_source optimizer_endpoint optimizer_key optimizer_label
  optimizer_source="${OPTIMIZER_SOURCE:-ark}"
  case "${optimizer_source}" in
    ark|volcano_ark|volcengine_ark)
      optimizer_source="ark"
      optimizer_endpoint="${ARK_BASE_URL:-https://ark.cn-beijing.volces.com/api/v3}"
      optimizer_key="${ARK_API_KEY:-}"
      optimizer_label="Volcano Ark"
      export OPTIMIZER_MODEL="${OPTIMIZER_MODEL:-deepseek-v4-pro-260425}"
      if [[ -z "${optimizer_key}" ]] && ! comparison_truthy "${DRY_RUN:-0}"; then
        comparison_fail "ARK_API_KEY is required for Volcano Ark optimizer."
      fi
      ;;
    deepseek_official|deepseek|official)
      optimizer_source="deepseek_official"
      optimizer_endpoint="${DEEPSEEK_BASE_URL:-${DEEPSEEK_OFFICIAL_BASE_URL:-https://api.deepseek.com}}"
      optimizer_key="${DEEPSEEK_API_KEY:-${DS_API_KEY:-${DEEPSEEK_OFFICIAL_API_KEY:-}}}"
      optimizer_label="DeepSeek official"
      export DEEPSEEK_API_KEY="${optimizer_key}"
      export OPTIMIZER_MODEL="${OPTIMIZER_MODEL:-${DEEPSEEK_MODEL:-deepseek-v4-pro}}"
      if [[ -z "${optimizer_key}" ]] && ! comparison_truthy "${DRY_RUN:-0}"; then
        comparison_fail "DEEPSEEK_API_KEY is required for DeepSeek official optimizer. DS_API_KEY and DEEPSEEK_OFFICIAL_API_KEY are also accepted."
      fi
      ;;
    *)
      comparison_fail "Unknown OPTIMIZER_SOURCE='${optimizer_source}'. Use 'ark' or 'deepseek_official'."
      ;;
  esac

  export OPTIMIZER_SOURCE="${optimizer_source}"
  export OPTIMIZER_PROVIDER_LABEL="${optimizer_label}"
  export TARGET_MODEL="${TARGET_MODEL:-Qwen3.6-35B-A3B}"

  # Optimizer through SkillOpt's OpenAI-compatible backend. Set these
  # explicitly from OPTIMIZER_SOURCE so stale AZURE_OPENAI_* variables from a
  # previous run cannot silently mix Ark and DeepSeek official sources.
  export AZURE_OPENAI_ENDPOINT="${optimizer_endpoint}"
  export AZURE_OPENAI_API_KEY="${optimizer_key}"
  export AZURE_OPENAI_AUTH_MODE=openai_compatible
  export AZURE_OPENAI_API_VERSION=openai-compat
  export OPTIMIZER_AZURE_OPENAI_ENDPOINT="${optimizer_endpoint}"
  export OPTIMIZER_AZURE_OPENAI_API_KEY="${optimizer_key}"
  export OPTIMIZER_AZURE_OPENAI_AUTH_MODE=openai_compatible
  export OPTIMIZER_AZURE_OPENAI_API_VERSION=openai-compat
  export DEEPSEEK_THINKING="${DEEPSEEK_THINKING:-disabled}"
  export DEEPSEEK_OFFICIAL_THINKING="${DEEPSEEK_OFFICIAL_THINKING:-${DEEPSEEK_THINKING}}"
  export REASONING_EFFORT=""
  export REWRITE_REASONING_EFFORT=""

  # Target: local Qwen3.6-35B-A3B served by vLLM. This model cannot use the
  # single-L20 4B defaults; keep method parameters identical while using a
  # practical 4xL20 vLLM profile by default.
  export MODEL_PATH="${MODEL_PATH:-/ai-app-vepfs/models/Qwen3.6-35B-A3B}"
  export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3.6-35B-A3B}"
  export VLLM_PORT="${VLLM_PORT:-59317}"
  export LOCAL_BASE_URL="${LOCAL_BASE_URL:-http://127.0.0.1:${VLLM_PORT}/v1}"
  export LOCAL_API_KEY="${LOCAL_API_KEY:-dummy}"
  export QWEN_CHAT_BASE_URL="${QWEN_CHAT_BASE_URL:-${LOCAL_BASE_URL}}"
  export QWEN_CHAT_API_KEY="${QWEN_CHAT_API_KEY:-${LOCAL_API_KEY}}"
  export QWEN_CHAT_MODEL="${QWEN_CHAT_MODEL:-${TARGET_MODEL}}"
  export QWEN_CHAT_TEMPERATURE="${QWEN_CHAT_TEMPERATURE:-0.2}"
  export QWEN_CHAT_TIMEOUT_SECONDS="${QWEN_CHAT_TIMEOUT_SECONDS:-1800}"
  export QWEN_CHAT_MAX_TOKENS="${QWEN_CHAT_MAX_TOKENS:-16384}"
  export QWEN_CHAT_ENABLE_THINKING="${QWEN_CHAT_ENABLE_THINKING:-false}"
  export TARGET_QWEN_CHAT_BASE_URL="${TARGET_QWEN_CHAT_BASE_URL:-${LOCAL_BASE_URL}}"
  export TARGET_QWEN_CHAT_API_KEY="${TARGET_QWEN_CHAT_API_KEY:-${LOCAL_API_KEY}}"
  export TARGET_QWEN_CHAT_MODEL="${TARGET_QWEN_CHAT_MODEL:-${TARGET_MODEL}}"
  export TARGET_QWEN_CHAT_TEMPERATURE="${TARGET_QWEN_CHAT_TEMPERATURE:-${QWEN_CHAT_TEMPERATURE}}"
  export TARGET_QWEN_CHAT_TIMEOUT_SECONDS="${TARGET_QWEN_CHAT_TIMEOUT_SECONDS:-${QWEN_CHAT_TIMEOUT_SECONDS}}"
  export TARGET_QWEN_CHAT_MAX_TOKENS="${TARGET_QWEN_CHAT_MAX_TOKENS:-${QWEN_CHAT_MAX_TOKENS}}"
  export TARGET_QWEN_CHAT_ENABLE_THINKING="${TARGET_QWEN_CHAT_ENABLE_THINKING:-${QWEN_CHAT_ENABLE_THINKING}}"

  export START_VLLM="${START_VLLM:-1}"
  export STOP_VLLM_ON_EXIT="${STOP_VLLM_ON_EXIT:-1}"
  export VLLM_HOST="${VLLM_HOST:-0.0.0.0}"
  export MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
  export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
  export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-32}"
  export VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-32768}"
  export VLLM_DTYPE="${VLLM_DTYPE:-bfloat16}"
  export VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-4}"
  export VLLM_ENABLE_AUTO_TOOL_CHOICE="${VLLM_ENABLE_AUTO_TOOL_CHOICE:-0}"
  # Qwen3.x emits tool calls in the XML grammar
  # (<tool_call><function=name><parameter=key>value</parameter></function></tool_call>).
  # The `hermes` parser expects JSON *inside* <tool_call>...</tool_call> and
  # crashes with "JSONDecodeError: Expecting value" on the XML body, so vLLM
  # returns no structured tool_calls and tool-using tasks (OfficeQA) never
  # retrieve evidence. Use the matching XML parser instead.
  export VLLM_TOOL_CALL_PARSER="${VLLM_TOOL_CALL_PARSER:-qwen3_coder}"
  export VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"

  if comparison_truthy "${START_VLLM}"; then
    local visible_devices
    visible_devices="${QWEN_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-0,1,2,3}}"
    visible_devices="${visible_devices// /}"
    [[ -n "${visible_devices}" ]] || comparison_fail "No GPU selected"
    # Count the selected GPUs and require the tensor-parallel size to match, so
    # a single vLLM engine shards Qwen3.6-35B-A3B across exactly the requested
    # cards. Default TP=4 is the practical L20 profile; TP=2 is usually only
    # viable on larger-memory cards such as H20.
    local device_count
    device_count="$(awk -F',' '{print NF}' <<<"${visible_devices}")"
    if [[ "${device_count}" != "${VLLM_TENSOR_PARALLEL_SIZE}" ]]; then
      comparison_fail "CUDA devices '${visible_devices}' (count=${device_count}) must match VLLM_TENSOR_PARALLEL_SIZE=${VLLM_TENSOR_PARALLEL_SIZE}"
    fi
    export QWEN_CUDA_VISIBLE_DEVICES="${visible_devices}"
    if command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
      local runtime_gpu_count
      runtime_gpu_count="$(
        env CUDA_VISIBLE_DEVICES="${QWEN_CUDA_VISIBLE_DEVICES}" "${PYTHON_BIN}" - <<'PY' 2>/dev/null || true
try:
    import torch
except Exception:
    raise SystemExit(0)
print(torch.cuda.device_count())
PY
      )"
      if [[ -n "${runtime_gpu_count}" ]] && [[ "${runtime_gpu_count}" != "${VLLM_TENSOR_PARALLEL_SIZE}" ]]; then
        comparison_fail "Runtime exposes ${runtime_gpu_count} CUDA GPU(s), but VLLM_TENSOR_PARALLEL_SIZE=${VLLM_TENSOR_PARALLEL_SIZE}. Request a 4-GPU allocation or lower both CUDA_VISIBLE_DEVICES and TP for a model that fits."
      fi
    fi
  fi
}

VLLM_PID=""
STARTED_VLLM=0
DATASET_PIDS=()
DATASET_NAMES=()
DATASET_OUTPUTS=()

cleanup_vllm() {
  local pid
  for pid in "${DATASET_PIDS[@]:-}"; do
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
    fi
  done
  if [[ "${STARTED_VLLM}" == "1" ]] && comparison_truthy "${STOP_VLLM_ON_EXIT}"; then
    if [[ -n "${VLLM_PID}" ]] && kill -0 "${VLLM_PID}" 2>/dev/null; then
      echo "[vllm] stopping pid=${VLLM_PID}"
      kill "${VLLM_PID}" 2>/dev/null || true
      wait "${VLLM_PID}" 2>/dev/null || true
    fi
  fi
}

endpoint_ready() {
  BENCH_BASE_URL="${LOCAL_BASE_URL}" BENCH_API_KEY="${LOCAL_API_KEY}" \
    "${PYTHON_BIN}" - <<'PY'
import os
import urllib.request

request = urllib.request.Request(
    os.environ["BENCH_BASE_URL"].rstrip("/") + "/models",
    headers={"Authorization": f"Bearer {os.environ['BENCH_API_KEY']}"},
)
try:
    with urllib.request.urlopen(request, timeout=3) as response:
        raise SystemExit(0 if response.status == 200 else 1)
except Exception:
    raise SystemExit(1)
PY
}

start_local_vllm() {
  if comparison_truthy "${DRY_RUN:-0}"; then
    echo "[dry-run] would start or reuse vLLM at ${LOCAL_BASE_URL}"
    return
  fi
  if endpoint_ready; then
    echo "[vllm] reusing ready endpoint: ${LOCAL_BASE_URL}"
    return
  fi
  comparison_truthy "${START_VLLM}" || comparison_fail "Endpoint is not ready and START_VLLM=${START_VLLM}"
  command -v vllm >/dev/null 2>&1 || comparison_fail "vllm command not found"
  [[ -d "${MODEL_PATH}" ]] || comparison_fail "MODEL_PATH not found: ${MODEL_PATH}"
  mkdir -p "${OUT_BASE}/vllm"

  local vllm_args=(
    serve "${MODEL_PATH}"
    --served-model-name "${SERVED_MODEL_NAME}"
    --host "${VLLM_HOST}"
    --port "${VLLM_PORT}"
    --trust-remote-code
    --dtype "${VLLM_DTYPE}"
    --tensor-parallel-size "${VLLM_TENSOR_PARALLEL_SIZE}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
    --max-model-len "${MAX_MODEL_LEN}"
    --max-num-seqs "${VLLM_MAX_NUM_SEQS}"
    --max-num-batched-tokens "${VLLM_MAX_NUM_BATCHED_TOKENS}"
    --enable-prefix-caching
    --enable-chunked-prefill
  )
  if comparison_truthy "${VLLM_ENABLE_AUTO_TOOL_CHOICE}"; then
    vllm_args+=(--enable-auto-tool-choice --tool-call-parser "${VLLM_TOOL_CALL_PARSER}")
  fi
  if [[ -n "${VLLM_EXTRA_ARGS}" ]]; then
    # shellcheck disable=SC2206
    local extra_args=(${VLLM_EXTRA_ARGS})
    vllm_args+=("${extra_args[@]}")
  fi

  local vllm_log="${OUT_BASE}/vllm/vllm.log"
  echo "[vllm] starting ${SERVED_MODEL_NAME} on CUDA_VISIBLE_DEVICES=${QWEN_CUDA_VISIBLE_DEVICES}"
  env CUDA_VISIBLE_DEVICES="${QWEN_CUDA_VISIBLE_DEVICES}" \
    nohup vllm "${vllm_args[@]}" > "${vllm_log}" 2>&1 &
  VLLM_PID=$!
  STARTED_VLLM=1
  echo "${VLLM_PID}" > "${OUT_BASE}/vllm/vllm.pid"

  local deadline=$((SECONDS + ${VLLM_STARTUP_TIMEOUT_SECONDS:-1200}))
  until endpoint_ready; do
    if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
      tail -n 100 "${vllm_log}" >&2 || true
      comparison_fail "vLLM exited during startup"
    fi
    if (( SECONDS >= deadline )); then
      tail -n 100 "${vllm_log}" >&2 || true
      comparison_fail "Timed out waiting for vLLM"
    fi
    sleep 3
  done
  echo "[vllm] ready pid=${VLLM_PID} log=${vllm_log}"
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
  # per-dataset launchers can stay identical to the SkillOpt-main reference.
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

  local -a cmd=(
    "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/train.py"
    --config "${config_path}"
    --optimizer_model "${OPTIMIZER_MODEL}"
    --target_model "${TARGET_MODEL}"
    --optimizer_backend openai_chat
    --target_backend qwen_chat
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
  # --cfg-options is nargs="+"; keep it last so it does not swallow later flags.
  if (( ${#cfg_opts[@]} )); then
    cmd+=(--cfg-options "${cfg_opts[@]}")
  fi

  echo "------------------------------------------------------------"
  echo "[train] dataset=${dataset} output=${output_dir}"
  echo "[cmd] cd ${DATA_PROJECT_ROOT} &&$(quote_cmd "${cmd[@]}")"
  if comparison_truthy "${DRY_RUN:-0}"; then
    return
  fi
  (cd "${DATA_PROJECT_ROOT}" && "${cmd[@]}")
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
    "qwen chat call failed",
    "connection refused",
    "timed out",
    "timeout",
    "task-timeout",
    "api request failed",
    "http 429",
    "http 500",
    "http 502",
    "http 503",
    "http 504",
    "no choices",
    "non-json response",
    "tool request",
    "tool_call",
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
  local index pid name exit_code alive
  local poll_seconds="${VLLM_HEALTHCHECK_INTERVAL_SECONDS:-5}"

  if [[ -z "${DATASET_PIDS[*]:-}" ]]; then
    return 0
  fi

  # A rollout converts target-model call failures into zero-score records. If
  # vLLM dies, fail the whole launcher promptly instead of silently producing
  # an apparently valid all-zero experiment for hours.
  while true; do
    if [[ "${STARTED_VLLM}" == "1" ]] && [[ -n "${VLLM_PID}" ]] \
      && ! kill -0 "${VLLM_PID}" 2>/dev/null; then
      echo "[fatal] vLLM exited while datasets were running; stopping all datasets" >&2
      tail -n 80 "${OUT_BASE}/vllm/vllm.log" >&2 || true
      for pid in "${DATASET_PIDS[@]}"; do
        if kill -0 "${pid}" 2>/dev/null; then
          kill "${pid}" 2>/dev/null || true
        fi
      done
      for pid in "${DATASET_PIDS[@]}"; do
        wait "${pid}" 2>/dev/null || true
      done
      return 1
    fi

    alive=0
    for pid in "${DATASET_PIDS[@]}"; do
      if kill -0 "${pid}" 2>/dev/null; then
        alive=$((alive + 1))
      fi
    done
    if (( alive == 0 )); then
      break
    fi
    sleep "${poll_seconds}"
  done

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
  echo "  target:             ${TARGET_MODEL} (local vLLM)"
  echo "  model path:         ${MODEL_PATH}"
  echo "  CUDA devices:       ${QWEN_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-unset}}"
  echo "  tensor parallel:    ${VLLM_TENSOR_PARALLEL_SIZE}"
  echo "  max model len:      ${MAX_MODEL_LEN}"
  echo "  target temperature: ${TARGET_QWEN_CHAT_TEMPERATURE}"
  echo "  target thinking:    ${TARGET_QWEN_CHAT_ENABLE_THINKING}"
  echo "  GPU memory target:  ${GPU_MEMORY_UTILIZATION}"
  echo "  vLLM max seqs:      ${VLLM_MAX_NUM_SEQS}"
  echo "  batched tokens:     ${VLLM_MAX_NUM_BATCHED_TOKENS}"
  echo "  method parameters:  inherited from configs/<dataset>/default.yaml"
  echo "  shared data:        ${DATA_ROOT}"
  echo "  output:             ${OUT_BASE}"
  echo "============================================================"
}

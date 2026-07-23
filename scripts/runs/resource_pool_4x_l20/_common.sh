#!/usr/bin/env bash

# Shared setup for resource-pool launchers. This file is sourced by the four
# runnable scripts; do not execute it directly.

RESOURCE_POOL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${RESOURCE_POOL_DIR}/../../.." && pwd)"

resource_fail() {
  echo "ERROR: $*" >&2
  exit 1
}

resource_truthy() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

# Probe whether the ALFWorld Python stack is importable in the run interpreter.
# Returns 0 when alfworld + gymnasium + omegaconf and our adapter all import.
alfworld_imports_ok() {
  local python_bin="${PYTHON_BIN:-python}"
  PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}" "${python_bin}" - <<'PY' >/dev/null 2>&1
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
  local python_bin="${PYTHON_BIN:-python}"
  export ALFWORLD_DATA="${ALFWORLD_DATA:-${PROJECT_ROOT}/data/alfworld}"
  if resource_truthy "${DRY_RUN:-0}"; then
    return
  fi
  command -v "${python_bin}" >/dev/null 2>&1 || resource_fail "Python interpreter not found: ${python_bin}"
  [[ -d "${ALFWORLD_DATA}/json_2.1.1" ]] || resource_fail \
    "ALFWorld data not found at ${ALFWORLD_DATA}/json_2.1.1 (unpack the split data bundle first)"

  if alfworld_imports_ok; then
    return
  fi

  if resource_truthy "${ALFWORLD_AUTO_INSTALL:-1}"; then
    echo "[deps] ALFWorld Python deps missing — installing .[alfworld] + omegaconf json_repair ..."
    ( cd "${PROJECT_ROOT}" \
        && "${python_bin}" -m pip install -e ".[alfworld]" \
        && "${python_bin}" -m pip install omegaconf json_repair ) \
      || resource_fail "ALFWorld dependency install failed (see pip output above)"
  fi

  alfworld_imports_ok || resource_fail \
    "ALFWorld Python env still not ready. Run: cd ${PROJECT_ROOT} && ${python_bin} -m pip install -e \".[alfworld]\" && ${python_bin} -m pip install omegaconf json_repair"
}

configure_single_l20() {
  local visible_devices
  visible_devices="${QWEN_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-0}}"
  visible_devices="${visible_devices// /}"
  [[ -n "${visible_devices}" ]] || resource_fail "No GPU is visible. Set CUDA_VISIBLE_DEVICES."
  case "${visible_devices}" in
    *,*) resource_fail "Exactly one GPU is required; got CUDA devices '${visible_devices}'." ;;
  esac

  export L20_GPU="${visible_devices}"
  export QWEN_CUDA_VISIBLE_DEVICES="${visible_devices}"
  export VLLM_TENSOR_PARALLEL_SIZE=1
  export MODEL_PATH="${MODEL_PATH:-/ai-app-vepfs/modelscope/models/Qwen/Qwen3.5-4B}"
  export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen/Qwen3.5-4B}"
  export TARGET_MODEL="${TARGET_MODEL:-${SERVED_MODEL_NAME}}"
  export VLLM_PORT="${VLLM_PORT:-59317}"
  export QWEN_CHAT_BASE_URL="${QWEN_CHAT_BASE_URL:-http://127.0.0.1:${VLLM_PORT}/v1}"
  export QWEN_CHAT_API_KEY="${QWEN_CHAT_API_KEY:-dummy}"
  export QWEN_CHAT_MODEL="${QWEN_CHAT_MODEL:-${SERVED_MODEL_NAME}}"
  export MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
  export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
  export VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-65536}"
  export VLLM_ENABLE_CHUNKED_PREFILL="${VLLM_ENABLE_CHUNKED_PREFILL:-1}"
  export START_VLLM="${START_VLLM:-1}"
  export STOP_VLLM_ON_EXIT="${STOP_VLLM_ON_EXIT:-1}"
  export TARGET_QWEN_CHAT_ENABLE_THINKING="${TARGET_QWEN_CHAT_ENABLE_THINKING:-false}"
}

configure_deepseek_official() {
  local key
  key="${DEEPSEEK_API_KEY:-${DS_API_KEY:-${DEEPSEEK_OFFICIAL_API_KEY:-}}}"
  if [[ -z "${key}" ]] && ! resource_truthy "${DRY_RUN:-0}"; then
    resource_fail "DEEPSEEK_API_KEY is required for this task group."
  fi

  export AZURE_OPENAI_ENDPOINT="${DEEPSEEK_BASE_URL:-https://api.deepseek.com}"
  export AZURE_OPENAI_API_KEY="${key}"
  export AZURE_OPENAI_AUTH_MODE=openai_compatible
  export AZURE_OPENAI_API_VERSION=openai-compat
  export OPTIMIZER_AZURE_OPENAI_ENDPOINT="${AZURE_OPENAI_ENDPOINT}"
  export OPTIMIZER_AZURE_OPENAI_API_KEY="${AZURE_OPENAI_API_KEY}"
  export OPTIMIZER_AZURE_OPENAI_AUTH_MODE="${AZURE_OPENAI_AUTH_MODE}"
  export OPTIMIZER_AZURE_OPENAI_API_VERSION="${AZURE_OPENAI_API_VERSION}"
  export OPTIMIZER_MODEL="${DEEPSEEK_OFFICIAL_MODEL:-deepseek-v4-pro}"
  export DEEPSEEK_THINKING="${DEEPSEEK_THINKING:-disabled}"
  export REASONING_EFFORT="${REASONING_EFFORT:-}"
}

configure_volcano_ark() {
  local key
  key="${ARK_API_KEY:-}"
  if [[ -z "${key}" ]] && ! resource_truthy "${DRY_RUN:-0}"; then
    resource_fail "ARK_API_KEY is required for this task group."
  fi

  export AZURE_OPENAI_ENDPOINT="${ARK_BASE_URL:-https://ark.cn-beijing.volces.com/api/v3}"
  export AZURE_OPENAI_API_KEY="${key}"
  export AZURE_OPENAI_AUTH_MODE=openai_compatible
  export AZURE_OPENAI_API_VERSION="${ARK_API_VERSION:-2024-12-01-preview}"
  export OPTIMIZER_AZURE_OPENAI_ENDPOINT="${AZURE_OPENAI_ENDPOINT}"
  export OPTIMIZER_AZURE_OPENAI_API_KEY="${AZURE_OPENAI_API_KEY}"
  export OPTIMIZER_AZURE_OPENAI_AUTH_MODE="${AZURE_OPENAI_AUTH_MODE}"
  export OPTIMIZER_AZURE_OPENAI_API_VERSION="${AZURE_OPENAI_API_VERSION}"
  export OPTIMIZER_MODEL="${ARK_OPTIMIZER_MODEL:-deepseek-v4-pro-260425}"
  export DEEPSEEK_THINKING="${DEEPSEEK_THINKING:-disabled}"
  export REASONING_EFFORT="${REASONING_EFFORT:-}"
}

print_resource_header() {
  local group="$1"
  local datasets="$2"
  local source="$3"
  echo "============================================================"
  echo "  Resource-pool group: ${group}"
  echo "============================================================"
  echo "  datasets:          ${datasets}"
  echo "  optimizer source:  ${source}"
  echo "  optimizer model:   ${OPTIMIZER_MODEL}"
  echo "  local target:      ${TARGET_MODEL}"
  echo "  cuda device:       ${QWEN_CUDA_VISIBLE_DEVICES}"
  echo "  vllm endpoint:     ${QWEN_CHAT_BASE_URL}"
  echo "  tensor parallel:   ${VLLM_TENSOR_PARALLEL_SIZE}"
  echo "  vllm max seqs:     ${VLLM_MAX_NUM_SEQS}"
  echo "============================================================"
}

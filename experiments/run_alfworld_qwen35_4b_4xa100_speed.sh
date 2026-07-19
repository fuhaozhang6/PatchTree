#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/run_alfworld_qwen35_4b_4xa100_multi_endpoint_speed.sh" "$@"

# 4xA100 ALFWorld speed probe for local Qwen3.5-4B.
# Run from anywhere inside the uploaded SkillOpt-Tree checkout:
#   bash experiments/run_alfworld_qwen35_4b_4xa100_speed.sh
#
# Useful overrides:
#   ALFWORLD_ENV_NUM=32 bash experiments/run_alfworld_qwen35_4b_4xa100_speed.sh  # quick smoke
#   ALFWORLD_WORKERS=256 ALFWORLD_MAX_API_WORKERS=256 bash experiments/run_alfworld_qwen35_4b_4xa100_speed.sh
#   STOP_VLLM_ON_EXIT=0 bash experiments/run_alfworld_qwen35_4b_4xa100_speed.sh  # keep vLLM alive

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHON_BIN="${PYTHON_BIN:-python}"

# The uploaded data package already contains data/alfworld and data/alfworld_path_split.
export ALFWORLD_DATA="${ALFWORLD_DATA:-${PROJECT_ROOT}/data/alfworld}"
export ALFWORLD_SPLIT_DIR="${ALFWORLD_SPLIT_DIR:-${PROJECT_ROOT}/data/alfworld_path_split}"
export ALFWORLD_SPLIT="${ALFWORLD_SPLIT:-test}"
export ALFWORLD_ENV_NUM="${ALFWORLD_ENV_NUM:-0}"  # 0 means full split; local test split has 134 items.
export ALFWORLD_MAX_STEPS="${ALFWORLD_MAX_STEPS:-50}"

# High enough to avoid Python-side throttling while vLLM batches requests.
export ALFWORLD_WORKERS="${ALFWORLD_WORKERS:-192}"
export ALFWORLD_MAX_API_WORKERS="${ALFWORLD_MAX_API_WORKERS:-192}"
export ALFWORLD_TARGET_MAX_COMPLETION_TOKENS="${ALFWORLD_TARGET_MAX_COMPLETION_TOKENS:-2048}"

# Local Qwen/vLLM service. Qwen3.5-4B fits comfortably on one A100, so 4-way
# data parallelism gives better throughput and avoids fragile tensor-parallel
# startup for this small model.
export MODEL_PATH="${MODEL_PATH:-/ai-car-vepfs1/ai_car/share/model/Qwen/Qwen3.5-4B}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen/Qwen3.5-4B}"
export QWEN_CHAT_MODEL="${QWEN_CHAT_MODEL:-${SERVED_MODEL_NAME}}"
export QWEN_CUDA_VISIBLE_DEVICES="${QWEN_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-0,1,2,3}}"
export VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-1}"
export VLLM_DATA_PARALLEL_SIZE="${VLLM_DATA_PARALLEL_SIZE:-4}"
export VLLM_PORT="${VLLM_PORT:-39217}"
export QWEN_CHAT_BASE_URL="${QWEN_CHAT_BASE_URL:-http://127.0.0.1:${VLLM_PORT}/v1}"
export QWEN_CHAT_API_KEY="${QWEN_CHAT_API_KEY:-dummy}"
export QWEN_CHAT_TIMEOUT_SECONDS="${QWEN_CHAT_TIMEOUT_SECONDS:-300}"
export QWEN_SMOKE_TIMEOUT_SECONDS="${QWEN_SMOKE_TIMEOUT_SECONDS:-300}"
export TARGET_QWEN_CHAT_TEMPERATURE="${TARGET_QWEN_CHAT_TEMPERATURE:-0.2}"
export TARGET_QWEN_CHAT_ENABLE_THINKING="${TARGET_QWEN_CHAT_ENABLE_THINKING:-false}"

# Throughput-oriented vLLM defaults for long ALFWorld prompts/responses.
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.88}"
export VLLM_DTYPE="${VLLM_DTYPE:-bfloat16}"
export VLLM_STARTUP_TIMEOUT_SECONDS="${VLLM_STARTUP_TIMEOUT_SECONDS:-1200}"
export VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:---data-parallel-size ${VLLM_DATA_PARALLEL_SIZE} --max-num-seqs 128 --max-num-batched-tokens 32768}"
export START_VLLM="${START_VLLM:-1}"

export TS="${TS:-qwen35_4b_4xa100_alfworld_speed_$(date +%Y%m%d_%H%M%S)}"
export OUT_BASE="${OUT_BASE:-${PROJECT_ROOT}/outputs/${TS}}"

echo "============================================================"
echo "  SkillOpt ALFWorld 4xA100 Qwen Speed Test"
echo "============================================================"
echo "  project_root:      ${PROJECT_ROOT}"
echo "  model_path:        ${MODEL_PATH}"
echo "  cuda_devices:      ${QWEN_CUDA_VISIBLE_DEVICES}"
echo "  tensor_parallel:   ${VLLM_TENSOR_PARALLEL_SIZE}"
echo "  data_parallel:     ${VLLM_DATA_PARALLEL_SIZE}"
echo "  qwen_url:          ${QWEN_CHAT_BASE_URL}"
echo "  vllm_extra_args:   ${VLLM_EXTRA_ARGS}"
echo "  split:             ${ALFWORLD_SPLIT}"
echo "  env_num:           ${ALFWORLD_ENV_NUM} (0 means full split)"
echo "  env_workers:       ${ALFWORLD_WORKERS}"
echo "  max_api_workers:   ${ALFWORLD_MAX_API_WORKERS}"
echo "  max_steps:         ${ALFWORLD_MAX_STEPS}"
echo "  max_tokens:        ${ALFWORLD_TARGET_MAX_COMPLETION_TOKENS}"
echo "  alfworld_data:     ${ALFWORLD_DATA}"
echo "  split_dir:         ${ALFWORLD_SPLIT_DIR}"
echo "  out_base:          ${OUT_BASE}"
echo "============================================================"

[[ -d "${MODEL_PATH}" ]] || {
  echo "ERROR: MODEL_PATH not found: ${MODEL_PATH}" >&2
  exit 1
}
[[ -d "${ALFWORLD_DATA}/json_2.1.1" ]] || {
  echo "ERROR: ALFWorld data not found under ${ALFWORLD_DATA}/json_2.1.1" >&2
  exit 1
}
[[ -f "${ALFWORLD_SPLIT_DIR}/${ALFWORLD_SPLIT}/items.json" ]] || {
  echo "ERROR: split items not found: ${ALFWORLD_SPLIT_DIR}/${ALFWORLD_SPLIT}/items.json" >&2
  exit 1
}

if [[ "${START_VLLM}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
  "${PYTHON_BIN}" - "${VLLM_PORT}" "${VLLM_DATA_PARALLEL_SIZE}" <<'PY'
import socket
import sys

base = int(sys.argv[1])
n = max(int(sys.argv[2]), 1)
busy = []
for port in range(base, base + n):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind(("127.0.0.1", port))
    except OSError:
        busy.append(port)
    finally:
        sock.close()

if busy:
    raise SystemExit(
        "ERROR: vLLM port(s) already in use: "
        + ", ".join(map(str, busy))
        + ". Try: VLLM_PORT=49317 QWEN_CHAT_BASE_URL=http://127.0.0.1:49317/v1"
    )
PY
fi

bash experiments/searchqa_prompt_latency/run_alfworld_qwen_speed_workers128.sh "$@"

#!/usr/bin/env bash
set -euo pipefail

# Optional final stage: evaluate each discovered shared patch on its held-out
# members and nearest outside-cluster boundary samples. This is the only stage
# that starts local Qwen/vLLM.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "${HERE}/_common.sh"

SHARD_COUNT="${SHARD_COUNT:-2}"
TAXONOMY_PATH="${TAXONOMY_PATH:-${OUT_BASE}/taxonomy/blind_revision_taxonomy.json}"
VALIDATION_DIR="${VALIDATION_DIR:-${OUT_BASE}/transfer_validation}"
VLLM_PORT="${VLLM_PORT:-59317}"
QWEN_CHAT_BASE_URL="${QWEN_CHAT_BASE_URL:-http://127.0.0.1:${VLLM_PORT}/v1}"
MODEL_PATH="${MODEL_PATH:-/ai-app-vepfs/modelscope/models/Qwen/Qwen3.5-4B}"
TARGET_MODEL="${TARGET_MODEL:-Qwen/Qwen3.5-4B}"
CUDA_DEVICE="${CUDA_VISIBLE_DEVICES:-0}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-128}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
START_VLLM="${START_VLLM:-1}"
STOP_VLLM_ON_EXIT="${STOP_VLLM_ON_EXIT:-1}"
VLLM_LOG="${LOG_BASE}/vllm_transfer_validation.log"
VLLM_PID=""

endpoint_ready() {
  "${PYTHON_BIN}" - "${QWEN_CHAT_BASE_URL}" <<'PY'
import sys
import urllib.request
try:
    with urllib.request.urlopen(sys.argv[1].rstrip("/") + "/models", timeout=5) as response:
        raise SystemExit(0 if response.status == 200 else 1)
except Exception:
    raise SystemExit(1)
PY
}

cleanup() {
  if blind_truthy "${STOP_VLLM_ON_EXIT}" && [[ -n "${VLLM_PID}" ]] \
    && kill -0 "${VLLM_PID}" 2>/dev/null; then
    kill "${VLLM_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

start_vllm() {
  if endpoint_ready; then
    echo "[vllm] reusing ${QWEN_CHAT_BASE_URL}"
    return
  fi
  blind_truthy "${START_VLLM}" || blind_fail "vLLM endpoint is not ready"
  [[ -d "${MODEL_PATH}" ]] || blind_fail "MODEL_PATH not found: ${MODEL_PATH}"
  echo "[vllm] starting Qwen on GPU ${CUDA_DEVICE}; log=${VLLM_LOG}"
  env CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" nohup vllm serve "${MODEL_PATH}" \
    --served-model-name "${TARGET_MODEL}" \
    --host 0.0.0.0 \
    --port "${VLLM_PORT}" \
    --trust-remote-code \
    --dtype bfloat16 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.90}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --max-num-seqs "${VLLM_MAX_NUM_SEQS}" \
    --max-num-batched-tokens "${VLLM_MAX_NUM_BATCHED_TOKENS:-65536}" \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --reasoning-parser "${VLLM_REASONING_PARSER:-qwen3}" \
    > "${VLLM_LOG}" 2>&1 &
  VLLM_PID=$!
  for second in $(seq 1 "${VLLM_WAIT_SECONDS:-900}"); do
    endpoint_ready && { echo "[vllm] ready after ${second}s"; return; }
    kill -0 "${VLLM_PID}" 2>/dev/null || {
      tail -n 100 "${VLLM_LOG}" || true
      blind_fail "vLLM exited before readiness"
    }
    sleep 1
  done
  blind_fail "timed out waiting for vLLM"
}

args=(
  "${PYTHON_BIN}" -u scripts/tools/validate_searchqa_blind_transfer.py
  --taxonomy "${TAXONOMY_PATH}"
  --config "${PROJECT_ROOT}/configs/searchqa/default.yaml"
  --output-dir "${VALIDATION_DIR}"
  --target-model "${TARGET_MODEL}"
  --target-base-url "${QWEN_CHAT_BASE_URL}"
  --target-workers "${TARGET_WORKERS:-128}"
  --batch-size "${BATCH_SIZE:-100}"
  --repeats "${REPEATS:-3}"
  --target-max-tokens "${TARGET_MAX_TOKENS:-4096}"
  --max-holdout-per-type "${MAX_HOLDOUT_PER_TYPE:-40}"
  --max-boundary-per-type "${MAX_BOUNDARY_PER_TYPE:-8}"
  --min-holdout-samples "${MIN_HOLDOUT_SAMPLES:-10}"
  --min-boundary-samples "${MIN_BOUNDARY_SAMPLES:-4}"
)
for ((index=0; index<SHARD_COUNT; index++)); do
  args+=(--cards "${OUT_BASE}/extract_shard${index}of${SHARD_COUNT}")
done
if blind_truthy "${DRY_RUN}"; then
  args+=(--dry-run)
else
  start_vllm
fi

"${args[@]}" 2>&1 | tee "${LOG_BASE}/transfer_validation.log"

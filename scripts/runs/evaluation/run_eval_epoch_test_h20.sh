#!/usr/bin/env bash
set -euo pipefail

# Evaluate each epoch's final skill on the LiveMath TEST split.
#
#   target / student = local Qwen3.5-4B served by ONE single-GPU vLLM (H20)
#   no optimizer is needed here — this is pure evaluation.
#
# What it does:
#   1. Boots a single-H20 vLLM server (TP=1) with large concurrency.
#   2. Runs scripts/tools/eval_epoch_skills_test.py, which:
#        - reads history.json to find each epoch's last global step,
#        - loads skills/skill_v{last_step}.md (already reflects the epoch-level
#          tail-bank computation, since the trainer re-saves that file at the
#          end of each epoch),
#        - de-duplicates identical skills and evaluates each unique one on the
#          test split (valid_unseen),
#        - reads the per-epoch val score from history.json,
#        - writes a val-vs-test comparison table (md + csv + json).
#
# Typical usage on the H20 node:
#   cd /ai-app-vepfs/zhangfuhao/eval/demo3/SkillOpt-Tree
#   bash scripts/runs/evaluation/run_eval_epoch_test_h20.sh
#
# Useful overrides:
#   H20_GPU=0 WORKERS=192 bash scripts/runs/evaluation/run_eval_epoch_test_h20.sh
#   TEST_ENV_NUM=40 bash scripts/runs/evaluation/run_eval_epoch_test_h20.sh          # smaller/faster
#   EPOCHS="1,4,8,12,16" bash scripts/runs/evaluation/run_eval_epoch_test_h20.sh     # subset of epochs
#   START_VLLM=0 QWEN_CHAT_BASE_URL=http://127.0.0.1:8000/v1 bash scripts/runs/evaluation/run_eval_epoch_test_h20.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

fail() { echo "ERROR: $*" >&2; exit 1; }
truthy() { case "${1:-}" in 1|true|TRUE|yes|YES|on|ON) return 0 ;; *) return 1 ;; esac; }

command -v "${PYTHON_BIN}" >/dev/null 2>&1 || fail "Python interpreter not found: ${PYTHON_BIN}"

# ── What to evaluate ─────────────────────────────────────────────────────────
RUN_NAME="${RUN_NAME:-livemath_l20_qwen35_4b_dsv4pro_16ep_20260715_031426}"
OUT_ROOT="${OUT_ROOT:-/ai-app-vepfs/zhangfuhao/eval/demo3/SkillOpt-Tree/outputs/${RUN_NAME}/livemath}"
LIVEMATH_SPLIT_DIR="${LIVEMATH_SPLIT_DIR:-${PROJECT_ROOT}/data/livemathematicianbench_split}"
RESULT_DIR="${RESULT_DIR:-${OUT_ROOT}/epoch_test_eval}"
EPOCHS="${EPOCHS:-}"          # empty = all epochs; else "1,4,8,16"
EVAL_INIT="${EVAL_INIT:-true}" # also report the initial skill as epoch 0 baseline
TEST_ENV_NUM="${TEST_ENV_NUM:-0}"  # 0 = full test split
SEED="${SEED:-42}"

[[ -f "${OUT_ROOT}/history.json" ]] || fail "history.json not found: ${OUT_ROOT}/history.json"
[[ -d "${OUT_ROOT}/skills" ]] || fail "skills dir not found: ${OUT_ROOT}/skills"
[[ -d "${LIVEMATH_SPLIT_DIR}/test" ]] || fail "LiveMath test split not found: ${LIVEMATH_SPLIT_DIR}/test"

# ── Target: local Qwen3.5-4B served by ONE H20 GPU (TP=1) ────────────────────
MODEL_PATH="${MODEL_PATH:-/ai-app-vepfs/modelscope/models/Qwen/Qwen3.5-4B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen/Qwen3.5-4B}"
TARGET_MODEL="${TARGET_MODEL:-${SERVED_MODEL_NAME}}"
H20_GPU="${H20_GPU:-0}"
QWEN_CUDA_VISIBLE_DEVICES="${QWEN_CUDA_VISIBLE_DEVICES:-${H20_GPU}}"
VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-1}"   # single H20
VLLM_HOST="${VLLM_HOST:-0.0.0.0}"
VLLM_PORT="${VLLM_PORT:-8000}"
QWEN_CHAT_BASE_URL="${QWEN_CHAT_BASE_URL:-http://127.0.0.1:${VLLM_PORT}/v1}"
QWEN_CHAT_API_KEY="${QWEN_CHAT_API_KEY:-dummy}"

# H20 has plenty of memory: push concurrency up hard.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-256}"
VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-65536}"
VLLM_DTYPE="${VLLM_DTYPE:-bfloat16}"
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:---max-num-seqs ${VLLM_MAX_NUM_SEQS} --max-num-batched-tokens ${VLLM_MAX_NUM_BATCHED_TOKENS}}"
START_VLLM="${START_VLLM:-1}"
STOP_VLLM_ON_EXIT="${STOP_VLLM_ON_EXIT:-1}"
VLLM_WAIT_SECONDS="${VLLM_WAIT_SECONDS:-900}"

# Rollout concurrency (ThreadPoolExecutor in run_batch). Keep <= vLLM max-num-seqs.
WORKERS="${WORKERS:-192}"
TARGET_QWEN_CHAT_MAX_TOKENS="${TARGET_QWEN_CHAT_MAX_TOKENS:-16384}"
TARGET_QWEN_CHAT_TEMPERATURE="${TARGET_QWEN_CHAT_TEMPERATURE:-0.2}"
TARGET_QWEN_CHAT_TIMEOUT_SECONDS="${TARGET_QWEN_CHAT_TIMEOUT_SECONDS:-300}"
TARGET_QWEN_CHAT_ENABLE_THINKING="${TARGET_QWEN_CHAT_ENABLE_THINKING:-false}"
MAX_TURNS="${MAX_TURNS:-1}"

TS="${TS:-eval_epoch_test_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs/${TS}}"
mkdir -p "${LOG_DIR}"
VLLM_LOG="${LOG_DIR}/vllm_qwen.log"
VLLM_PID_FILE="${LOG_DIR}/vllm.pid"

echo "============================================================"
echo "  Per-epoch skill -> TEST eval  (single H20, TP=1)"
echo "============================================================"
echo "  out_root:        ${OUT_ROOT}"
echo "  split_dir:       ${LIVEMATH_SPLIT_DIR}"
echo "  result_dir:      ${RESULT_DIR}"
echo "  epochs filter:   ${EPOCHS:-<all>}"
echo "  eval_init:       ${EVAL_INIT}"
echo "  test_env_num:    ${TEST_ENV_NUM} (0 = full test split)"
echo "  model_path:      ${MODEL_PATH}"
echo "  served_name:     ${SERVED_MODEL_NAME}"
echo "  gpu:             ${QWEN_CUDA_VISIBLE_DEVICES}"
echo "  tensor_parallel: ${VLLM_TENSOR_PARALLEL_SIZE}"
echo "  qwen_url:        ${QWEN_CHAT_BASE_URL}"
echo "  max_model_len:   ${MAX_MODEL_LEN}"
echo "  max_num_seqs:    ${VLLM_MAX_NUM_SEQS}"
echo "  rollout workers: ${WORKERS}"
echo "  log_dir:         ${LOG_DIR}"
echo "============================================================"

# ── vLLM lifecycle helpers ───────────────────────────────────────────────────
endpoint_ready() {
  "${PYTHON_BIN}" - "${QWEN_CHAT_BASE_URL}" "${QWEN_CHAT_API_KEY}" <<'PY'
import sys, urllib.request
base, api_key = sys.argv[1].rstrip("/"), sys.argv[2]
req = urllib.request.Request(f"{base}/models",
    headers={"Authorization": f"Bearer {api_key}"}, method="GET")
try:
    with urllib.request.urlopen(req, timeout=5) as resp:
        raise SystemExit(0 if resp.status == 200 else 1)
except Exception:
    raise SystemExit(1)
PY
}

qwen_smoke() {
  "${PYTHON_BIN}" - "${QWEN_CHAT_BASE_URL}" "${QWEN_CHAT_API_KEY}" "${SERVED_MODEL_NAME}" <<'PY'
import json, sys, urllib.request
base, api_key, model = sys.argv[1].rstrip("/"), sys.argv[2], sys.argv[3]
payload = {"model": model,
    "messages": [{"role": "user", "content": "Say hello in one short sentence."}],
    "max_tokens": 64, "temperature": 0.2,
    "chat_template_kwargs": {"enable_thinking": False}}
req = urllib.request.Request(f"{base}/chat/completions",
    data=json.dumps(payload).encode(),
    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    method="POST")
with urllib.request.urlopen(req, timeout=120) as resp:
    data = json.loads(resp.read().decode())
choice = (data.get("choices") or [{}])[0]
content = (choice.get("message") or {}).get("content") or ""
print(f"[smoke/qwen] finish_reason={choice.get('finish_reason')} content_len={len(content)}")
print(f"[smoke/qwen] content_preview={content[:100]!r}")
PY
}

cleanup() {
  if truthy "${STOP_VLLM_ON_EXIT}" && [[ -f "${VLLM_PID_FILE}" ]]; then
    local pid; pid="$(cat "${VLLM_PID_FILE}")"
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      echo "[cleanup] stopping vLLM pid=${pid}"
      kill "${pid}" 2>/dev/null || true
    fi
  fi
}
trap cleanup EXIT

# ── boot vLLM ────────────────────────────────────────────────────────────────
if endpoint_ready; then
  echo "[check] Existing Qwen endpoint is ready: ${QWEN_CHAT_BASE_URL}"
else
  truthy "${START_VLLM}" || fail "Qwen endpoint not ready and START_VLLM=${START_VLLM}."
  [[ -d "${MODEL_PATH}" ]] || fail "MODEL_PATH not found: ${MODEL_PATH}"
  echo "[vllm] Starting local Qwen on CUDA_VISIBLE_DEVICES=${QWEN_CUDA_VISIBLE_DEVICES}..."
  vllm_args=(
    serve "${MODEL_PATH}"
    --served-model-name "${SERVED_MODEL_NAME}"
    --host "${VLLM_HOST}"
    --port "${VLLM_PORT}"
    --trust-remote-code
    --dtype "${VLLM_DTYPE}"
    --tensor-parallel-size "${VLLM_TENSOR_PARALLEL_SIZE}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
    --max-model-len "${MAX_MODEL_LEN}"
    --enable-prefix-caching
  )
  if [[ -n "${VLLM_EXTRA_ARGS}" ]]; then
    # shellcheck disable=SC2206
    extra_vllm_args=(${VLLM_EXTRA_ARGS})
    vllm_args+=("${extra_vllm_args[@]}")
  fi
  env CUDA_VISIBLE_DEVICES="${QWEN_CUDA_VISIBLE_DEVICES}" \
    nohup vllm "${vllm_args[@]}" > "${VLLM_LOG}" 2>&1 &
  VLLM_PID=$!
  echo "${VLLM_PID}" > "${VLLM_PID_FILE}"
  echo "[vllm] pid=${VLLM_PID}  log=${VLLM_LOG}"
  echo "[vllm] Waiting for endpoint (up to ${VLLM_WAIT_SECONDS}s)..."
  for i in $(seq 1 "${VLLM_WAIT_SECONDS}"); do
    if endpoint_ready; then
      echo "[check] Qwen endpoint OK after ${i}s: ${QWEN_CHAT_BASE_URL}"
      break
    fi
    if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
      tail -n 80 "${VLLM_LOG}" || true
      fail "vLLM process exited before endpoint became ready."
    fi
    sleep 1
    if [[ "${i}" == "${VLLM_WAIT_SECONDS}" ]]; then
      tail -n 80 "${VLLM_LOG}" || true
      fail "Timed out waiting for vLLM endpoint."
    fi
  done
fi
qwen_smoke

# ── run the evaluator ────────────────────────────────────────────────────────
export TARGET_QWEN_CHAT_BASE_URL="${QWEN_CHAT_BASE_URL}"
export TARGET_QWEN_CHAT_API_KEY="${QWEN_CHAT_API_KEY}"

eval_args=(
  "${PYTHON_BIN}" -u "${PROJECT_ROOT}/scripts/tools/eval_epoch_skills_test.py"
  --out_root "${OUT_ROOT}"
  --split_dir "${LIVEMATH_SPLIT_DIR}"
  --result_dir "${RESULT_DIR}"
  --target_model "${TARGET_MODEL}"
  --base_url "${QWEN_CHAT_BASE_URL}"
  --api_key "${QWEN_CHAT_API_KEY}"
  --temperature "${TARGET_QWEN_CHAT_TEMPERATURE}"
  --timeout_seconds "${TARGET_QWEN_CHAT_TIMEOUT_SECONDS}"
  --enable_thinking "${TARGET_QWEN_CHAT_ENABLE_THINKING}"
  --max_completion_tokens "${TARGET_QWEN_CHAT_MAX_TOKENS}"
  --workers "${WORKERS}"
  --test_env_num "${TEST_ENV_NUM}"
  --seed "${SEED}"
  --max_turns "${MAX_TURNS}"
  --eval_init "${EVAL_INIT}"
)
[[ -n "${EPOCHS}" ]] && eval_args+=(--epochs "${EPOCHS}")

echo "[eval] $(printf ' %q' "${eval_args[@]}")"
"${eval_args[@]}" 2>&1 | tee "${LOG_DIR}/eval_epoch_test.log"

echo ""
echo "[done] results under: ${RESULT_DIR}"
echo "       table: ${RESULT_DIR}/epoch_eval_table.md"

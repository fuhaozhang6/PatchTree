#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export TARGET_QWEN_CHAT_ROLLOUT_RETRIES="${TARGET_QWEN_CHAT_ROLLOUT_RETRIES:-1}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DRY_RUN="${DRY_RUN:-0}"

fail() { echo "ERROR: $*" >&2; exit 1; }
truthy() { case "${1:-}" in 1|true|TRUE|yes|YES|on|ON) return 0 ;; *) return 1 ;; esac; }

REPLAY_DIR="${REPLAY_DIR:-${PROJECT_ROOT}/outputs/searchqa_tree_verdict_p6_fixed}"
SEARCHQA_SPLIT_DIR="${SEARCHQA_SPLIT_DIR:-${PROJECT_ROOT}/data/searchqa_split}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
RESULT_ROOT="${RESULT_ROOT:-${PROJECT_ROOT}/outputs/searchqa_tree_verdict_eval_${RUN_STAMP}}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs/searchqa_tree_verdict_eval_${RUN_STAMP}}"

MODEL_PATH="${MODEL_PATH:-/ai-app-vepfs/modelscope/models/Qwen/Qwen3.5-4B}"
TARGET_MODEL="${TARGET_MODEL:-Qwen/Qwen3.5-4B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-${TARGET_MODEL}}"
QWEN_CUDA_VISIBLE_DEVICES="${QWEN_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-0}}"
VLLM_HOST="${VLLM_HOST:-0.0.0.0}"
VLLM_PORT="${VLLM_PORT:-59317}"
QWEN_CHAT_BASE_URL="${QWEN_CHAT_BASE_URL:-http://127.0.0.1:${VLLM_PORT}/v1}"
QWEN_CHAT_API_KEY="${QWEN_CHAT_API_KEY:-dummy}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-128}"
VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-65536}"
WORKERS="${WORKERS:-128}"
TARGET_MAX_TOKENS="${TARGET_MAX_TOKENS:-4096}"
TARGET_TEMPERATURE="${TARGET_TEMPERATURE:-0}"
TARGET_TIMEOUT="${TARGET_TIMEOUT:-300}"
START_VLLM="${START_VLLM:-1}"
STOP_VLLM_ON_EXIT="${STOP_VLLM_ON_EXIT:-1}"
VLLM_WAIT_SECONDS="${VLLM_WAIT_SECONDS:-900}"
SEED="${SEED:-42}"
RETRY_FAILED="${RETRY_FAILED:-1}"

case "${QWEN_CUDA_VISIBLE_DEVICES// /}" in
  "") fail "Set CUDA_VISIBLE_DEVICES or QWEN_CUDA_VISIBLE_DEVICES." ;;
  *,*) fail "Exactly one GPU is required; got ${QWEN_CUDA_VISIBLE_DEVICES}." ;;
esac
[[ "${WORKERS}" -le "${VLLM_MAX_NUM_SEQS}" ]] || \
  fail "WORKERS=${WORKERS} exceeds VLLM_MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS}."

echo "============================================================"
echo "  SearchQA Tree Verdict: one-GPU evaluation"
echo "============================================================"
echo "  replay_dir:     ${REPLAY_DIR}"
echo "  result_root:    ${RESULT_ROOT}"
echo "  split_dir:      ${SEARCHQA_SPLIT_DIR}"
echo "  model/path:     ${TARGET_MODEL} / ${MODEL_PATH}"
echo "  gpu/url:        ${QWEN_CUDA_VISIBLE_DEVICES} / ${QWEN_CHAT_BASE_URL}"
echo "  workers/seqs:   ${WORKERS}/${VLLM_MAX_NUM_SEQS}"
echo "  model/tokens:   ${MAX_MODEL_LEN}/${TARGET_MAX_TOKENS}"
echo "  temperature:    ${TARGET_TEMPERATURE}"
echo "  protocol:       Parent + Flat + Leaf + Tree, then conditional Top-down"
echo "============================================================"

if truthy "${DRY_RUN}"; then
  echo "[dry-run] prepare manifest: ${REPLAY_DIR}/main/skill_manifest.tsv"
  echo "[dry-run] main val/test -> ${RESULT_ROOT}/main_eval"
  echo "[dry-run] top-down phase1 val -> finalize -> combo val -> conditional test"
  exit 0
fi

[[ -f "${REPLAY_DIR}/replay_manifest.json" ]] || \
  fail "Frozen replay is missing. Run run_01_prepare_fixed_candidates.sh first."
[[ -f "${REPLAY_DIR}/main/skill_manifest.tsv" ]] || \
  fail "Main skill manifest is missing."
[[ -f "${REPLAY_DIR}/topdown/phase1_val_manifest.tsv" ]] || \
  fail "Top-down phase1 manifest is missing."
[[ -f "${REPLAY_DIR}/topdown/global_step.txt" ]] || \
  fail "Top-down global_step.txt is missing."
[[ -d "${SEARCHQA_SPLIT_DIR}/val" && -d "${SEARCHQA_SPLIT_DIR}/test" ]] || \
  fail "SearchQA split directory is incomplete: ${SEARCHQA_SPLIT_DIR}"
[[ -d "${MODEL_PATH}" ]] || fail "MODEL_PATH not found: ${MODEL_PATH}"
mkdir -p "${RESULT_ROOT}" "${LOG_DIR}"

started_vllm_pid=""
VLLM_LOG="${LOG_DIR}/vllm_qwen.log"

endpoint_ready() {
  "${PYTHON_BIN}" - "${QWEN_CHAT_BASE_URL}" "${QWEN_CHAT_API_KEY}" "${TARGET_MODEL}" <<'PY'
import json
import sys
import urllib.request

base, key, expected = sys.argv[1].rstrip("/"), sys.argv[2], sys.argv[3]
request = urllib.request.Request(
    f"{base}/models",
    headers={"Authorization": f"Bearer {key}"},
)
try:
    with urllib.request.urlopen(request, timeout=5) as response:
        payload = json.load(response)
    ids = {str(row.get("id")) for row in payload.get("data", [])}
    raise SystemExit(0 if expected in ids else 1)
except Exception:
    raise SystemExit(1)
PY
}

cleanup() {
  if truthy "${STOP_VLLM_ON_EXIT}" && [[ -n "${started_vllm_pid}" ]]; then
    if kill -0 "${started_vllm_pid}" 2>/dev/null; then
      echo "[cleanup] stopping vLLM pid=${started_vllm_pid}"
      kill "${started_vllm_pid}" 2>/dev/null || true
    fi
  fi
}
trap cleanup EXIT

if endpoint_ready; then
  echo "[vllm] reusing endpoint after model-ID verification: ${QWEN_CHAT_BASE_URL}"
else
  truthy "${START_VLLM}" || fail "Endpoint unavailable and START_VLLM=${START_VLLM}."
  vllm_args=(
    serve "${MODEL_PATH}"
    --served-model-name "${SERVED_MODEL_NAME}"
    --host "${VLLM_HOST}"
    --port "${VLLM_PORT}"
    --trust-remote-code
    --dtype bfloat16
    --tensor-parallel-size 1
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
    --max-model-len "${MAX_MODEL_LEN}"
    --enable-prefix-caching
    --enable-chunked-prefill
    --max-num-seqs "${VLLM_MAX_NUM_SEQS}"
    --max-num-batched-tokens "${VLLM_MAX_NUM_BATCHED_TOKENS}"
    --reasoning-parser qwen3
  )
  env CUDA_VISIBLE_DEVICES="${QWEN_CUDA_VISIBLE_DEVICES}" \
    vllm "${vllm_args[@]}" > "${VLLM_LOG}" 2>&1 &
  started_vllm_pid=$!
  echo "[vllm] pid=${started_vllm_pid} log=${VLLM_LOG}"
  for ((i=1; i<=VLLM_WAIT_SECONDS; i++)); do
    if endpoint_ready; then
      echo "[vllm] ready after ${i}s"
      break
    fi
    if ! kill -0 "${started_vllm_pid}" 2>/dev/null; then
      tail -n 100 "${VLLM_LOG}" || true
      fail "vLLM exited before readiness."
    fi
    if ((i == VLLM_WAIT_SECONDS)); then
      tail -n 100 "${VLLM_LOG}" || true
      fail "Timed out waiting for vLLM."
    fi
    sleep 1
  done
fi

run_eval() {
  local manifest="$1"
  local out_root="$2"
  local split="$3"
  local -a eval_cmd=(
    "${PYTHON_BIN}" -u
    "${PROJECT_ROOT}/scripts/tools/eval_searchqa_skill_manifest.py"
    --manifest "${manifest}"
    --out-root "${out_root}"
    --split-dir "${SEARCHQA_SPLIT_DIR}"
    --split "${split}"
    --target-model "${TARGET_MODEL}"
    --base-url "${QWEN_CHAT_BASE_URL}"
    --api-key "${QWEN_CHAT_API_KEY}"
    --temperature "${TARGET_TEMPERATURE}"
    --max-tokens "${TARGET_MAX_TOKENS}"
    --timeout "${TARGET_TIMEOUT}"
    --workers "${WORKERS}"
    --seed "${SEED}"
    --max-turns 1
  )
  if truthy "${RETRY_FAILED}"; then
    eval_cmd+=(--retry-failed)
  fi
  "${eval_cmd[@]}"
}

MAIN_MANIFEST="${REPLAY_DIR}/main/skill_manifest.tsv"
MAIN_EVAL_ROOT="${RESULT_ROOT}/main_eval"
run_eval "${MAIN_MANIFEST}" "${MAIN_EVAL_ROOT}" val \
  2>&1 | tee "${LOG_DIR}/main_val.log"
run_eval "${MAIN_MANIFEST}" "${MAIN_EVAL_ROOT}" test \
  2>&1 | tee "${LOG_DIR}/main_test.log"
"${PYTHON_BIN}" -u "${PROJECT_ROOT}/scripts/tools/analyze_searchqa_tree_verdict.py" \
  --replay-dir "${REPLAY_DIR}" \
  --eval-root "${MAIN_EVAL_ROOT}" \
  --out-dir "${RESULT_ROOT}/verdict" \
  2>&1 | tee "${LOG_DIR}/main_analysis.log"

TOPDOWN_ROOT="${RESULT_ROOT}/topdown"
PHASE1_EVAL_ROOT="${TOPDOWN_ROOT}/phase1_eval"
TOPDOWN_GLOBAL_STEP="$(<"${REPLAY_DIR}/topdown/global_step.txt")"
[[ "${TOPDOWN_GLOBAL_STEP}" =~ ^[0-9]+$ ]] || \
  fail "Invalid top-down global step: ${TOPDOWN_GLOBAL_STEP}"
run_eval \
  "${REPLAY_DIR}/topdown/phase1_val_manifest.tsv" \
  "${PHASE1_EVAL_ROOT}" val \
  2>&1 | tee "${LOG_DIR}/topdown_phase1_val.log"

finalize_cmd=(
  "${PYTHON_BIN}" -u
  "${PROJECT_ROOT}/scripts/tools/finalize_searchqa_topdown.py"
  --merge-artifact "${REPLAY_DIR}/topdown/source_tree_artifact.json"
  --parent-skill "${REPLAY_DIR}/topdown/parent_skill.md"
  --root-skill "${REPLAY_DIR}/topdown/rejected_root_skill.md"
  --parent-results "${PHASE1_EVAL_ROOT}/td_parent/val/results.jsonl"
  --root-results "${PHASE1_EVAL_ROOT}/td_root/val/results.jsonl"
  --val-protocol "${PHASE1_EVAL_ROOT}/td_parent/val/protocol.json"
  --base-seed "${SEED}"
  --global-step "${TOPDOWN_GLOBAL_STEP}"
  --subset-size 40
  --mixed-weight 0.5
  --out-dir "${TOPDOWN_ROOT}/finalize"
)
while IFS=$'\t' read -r child_id run_name; do
  [[ "${child_id}" == "child_id" ]] && continue
  finalize_cmd+=(
    --child-result
    "${child_id}=${PHASE1_EVAL_ROOT}/${run_name}/val/results.jsonl"
  )
done < "${REPLAY_DIR}/topdown/child_manifest.tsv"
"${finalize_cmd[@]}" 2>&1 | tee "${LOG_DIR}/topdown_finalize.log"

run_eval \
  "${TOPDOWN_ROOT}/finalize/phase2_combo_manifest.tsv" \
  "${TOPDOWN_ROOT}/phase2_eval" val \
  2>&1 | tee "${LOG_DIR}/topdown_combo_val.log"

set +e
"${PYTHON_BIN}" -u "${PROJECT_ROOT}/scripts/tools/check_searchqa_topdown_gate.py" \
  --finalize-report "${TOPDOWN_ROOT}/finalize/topdown_finalize_report.json" \
  --combo-results "${TOPDOWN_ROOT}/phase2_eval/td_combo/val/results.jsonl" \
  --out-dir "${TOPDOWN_ROOT}/gate" \
  2>&1 | tee "${LOG_DIR}/topdown_gate.log"
gate_status=${PIPESTATUS[0]}
set -e
if ((gate_status == 0)); then
  run_eval \
    "${TOPDOWN_ROOT}/gate/topdown_test_manifest.tsv" \
    "${TOPDOWN_ROOT}/test_eval" test \
    2>&1 | tee "${LOG_DIR}/topdown_test.log"
elif ((gate_status == 3)); then
  echo "[top-down] Full-val gate/preconditions failed; TEST correctly skipped."
else
  fail "Top-down gate checker failed with exit=${gate_status}."
fi

"${PYTHON_BIN}" -u \
  "${PROJECT_ROOT}/scripts/tools/analyze_searchqa_topdown_verdict.py" \
  --topdown-root "${TOPDOWN_ROOT}" \
  --out-dir "${RESULT_ROOT}/verdict" \
  2>&1 | tee "${LOG_DIR}/topdown_analysis.log"

echo ""
echo "============================================================"
echo "  SearchQA Tree Verdict finished"
echo "  results: ${RESULT_ROOT}"
echo "  report:  ${RESULT_ROOT}/verdict/main_verdict.md"
echo "  topdown: ${RESULT_ROOT}/verdict/topdown_verdict.md"
echo "  logs:    ${LOG_DIR}"
echo "============================================================"

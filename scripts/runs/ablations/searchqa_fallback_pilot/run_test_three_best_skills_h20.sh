#!/usr/bin/env bash
set -euo pipefail

# Evaluate the three completed SearchQA pilot best skills on the full test1400.
# One local Qwen3.5-4B vLLM is shared on a single H20. Skills run sequentially
# to preserve prefix-cache locality and avoid cross-skill GPU contention.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export TARGET_QWEN_CHAT_ROLLOUT_RETRIES="${TARGET_QWEN_CHAT_ROLLOUT_RETRIES:-1}"
PYTHON_BIN="${PYTHON_BIN:-python}"

fail() { echo "ERROR: $*" >&2; exit 1; }
truthy() { case "${1:-}" in 1|true|TRUE|yes|YES|on|ON) return 0 ;; *) return 1 ;; esac; }

command -v "${PYTHON_BIN}" >/dev/null 2>&1 || \
  fail "Python interpreter not found: ${PYTHON_BIN}"

# ── Skills and output roots ─────────────────────────────────────────────────
OUTPUTS_ROOT="${OUTPUTS_ROOT:-${PROJECT_ROOT}/outputs}"
SKILL_MANIFEST="${SKILL_MANIFEST:-}"
SEARCHQA_SPLIT_DIR="${SEARCHQA_SPLIT_DIR:-${PROJECT_ROOT}/data/searchqa_split}"
TEST_ENV_NUM="${TEST_ENV_NUM:-0}"  # 0 = complete SearchQA test1400
SEED="${SEED:-42}"

RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
RESULT_ROOT="${RESULT_ROOT:-${PROJECT_ROOT}/outputs/searchqa_fallback_pilot_three_test_h20_${RUN_STAMP}}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs/searchqa_fallback_pilot_three_test_h20_${RUN_STAMP}}"
MANIFEST_PATH="${RESULT_ROOT}/skill_manifest.tsv"
mkdir -p "${RESULT_ROOT}" "${LOG_DIR}"

write_default_manifest() {
  printf 'run_name\trelative_skill_path\n'
  printf 'p1_d2_fallback_off\tsearchqa_fallback_pilot_p1_d2_fallback_off_seed42_20260717_175305/searchqa/best_skill.md\n'
  printf 'p2_d2_fallback_on\tsearchqa_fallback_pilot_p2_d2_fallback_on_seed42_20260717_175923/searchqa/best_skill.md\n'
  printf 'p3_d3_fallback_off\tsearchqa_fallback_pilot_p3_d3_fallback_off_seed42_20260717_175939/searchqa/best_skill.md\n'
}

if [[ -n "${SKILL_MANIFEST}" ]]; then
  [[ -f "${SKILL_MANIFEST}" ]] || fail "Skill manifest not found: ${SKILL_MANIFEST}"
  cp "${SKILL_MANIFEST}" "${MANIFEST_PATH}"
else
  write_default_manifest > "${MANIFEST_PATH}"
fi

if ! truthy "${DRY_RUN:-0}"; then
  [[ -d "${SEARCHQA_SPLIT_DIR}/test" ]] || \
    fail "SearchQA test split not found: ${SEARCHQA_SPLIT_DIR}/test"
  while IFS=$'\t' read -r run_name relative_skill_path; do
    [[ "${run_name}" == "run_name" ]] && continue
    [[ -n "${run_name}" && -n "${relative_skill_path}" ]] || \
      fail "Malformed manifest row: ${run_name} ${relative_skill_path}"
    [[ -f "${OUTPUTS_ROOT}/${relative_skill_path}" ]] || \
      fail "best_skill.md not found for ${run_name}: ${OUTPUTS_ROOT}/${relative_skill_path}"
  done < "${MANIFEST_PATH}"
fi

# ── One-H20 local Qwen/vLLM profile ─────────────────────────────────────────
MODEL_PATH="${MODEL_PATH:-/ai-app-vepfs/modelscope/models/Qwen/Qwen3.5-4B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen/Qwen3.5-4B}"
TARGET_MODEL="${TARGET_MODEL:-${SERVED_MODEL_NAME}}"
QWEN_CUDA_VISIBLE_DEVICES="${QWEN_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-0}}"
case "${QWEN_CUDA_VISIBLE_DEVICES// /}" in
  "") fail "No GPU selected; set CUDA_VISIBLE_DEVICES or QWEN_CUDA_VISIBLE_DEVICES" ;;
  *,*) fail "This evaluator requires exactly one H20; got ${QWEN_CUDA_VISIBLE_DEVICES}" ;;
esac

VLLM_HOST="${VLLM_HOST:-0.0.0.0}"
VLLM_PORT="${VLLM_PORT:-59317}"
QWEN_CHAT_BASE_URL="${QWEN_CHAT_BASE_URL:-http://127.0.0.1:${VLLM_PORT}/v1}"
QWEN_CHAT_API_KEY="${QWEN_CHAT_API_KEY:-dummy}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-256}"
VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-65536}"
VLLM_DTYPE="${VLLM_DTYPE:-bfloat16}"
VLLM_REASONING_PARSER="${VLLM_REASONING_PARSER:-qwen3}"
START_VLLM="${START_VLLM:-1}"
STOP_VLLM_ON_EXIT="${STOP_VLLM_ON_EXIT:-1}"
VLLM_WAIT_SECONDS="${VLLM_WAIT_SECONDS:-900}"

# SearchQA answers are short. A 4096 completion cap is ample and reduces the
# scheduling/KV reservation pressure compared with the training cap of 16384.
WORKERS="${WORKERS:-256}"
TARGET_QWEN_CHAT_MAX_TOKENS="${TARGET_QWEN_CHAT_MAX_TOKENS:-4096}"
TARGET_QWEN_CHAT_TEMPERATURE="${TARGET_QWEN_CHAT_TEMPERATURE:-0.2}"
TARGET_QWEN_CHAT_TIMEOUT_SECONDS="${TARGET_QWEN_CHAT_TIMEOUT_SECONDS:-300}"
SEARCHQA_EXEC_TIMEOUT="${SEARCHQA_EXEC_TIMEOUT:-300}"
TARGET_QWEN_CHAT_ENABLE_THINKING="${TARGET_QWEN_CHAT_ENABLE_THINKING:-false}"
MAX_TURNS="${MAX_TURNS:-1}"

for value_name in WORKERS VLLM_MAX_NUM_SEQS VLLM_MAX_NUM_BATCHED_TOKENS MAX_MODEL_LEN; do
  value="${!value_name}"
  [[ "${value}" =~ ^[0-9]+$ ]] || fail "${value_name} must be an integer: ${value}"
  [[ "${value}" -ge 1 ]] || fail "${value_name} must be >= 1"
done
[[ "${WORKERS}" -le "${VLLM_MAX_NUM_SEQS}" ]] || \
  fail "WORKERS=${WORKERS} exceeds VLLM_MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS}"

VLLM_LOG="${LOG_DIR}/vllm_qwen.log"
VLLM_PID_FILE="${LOG_DIR}/vllm.pid"

endpoint_ready() {
  "${PYTHON_BIN}" - "${QWEN_CHAT_BASE_URL}" "${QWEN_CHAT_API_KEY}" <<'PY'
import sys
import urllib.request

base, api_key = sys.argv[1].rstrip("/"), sys.argv[2]
request = urllib.request.Request(
    f"{base}/models",
    headers={"Authorization": f"Bearer {api_key}"},
    method="GET",
)
try:
    with urllib.request.urlopen(request, timeout=5) as response:
        raise SystemExit(0 if response.status == 200 else 1)
except Exception:
    raise SystemExit(1)
PY
}

cleanup() {
  if truthy "${STOP_VLLM_ON_EXIT}" && [[ -f "${VLLM_PID_FILE}" ]]; then
    local pid
    pid="$(<"${VLLM_PID_FILE}")"
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      echo "[cleanup] stopping vLLM pid=${pid}"
      kill "${pid}" 2>/dev/null || true
    fi
  fi
}
trap cleanup EXIT

echo "============================================================"
echo "  SearchQA pilot: three best skills -> full TEST on one H20"
echo "============================================================"
echo "  outputs_root:   ${OUTPUTS_ROOT}"
echo "  split_dir:      ${SEARCHQA_SPLIT_DIR}"
echo "  test_env_num:   ${TEST_ENV_NUM} (0 = full test1400)"
echo "  result_root:    ${RESULT_ROOT}"
echo "  target/model:   ${TARGET_MODEL} / ${MODEL_PATH}"
echo "  cuda/url:       ${QWEN_CUDA_VISIBLE_DEVICES} / ${QWEN_CHAT_BASE_URL}"
echo "  workers/seqs:   ${WORKERS}/${VLLM_MAX_NUM_SEQS}"
echo "  batch_tokens:   ${VLLM_MAX_NUM_BATCHED_TOKENS}"
echo "  model_len/mem:  ${MAX_MODEL_LEN}/${GPU_MEMORY_UTILIZATION}"
echo "  output_tokens:  ${TARGET_QWEN_CHAT_MAX_TOKENS}"
echo "  temperature:    ${TARGET_QWEN_CHAT_TEMPERATURE}"
echo "============================================================"

if truthy "${DRY_RUN:-0}"; then
  echo "[dry-run] vLLM startup and file preflight are skipped."
elif endpoint_ready; then
  echo "[check] reusing ready Qwen endpoint: ${QWEN_CHAT_BASE_URL}"
else
  truthy "${START_VLLM}" || \
    fail "Qwen endpoint is not ready and START_VLLM=${START_VLLM}"
  [[ -d "${MODEL_PATH}" ]] || fail "MODEL_PATH not found: ${MODEL_PATH}"

  vllm_args=(
    serve "${MODEL_PATH}"
    --served-model-name "${SERVED_MODEL_NAME}"
    --host "${VLLM_HOST}"
    --port "${VLLM_PORT}"
    --trust-remote-code
    --dtype "${VLLM_DTYPE}"
    --tensor-parallel-size 1
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
    --max-model-len "${MAX_MODEL_LEN}"
    --enable-prefix-caching
    --enable-chunked-prefill
    --max-num-seqs "${VLLM_MAX_NUM_SEQS}"
    --max-num-batched-tokens "${VLLM_MAX_NUM_BATCHED_TOKENS}"
  )
  if [[ -n "${VLLM_REASONING_PARSER}" ]]; then
    vllm_args+=(--reasoning-parser "${VLLM_REASONING_PARSER}")
  fi

  echo "[vllm] starting Qwen on CUDA_VISIBLE_DEVICES=${QWEN_CUDA_VISIBLE_DEVICES}"
  env CUDA_VISIBLE_DEVICES="${QWEN_CUDA_VISIBLE_DEVICES}" \
    nohup vllm "${vllm_args[@]}" > "${VLLM_LOG}" 2>&1 &
  vllm_pid=$!
  echo "${vllm_pid}" > "${VLLM_PID_FILE}"
  echo "[vllm] pid=${vllm_pid} log=${VLLM_LOG}"

  for ((i = 1; i <= VLLM_WAIT_SECONDS; i++)); do
    if endpoint_ready; then
      echo "[check] Qwen endpoint ready after ${i}s"
      break
    fi
    if ! kill -0 "${vllm_pid}" 2>/dev/null; then
      tail -n 80 "${VLLM_LOG}" || true
      fail "vLLM exited before the endpoint became ready"
    fi
    if ((i == VLLM_WAIT_SECONDS)); then
      tail -n 80 "${VLLM_LOG}" || true
      fail "timed out waiting for vLLM endpoint"
    fi
    sleep 1
  done
fi

# ── Sequential, resume-aware best-skill evaluation ──────────────────────────
failed=0
suite_started_at="$(date +%s)"
while IFS=$'\t' read -r run_name relative_skill_path; do
  [[ "${run_name}" == "run_name" ]] && continue
  skill_path="${OUTPUTS_ROOT}/${relative_skill_path}"
  run_result_dir="${RESULT_ROOT}/${run_name}"
  run_log="${LOG_DIR}/${run_name}.log"
  mkdir -p "${run_result_dir}"

  eval_cmd=(
    "${PYTHON_BIN}" -u "${PROJECT_ROOT}/scripts/cli/eval_only.py"
    --config "${PROJECT_ROOT}/configs/searchqa/default.yaml"
    --skill "${skill_path}"
    --split test
    --out_root "${run_result_dir}"
    --target_backend qwen_chat
    --target_model "${TARGET_MODEL}"
    --target_qwen_chat_base_url "${QWEN_CHAT_BASE_URL}"
    --target_qwen_chat_api_key "${QWEN_CHAT_API_KEY}"
    --target_qwen_chat_temperature "${TARGET_QWEN_CHAT_TEMPERATURE}"
    --target_qwen_chat_timeout_seconds "${TARGET_QWEN_CHAT_TIMEOUT_SECONDS}"
    --target_qwen_chat_max_tokens "${TARGET_QWEN_CHAT_MAX_TOKENS}"
    --target_qwen_chat_enable_thinking "${TARGET_QWEN_CHAT_ENABLE_THINKING}"
    --split_mode split_dir
    --split_dir "${SEARCHQA_SPLIT_DIR}"
    --workers "${WORKERS}"
    --seed "${SEED}"
    --test_env_num "${TEST_ENV_NUM}"
    --max_turns "${MAX_TURNS}"
    --cfg-options
      "env.exec_timeout=${SEARCHQA_EXEC_TIMEOUT}"
      "env.max_completion_tokens=${TARGET_QWEN_CHAT_MAX_TOKENS}"
      "env.limit=0"
  )

  echo ""
  echo "[eval] ${run_name}"
  echo "       skill=${skill_path}"
  echo "       output=${run_result_dir}"
  if truthy "${DRY_RUN:-0}"; then
    printf '[dry-run]'
    printf ' %q' "${eval_cmd[@]}"
    printf '\n'
    continue
  fi

  run_started_at="$(date +%s)"
  set +e
  "${eval_cmd[@]}" 2>&1 | tee "${run_log}"
  status=${PIPESTATUS[0]}
  set -e
  elapsed="$(( $(date +%s) - run_started_at ))"
  echo "${elapsed}" > "${run_result_dir}/elapsed_seconds.txt"
  if ((status != 0)); then
    echo "ERROR: ${run_name} evaluation failed with exit=${status}" >&2
    failed=1
  fi
done < "${MANIFEST_PATH}"

if truthy "${DRY_RUN:-0}"; then
  echo "[dry-run] manifest=${MANIFEST_PATH}"
  exit 0
fi

suite_elapsed="$(( $(date +%s) - suite_started_at ))"

# Aggregate eval_summary.json into machine-readable and Markdown tables.
"${PYTHON_BIN}" - "${RESULT_ROOT}" "${MANIFEST_PATH}" "${suite_elapsed}" <<'PY'
import csv
import json
import os
import sys

result_root, manifest_path, suite_elapsed = sys.argv[1:]
rows = []
with open(manifest_path, newline="") as file:
    for manifest_row in csv.DictReader(file, delimiter="\t"):
        name = manifest_row["run_name"]
        run_dir = os.path.join(result_root, name)
        summary_path = os.path.join(run_dir, "eval_summary.json")
        elapsed_path = os.path.join(run_dir, "elapsed_seconds.txt")
        elapsed = None
        if os.path.exists(elapsed_path):
            with open(elapsed_path) as elapsed_file:
                elapsed = int(elapsed_file.read().strip())
        if os.path.exists(summary_path):
            with open(summary_path) as summary_file:
                summary = json.load(summary_file)
            rows.append({
                "run_name": name,
                "n_items": summary.get("n_items"),
                "test_hard": summary.get("hard"),
                "test_soft": summary.get("soft"),
                "elapsed_seconds": elapsed,
                "skill": summary.get("skill"),
                "status": "ok",
            })
        else:
            rows.append({
                "run_name": name,
                "n_items": None,
                "test_hard": None,
                "test_soft": None,
                "elapsed_seconds": elapsed,
                "skill": manifest_row["relative_skill_path"],
                "status": "missing",
            })

json_path = os.path.join(result_root, "three_test_summary.json")
csv_path = os.path.join(result_root, "three_test_summary.csv")
md_path = os.path.join(result_root, "three_test_summary.md")
with open(json_path, "w") as file:
    json.dump(
        {"suite_elapsed_seconds": int(suite_elapsed), "runs": rows},
        file,
        indent=2,
        ensure_ascii=False,
    )
with open(csv_path, "w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
with open(md_path, "w") as file:
    file.write("# SearchQA fallback pilot: full test results\n\n")
    file.write(f"Suite wall time: {int(suite_elapsed)} seconds.\n\n")
    file.write("| run | n | test hard | test soft | seconds | status |\n")
    file.write("|---|---:|---:|---:|---:|---|\n")
    for row in rows:
        hard = "-" if row["test_hard"] is None else f'{row["test_hard"]:.4f}'
        soft = "-" if row["test_soft"] is None else f'{row["test_soft"]:.4f}'
        n_items = "-" if row["n_items"] is None else str(row["n_items"])
        elapsed = "-" if row["elapsed_seconds"] is None else str(row["elapsed_seconds"])
        file.write(
            f'| {row["run_name"]} | {n_items} | {hard} | {soft} | '
            f'{elapsed} | {row["status"]} |\n'
        )
print(f"[summary] {md_path}")
PY

echo ""
echo "============================================================"
echo "  SearchQA three-skill TEST evaluation finished"
echo "  results: ${RESULT_ROOT}"
echo "  logs:    ${LOG_DIR}"
echo "  seconds: ${suite_elapsed}"
echo "============================================================"

((failed == 0)) || exit 1

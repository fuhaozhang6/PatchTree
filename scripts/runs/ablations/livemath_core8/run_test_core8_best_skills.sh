#!/usr/bin/env bash
set -euo pipefail

# Evaluate all eight LiveMath core-8 best_skill.md files on the TEST split.
# A single local Qwen3.5-4B vLLM is reused, while skills run sequentially so
# their scores are not affected by cross-skill GPU contention.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export TARGET_QWEN_CHAT_ROLLOUT_RETRIES="${TARGET_QWEN_CHAT_ROLLOUT_RETRIES:-1}"
PYTHON_BIN="${PYTHON_BIN:-python}"

fail() { echo "ERROR: $*" >&2; exit 1; }
truthy() { case "${1:-}" in 1|true|TRUE|yes|YES|on|ON) return 0 ;; *) return 1 ;; esac; }

command -v "${PYTHON_BIN}" >/dev/null 2>&1 || fail "Python interpreter not found: ${PYTHON_BIN}"

# ── Inputs and result roots ─────────────────────────────────────────────────
# SKILL_ROOT/SKILL_MANIFEST make this evaluator reusable for later LiveMath
# ablation suites. CORE8_ROOT remains supported for backward compatibility.
SKILL_ROOT="${SKILL_ROOT:-${CORE8_ROOT:-/ai-app-vepfs/zhangfuhao/skill/SkillOpt-Tree/outputs/livemath_core8_pairs}}"
SKILL_MANIFEST="${SKILL_MANIFEST:-}"
EVAL_SUITE_SLUG="${EVAL_SUITE_SLUG:-livemath_core8}"
EVAL_SUITE_TITLE="${EVAL_SUITE_TITLE:-LiveMath core-8}"
EVAL_SUMMARY_STEM="${EVAL_SUMMARY_STEM:-core8_test_summary}"
LIVEMATH_SPLIT_DIR="${LIVEMATH_SPLIT_DIR:-${PROJECT_ROOT}/data/livemathematicianbench_split}"
TEST_ENV_NUM="${TEST_ENV_NUM:-0}"  # 0 = all 124 test examples
SEED="${SEED:-42}"

RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
RESULT_ROOT="${RESULT_ROOT:-${PROJECT_ROOT}/outputs/${EVAL_SUITE_SLUG}_test_${RUN_STAMP}}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs/${EVAL_SUITE_SLUG}_test_${RUN_STAMP}}"
MANIFEST_PATH="${RESULT_ROOT}/skill_manifest.tsv"
mkdir -p "${RESULT_ROOT}" "${LOG_DIR}"

write_manifest() {
  printf 'run_name\trelative_skill_path\n'
  printf 'r1_b8_d2\tr1_r4_seed42_20260716_161711/r1_b8_d2/livemath/best_skill.md\n'
  printf 'r2_b8_d2\tr2_base_seed42_20260716_162410/r2_b8_d2/livemath/best_skill.md\n'
  printf 'r3_b8_d2\tr2_base_seed42_20260716_162410/base_r3_b8_d2/livemath/best_skill.md\n'
  printf 'r4_b8_d2\tr1_r4_seed42_20260716_161711/r4_b8_d2/livemath/best_skill.md\n'
  printf 'r3_b4_d2\tb4_b32_seed42_20260716_162509/r3_b4_d2/livemath/best_skill.md\n'
  printf 'r3_b16_d2\tb16_d3_seed42_20260716_162509/r3_b16_d2/livemath/best_skill.md\n'
  printf 'r3_b32_d2\tb4_b32_seed42_20260716_162509/r3_b32_d2/livemath/best_skill.md\n'
  printf 'r3_b8_d3\tb16_d3_seed42_20260716_162509/r3_b8_d3/livemath/best_skill.md\n'
}
if [[ -n "${SKILL_MANIFEST}" ]]; then
  [[ -f "${SKILL_MANIFEST}" ]] || fail "Skill manifest not found: ${SKILL_MANIFEST}"
  cp "${SKILL_MANIFEST}" "${MANIFEST_PATH}"
else
  write_manifest > "${MANIFEST_PATH}"
fi

if ! truthy "${DRY_RUN:-0}"; then
  [[ -d "${LIVEMATH_SPLIT_DIR}/test" ]] || fail "LiveMath test split not found: ${LIVEMATH_SPLIT_DIR}/test"
  while IFS=$'\t' read -r run_name relative_skill_path; do
    [[ "${run_name}" == "run_name" ]] && continue
    [[ -f "${SKILL_ROOT}/${relative_skill_path}" ]] || \
      fail "best_skill.md not found for ${run_name}: ${SKILL_ROOT}/${relative_skill_path}"
  done < "${MANIFEST_PATH}"
fi

# ── Local Qwen/vLLM ─────────────────────────────────────────────────────────
MODEL_PATH="${MODEL_PATH:-/ai-app-vepfs/modelscope/models/Qwen/Qwen3.5-4B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen/Qwen3.5-4B}"
TARGET_MODEL="${TARGET_MODEL:-${SERVED_MODEL_NAME}}"
QWEN_CUDA_VISIBLE_DEVICES="${QWEN_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-0}}"
VLLM_HOST="${VLLM_HOST:-0.0.0.0}"
VLLM_PORT="${VLLM_PORT:-59317}"
QWEN_CHAT_BASE_URL="${QWEN_CHAT_BASE_URL:-http://127.0.0.1:${VLLM_PORT}/v1}"
QWEN_CHAT_API_KEY="${QWEN_CHAT_API_KEY:-dummy}"

MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-128}"
VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-65536}"
VLLM_DTYPE="${VLLM_DTYPE:-bfloat16}"
START_VLLM="${START_VLLM:-1}"
STOP_VLLM_ON_EXIT="${STOP_VLLM_ON_EXIT:-1}"
VLLM_WAIT_SECONDS="${VLLM_WAIT_SECONDS:-900}"

# Training used 300 seconds. Test defaults to 600 seconds because the pilot
# showed slow requests being scored as wrong even though vLLM later returned
# HTTP 200. Override both values to 300 for exact training-time parity.
WORKERS="${WORKERS:-96}"
TARGET_QWEN_CHAT_MAX_TOKENS="${TARGET_QWEN_CHAT_MAX_TOKENS:-16384}"
TARGET_QWEN_CHAT_TEMPERATURE="${TARGET_QWEN_CHAT_TEMPERATURE:-0.2}"
TARGET_QWEN_CHAT_TIMEOUT_SECONDS="${TARGET_QWEN_CHAT_TIMEOUT_SECONDS:-600}"
LIVEMATH_EXEC_TIMEOUT="${LIVEMATH_EXEC_TIMEOUT:-600}"
TARGET_QWEN_CHAT_ENABLE_THINKING="${TARGET_QWEN_CHAT_ENABLE_THINKING:-false}"
MAX_TURNS="${MAX_TURNS:-1}"

[[ "${WORKERS}" =~ ^[0-9]+$ ]] || fail "WORKERS must be an integer: ${WORKERS}"
[[ "${WORKERS}" -ge 1 ]] || fail "WORKERS must be >= 1"
[[ "${WORKERS}" -le "${VLLM_MAX_NUM_SEQS}" ]] || \
  fail "WORKERS=${WORKERS} exceeds VLLM_MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS}"

VLLM_LOG="${LOG_DIR}/vllm_qwen.log"
VLLM_PID_FILE="${LOG_DIR}/vllm.pid"

endpoint_ready() {
  "${PYTHON_BIN}" - "${QWEN_CHAT_BASE_URL}" "${QWEN_CHAT_API_KEY}" <<'PY'
import sys
import urllib.request

base, api_key = sys.argv[1].rstrip("/"), sys.argv[2]
req = urllib.request.Request(
    f"{base}/models",
    headers={"Authorization": f"Bearer {api_key}"},
    method="GET",
)
try:
    with urllib.request.urlopen(req, timeout=5) as resp:
        raise SystemExit(0 if resp.status == 200 else 1)
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
echo "  ${EVAL_SUITE_TITLE} best skills -> full TEST"
echo "============================================================"
echo "  skill_root:      ${SKILL_ROOT}"
echo "  split_dir:       ${LIVEMATH_SPLIT_DIR}"
echo "  test_env_num:    ${TEST_ENV_NUM} (0 = full test)"
echo "  result_root:     ${RESULT_ROOT}"
echo "  target:          ${TARGET_MODEL}"
echo "  model_path:      ${MODEL_PATH}"
echo "  qwen_url:        ${QWEN_CHAT_BASE_URL}"
echo "  cuda_devices:    ${QWEN_CUDA_VISIBLE_DEVICES}"
echo "  workers/seqs:    ${WORKERS}/${VLLM_MAX_NUM_SEQS}"
echo "  timeout:         request=${TARGET_QWEN_CHAT_TIMEOUT_SECONDS}s exec=${LIVEMATH_EXEC_TIMEOUT}s"
echo "  temperature:     ${TARGET_QWEN_CHAT_TEMPERATURE}"
echo "============================================================"

if truthy "${DRY_RUN:-0}"; then
  echo "[dry-run] vLLM startup is skipped."
elif endpoint_ready; then
  echo "[check] existing Qwen endpoint is ready: ${QWEN_CHAT_BASE_URL}"
else
  truthy "${START_VLLM}" || fail "Qwen endpoint is not ready and START_VLLM=${START_VLLM}"
  [[ -d "${MODEL_PATH}" ]] || fail "MODEL_PATH not found: ${MODEL_PATH}"
  echo "[vllm] starting Qwen on CUDA_VISIBLE_DEVICES=${QWEN_CUDA_VISIBLE_DEVICES}"
  env CUDA_VISIBLE_DEVICES="${QWEN_CUDA_VISIBLE_DEVICES}" \
    nohup vllm serve "${MODEL_PATH}" \
      --served-model-name "${SERVED_MODEL_NAME}" \
      --host "${VLLM_HOST}" \
      --port "${VLLM_PORT}" \
      --trust-remote-code \
      --dtype "${VLLM_DTYPE}" \
      --tensor-parallel-size 1 \
      --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
      --max-model-len "${MAX_MODEL_LEN}" \
      --enable-prefix-caching \
      --enable-chunked-prefill \
      --max-num-seqs "${VLLM_MAX_NUM_SEQS}" \
      --max-num-batched-tokens "${VLLM_MAX_NUM_BATCHED_TOKENS}" \
      > "${VLLM_LOG}" 2>&1 &
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

# ── Sequential eval-only runs ───────────────────────────────────────────────
failed=0
while IFS=$'\t' read -r run_name relative_skill_path; do
  [[ "${run_name}" == "run_name" ]] && continue
  skill_path="${SKILL_ROOT}/${relative_skill_path}"
  run_result_dir="${RESULT_ROOT}/${run_name}"
  run_log="${LOG_DIR}/${run_name}.log"

  eval_cmd=(
    "${PYTHON_BIN}" -u "${PROJECT_ROOT}/scripts/cli/eval_only.py"
    --config "${PROJECT_ROOT}/configs/livemathematicianbench/default.yaml"
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
    --split_dir "${LIVEMATH_SPLIT_DIR}"
    --workers "${WORKERS}"
    --seed "${SEED}"
    --test_env_num "${TEST_ENV_NUM}"
    --max_turns "${MAX_TURNS}"
    --cfg-options
      "env.exec_timeout=${LIVEMATH_EXEC_TIMEOUT}"
      "env.max_completion_tokens=${TARGET_QWEN_CHAT_MAX_TOKENS}"
      "env.limit=0"
      "env.shuffle_choices=true"
      "env.use_theorem=false"
      "env.use_sketch=false"
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

  set +e
  "${eval_cmd[@]}" 2>&1 | tee "${run_log}"
  status=${PIPESTATUS[0]}
  set -e
  if ((status != 0)); then
    echo "ERROR: ${run_name} evaluation failed with exit=${status}" >&2
    failed=1
  fi
done < "${MANIFEST_PATH}"

if truthy "${DRY_RUN:-0}"; then
  echo "[dry-run] manifest=${MANIFEST_PATH}"
  exit 0
fi

# Aggregate all eval_summary.json files into Markdown, CSV, and JSON.
"${PYTHON_BIN}" - "${RESULT_ROOT}" "${MANIFEST_PATH}" "${EVAL_SUMMARY_STEM}" "${EVAL_SUITE_TITLE}" <<'PY'
import csv
import json
import os
import sys

result_root, manifest_path, summary_stem, suite_title = sys.argv[1:]
rows = []
with open(manifest_path, newline="") as f:
    for row in csv.DictReader(f, delimiter="\t"):
        name = row["run_name"]
        summary_path = os.path.join(result_root, name, "eval_summary.json")
        if os.path.exists(summary_path):
            with open(summary_path) as sf:
                summary = json.load(sf)
            rows.append({
                "run_name": name,
                "n_items": summary.get("n_items"),
                "test_hard": summary.get("hard"),
                "test_soft": summary.get("soft"),
                "skill": summary.get("skill"),
                "status": "ok",
            })
        else:
            rows.append({
                "run_name": name,
                "n_items": None,
                "test_hard": None,
                "test_soft": None,
                "skill": row["relative_skill_path"],
                "status": "missing",
            })

with open(os.path.join(result_root, f"{summary_stem}.json"), "w") as f:
    json.dump(rows, f, indent=2, ensure_ascii=False)

with open(os.path.join(result_root, f"{summary_stem}.csv"), "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

md_path = os.path.join(result_root, f"{summary_stem}.md")
with open(md_path, "w") as f:
    f.write(f"# {suite_title} test results\n\n")
    f.write("| run | n | test hard | test soft | status |\n")
    f.write("|---|---:|---:|---:|---|\n")
    for row in rows:
        hard = "-" if row["test_hard"] is None else f'{row["test_hard"]:.4f}'
        soft = "-" if row["test_soft"] is None else f'{row["test_soft"]:.4f}'
        n_items = "-" if row["n_items"] is None else str(row["n_items"])
        f.write(f'| {row["run_name"]} | {n_items} | {hard} | {soft} | {row["status"]} |\n')

print(f"[summary] {md_path}")
PY

echo ""
echo "============================================================"
echo "  ${EVAL_SUITE_TITLE} TEST evaluation finished"
echo "  results: ${RESULT_ROOT}"
echo "  logs:    ${LOG_DIR}"
echo "============================================================"

((failed == 0)) || exit 1

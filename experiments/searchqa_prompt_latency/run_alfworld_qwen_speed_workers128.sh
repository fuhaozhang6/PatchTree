#!/usr/bin/env bash
set -euo pipefail

# ALFWorld speed probe with local Qwen target.
# This is eval-only: it runs the initial ALFWorld skill on a selected split and
# reports wall time, per-episode time, hard/soft score, and rollout artifacts.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
OUT_BASE="${OUT_BASE:-outputs/alfworld_qwen_speed_workers128_${TS}}"
LOG_DIR="${LOG_DIR:-${OUT_BASE}/logs}"
SERVER_OUT_DIR="${SERVER_OUT_DIR:-${OUT_BASE}/server}"
RUN_LOG="${RUN_LOG:-${LOG_DIR}/alfworld_eval.log}"
SUMMARY_JSON="${SUMMARY_JSON:-${OUT_BASE}/speed_summary.json}"
SUMMARY_MD="${SUMMARY_MD:-${OUT_BASE}/speed_summary.md}"

ALFWORLD_CONFIG="${ALFWORLD_CONFIG:-configs/alfworld/default.yaml}"
ALFWORLD_SPLIT_DIR="${ALFWORLD_SPLIT_DIR:-data/alfworld_path_split}"
ALFWORLD_DATA="${ALFWORLD_DATA:-${PROJECT_ROOT}/data/alfworld}"
ALFWORLD_SKILL="${ALFWORLD_SKILL:-skillopt/envs/alfworld/skills/initial.md}"
ALFWORLD_SPLIT="${ALFWORLD_SPLIT:-test}"
ALFWORLD_ENV_NUM="${ALFWORLD_ENV_NUM:-0}"
ALFWORLD_WORKERS="${ALFWORLD_WORKERS:-128}"
ALFWORLD_MAX_API_WORKERS="${ALFWORLD_MAX_API_WORKERS:-128}"
ALFWORLD_MAX_STEPS="${ALFWORLD_MAX_STEPS:-50}"
ALFWORLD_TARGET_MAX_COMPLETION_TOKENS="${ALFWORLD_TARGET_MAX_COMPLETION_TOKENS:-2048}"

QWEN_CHAT_BASE_URL="${QWEN_CHAT_BASE_URL:-http://127.0.0.1:${VLLM_PORT:-8000}/v1}"
QWEN_CHAT_API_KEY="${QWEN_CHAT_API_KEY:-dummy}"
QWEN_CHAT_MODEL="${QWEN_CHAT_MODEL:-${SERVED_MODEL_NAME:-Qwen/Qwen3.5-4B}}"
QWEN_CHAT_TIMEOUT_SECONDS="${QWEN_CHAT_TIMEOUT_SECONDS:-240}"
TARGET_QWEN_CHAT_TEMPERATURE="${TARGET_QWEN_CHAT_TEMPERATURE:-0.2}"
TARGET_QWEN_CHAT_ENABLE_THINKING="${TARGET_QWEN_CHAT_ENABLE_THINKING:-false}"
START_VLLM="${START_VLLM:-1}"
STOP_VLLM_ON_EXIT="${STOP_VLLM_ON_EXIT:-1}"
DRY_RUN="${DRY_RUN:-0}"

mkdir -p "${OUT_BASE}" "${LOG_DIR}"

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

truthy() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

cleanup() {
  if truthy "${STOP_VLLM_ON_EXIT}"; then
    local pid_file="${SERVER_OUT_DIR}/logs/vllm.pid"
    if [[ -f "${pid_file}" ]]; then
      local pid
      pid="$(cat "${pid_file}" 2>/dev/null || true)"
      if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
        echo "[vllm] stopping pid=${pid}"
        kill "${pid}" || true
      fi
    fi
  fi
}
trap cleanup EXIT

check_alfworld_split() {
  local split
  [[ -d "${ALFWORLD_SPLIT_DIR}" ]] || return 1
  for split in train val test; do
    [[ -f "${ALFWORLD_SPLIT_DIR}/${split}/items.json" ]] || return 1
  done
}

check_alfworld_payload() {
  "${PYTHON_BIN}" - "${ALFWORLD_SPLIT_DIR}" "${ALFWORLD_DATA}" <<'PY'
import json
import os
import sys

split_dir, data_root = sys.argv[1:]
for split in ("train", "val", "test"):
    path = os.path.join(split_dir, split, "items.json")
    with open(path, encoding="utf-8") as f:
        items = json.load(f)
    if not isinstance(items, list) or not items:
        raise SystemExit(f"No items in {path}")
    gamefile = str((items[0] or {}).get("gamefile") or "").strip()
    if not gamefile:
        raise SystemExit(f"Missing gamefile in {path}")
    full = gamefile if os.path.isabs(gamefile) else os.path.join(data_root, gamefile)
    if not os.path.exists(full):
        raise SystemExit(f"Missing ALFWorld gamefile referenced by {path}: {full}")
root = os.path.join(data_root, "json_2.1.1")
if not os.path.isdir(root):
    raise SystemExit(f"Missing ALFWorld json_2.1.1 directory: {root}")
PY
}

check_alfworld_python_deps() {
  "${PYTHON_BIN}" - <<'PY'
missing = []
for name in ("alfworld", "gymnasium", "omegaconf"):
    try:
        __import__(name)
    except Exception:
        missing.append(name)
if missing:
    raise SystemExit("Missing ALFWorld Python dependencies: " + ", ".join(missing))
PY
}

split_count() {
  "${PYTHON_BIN}" - "${ALFWORLD_SPLIT_DIR}" "${ALFWORLD_SPLIT}" "${ALFWORLD_ENV_NUM}" <<'PY'
import json
import os
import sys

split_dir, split, env_num = sys.argv[1], sys.argv[2], int(sys.argv[3])
alias = {"valid_seen": "val", "selection": "val", "valid_unseen": "test"}.get(split, split)
path = os.path.join(split_dir, alias, "items.json")
items = json.load(open(path, encoding="utf-8"))
print(min(env_num, len(items)) if env_num > 0 else len(items))
PY
}

[[ -f "${ALFWORLD_CONFIG}" ]] || fail "Missing config: ${ALFWORLD_CONFIG}"
[[ -f "${ALFWORLD_SKILL}" ]] || fail "Missing skill: ${ALFWORLD_SKILL}"
if ! check_alfworld_split; then
  if truthy "${DRY_RUN}"; then
    echo "[dry-run][warn] missing split_dir: ${ALFWORLD_SPLIT_DIR}/{train,val,test}/items.json" >&2
  else
    fail "Missing split_dir: ${ALFWORLD_SPLIT_DIR}/{train,val,test}/items.json"
  fi
elif ! check_alfworld_payload; then
  if truthy "${DRY_RUN}"; then
    echo "[dry-run][warn] ALFWorld split references missing payload. Check ALFWORLD_DATA=${ALFWORLD_DATA}" >&2
  else
    fail "ALFWorld split references missing payload. Check ALFWORLD_DATA=${ALFWORLD_DATA}"
  fi
fi

if ! check_alfworld_python_deps; then
  if truthy "${DRY_RUN}"; then
    echo "[dry-run][warn] Missing ALFWorld Python dependencies for ${PYTHON_BIN}" >&2
  else
    fail "Missing ALFWorld Python dependencies for ${PYTHON_BIN}"
  fi
fi

EVAL_ITEMS="$(split_count)"

echo "============================================================"
echo "  ALFWorld Qwen Speed Probe"
echo "============================================================"
echo "  project:          ${PROJECT_ROOT}"
echo "  out_base:         ${OUT_BASE}"
echo "  split:            ${ALFWORLD_SPLIT}"
echo "  eval_items:       ${EVAL_ITEMS}"
echo "  env_workers:      ${ALFWORLD_WORKERS}"
echo "  max_api_workers:  ${ALFWORLD_MAX_API_WORKERS}"
echo "  max_steps:        ${ALFWORLD_MAX_STEPS}"
echo "  max_tokens:       ${ALFWORLD_TARGET_MAX_COMPLETION_TOKENS}"
echo "  qwen_url:         ${QWEN_CHAT_BASE_URL}"
echo "  qwen_model:       ${QWEN_CHAT_MODEL}"
echo "  dry_run:          ${DRY_RUN}"
echo "============================================================"

if truthy "${DRY_RUN}"; then
  echo "[dry-run] Skip vLLM startup and ALFWorld rollout."
  exit 0
fi

echo "[step 1/3] start or reuse Qwen vLLM"
START_ONLY=1 \
START_VLLM="${START_VLLM}" \
STOP_VLLM_ON_EXIT=0 \
OUT_DIR="${SERVER_OUT_DIR}" \
QWEN_CHAT_BASE_URL="${QWEN_CHAT_BASE_URL}" \
QWEN_CHAT_API_KEY="${QWEN_CHAT_API_KEY}" \
QWEN_CHAT_MODEL="${QWEN_CHAT_MODEL}" \
QWEN_CHAT_TIMEOUT_SECONDS="${QWEN_CHAT_TIMEOUT_SECONDS}" \
TARGET_QWEN_CHAT_TEMPERATURE="${TARGET_QWEN_CHAT_TEMPERATURE}" \
TARGET_QWEN_CHAT_ENABLE_THINKING="${TARGET_QWEN_CHAT_ENABLE_THINKING}" \
  bash experiments/searchqa_prompt_latency/run_with_vllm.sh

echo "[step 2/3] run ALFWorld eval-only rollout"
start_epoch="$(date +%s)"
set +e
"${PYTHON_BIN}" -u scripts/cli/eval_only.py \
  --config "${ALFWORLD_CONFIG}" \
  --skill "${ALFWORLD_SKILL}" \
  --split "${ALFWORLD_SPLIT}" \
  --out_root "${OUT_BASE}/eval" \
  --target_backend qwen_chat \
  --target_model "${QWEN_CHAT_MODEL}" \
  --split_dir "${ALFWORLD_SPLIT_DIR}" \
  --workers "${ALFWORLD_WORKERS}" \
  --max_api_workers "${ALFWORLD_MAX_API_WORKERS}" \
  --test_env_num "${ALFWORLD_ENV_NUM}" \
  --cfg-options \
    "env.split_mode=split_dir" \
    "env.split_dir=${ALFWORLD_SPLIT_DIR}" \
    "env.max_steps=${ALFWORLD_MAX_STEPS}" \
    "env.max_completion_tokens=${ALFWORLD_TARGET_MAX_COMPLETION_TOKENS}" \
    "model.target_qwen_chat_base_url=${QWEN_CHAT_BASE_URL}" \
    "model.target_qwen_chat_api_key=${QWEN_CHAT_API_KEY}" \
    "model.target_qwen_chat_temperature=${TARGET_QWEN_CHAT_TEMPERATURE}" \
    "model.target_qwen_chat_timeout_seconds=${QWEN_CHAT_TIMEOUT_SECONDS}" \
    "model.target_qwen_chat_max_tokens=${ALFWORLD_TARGET_MAX_COMPLETION_TOKENS}" \
    "model.target_qwen_chat_enable_thinking=${TARGET_QWEN_CHAT_ENABLE_THINKING}" \
  2>&1 | tee "${RUN_LOG}"
status=${PIPESTATUS[0]}
set -e
end_epoch="$(date +%s)"
elapsed_s=$((end_epoch - start_epoch))

echo "[step 3/3] summarize"
"${PYTHON_BIN}" - "${OUT_BASE}" "${SUMMARY_JSON}" "${SUMMARY_MD}" "${elapsed_s}" "${status}" "${EVAL_ITEMS}" <<'PY'
import json
import sys
from pathlib import Path

out_base = Path(sys.argv[1])
summary_json = Path(sys.argv[2])
summary_md = Path(sys.argv[3])
elapsed_s = int(sys.argv[4])
status = int(sys.argv[5])
expected_items = int(sys.argv[6])

eval_summary_path = out_base / "eval" / "eval_summary.json"
results_path = out_base / "eval" / "results.jsonl"
eval_summary = {}
if eval_summary_path.exists():
    eval_summary = json.loads(eval_summary_path.read_text(encoding="utf-8"))

results = []
if results_path.exists():
    with results_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))

turn_counts = []
successes = 0
for row in results:
    successes += int(bool(row.get("hard") or row.get("success")))
    traj = row.get("trajectory") or row.get("traj") or row.get("steps") or []
    if isinstance(traj, list):
        turn_counts.append(len(traj))

n = len(results) or int(eval_summary.get("n_items") or 0)
hard = eval_summary.get("hard")
soft = eval_summary.get("soft")
episodes_per_s = (n / elapsed_s) if elapsed_s > 0 and n else None
seconds_per_episode = (elapsed_s / n) if n else None

summary = {
    "status": status,
    "elapsed_s": elapsed_s,
    "expected_items": expected_items,
    "n_items": n,
    "hard": hard,
    "soft": soft,
    "episodes_per_s": episodes_per_s,
    "seconds_per_episode": seconds_per_episode,
    "results_path": str(results_path),
    "eval_summary_path": str(eval_summary_path),
}
summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

def fmt(value, digits=4):
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)

lines = [
    "# ALFWorld Qwen Speed Summary",
    "",
    f"- status: `{status}`",
    f"- elapsed_s: `{elapsed_s}`",
    f"- n_items: `{n}`",
    f"- hard: `{fmt(hard)}`",
    f"- soft: `{fmt(soft)}`",
    f"- episodes_per_s: `{fmt(episodes_per_s)}`",
    f"- seconds_per_episode: `{fmt(seconds_per_episode)}`",
    f"- results: `{results_path}`",
    f"- eval_summary: `{eval_summary_path}`",
]
summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines), flush=True)
PY

if [[ "${status}" -ne 0 ]]; then
  echo "[failed] ALFWorld speed probe failed. See ${RUN_LOG}" >&2
  exit "${status}"
fi

echo "[done] summary=${SUMMARY_MD}"

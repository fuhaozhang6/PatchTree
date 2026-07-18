#!/usr/bin/env bash
set -euo pipefail

# Launch a small PatchTree ALFWorld smoke run with Doubao Seed models via Ark API.
#
# Default role split:
#   target / student    = Doubao Seed 1.6 Flash
#   optimizer / teacher = Doubao Seed 2.0 Pro
#
# ALFWorld episodes can require many target calls. Defaults are therefore tiny
# and are meant for smoke testing only.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
# ALFWorld's AlfredTWEnv prints a tqdm progress bar while scanning game files on
# every env init (494 / 3553 / 8810 items), which floods the log. Disable tqdm
# bars in this process and the child env workers. Override with TQDM_DISABLE=0.
export TQDM_DISABLE="${TQDM_DISABLE:-1}"

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

quote_cmd() {
  local arg
  for arg in "$@"; do
    printf ' %q' "${arg}"
  done
  printf '\n'
}

command -v "${PYTHON_BIN}" >/dev/null 2>&1 || fail "Python interpreter not found: ${PYTHON_BIN}"

# Ark OpenAI-compatible API. The trainer reads these from env/config.
export AZURE_OPENAI_ENDPOINT="${AZURE_OPENAI_ENDPOINT:-https://ark.cn-beijing.volces.com/api/v3}"
export AZURE_OPENAI_API_KEY="${AZURE_OPENAI_API_KEY:-${ARK_API_KEY:-}}"
export AZURE_OPENAI_AUTH_MODE="${AZURE_OPENAI_AUTH_MODE:-openai_compatible}"
export AZURE_OPENAI_API_VERSION="${AZURE_OPENAI_API_VERSION:-2024-12-01-preview}"

export OPTIMIZER_AZURE_OPENAI_ENDPOINT="${OPTIMIZER_AZURE_OPENAI_ENDPOINT:-${AZURE_OPENAI_ENDPOINT}}"
export OPTIMIZER_AZURE_OPENAI_API_KEY="${OPTIMIZER_AZURE_OPENAI_API_KEY:-${AZURE_OPENAI_API_KEY}}"
export OPTIMIZER_AZURE_OPENAI_AUTH_MODE="${OPTIMIZER_AZURE_OPENAI_AUTH_MODE:-${AZURE_OPENAI_AUTH_MODE}}"
export OPTIMIZER_AZURE_OPENAI_API_VERSION="${OPTIMIZER_AZURE_OPENAI_API_VERSION:-${AZURE_OPENAI_API_VERSION}}"
export TARGET_AZURE_OPENAI_ENDPOINT="${TARGET_AZURE_OPENAI_ENDPOINT:-${AZURE_OPENAI_ENDPOINT}}"
export TARGET_AZURE_OPENAI_API_KEY="${TARGET_AZURE_OPENAI_API_KEY:-${AZURE_OPENAI_API_KEY}}"
export TARGET_AZURE_OPENAI_AUTH_MODE="${TARGET_AZURE_OPENAI_AUTH_MODE:-${AZURE_OPENAI_AUTH_MODE}}"
export TARGET_AZURE_OPENAI_API_VERSION="${TARGET_AZURE_OPENAI_API_VERSION:-${AZURE_OPENAI_API_VERSION}}"

OPTIMIZER_MODEL="${OPTIMIZER_MODEL:-doubao-seed-2-0-pro-260215}"
TARGET_MODEL="${TARGET_MODEL:-doubao-seed-1-6-flash-250828}"
OPTIMIZER_BACKEND="${OPTIMIZER_BACKEND:-openai_chat}"
TARGET_BACKEND="${TARGET_BACKEND:-openai_chat}"
OPTIMIZER_SMOKE_REASONING_EFFORT="${OPTIMIZER_SMOKE_REASONING_EFFORT:-minimal}"
TARGET_QWEN_CHAT_BASE_URL="${TARGET_QWEN_CHAT_BASE_URL:-${QWEN_CHAT_BASE_URL:-}}"
TARGET_QWEN_CHAT_API_KEY="${TARGET_QWEN_CHAT_API_KEY:-${QWEN_CHAT_API_KEY:-dummy}}"
TARGET_QWEN_CHAT_TEMPERATURE="${TARGET_QWEN_CHAT_TEMPERATURE:-${QWEN_CHAT_TEMPERATURE:-0.2}}"
TARGET_QWEN_CHAT_TIMEOUT_SECONDS="${TARGET_QWEN_CHAT_TIMEOUT_SECONDS:-${QWEN_CHAT_TIMEOUT_SECONDS:-120}}"
TARGET_QWEN_CHAT_MAX_TOKENS="${TARGET_QWEN_CHAT_MAX_TOKENS:-${QWEN_CHAT_MAX_TOKENS:-16384}}"
TARGET_QWEN_CHAT_ENABLE_THINKING="${TARGET_QWEN_CHAT_ENABLE_THINKING:-${QWEN_CHAT_ENABLE_THINKING:-false}}"

# Small smoke defaults. Override from the shell for larger runs.
API_MAX_CONCURRENCY="${API_MAX_CONCURRENCY:-4}"
WORKERS="${WORKERS:-2}"
MAX_API_WORKERS="${MAX_API_WORKERS:-2}"
ANALYST_WORKERS="${ANALYST_WORKERS:-4}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-2}"
LIMIT="${LIMIT:-2}"
SEED="${SEED:-42}"
DRY_RUN="${DRY_RUN:-0}"
WAIT_FOR_JOB="${WAIT_FOR_JOB:-1}"

LR_SCHEDULER="${LR_SCHEDULER:-cosine}"
LR_CONTROL_MODE="${LR_CONTROL_MODE:-fixed}"
EDIT_BUDGET="${EDIT_BUDGET:-2}"
MIN_EDIT_BUDGET="${MIN_EDIT_BUDGET:-1}"
USE_GATE="${USE_GATE:-true}"
GATE_METRIC="${GATE_METRIC:-mixed}"
GATE_MIXED_WEIGHT="${GATE_MIXED_WEIGHT:-0.5}"
SEL_ENV_NUM="${SEL_ENV_NUM:-2}"
TEST_ENV_NUM="${TEST_ENV_NUM:-0}"
EVAL_TEST="${EVAL_TEST:-false}"
REASONING_EFFORT="${REASONING_EFFORT:-}"
TARGET_MAX_COMPLETION_TOKENS="${TARGET_MAX_COMPLETION_TOKENS:-2048}"
MAX_STEPS="${MAX_STEPS:-8}"

# PatchTree controls.
TYPE_GUIDED_MIN_SUPPORT="${TYPE_GUIDED_MIN_SUPPORT:-1}"
TYPE_GUIDED_MAX_LEAF_GROUPS="${TYPE_GUIDED_MAX_LEAF_GROUPS:-4}"
TYPE_GUIDED_LEAF_FALLBACK="${TYPE_GUIDED_LEAF_FALLBACK:-false}"
TYPE_GUIDED_TREE_DEPTH="${TYPE_GUIDED_TREE_DEPTH:-2}"
TYPE_GUIDED_ROLLOUT_REPEATS="${TYPE_GUIDED_ROLLOUT_REPEATS:-2}"
TYPE_GUIDED_TAU_SUCC="${TYPE_GUIDED_TAU_SUCC:-1.0}"
TYPE_GUIDED_MAX_PATCH_RECORDS="${TYPE_GUIDED_MAX_PATCH_RECORDS:-4}"
TYPE_GUIDED_FALLBACK_TOP_K="${TYPE_GUIDED_FALLBACK_TOP_K:-0}"
TYPE_GUIDED_FALLBACK_TAU_CHILD="${TYPE_GUIDED_FALLBACK_TAU_CHILD:-0.0}"
TYPE_GUIDED_PATCH_RECORD_WORKERS="${TYPE_GUIDED_PATCH_RECORD_WORKERS:-${ANALYST_WORKERS}}"
TYPE_GUIDED_CLUSTERING="${TYPE_GUIDED_CLUSTERING:-true}"
TYPE_GUIDED_CLUSTER_TARGET_SIZE="${TYPE_GUIDED_CLUSTER_TARGET_SIZE:-2}"
TYPE_GUIDED_CLUSTER_MAX_SIZE="${TYPE_GUIDED_CLUSTER_MAX_SIZE:-4}"

TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
RUN_NAME="${RUN_NAME:-Seed API smoke}"
RUN_SLUG="${RUN_SLUG:-skillopt_alfworld_seed_api_smoke}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs/${RUN_SLUG}_${TS}}"
OUT_ROOT="${OUT_ROOT:-${PROJECT_ROOT}/outputs/${RUN_SLUG}_${TS}}"
TYPE_GUIDED_CACHE_DIR="${TYPE_GUIDED_CACHE_DIR:-${OUT_ROOT}/type_guided_cache}"

ALFWORLD_CONFIG="${ALFWORLD_CONFIG:-${PROJECT_ROOT}/configs/alfworld/default.yaml}"
ALFWORLD_SPLIT_DIR="${ALFWORLD_SPLIT_DIR:-${PROJECT_ROOT}/data/alfworld_path_split}"
ALFWORLD_SKILL_INIT="${ALFWORLD_SKILL_INIT:-${PROJECT_ROOT}/skillopt/envs/alfworld/skills/initial.md}"
DEFAULT_ALFWORLD_DATA="${PROJECT_ROOT}/data/alfworld"
if [[ -n "${ALFWORLD_DATA:-}" && "${ALFWORLD_DATA}" != "${DEFAULT_ALFWORLD_DATA}" ]]; then
  echo "[info] Ignoring inherited ALFWORLD_DATA=${ALFWORLD_DATA}; using ${DEFAULT_ALFWORLD_DATA}" >&2
fi
export ALFWORLD_DATA="${DEFAULT_ALFWORLD_DATA}"

[[ "${GATE_METRIC}" == "hard" || "${GATE_METRIC}" == "soft" || "${GATE_METRIC}" == "mixed" ]] || fail "GATE_METRIC must be hard, soft, or mixed."
[[ "${TYPE_GUIDED_TREE_DEPTH}" =~ ^[0-9]+$ ]] || fail "TYPE_GUIDED_TREE_DEPTH must be an integer."
[[ "${OPTIMIZER_BACKEND}" == "openai_chat" || "${OPTIMIZER_BACKEND}" == "qwen_chat" || "${OPTIMIZER_BACKEND}" == "claude_chat" || "${OPTIMIZER_BACKEND}" == "minimax_chat" ]] || fail "Unsupported OPTIMIZER_BACKEND=${OPTIMIZER_BACKEND}."
[[ "${TARGET_BACKEND}" == "openai_chat" || "${TARGET_BACKEND}" == "qwen_chat" || "${TARGET_BACKEND}" == "claude_chat" || "${TARGET_BACKEND}" == "minimax_chat" || "${TARGET_BACKEND}" == "codex_exec" || "${TARGET_BACKEND}" == "claude_code_exec" ]] || fail "Unsupported TARGET_BACKEND=${TARGET_BACKEND}."
if [[ "${TARGET_BACKEND}" == "qwen_chat" && -z "${TARGET_QWEN_CHAT_BASE_URL}" ]]; then
  fail "TARGET_BACKEND=qwen_chat requires TARGET_QWEN_CHAT_BASE_URL or QWEN_CHAT_BASE_URL."
fi
[[ "${API_MAX_CONCURRENCY}" =~ ^[0-9]+$ ]] || fail "API_MAX_CONCURRENCY must be an integer."
[[ "${WORKERS}" =~ ^[0-9]+$ ]] || fail "WORKERS must be an integer."
[[ "${MAX_API_WORKERS}" =~ ^[0-9]+$ ]] || fail "MAX_API_WORKERS must be an integer."
[[ "${ANALYST_WORKERS}" =~ ^[0-9]+$ ]] || fail "ANALYST_WORKERS must be an integer."
[[ "${WORKERS}" -ge 1 ]] || fail "WORKERS must be >= 1."
[[ "${MAX_API_WORKERS}" -ge 1 ]] || fail "MAX_API_WORKERS must be >= 1."
[[ "${ANALYST_WORKERS}" -ge 1 ]] || fail "ANALYST_WORKERS must be >= 1."
[[ "${MAX_API_WORKERS}" -le "${API_MAX_CONCURRENCY}" ]] || fail "MAX_API_WORKERS=${MAX_API_WORKERS} exceeds API_MAX_CONCURRENCY=${API_MAX_CONCURRENCY}."
[[ "${ANALYST_WORKERS}" -le "${API_MAX_CONCURRENCY}" ]] || fail "ANALYST_WORKERS=${ANALYST_WORKERS} exceeds API_MAX_CONCURRENCY=${API_MAX_CONCURRENCY}."
[[ -f "${ALFWORLD_CONFIG}" ]] || fail "Missing config: ${ALFWORLD_CONFIG}"
[[ -f "${ALFWORLD_SKILL_INIT}" ]] || fail "Missing initial skill: ${ALFWORLD_SKILL_INIT}"

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
    first = items[0]
    gamefile = str(first.get("gamefile") or "").strip()
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
    raise SystemExit(
        "Missing ALFWorld Python dependencies: "
        + ", ".join(missing)
        + ". Install with: pip install -e '.[alfworld]' && pip install 'alfworld[full]'"
    )
PY
}

if ! check_alfworld_split; then
  if truthy "${DRY_RUN}"; then
    echo "[dry-run][warn] missing split_dir: ${ALFWORLD_SPLIT_DIR}/{train,val,test}/items.json" >&2
  else
    fail "Missing split_dir: ${ALFWORLD_SPLIT_DIR}/{train,val,test}/items.json"
  fi
elif ! check_alfworld_payload; then
  if truthy "${DRY_RUN}"; then
    echo "[dry-run][warn] ALFWorld payload check failed; real run will fail unless data paths are fixed." >&2
  else
    fail "ALFWorld split references missing game files. Check ALFWORLD_DATA=${ALFWORLD_DATA}"
  fi
fi

if ! check_alfworld_python_deps; then
  if truthy "${DRY_RUN}"; then
    echo "[dry-run][warn] ALFWorld Python dependency check failed; real run will fail unless dependencies are installed." >&2
  else
    fail "Missing ALFWorld Python dependencies for ${PYTHON_BIN}."
  fi
fi

mkdir -p "${LOG_DIR}" "${OUT_ROOT}"

RUN_LOG="${LOG_DIR}/launcher.log"
TRAIN_LOG="${LOG_DIR}/alfworld.log"
STATUS_FILE="${LOG_DIR}/alfworld.status"
DONE_FILE="${LOG_DIR}/completed.tsv"
: > "${DONE_FILE}"

echo "============================================================"
echo "  SkillOpt-Tree ALFWorld: ${RUN_NAME}"
echo "============================================================"
echo "  project:         ${PROJECT_ROOT}"
echo "  optimizer:       ${OPTIMIZER_MODEL}"
echo "  target:          ${TARGET_MODEL}"
echo "  opt_backend:     ${OPTIMIZER_BACKEND}"
echo "  target_backend:  ${TARGET_BACKEND}"
if [[ "${TARGET_BACKEND}" == "qwen_chat" ]]; then
  echo "  qwen_url:        ${TARGET_QWEN_CHAT_BASE_URL}"
fi
echo "  api_endpoint:    ${AZURE_OPENAI_ENDPOINT}"
echo "  alfworld_data:   ${ALFWORLD_DATA}"
echo "  split_dir:       ${ALFWORLD_SPLIT_DIR}"
echo "  skill_init:      ${ALFWORLD_SKILL_INIT}"
echo "  api_limit:       ${API_MAX_CONCURRENCY}"
echo "  workers:         ${WORKERS}"
echo "  max_api_workers: ${MAX_API_WORKERS}"
echo "  analyst_workers: ${ANALYST_WORKERS}"
echo "  batch_size:      ${BATCH_SIZE}"
echo "  minibatch_size:  ${MINIBATCH_SIZE}"
echo "  merge_batch:     ${MERGE_BATCH_SIZE}"
echo "  max_steps:       ${MAX_STEPS}"
echo "  gate_metric:     ${GATE_METRIC}"
echo "  sel_env_num:     ${SEL_ENV_NUM}"
echo "  limit:           ${LIMIT} (0 means full split)"
echo "  method:          PatchTree"
echo "  tg_tree_depth:   ${TYPE_GUIDED_TREE_DEPTH}"
echo "  dry_run:         ${DRY_RUN}"
echo "  out_root:        ${OUT_ROOT}"
echo "  log_dir:         ${LOG_DIR}"
echo "============================================================"
{
  echo "============================================================"
  echo "  SkillOpt-Tree ALFWorld Seed API smoke launcher started at $(date)"
  echo "  project=${PROJECT_ROOT}"
  echo "  optimizer=${OPTIMIZER_MODEL}"
  echo "  target=${TARGET_MODEL}"
  echo "  alfworld_data=${ALFWORLD_DATA}"
  echo "  split_dir=${ALFWORLD_SPLIT_DIR}"
  echo "  api_limit=${API_MAX_CONCURRENCY}"
  echo "  dry_run=${DRY_RUN}"
  echo "============================================================"
} > "${RUN_LOG}"

smoke_optimizer() {
  "${PYTHON_BIN}" - "${OPTIMIZER_AZURE_OPENAI_ENDPOINT}" "${OPTIMIZER_AZURE_OPENAI_API_KEY}" "${OPTIMIZER_MODEL}" "${OPTIMIZER_SMOKE_REASONING_EFFORT}" <<'PY'
import json
import os
import sys
import urllib.request

endpoint, api_key, model, reasoning_effort = sys.argv[1:]
payload = {
    "model": model,
    "messages": [{"role": "user", "content": "Return exactly OK for optimizer smoke test."}],
    "max_tokens": 256,
    "temperature": 0.0,
}
is_deepseek = "api.deepseek.com" in endpoint.lower() and model.lower().startswith("deepseek-")
# This is a raw HTTP request, not the OpenAI SDK, so there is no `extra_body`
# flattening: the DeepSeek `thinking` switch must sit at the top level of the
# body. The DeepSeek-official API accepts `reasoning_effort` and `thinking`
# together, so both may be sent.
if reasoning_effort:
    payload["reasoning_effort"] = reasoning_effort
thinking = (os.environ.get("DEEPSEEK_THINKING") or os.environ.get("DEEPSEEK_OFFICIAL_THINKING") or "").strip().lower()
if is_deepseek:
    if thinking in {"1", "true", "yes", "on", "enable", "enabled"}:
        payload["thinking"] = {"type": "enabled"}
    elif thinking in {"0", "false", "no", "off", "disable", "disabled"}:
        payload["thinking"] = {"type": "disabled"}

req = urllib.request.Request(
    endpoint.rstrip("/") + "/chat/completions",
    data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
    headers={
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    },
    method="POST",
)
with urllib.request.urlopen(req, timeout=120) as resp:
    data = json.loads(resp.read().decode("utf-8"))
choice = (data.get("choices") or [{}])[0]
message = choice.get("message") or {}
content = message.get("content") or ""
finish_reason = choice.get("finish_reason")
print(f"[smoke/optimizer] model={model} finish_reason={finish_reason} content_preview={content[:80]!r}")
if not str(content).strip():
    if finish_reason == "length":
        raise SystemExit(
            f"[smoke/optimizer] empty content for model={model}: response hit the "
            f"max_tokens limit before emitting any content. This usually means a "
            f"reasoning model consumed the budget while thinking. Lower or unset "
            f"OPTIMIZER_SMOKE_REASONING_EFFORT (currently={reasoning_effort!r}) or "
            f"raise the probe max_tokens."
        )
    raise SystemExit(f"[smoke/optimizer] empty content for model={model}")
PY
}

smoke_target() {
  "${PYTHON_BIN}" - "${TARGET_AZURE_OPENAI_ENDPOINT}" "${TARGET_AZURE_OPENAI_API_KEY}" "${TARGET_MODEL}" <<'PY'
import json
import re
import sys
import urllib.request

endpoint, api_key, model = sys.argv[1:]
payload = {
    "model": model,
    "messages": [
        {
            "role": "system",
            "content": "You are an expert agent operating in the ALFRED Embodied Environment.",
        },
        {
            "role": "user",
            "content": (
                "This is an ALFWorld smoke test. Reply with exactly one action in this format:\n"
                "<think>brief</think><action>look</action>"
            ),
        },
    ],
    "max_tokens": 64,
    "temperature": 0.0,
}
req = urllib.request.Request(
    endpoint.rstrip("/") + "/chat/completions",
    data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
    headers={
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    },
    method="POST",
)
with urllib.request.urlopen(req, timeout=120) as resp:
    data = json.loads(resp.read().decode("utf-8"))
choice = (data.get("choices") or [{}])[0]
message = choice.get("message") or {}
content = message.get("content") or ""
print(f"[smoke/target] model={model} finish_reason={choice.get('finish_reason')} content_preview={content[:120]!r}")
if not re.search(r"<action>.*?</action>", str(content), re.DOTALL):
    raise SystemExit(f"[smoke/target] missing <action>...</action> for model={model}")
PY
}

if truthy "${DRY_RUN}"; then
  echo "[dry-run] Skip model smoke tests."
  echo "[dry-run] Skip model smoke tests." >> "${RUN_LOG}"
else
  if [[ "${OPTIMIZER_BACKEND}" == "openai_chat" ]]; then
    [[ -n "${AZURE_OPENAI_API_KEY}" ]] || fail "AZURE_OPENAI_API_KEY is empty. Export ARK_API_KEY or AZURE_OPENAI_API_KEY first."
    smoke_optimizer 2>&1 | tee -a "${RUN_LOG}"
    status=${PIPESTATUS[0]}
    [[ "${status}" == "0" ]] || fail "Optimizer smoke test failed."
  fi
  if [[ "${TARGET_BACKEND}" == "openai_chat" ]]; then
    smoke_target 2>&1 | tee -a "${RUN_LOG}"
    status=${PIPESTATUS[0]}
    [[ "${status}" == "0" ]] || fail "ALFWorld target smoke test failed."
  elif [[ "${TARGET_BACKEND}" == "qwen_chat" ]]; then
    echo "[smoke/target] qwen_chat smoke should be handled by the Qwen launcher wrapper: ${TARGET_QWEN_CHAT_BASE_URL}" | tee -a "${RUN_LOG}"
  fi
fi

CFG_OPTIONS=(
  "evaluation.gate_metric=${GATE_METRIC}"
  "evaluation.gate_mixed_weight=${GATE_MIXED_WEIGHT}"
  "env.max_completion_tokens=${TARGET_MAX_COMPLETION_TOKENS}"
)

CMD=(
  "${PYTHON_BIN}" -u "${PROJECT_ROOT}/scripts/cli/train.py"
  --config "${ALFWORLD_CONFIG}"
  --optimizer_backend "${OPTIMIZER_BACKEND}"
  --target_backend "${TARGET_BACKEND}"
  --optimizer_model "${OPTIMIZER_MODEL}"
  --target_model "${TARGET_MODEL}"
  --reasoning_effort "${REASONING_EFFORT}"
  --num_epochs "${NUM_EPOCHS}"
  --batch_size "${BATCH_SIZE}"
  --workers "${WORKERS}"
  --max_api_workers "${MAX_API_WORKERS}"
  --analyst_workers "${ANALYST_WORKERS}"
  --seed "${SEED}"
  --edit_budget "${EDIT_BUDGET}"
  --min_edit_budget "${MIN_EDIT_BUDGET}"
  --lr_scheduler "${LR_SCHEDULER}"
  --lr_control_mode "${LR_CONTROL_MODE}"
  --use_gate "${USE_GATE}"
  --sel_env_num "${SEL_ENV_NUM}"
  --test_env_num "${TEST_ENV_NUM}"
  --eval_test "${EVAL_TEST}"
  --type_guided_min_support "${TYPE_GUIDED_MIN_SUPPORT}"
  --type_guided_max_leaf_groups "${TYPE_GUIDED_MAX_LEAF_GROUPS}"
  --type_guided_tree_depth "${TYPE_GUIDED_TREE_DEPTH}"
  --type_guided_leaf_fallback "${TYPE_GUIDED_LEAF_FALLBACK}"
  --type_guided_rollout_repeats "${TYPE_GUIDED_ROLLOUT_REPEATS}"
  --type_guided_tau_succ "${TYPE_GUIDED_TAU_SUCC}"
  --type_guided_max_patch_records "${TYPE_GUIDED_MAX_PATCH_RECORDS}"
  --type_guided_fallback_top_k "${TYPE_GUIDED_FALLBACK_TOP_K}"
  --type_guided_fallback_tau_child "${TYPE_GUIDED_FALLBACK_TAU_CHILD}"
  --type_guided_cache_dir "${TYPE_GUIDED_CACHE_DIR}"
  --type_guided_patch_record_workers "${TYPE_GUIDED_PATCH_RECORD_WORKERS}"
  --type_guided_clustering "${TYPE_GUIDED_CLUSTERING}"
  --type_guided_cluster_target_size "${TYPE_GUIDED_CLUSTER_TARGET_SIZE}"
  --type_guided_cluster_max_size "${TYPE_GUIDED_CLUSTER_MAX_SIZE}"
  --max_steps "${MAX_STEPS}"
  --split_mode split_dir
  --split_dir "${ALFWORLD_SPLIT_DIR}"
  --skill_init "${ALFWORLD_SKILL_INIT}"
  --out_root "${OUT_ROOT}"
)

if [[ "${TARGET_BACKEND}" == "qwen_chat" ]]; then
  CMD+=(
    --target_qwen_chat_base_url "${TARGET_QWEN_CHAT_BASE_URL}"
    --target_qwen_chat_api_key "${TARGET_QWEN_CHAT_API_KEY}"
    --target_qwen_chat_temperature "${TARGET_QWEN_CHAT_TEMPERATURE}"
    --target_qwen_chat_timeout_seconds "${TARGET_QWEN_CHAT_TIMEOUT_SECONDS}"
    --target_qwen_chat_max_tokens "${TARGET_QWEN_CHAT_MAX_TOKENS}"
    --target_qwen_chat_enable_thinking "${TARGET_QWEN_CHAT_ENABLE_THINKING}"
  )
fi

if [[ "${LIMIT}" != "0" ]]; then
  CMD+=(--limit "${LIMIT}")
fi


CMD+=(
  --cfg-options
  "${CFG_OPTIONS[@]}"
)

{
  printf '[cmd]'
  quote_cmd "${CMD[@]}"
} | tee -a "${RUN_LOG}"

if truthy "${DRY_RUN}"; then
  printf '[dry-run]'
  quote_cmd "${CMD[@]}"
  exit 0
fi

rm -f "${STATUS_FILE}"

if truthy "${WAIT_FOR_JOB}"; then
  set +e
  {
    echo "[start] $(date)"
    printf '[cmd]'
    quote_cmd "${CMD[@]}"
    "${CMD[@]}"
    status=$?
    echo "[exit] ${status} $(date)"
    exit "${status}"
  } 2>&1 | tee "${TRAIN_LOG}"
  status=${PIPESTATUS[0]}
  set -e
  echo "${status}" > "${STATUS_FILE}"
  printf 'alfworld\t%s\t%s\n' "${status}" "${TRAIN_LOG}" > "${DONE_FILE}"
  if [[ "${status}" != "0" ]]; then
    echo "[failed] ALFWorld run failed. See ${TRAIN_LOG}" | tee -a "${RUN_LOG}" >&2
    exit "${status}"
  fi
  echo "[done] ALFWorld run completed successfully. Log: ${TRAIN_LOG}" | tee -a "${RUN_LOG}"
else
  {
    echo "[start] $(date)"
    printf '[cmd]'
    quote_cmd "${CMD[@]}"
    set +e
    "${CMD[@]}"
    status=$?
    set -e
    echo "[exit] ${status} $(date)"
    echo "${status}" > "${STATUS_FILE}"
    exit "${status}"
  } > "${TRAIN_LOG}" 2>&1 &
  pid=$!
  printf 'alfworld\t%s\t%s\n' "${pid}" "${TRAIN_LOG}" > "${DONE_FILE}"
  echo "[background] launched ALFWorld job pid=${pid}. Log: ${TRAIN_LOG}" | tee -a "${RUN_LOG}"
fi

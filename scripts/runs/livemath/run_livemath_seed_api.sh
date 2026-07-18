#!/usr/bin/env bash
set -euo pipefail

# Launch SkillOpt-Tree LiveMathematicianBench training with Doubao Seed models.
#
# Default role split:
#   target / student    = Doubao Seed 1.6 Flash
#   optimizer / teacher = Doubao Seed 2.0 Pro
#
# LiveMath is text-only MCQ, so this script intentionally avoids DocVQA image
# handling. By default it uses the raw LiveMath JSON files and lets the loader
# materialize the canonical 2:1:7 split under the run output directory.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

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
OPTIMIZER_SMOKE_REASONING_EFFORT="${OPTIMIZER_SMOKE_REASONING_EFFORT:-minimal}"

API_MAX_CONCURRENCY="${API_MAX_CONCURRENCY:-32}"
WORKERS="${WORKERS:-${API_MAX_CONCURRENCY}}"
ANALYST_WORKERS="${ANALYST_WORKERS:-${API_MAX_CONCURRENCY}}"
NUM_EPOCHS="${NUM_EPOCHS:-2}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LIMIT="${LIMIT:-0}"
SEED="${SEED:-42}"
DRY_RUN="${DRY_RUN:-0}"
WAIT_FOR_JOB="${WAIT_FOR_JOB:-1}"

LR_SCHEDULER="${LR_SCHEDULER:-cosine}"
LR_CONTROL_MODE="${LR_CONTROL_MODE:-fixed}"
EDIT_BUDGET="${EDIT_BUDGET:-4}"
MIN_EDIT_BUDGET="${MIN_EDIT_BUDGET:-2}"
USE_GATE="${USE_GATE:-true}"
GATE_METRIC="${GATE_METRIC:-mixed}"
GATE_MIXED_WEIGHT="${GATE_MIXED_WEIGHT:-0.5}"
SEL_ENV_NUM="${SEL_ENV_NUM:-18}"
TEST_ENV_NUM="${TEST_ENV_NUM:-0}"
EVAL_TEST="${EVAL_TEST:-false}"
REASONING_EFFORT="${REASONING_EFFORT:-}"
TARGET_MAX_COMPLETION_TOKENS="${TARGET_MAX_COMPLETION_TOKENS:-8192}"
MAX_TURNS="${MAX_TURNS:-1}"

# LiveMath data controls.
DEFAULT_RAW="${PROJECT_ROOT}/data/raw/livemath"
if [[ ! -e "${DEFAULT_RAW}" && -e "/Users/bytedance/Documents/codes/Opt/M/data/raw/livemath" ]]; then
  DEFAULT_RAW="/Users/bytedance/Documents/codes/Opt/M/data/raw/livemath"
fi
LIVEMATH_CONFIG="${LIVEMATH_CONFIG:-${PROJECT_ROOT}/configs/livemathematicianbench/default.yaml}"
LIVEMATH_SKILL_INIT="${LIVEMATH_SKILL_INIT:-${PROJECT_ROOT}/skillopt/envs/livemathematicianbench/skills/initial.md}"
LIVEMATH_SPLIT_MODE="${LIVEMATH_SPLIT_MODE:-ratio}"
LIVEMATH_DATA_PATH="${LIVEMATH_DATA_PATH:-${DEFAULT_RAW}}"
LIVEMATH_SPLIT_DIR="${LIVEMATH_SPLIT_DIR:-${PROJECT_ROOT}/data/livemathematicianbench_split}"
LIVEMATH_SPLIT_OUTPUT_DIR="${LIVEMATH_SPLIT_OUTPUT_DIR:-}"
SPLIT_RATIO="${SPLIT_RATIO:-2:1:7}"
SPLIT_SEED="${SPLIT_SEED:-42}"
SHUFFLE_CHOICES="${SHUFFLE_CHOICES:-true}"
USE_THEOREM="${USE_THEOREM:-false}"
USE_SKETCH="${USE_SKETCH:-false}"

TYPE_GUIDED_LEAF_FALLBACK="${TYPE_GUIDED_LEAF_FALLBACK:-true}"
TYPE_GUIDED_TREE_DEPTH="${TYPE_GUIDED_TREE_DEPTH:-2}"

TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs/skillopt_livemath_seed_api_${TS}}"
OUT_ROOT="${OUT_ROOT:-${PROJECT_ROOT}/outputs/skillopt_livemath_seed_api_${TS}}"

[[ "${GATE_METRIC}" == "hard" || "${GATE_METRIC}" == "soft" || "${GATE_METRIC}" == "mixed" ]] || fail "GATE_METRIC must be hard, soft, or mixed."
[[ "${LIVEMATH_SPLIT_MODE}" == "ratio" || "${LIVEMATH_SPLIT_MODE}" == "split_dir" ]] || fail "LIVEMATH_SPLIT_MODE must be ratio or split_dir."
[[ "${API_MAX_CONCURRENCY}" =~ ^[0-9]+$ ]] || fail "API_MAX_CONCURRENCY must be an integer."
[[ "${WORKERS}" =~ ^[0-9]+$ ]] || fail "WORKERS must be an integer."
[[ "${ANALYST_WORKERS}" =~ ^[0-9]+$ ]] || fail "ANALYST_WORKERS must be an integer."
[[ "${WORKERS}" -ge 1 ]] || fail "WORKERS must be >= 1."
[[ "${ANALYST_WORKERS}" -ge 1 ]] || fail "ANALYST_WORKERS must be >= 1."
[[ "${WORKERS}" -le "${API_MAX_CONCURRENCY}" ]] || fail "WORKERS=${WORKERS} exceeds API_MAX_CONCURRENCY=${API_MAX_CONCURRENCY}."
[[ "${ANALYST_WORKERS}" -le "${API_MAX_CONCURRENCY}" ]] || fail "ANALYST_WORKERS=${ANALYST_WORKERS} exceeds API_MAX_CONCURRENCY=${API_MAX_CONCURRENCY}."
[[ -f "${LIVEMATH_CONFIG}" ]] || fail "Missing config: ${LIVEMATH_CONFIG}"
[[ -f "${LIVEMATH_SKILL_INIT}" ]] || fail "Missing initial skill: ${LIVEMATH_SKILL_INIT}"

check_livemath_split() {
  local split
  [[ -d "${LIVEMATH_SPLIT_DIR}" ]] || return 1
  for split in train val test; do
    [[ -d "${LIVEMATH_SPLIT_DIR}/${split}" ]] || return 1
    compgen -G "${LIVEMATH_SPLIT_DIR}/${split}/*.json" >/dev/null || return 1
  done
}

check_livemath_split_payload() {
  "${PYTHON_BIN}" - "${LIVEMATH_SPLIT_DIR}" <<'PY'
import glob
import json
import os
import sys

split_dir = sys.argv[1]
path = sorted(glob.glob(os.path.join(split_dir, "train", "*.json")))[0]
with open(path, encoding="utf-8") as f:
    data = json.load(f)
if not isinstance(data, list) or not data:
    raise SystemExit(f"No train items in {path}")
item = data[0]
missing = [key for key in ("question", "choices", "correct_choice") if key not in item]
if missing:
    raise SystemExit(
        "LiveMath split appears to be an ID manifest, not runnable payload. "
        f"Missing fields {missing} in {path}. Use LIVEMATH_SPLIT_MODE=ratio "
        "with LIVEMATH_DATA_PATH pointing to raw qa_*_final.json files, or materialize full split items."
    )
PY
}

check_livemath_raw() {
  "${PYTHON_BIN}" - "${LIVEMATH_DATA_PATH}" <<'PY'
import glob
import os
import sys

data_path = sys.argv[1]
if os.path.isfile(data_path):
    files = [data_path]
elif os.path.isdir(data_path):
    files = glob.glob(os.path.join(data_path, "**", "qa_*_final.json"), recursive=True)
else:
    files = []
if not files:
    raise SystemExit(f"No qa_*_final.json files found under {data_path}")
PY
}

if [[ "${LIVEMATH_SPLIT_MODE}" == "split_dir" ]]; then
  if ! check_livemath_split; then
    if truthy "${DRY_RUN}"; then
      echo "[dry-run][warn] missing runnable split_dir: ${LIVEMATH_SPLIT_DIR}/{train,val,test}/*.json" >&2
    else
      fail "Missing runnable split_dir: ${LIVEMATH_SPLIT_DIR}/{train,val,test}/*.json"
    fi
  elif ! check_livemath_split_payload; then
    if truthy "${DRY_RUN}"; then
      echo "[dry-run][warn] split_dir payload check failed; real run will fail unless split is materialized." >&2
    else
      fail "LiveMath split_dir is not materialized with question/choices/correct_choice."
    fi
  fi
else
  if ! check_livemath_raw; then
    if truthy "${DRY_RUN}"; then
      echo "[dry-run][warn] missing raw LiveMath data: ${LIVEMATH_DATA_PATH}" >&2
    else
      fail "Missing raw LiveMath data. Set LIVEMATH_DATA_PATH to a qa_*_final.json file or directory."
    fi
  fi
fi

mkdir -p "${LOG_DIR}" "${OUT_ROOT}"

RUN_LOG="${LOG_DIR}/launcher.log"
TRAIN_LOG="${LOG_DIR}/livemath.log"
STATUS_FILE="${LOG_DIR}/livemath.status"
DONE_FILE="${LOG_DIR}/completed.tsv"
: > "${DONE_FILE}"

echo "============================================================"
echo "  SkillOpt-Tree LiveMath: Seed API"
echo "============================================================"
echo "  project:         ${PROJECT_ROOT}"
echo "  optimizer:       ${OPTIMIZER_MODEL}"
echo "  target:          ${TARGET_MODEL}"
echo "  api_endpoint:    ${AZURE_OPENAI_ENDPOINT}"
echo "  split_mode:      ${LIVEMATH_SPLIT_MODE}"
echo "  data_path:       ${LIVEMATH_DATA_PATH}"
echo "  split_dir:       ${LIVEMATH_SPLIT_DIR}"
echo "  skill_init:      ${LIVEMATH_SKILL_INIT}"
echo "  api_limit:       ${API_MAX_CONCURRENCY}"
echo "  workers:         ${WORKERS}"
echo "  analyst_workers: ${ANALYST_WORKERS}"
echo "  batch_size:      ${BATCH_SIZE}"
echo "  minibatch_size:  ${MINIBATCH_SIZE}"
echo "  merge_batch:     ${MERGE_BATCH_SIZE}"
echo "  gate_metric:     ${GATE_METRIC}"
echo "  sel_env_num:     ${SEL_ENV_NUM}"
echo "  limit:           ${LIMIT} (0 means full split)"
echo "  method:          PatchTree"
echo "  use_theorem:     ${USE_THEOREM}"
echo "  use_sketch:      ${USE_SKETCH}"
echo "  dry_run:         ${DRY_RUN}"
echo "  out_root:        ${OUT_ROOT}"
echo "  log_dir:         ${LOG_DIR}"
echo "============================================================"
{
  echo "============================================================"
  echo "  SkillOpt-Tree LiveMath launcher started at $(date)"
  echo "  project=${PROJECT_ROOT}"
  echo "  optimizer=${OPTIMIZER_MODEL}"
  echo "  target=${TARGET_MODEL}"
  echo "  split_mode=${LIVEMATH_SPLIT_MODE}"
  echo "  data_path=${LIVEMATH_DATA_PATH}"
  echo "  split_dir=${LIVEMATH_SPLIT_DIR}"
  echo "  api_limit=${API_MAX_CONCURRENCY}"
  echo "  dry_run=${DRY_RUN}"
  echo "============================================================"
} > "${RUN_LOG}"

resolve_livemath_smoke_sample() {
  "${PYTHON_BIN}" - "${LIVEMATH_SPLIT_MODE}" "${LIVEMATH_SPLIT_DIR}" "${LIVEMATH_DATA_PATH}" <<'PY'
import glob
import json
import os
import sys

split_mode, split_dir, data_path = sys.argv[1:]

def normalize_label(text):
    return str(text or "").strip().upper().rstrip(".):")

def coerce_choices(raw):
    labels = ["A", "B", "C", "D", "E", "F", "G"]
    if isinstance(raw, list):
        out = []
        for idx, item in enumerate(raw):
            if isinstance(item, dict):
                label = str(item.get("label") or labels[idx]).strip()
                text = str(item.get("text") or item.get("content") or "").strip()
            else:
                label = labels[idx]
                text = str(item).strip()
            if text:
                out.append({"label": label, "text": text})
        return out
    if isinstance(raw, dict):
        return [
            {"label": str(label).strip(), "text": str(raw[label]).strip()}
            for label in sorted(raw)
            if str(raw[label]).strip()
        ]
    return []

def normalize_item(item, source_path):
    mcq = item.get("mcq", {}) if isinstance(item.get("mcq"), dict) else {}
    question = str(mcq.get("question") or item.get("question") or "").strip()
    choices = coerce_choices(mcq.get("choices") or item.get("choices") or [])
    correct = mcq.get("correct_choice") or item.get("correct_choice") or {}
    if isinstance(correct, dict):
        correct_label = normalize_label(correct.get("label", ""))
        correct_text = str(correct.get("text") or "").strip()
    else:
        correct_label = normalize_label(correct)
        correct_text = ""
    if correct_label and not correct_text:
        for choice in choices:
            if normalize_label(choice["label"]) == correct_label:
                correct_text = choice["text"]
                break
    item_id = str(item.get("id") or f"{item.get('month', '')}:{item.get('no', '')}").strip(":")
    return {
        "id": item_id or os.path.basename(source_path),
        "question": question,
        "choices": choices,
        "correct_choice": {"label": correct_label, "text": correct_text},
    }

if split_mode == "split_dir":
    files = sorted(glob.glob(os.path.join(split_dir, "train", "*.json")))
else:
    if os.path.isfile(data_path):
        files = [data_path]
    else:
        files = sorted(glob.glob(os.path.join(data_path, "**", "qa_*_final.json"), recursive=True))
if not files:
    raise SystemExit("No LiveMath JSON files found for smoke sample")

for path in files:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        continue
    for raw in data:
        item = normalize_item(raw, path)
        if item["question"] and item["choices"] and item["correct_choice"]["label"]:
            print(json.dumps(item, ensure_ascii=False))
            raise SystemExit(0)
raise SystemExit("No usable LiveMath smoke sample found")
PY
}

smoke_optimizer() {
  "${PYTHON_BIN}" - "${OPTIMIZER_AZURE_OPENAI_ENDPOINT}" "${OPTIMIZER_AZURE_OPENAI_API_KEY}" "${OPTIMIZER_MODEL}" "${OPTIMIZER_SMOKE_REASONING_EFFORT}" <<'PY'
import json
import sys
import urllib.request

endpoint, api_key, model, reasoning_effort = sys.argv[1:]
payload = {
    "model": model,
    "messages": [{"role": "user", "content": "Return exactly OK for optimizer smoke test."}],
    "max_tokens": 32,
    "temperature": 0.0,
}
if reasoning_effort:
    payload["reasoning_effort"] = reasoning_effort

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
print(f"[smoke/optimizer] model={model} finish_reason={choice.get('finish_reason')} content_preview={content[:80]!r}")
if not str(content).strip():
    raise SystemExit(f"[smoke/optimizer] empty content for model={model}")
PY
}

smoke_livemath_target() {
  local sample_json
  sample_json="$(resolve_livemath_smoke_sample)"
  "${PYTHON_BIN}" - "${TARGET_AZURE_OPENAI_ENDPOINT}" "${TARGET_AZURE_OPENAI_API_KEY}" "${TARGET_MODEL}" "${sample_json}" <<'PY'
import json
import sys
import urllib.request

endpoint, api_key, model, sample_json = sys.argv[1:]
sample = json.loads(sample_json)
choices = "\n".join(f"{c['label']}. {c['text']}" for c in sample["choices"])
prompt = (
    "This is a LiveMathematicianBench smoke test. Solve the multiple-choice "
    "question and reply with only one final label inside <answer>...</answer>.\n\n"
    f"Question:\n{sample['question']}\n\nChoices:\n{choices}"
)
payload = {
    "model": model,
    "messages": [{"role": "user", "content": prompt}],
    "max_tokens": 256,
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
with urllib.request.urlopen(req, timeout=180) as resp:
    data = json.loads(resp.read().decode("utf-8"))
choice = (data.get("choices") or [{}])[0]
message = choice.get("message") or {}
content = message.get("content") or ""
print(
    f"[smoke/target] model={model} finish_reason={choice.get('finish_reason')} "
    f"id={sample.get('id')} content_preview={content[:120]!r}"
)
if not str(content).strip():
    raise SystemExit(f"[smoke/target] empty content for model={model}")
PY
}

if truthy "${DRY_RUN}"; then
  echo "[dry-run] Skip Ark smoke tests."
  echo "[dry-run] Skip Ark smoke tests." >> "${RUN_LOG}"
else
  [[ -n "${AZURE_OPENAI_API_KEY}" ]] || fail "AZURE_OPENAI_API_KEY is empty. Export ARK_API_KEY or AZURE_OPENAI_API_KEY first."
  smoke_optimizer 2>&1 | tee -a "${RUN_LOG}"
  status=${PIPESTATUS[0]}
  [[ "${status}" == "0" ]] || fail "Optimizer smoke test failed."
  smoke_livemath_target 2>&1 | tee -a "${RUN_LOG}"
  status=${PIPESTATUS[0]}
  [[ "${status}" == "0" ]] || fail "LiveMath target smoke test failed."
fi

CFG_OPTIONS=(
  "evaluation.gate_metric=${GATE_METRIC}"
  "evaluation.gate_mixed_weight=${GATE_MIXED_WEIGHT}"
  "env.max_completion_tokens=${TARGET_MAX_COMPLETION_TOKENS}"
)

CMD=(
  "${PYTHON_BIN}" -u "${PROJECT_ROOT}/scripts/cli/train.py"
  --config "${LIVEMATH_CONFIG}"
  --optimizer_backend openai_chat
  --target_backend openai_chat
  --optimizer_model "${OPTIMIZER_MODEL}"
  --target_model "${TARGET_MODEL}"
  --reasoning_effort "${REASONING_EFFORT}"
  --num_epochs "${NUM_EPOCHS}"
  --batch_size "${BATCH_SIZE}"
  --workers "${WORKERS}"
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
  --type_guided_tree_depth "${TYPE_GUIDED_TREE_DEPTH}"
  --type_guided_leaf_fallback "${TYPE_GUIDED_LEAF_FALLBACK}"
  --split_mode "${LIVEMATH_SPLIT_MODE}"
  --split_ratio "${SPLIT_RATIO}"
  --split_seed "${SPLIT_SEED}"
  --shuffle_choices "${SHUFFLE_CHOICES}"
  --use_theorem "${USE_THEOREM}"
  --use_sketch "${USE_SKETCH}"
  --max_turns "${MAX_TURNS}"
  --skill_init "${LIVEMATH_SKILL_INIT}"
  --out_root "${OUT_ROOT}"
)

if [[ "${LIVEMATH_SPLIT_MODE}" == "split_dir" ]]; then
  CMD+=(--split_dir "${LIVEMATH_SPLIT_DIR}")
else
  CMD+=(--data_path "${LIVEMATH_DATA_PATH}")
  if [[ -n "${LIVEMATH_SPLIT_OUTPUT_DIR}" ]]; then
    CMD+=(--split_output_dir "${LIVEMATH_SPLIT_OUTPUT_DIR}")
  fi
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
  printf 'livemath\t%s\t%s\n' "${status}" "${TRAIN_LOG}" > "${DONE_FILE}"
  if [[ "${status}" != "0" ]]; then
    echo "[failed] LiveMath run failed. See ${TRAIN_LOG}" | tee -a "${RUN_LOG}" >&2
    exit "${status}"
  fi
  echo "[done] LiveMath run completed successfully. Log: ${TRAIN_LOG}" | tee -a "${RUN_LOG}"
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
  printf 'livemath\t%s\t%s\n' "${pid}" "${TRAIN_LOG}" > "${DONE_FILE}"
  echo "[background] launched LiveMath job pid=${pid}. Log: ${TRAIN_LOG}" | tee -a "${RUN_LOG}"
fi

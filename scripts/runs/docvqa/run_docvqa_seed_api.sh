#!/usr/bin/env bash
set -euo pipefail

# Launch SkillOpt DocVQA training with Doubao Seed models via Ark API.
#
# Default role split:
#   target / student    = Doubao Seed 1.6 Flash
#   optimizer / teacher = Doubao Seed 2.0 Pro
#
# This script is specific to SkillOpt-Tree. It intentionally avoids DR2-only
# negative-objective flags and uses this project's train.py/config surface.

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

# Per-role overrides default to the shared Ark endpoint/key.
export OPTIMIZER_AZURE_OPENAI_ENDPOINT="${OPTIMIZER_AZURE_OPENAI_ENDPOINT:-${AZURE_OPENAI_ENDPOINT}}"
export OPTIMIZER_AZURE_OPENAI_API_KEY="${OPTIMIZER_AZURE_OPENAI_API_KEY:-${AZURE_OPENAI_API_KEY}}"
export OPTIMIZER_AZURE_OPENAI_AUTH_MODE="${OPTIMIZER_AZURE_OPENAI_AUTH_MODE:-${AZURE_OPENAI_AUTH_MODE}}"
export OPTIMIZER_AZURE_OPENAI_API_VERSION="${OPTIMIZER_AZURE_OPENAI_API_VERSION:-${AZURE_OPENAI_API_VERSION}}"
export TARGET_AZURE_OPENAI_ENDPOINT="${TARGET_AZURE_OPENAI_ENDPOINT:-${AZURE_OPENAI_ENDPOINT}}"
export TARGET_AZURE_OPENAI_API_KEY="${TARGET_AZURE_OPENAI_API_KEY:-${AZURE_OPENAI_API_KEY}}"
export TARGET_AZURE_OPENAI_AUTH_MODE="${TARGET_AZURE_OPENAI_AUTH_MODE:-${AZURE_OPENAI_AUTH_MODE}}"
export TARGET_AZURE_OPENAI_API_VERSION="${TARGET_AZURE_OPENAI_API_VERSION:-${AZURE_OPENAI_API_VERSION}}"

# Model split: stronger optimizer, cheaper multimodal target.
OPTIMIZER_MODEL="${OPTIMIZER_MODEL:-doubao-seed-2-0-pro-260215}"
TARGET_MODEL="${TARGET_MODEL:-doubao-seed-1-6-flash-250828}"
OPTIMIZER_SMOKE_REASONING_EFFORT="${OPTIMIZER_SMOKE_REASONING_EFFORT:-minimal}"

# Training defaults. Override any value from the shell, e.g.
#   LIMIT=16 NUM_EPOCHS=1 SEL_ENV_NUM=8 bash scripts/runs/docvqa/run_docvqa_seed_api.sh
API_MAX_CONCURRENCY="${API_MAX_CONCURRENCY:-32}"
WORKERS="${WORKERS:-${API_MAX_CONCURRENCY}}"
ANALYST_WORKERS="${ANALYST_WORKERS:-${API_MAX_CONCURRENCY}}"
NUM_EPOCHS="${NUM_EPOCHS:-2}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LIMIT="${LIMIT:-0}"
SEED="${SEED:-42}"
DRY_RUN="${DRY_RUN:-0}"
WAIT_FOR_JOB="${WAIT_FOR_JOB:-1}"

LR_SCHEDULER="${LR_SCHEDULER:-constant}"
LR_CONTROL_MODE="${LR_CONTROL_MODE:-fixed}"
EDIT_BUDGET="${EDIT_BUDGET:-999}"
MIN_EDIT_BUDGET="${MIN_EDIT_BUDGET:-999}"
USE_GATE="${USE_GATE:-true}"
GATE_METRIC="${GATE_METRIC:-mixed}"
GATE_MIXED_WEIGHT="${GATE_MIXED_WEIGHT:-0.5}"
SEL_ENV_NUM="${SEL_ENV_NUM:-32}"
TEST_ENV_NUM="${TEST_ENV_NUM:-0}"
EVAL_TEST="${EVAL_TEST:-false}"
REASONING_EFFORT="${REASONING_EFFORT:-}"
IMAGE_DETAIL="${IMAGE_DETAIL:-auto}"

# PatchTree controls.
TYPE_GUIDED_MIN_SUPPORT="${TYPE_GUIDED_MIN_SUPPORT:-2}"
TYPE_GUIDED_MAX_LEAF_GROUPS="${TYPE_GUIDED_MAX_LEAF_GROUPS:-8}"
TYPE_GUIDED_LEAF_FALLBACK="${TYPE_GUIDED_LEAF_FALLBACK:-true}"
TYPE_GUIDED_TREE_DEPTH="${TYPE_GUIDED_TREE_DEPTH:-2}"
TYPE_GUIDED_ROLLOUT_REPEATS="${TYPE_GUIDED_ROLLOUT_REPEATS:-3}"
TYPE_GUIDED_TAU_SUCC="${TYPE_GUIDED_TAU_SUCC:-1.0}"
TYPE_GUIDED_MAX_PATCH_RECORDS="${TYPE_GUIDED_MAX_PATCH_RECORDS:-32}"
TYPE_GUIDED_FALLBACK_EVAL_ALL_LEAVES="${TYPE_GUIDED_FALLBACK_EVAL_ALL_LEAVES:-true}"
TYPE_GUIDED_FALLBACK_TOP_K="${TYPE_GUIDED_FALLBACK_TOP_K:-0}"
TYPE_GUIDED_FALLBACK_TAU_CHILD="${TYPE_GUIDED_FALLBACK_TAU_CHILD:-0.0}"
TYPE_GUIDED_PATCH_RECORD_WORKERS="${TYPE_GUIDED_PATCH_RECORD_WORKERS:-${ANALYST_WORKERS}}"
TYPE_GUIDED_CLUSTERING="${TYPE_GUIDED_CLUSTERING:-false}"
TYPE_GUIDED_CLUSTER_TARGET_SIZE="${TYPE_GUIDED_CLUSTER_TARGET_SIZE:-4}"
TYPE_GUIDED_CLUSTER_MAX_SIZE="${TYPE_GUIDED_CLUSTER_MAX_SIZE:-8}"
TYPE_GUIDED_TAIL_BANK="${TYPE_GUIDED_TAIL_BANK:-true}"
TYPE_GUIDED_TAIL_MIN_SUPPORT="${TYPE_GUIDED_TAIL_MIN_SUPPORT:-2}"
TYPE_GUIDED_TAIL_MAX_RECORDS="${TYPE_GUIDED_TAIL_MAX_RECORDS:-32}"
TYPE_GUIDED_TAIL_MAX_LEAF_GROUPS="${TYPE_GUIDED_TAIL_MAX_LEAF_GROUPS:-4}"
TYPE_GUIDED_TAIL_WINDOW_EPOCHS="${TYPE_GUIDED_TAIL_WINDOW_EPOCHS:-1}"
TYPE_GUIDED_TAIL_REQUIRE_CROSS_STEP="${TYPE_GUIDED_TAIL_REQUIRE_CROSS_STEP:-true}"

TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs/skillopt_docvqa_seed_api_${TS}}"
OUT_ROOT="${OUT_ROOT:-${PROJECT_ROOT}/outputs/skillopt_docvqa_seed_api_${TS}}"
TYPE_GUIDED_CACHE_DIR="${TYPE_GUIDED_CACHE_DIR:-${OUT_ROOT}/type_guided_cache}"

DOCVQA_CONFIG="${DOCVQA_CONFIG:-${PROJECT_ROOT}/configs/docvqa/default.yaml}"
DOCVQA_SPLIT_DIR="${DOCVQA_SPLIT_DIR:-${PROJECT_ROOT}/data/docvqa/splits}"
DOCVQA_SKILL_INIT="${DOCVQA_SKILL_INIT:-${PROJECT_ROOT}/skillopt/envs/docvqa/skills/initial.md}"

[[ "${GATE_METRIC}" == "hard" || "${GATE_METRIC}" == "soft" || "${GATE_METRIC}" == "mixed" ]] || fail "GATE_METRIC must be hard, soft, or mixed."
[[ "${API_MAX_CONCURRENCY}" =~ ^[0-9]+$ ]] || fail "API_MAX_CONCURRENCY must be an integer."
[[ "${WORKERS}" =~ ^[0-9]+$ ]] || fail "WORKERS must be an integer."
[[ "${ANALYST_WORKERS}" =~ ^[0-9]+$ ]] || fail "ANALYST_WORKERS must be an integer."
[[ "${WORKERS}" -ge 1 ]] || fail "WORKERS must be >= 1."
[[ "${ANALYST_WORKERS}" -ge 1 ]] || fail "ANALYST_WORKERS must be >= 1."
[[ "${WORKERS}" -le "${API_MAX_CONCURRENCY}" ]] || fail "WORKERS=${WORKERS} exceeds API_MAX_CONCURRENCY=${API_MAX_CONCURRENCY}."
[[ "${ANALYST_WORKERS}" -le "${API_MAX_CONCURRENCY}" ]] || fail "ANALYST_WORKERS=${ANALYST_WORKERS} exceeds API_MAX_CONCURRENCY=${API_MAX_CONCURRENCY}."
[[ "${TYPE_GUIDED_TREE_DEPTH}" =~ ^[0-9]+$ ]] || fail "TYPE_GUIDED_TREE_DEPTH must be an integer."
[[ -f "${DOCVQA_CONFIG}" ]] || fail "Missing config: ${DOCVQA_CONFIG}"
[[ -f "${DOCVQA_SKILL_INIT}" ]] || fail "Missing initial skill: ${DOCVQA_SKILL_INIT}"

check_docvqa_split() {
  local split
  [[ -d "${DOCVQA_SPLIT_DIR}" ]] || return 1
  for split in train val test; do
    [[ -d "${DOCVQA_SPLIT_DIR}/${split}" ]] || return 1
    compgen -G "${DOCVQA_SPLIT_DIR}/${split}/*.csv" >/dev/null || return 1
  done
}

if ! check_docvqa_split; then
  if truthy "${DRY_RUN}"; then
    cat >&2 <<EOF
[dry-run][warn] DocVQA runnable CSV split is missing or incomplete:
  ${DOCVQA_SPLIT_DIR}/{train,val,test}/*.csv

Real runs require materialized DocVQA CSV files with fields such as:
  question, answer/ground_truth, image_path
EOF
  else
    cat >&2 <<EOF
ERROR: DocVQA runnable CSV split is missing or incomplete:
  ${DOCVQA_SPLIT_DIR}/{train,val,test}/*.csv

SkillOpt-Tree's DocVQA loader reads CSV files with fields such as:
  question, answer/ground_truth, image_path

The released data/docvqa_id_split directory is only an ID manifest and is not
directly runnable. Materialize DocVQA into ${DOCVQA_SPLIT_DIR}, or set
DOCVQA_SPLIT_DIR=/path/to/materialized/docvqa/splits.
EOF
    exit 1
  fi
fi

mkdir -p "${LOG_DIR}" "${OUT_ROOT}" "${TYPE_GUIDED_CACHE_DIR}"

RUN_LOG="${LOG_DIR}/launcher.log"
TRAIN_LOG="${LOG_DIR}/docvqa.log"
STATUS_FILE="${LOG_DIR}/docvqa.status"
DONE_FILE="${LOG_DIR}/completed.tsv"
: > "${DONE_FILE}"

echo "============================================================"
echo "  SkillOpt-Tree DocVQA: Seed API"
echo "============================================================"
echo "  project:         ${PROJECT_ROOT}"
echo "  optimizer:       ${OPTIMIZER_MODEL}"
echo "  target:          ${TARGET_MODEL}"
echo "  api_endpoint:    ${AZURE_OPENAI_ENDPOINT}"
echo "  split_dir:       ${DOCVQA_SPLIT_DIR}"
echo "  skill_init:      ${DOCVQA_SKILL_INIT}"
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
  echo "  tree_depth:      ${TYPE_GUIDED_TREE_DEPTH}"
echo "  dry_run:         ${DRY_RUN}"
echo "  out_root:        ${OUT_ROOT}"
echo "  log_dir:         ${LOG_DIR}"
echo "============================================================"
{
  echo "============================================================"
  echo "  SkillOpt-Tree DocVQA launcher started at $(date)"
  echo "  project=${PROJECT_ROOT}"
  echo "  optimizer=${OPTIMIZER_MODEL}"
  echo "  target=${TARGET_MODEL}"
  echo "  split_dir=${DOCVQA_SPLIT_DIR}"
  echo "  api_limit=${API_MAX_CONCURRENCY}"
  echo "  dry_run=${DRY_RUN}"
  echo "============================================================"
} > "${RUN_LOG}"

resolve_docvqa_smoke_sample() {
  "${PYTHON_BIN}" - "${DOCVQA_SPLIT_DIR}" "${PROJECT_ROOT}" <<'PY'
import csv
import json
import os
import sys

split_dir, project_root = sys.argv[1:]
train_dir = os.path.join(split_dir, "train")
csv_files = sorted(
    os.path.join(train_dir, name)
    for name in os.listdir(train_dir)
    if name.endswith(".csv")
)
if not csv_files:
    raise SystemExit(f"No CSV file found in {train_dir}")
with open(csv_files[0], encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        question = str(row.get("question") or "").strip()
        image_path = str(row.get("image_path") or "").strip()
        if "document_path:" in question and not image_path:
            question, image_path = question.split("document_path:", 1)
            question = question.strip()
            image_path = image_path.strip()
        if not question or not image_path:
            continue
        image_abs = image_path if os.path.isabs(image_path) else os.path.join(project_root, image_path)
        if not os.path.exists(image_abs):
            continue
        print(json.dumps({"question": question, "image_path": image_abs}, ensure_ascii=False))
        raise SystemExit(0)
raise SystemExit(f"No usable DocVQA smoke sample found in {csv_files[0]}")
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

smoke_docvqa_target() {
  local sample_json
  sample_json="$(resolve_docvqa_smoke_sample)"
  "${PYTHON_BIN}" - "${TARGET_AZURE_OPENAI_ENDPOINT}" "${TARGET_AZURE_OPENAI_API_KEY}" "${TARGET_MODEL}" "${sample_json}" "${IMAGE_DETAIL}" <<'PY'
import base64
import json
import mimetypes
import os
import sys
import urllib.request

endpoint, api_key, model, sample_json, image_detail = sys.argv[1:]
sample = json.loads(sample_json)
image_path = sample["image_path"]
question = sample["question"]

mime = mimetypes.guess_type(image_path)[0] or "image/png"
with open(image_path, "rb") as f:
    encoded = base64.b64encode(f.read()).decode("ascii")

image_url = {"url": f"data:{mime};base64,{encoded}"}
if image_detail and image_detail != "auto":
    image_url["detail"] = image_detail

payload = {
    "model": model,
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "This is a DocVQA smoke test. Inspect the document image and reply "
                        "with one short answer.\n\nQuestion: "
                        f"{question}"
                    ),
                },
                {"type": "image_url", "image_url": image_url},
            ],
        }
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
with urllib.request.urlopen(req, timeout=180) as resp:
    data = json.loads(resp.read().decode("utf-8"))
choice = (data.get("choices") or [{}])[0]
message = choice.get("message") or {}
content = message.get("content") or ""
print(
    f"[smoke/target] model={model} finish_reason={choice.get('finish_reason')} "
    f"image={os.path.basename(image_path)} content_preview={content[:120]!r}"
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
  smoke_docvqa_target 2>&1 | tee -a "${RUN_LOG}"
  status=${PIPESTATUS[0]}
  [[ "${status}" == "0" ]] || fail "DocVQA target smoke test failed."
fi

CFG_OPTIONS=(
  "evaluation.gate_metric=${GATE_METRIC}"
  "evaluation.gate_mixed_weight=${GATE_MIXED_WEIGHT}"
)

CMD=(
  "${PYTHON_BIN}" -u "${PROJECT_ROOT}/scripts/cli/train.py"
  --config "${DOCVQA_CONFIG}"
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
  --type_guided_min_support "${TYPE_GUIDED_MIN_SUPPORT}"
  --type_guided_max_leaf_groups "${TYPE_GUIDED_MAX_LEAF_GROUPS}"
  --type_guided_tree_depth "${TYPE_GUIDED_TREE_DEPTH}"
  --type_guided_leaf_fallback "${TYPE_GUIDED_LEAF_FALLBACK}"
  --type_guided_rollout_repeats "${TYPE_GUIDED_ROLLOUT_REPEATS}"
  --type_guided_tau_succ "${TYPE_GUIDED_TAU_SUCC}"
  --type_guided_max_patch_records "${TYPE_GUIDED_MAX_PATCH_RECORDS}"
  --type_guided_fallback_eval_all_leaves "${TYPE_GUIDED_FALLBACK_EVAL_ALL_LEAVES}"
  --type_guided_fallback_top_k "${TYPE_GUIDED_FALLBACK_TOP_K}"
  --type_guided_fallback_tau_child "${TYPE_GUIDED_FALLBACK_TAU_CHILD}"
  --type_guided_cache_dir "${TYPE_GUIDED_CACHE_DIR}"
  --type_guided_patch_record_workers "${TYPE_GUIDED_PATCH_RECORD_WORKERS}"
  --type_guided_clustering "${TYPE_GUIDED_CLUSTERING}"
  --type_guided_cluster_target_size "${TYPE_GUIDED_CLUSTER_TARGET_SIZE}"
  --type_guided_cluster_max_size "${TYPE_GUIDED_CLUSTER_MAX_SIZE}"
  --type_guided_tail_bank "${TYPE_GUIDED_TAIL_BANK}"
  --type_guided_tail_min_support "${TYPE_GUIDED_TAIL_MIN_SUPPORT}"
  --type_guided_tail_max_records "${TYPE_GUIDED_TAIL_MAX_RECORDS}"
  --type_guided_tail_max_leaf_groups "${TYPE_GUIDED_TAIL_MAX_LEAF_GROUPS}"
  --type_guided_tail_window_epochs "${TYPE_GUIDED_TAIL_WINDOW_EPOCHS}"
  --type_guided_tail_require_cross_step "${TYPE_GUIDED_TAIL_REQUIRE_CROSS_STEP}"
  --image_detail "${IMAGE_DETAIL}"
  --split_dir "${DOCVQA_SPLIT_DIR}"
  --skill_init "${DOCVQA_SKILL_INIT}"
  --out_root "${OUT_ROOT}"
)

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
  printf 'docvqa\t%s\t%s\n' "${status}" "${TRAIN_LOG}" > "${DONE_FILE}"
  if [[ "${status}" != "0" ]]; then
    echo "[failed] DocVQA run failed. See ${TRAIN_LOG}" | tee -a "${RUN_LOG}" >&2
    exit "${status}"
  fi
  echo "[done] DocVQA run completed successfully. Log: ${TRAIN_LOG}" | tee -a "${RUN_LOG}"
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
  printf 'docvqa\t%s\t%s\n' "${pid}" "${TRAIN_LOG}" > "${DONE_FILE}"
  echo "[background] launched DocVQA job pid=${pid}. Log: ${TRAIN_LOG}" | tee -a "${RUN_LOG}"
fi

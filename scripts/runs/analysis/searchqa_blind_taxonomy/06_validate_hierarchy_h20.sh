#!/usr/bin/env bash
set -euo pipefail

# Compare parent, child, and parent+child patches on the same SearchQA samples.
# This stage is target-model-only: it does not call the optimizer/analyst LLM.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "${HERE}/_common.sh"

SHARD_COUNT="${SHARD_COUNT:-2}"
TAXONOMY_PATH="${TAXONOMY_PATH:-${OUT_BASE}/taxonomy/blind_revision_taxonomy.json}"
HIERARCHY_DIR="${HIERARCHY_DIR:-${OUT_BASE}/hierarchy_validation}"
HIERARCHY_PAIRS="${HIERARCHY_PAIRS:-R_SEARCH_001:R_SEARCH_002:R_SEARCH_004 R_SEARCH_001:R_SEARCH_004:R_SEARCH_002}"

VLLM_PORT="${VLLM_PORT:-59317}"
QWEN_CHAT_BASE_URL="${QWEN_CHAT_BASE_URL:-http://127.0.0.1:${VLLM_PORT}/v1}"
MODEL_PATH="${MODEL_PATH:-/ai-app-vepfs/modelscope/models/Qwen/Qwen3.5-4B}"
TARGET_MODEL="${TARGET_MODEL:-Qwen/Qwen3.5-4B}"
TARGET_API_KEY="${TARGET_API_KEY:-dummy}"
CUDA_DEVICE="${CUDA_VISIBLE_DEVICES:-0}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-128}"
VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-65536}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
START_VLLM="${START_VLLM:-1}"
STOP_VLLM_ON_EXIT="${STOP_VLLM_ON_EXIT:-1}"
VLLM_LOG="${VLLM_LOG:-${LOG_BASE}/vllm_hierarchy_validation.log}"
VLLM_PID=""

TARGET_WORKERS="${TARGET_WORKERS:-128}"
TARGET_TEMPERATURE="${TARGET_TEMPERATURE:-0.2}"
TARGET_TIMEOUT_SECONDS="${TARGET_TIMEOUT_SECONDS:-240}"
TARGET_MAX_TOKENS="${TARGET_MAX_TOKENS:-4096}"
BATCH_SIZE="${BATCH_SIZE:-100}"
REPEATS="${REPEATS:-5}"
SEED="${SEED:-4242}"
MAX_CHILD_HOLDOUT="${MAX_CHILD_HOLDOUT:-40}"
MAX_PARENT_REFERENCE="${MAX_PARENT_REFERENCE:-20}"
SKILL_PATH="${SKILL_PATH:-}"

[[ -s "${TAXONOMY_PATH}" ]] || blind_fail "missing or empty taxonomy: ${TAXONOMY_PATH}"
(( SHARD_COUNT > 0 )) || blind_fail "SHARD_COUNT must be positive"
(( TARGET_WORKERS > 0 )) || blind_fail "TARGET_WORKERS must be positive"
(( VLLM_MAX_NUM_SEQS > 0 )) || blind_fail "VLLM_MAX_NUM_SEQS must be positive"
(( TARGET_WORKERS <= VLLM_MAX_NUM_SEQS )) \
  || blind_fail "TARGET_WORKERS=${TARGET_WORKERS} exceeds VLLM_MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS}"
[[ -n "${HIERARCHY_PAIRS//[[:space:]]/}" ]] || blind_fail "HIERARCHY_PAIRS is empty"

gpu_selection="${CUDA_VISIBLE_DEVICES:-0}"
case "${gpu_selection// /}" in
  "") blind_fail "set CUDA_VISIBLE_DEVICES to one H20 GPU" ;;
  *,*) blind_fail "hierarchy validation expects one H20 GPU; got CUDA_VISIBLE_DEVICES=${gpu_selection}" ;;
esac
export CUDA_VISIBLE_DEVICES="${gpu_selection}"

card_paths=()
for ((index=0; index<SHARD_COUNT; index++)); do
  card_dir="${OUT_BASE}/extract_shard${index}of${SHARD_COUNT}"
  card_path="${card_dir}/usable_mechanism_cards.jsonl"
  [[ -s "${card_path}" ]] || blind_fail "missing or empty blind cards: ${card_path}"
  card_paths+=("${card_path}")
done

read -r -a hierarchy_pairs <<< "${HIERARCHY_PAIRS}"
"${PYTHON_BIN}" - "${TAXONOMY_PATH}" "${HIERARCHY_PAIRS}" "${card_paths[@]}" <<'PY'
import json
import sys
from pathlib import Path

taxonomy_path = Path(sys.argv[1])
pairs = sys.argv[2].split()
card_paths = [Path(value) for value in sys.argv[3:]]

payload = json.loads(taxonomy_path.read_text(encoding="utf-8"))
types = list(payload.get("types") or [])
by_id = {str(row.get("type_id") or ""): row for row in types}
if not by_id or "" in by_id or len(by_id) != len(types):
    raise SystemExit("ERROR: taxonomy has zero, missing, or duplicate type_id values")

parsed_pairs = []
for raw in pairs:
    parts = raw.split(":")
    if len(parts) not in (2, 3) or any(not value for value in parts):
        raise SystemExit(
            f"ERROR: invalid hierarchy pair {raw!r}; expected "
            "PARENT_ID:CHILD_ID[:CONTROL_TYPE_ID]"
        )
    parent_id, child_id = parts[:2]
    control_id = parts[2] if len(parts) == 3 else None
    if len(set(parts)) != len(parts):
        raise SystemExit(f"ERROR: hierarchy pair repeats a type ID: {raw}")
    missing = [value for value in parts if value not in by_id]
    if missing:
        raise SystemExit(
            f"ERROR: hierarchy pair {raw} references unknown type(s): {', '.join(missing)}"
        )
    child_holdout = set(by_id[child_id].get("holdout_member_keys") or [])
    if not child_holdout:
        raise SystemExit(f"ERROR: child {child_id} has no held-out members")
    parsed_pairs.append((parent_id, child_id, control_id, len(child_holdout)))

card_keys = set()
for path in card_paths:
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = str(row.get("sample_key") or "")
            if not key:
                raise SystemExit(f"ERROR: missing sample_key in {path}:{line_no}")
            if key in card_keys:
                raise SystemExit(f"ERROR: duplicate sample_key across shards: {key}")
            card_keys.add(key)

unknown = set()
for type_row in types:
    for field in ("member_keys", "fit_member_keys", "holdout_member_keys"):
        unknown.update(set(type_row.get(field) or []) - card_keys)
if unknown:
    raise SystemExit(
        f"ERROR: taxonomy references {len(unknown)} unknown card key(s); "
        f"first={sorted(unknown)[0]}"
    )

pair_text = ", ".join(
    f"{parent}->{child}"
    + (f"[control={control}]" if control else "")
    + f"(child_holdout={count})"
    for parent, child, control, count in parsed_pairs
)
print(
    f"[preflight] cards={len(card_keys)} types={len(types)} "
    f"pairs={len(parsed_pairs)}: {pair_text}"
)
PY

endpoint_ready() {
  "${PYTHON_BIN}" - "${QWEN_CHAT_BASE_URL}" "${TARGET_MODEL}" <<'PY'
import json
import sys
import urllib.request

try:
    with urllib.request.urlopen(sys.argv[1].rstrip("/") + "/models", timeout=5) as response:
        if response.status != 200:
            raise SystemExit(1)
        payload = json.load(response)
        model_ids = {str(row.get("id") or "") for row in payload.get("data") or []}
        raise SystemExit(0 if sys.argv[2] in model_ids else 1)
except Exception:
    raise SystemExit(1)
PY
}

cleanup() {
  if blind_truthy "${STOP_VLLM_ON_EXIT}" && [[ -n "${VLLM_PID}" ]] \
    && kill -0 "${VLLM_PID}" 2>/dev/null; then
    echo "[vllm] stopping pid=${VLLM_PID}"
    kill "${VLLM_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

start_vllm() {
  if endpoint_ready; then
    echo "[vllm] reusing compatible endpoint ${QWEN_CHAT_BASE_URL}"
    return
  fi
  blind_truthy "${START_VLLM}" \
    || blind_fail "compatible vLLM endpoint is not ready: ${QWEN_CHAT_BASE_URL}"
  [[ -d "${MODEL_PATH}" ]] || blind_fail "MODEL_PATH not found: ${MODEL_PATH}"
  command -v vllm >/dev/null 2>&1 || blind_fail "vllm command not found"

  echo "[vllm] starting ${TARGET_MODEL} on GPU ${CUDA_DEVICE}; log=${VLLM_LOG}"
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
    --max-num-batched-tokens "${VLLM_MAX_NUM_BATCHED_TOKENS}" \
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
  "${PYTHON_BIN}" -u scripts/tools/validate_searchqa_blind_hierarchy.py
  --taxonomy "${TAXONOMY_PATH}"
  --config "${PROJECT_ROOT}/configs/searchqa/default.yaml"
  --output-dir "${HIERARCHY_DIR}"
  --target-model "${TARGET_MODEL}"
  --target-base-url "${QWEN_CHAT_BASE_URL}"
  --target-api-key "${TARGET_API_KEY}"
  --target-temperature "${TARGET_TEMPERATURE}"
  --target-timeout-seconds "${TARGET_TIMEOUT_SECONDS}"
  --target-max-tokens "${TARGET_MAX_TOKENS}"
  --target-workers "${TARGET_WORKERS}"
  --batch-size "${BATCH_SIZE}"
  --repeats "${REPEATS}"
  --seed "${SEED}"
  --max-child-holdout "${MAX_CHILD_HOLDOUT}"
  --max-parent-reference "${MAX_PARENT_REFERENCE}"
)
for card_path in "${card_paths[@]}"; do
  args+=(--cards "${card_path}")
done
for pair in "${hierarchy_pairs[@]}"; do
  args+=(--pair "${pair}")
done
if [[ -n "${SKILL_PATH}" ]]; then
  [[ -s "${SKILL_PATH}" ]] || blind_fail "missing or empty SKILL_PATH: ${SKILL_PATH}"
  args+=(--skill "${SKILL_PATH}")
fi

mkdir -p "${HIERARCHY_DIR}"
echo "============================================================"
echo "  SearchQA blind parent/child hierarchy validation"
echo "============================================================"
echo "  taxonomy:       ${TAXONOMY_PATH}"
echo "  pairs:          ${HIERARCHY_PAIRS}"
echo "  variants:       initial parent child parent_plus_child unrelated_control"
echo "  samples:        child<=${MAX_CHILD_HOLDOUT} parent_reference<=${MAX_PARENT_REFERENCE}"
echo "  repeats/seed:   ${REPEATS}/${SEED}"
echo "  workers/seqs:   ${TARGET_WORKERS}/${VLLM_MAX_NUM_SEQS}"
echo "  qwen_url:       ${QWEN_CHAT_BASE_URL}"
echo "  output:         ${HIERARCHY_DIR}"
echo "============================================================"

if blind_truthy "${DRY_RUN}"; then
  args+=(--dry-run)
else
  start_vllm
fi

"${args[@]}" 2>&1 | tee "${LOG_BASE}/hierarchy_validation.log"

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "${SCRIPT_DIR}/_common.sh"

suite_require_layout
if ! suite_truthy "${DRY_RUN}"; then
  mkdir -p "${SEED_ROOT}" "${LOG_ROOT}/vllm"
fi

visible="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
visible="${visible// /}"
IFS=',' read -r -a gpu_devices <<< "${visible}"
[[ "${#gpu_devices[@]}" -eq 4 ]] \
  || suite_fail "This suite requires exactly four visible GPUs; got '${visible}'"

endpoint_ready() {
  local base_url="$1"
  "${PYTHON_BIN}" - "${base_url}" "${SERVED_MODEL_NAME}" <<'PY'
import json
import sys
import urllib.request

try:
    with urllib.request.urlopen(
        urllib.request.Request(
            sys.argv[1].rstrip("/") + "/models",
            headers={"Authorization": "Bearer dummy"},
        ),
        timeout=5,
    ) as response:
        payload = json.loads(response.read().decode("utf-8"))
        model_ids = {
            str(row.get("id"))
            for row in payload.get("data", [])
            if isinstance(row, dict)
        }
        raise SystemExit(
            0 if response.status == 200 and sys.argv[2] in model_ids else 1
        )
except Exception:
    raise SystemExit(1)
PY
}

qwen_smoke() {
  local base_url="$1"
  "${PYTHON_BIN}" - "${base_url}" "${SERVED_MODEL_NAME}" <<'PY'
import json
import sys
import urllib.request

payload = {
    "model": sys.argv[2],
    "messages": [{"role": "user", "content": "Reply with OK."}],
    "temperature": 0.0,
    "max_tokens": 32,
    "chat_template_kwargs": {"enable_thinking": False},
}
request = urllib.request.Request(
    sys.argv[1].rstrip("/") + "/chat/completions",
    data=json.dumps(payload).encode("utf-8"),
    headers={
        "Authorization": "Bearer dummy",
        "Content-Type": "application/json",
    },
    method="POST",
)
with urllib.request.urlopen(request, timeout=120) as response:
    result = json.loads(response.read().decode("utf-8"))
choices = result.get("choices") or []
if not choices or not isinstance(choices[0].get("message"), dict):
    raise SystemExit("Qwen smoke test returned no assistant message")
print(
    "[qwen smoke] "
    f"endpoint={sys.argv[1]} "
    f"finish_reason={choices[0].get('finish_reason')}"
)
PY
}

vllm_pids=()
queue_pids=()

cleanup() {
  local pid
  for pid in "${queue_pids[@]:-}"; do
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
    fi
  done
  if suite_truthy "${STOP_VLLM_ON_EXIT}"; then
    for pid in "${vllm_pids[@]:-}"; do
      if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
        kill "${pid}" 2>/dev/null || true
      fi
    done
  fi
}
trap cleanup EXIT INT TERM

start_vllm() {
  local queue_id="$1"
  local gpu_device="$2"
  local port="$3"
  local base_url="http://127.0.0.1:${port}/v1"
  local log_file="${LOG_ROOT}/vllm/gpu_${queue_id}.log"

  vllm_cmd=(
    vllm serve "${MODEL_PATH}"
    --served-model-name "${SERVED_MODEL_NAME}"
    --host 0.0.0.0
    --port "${port}"
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
  if suite_truthy "${DRY_RUN}"; then
    printf '[vllm cmd queue=%s gpu=%s]' "${queue_id}" "${gpu_device}"
    suite_quote_cmd env "CUDA_VISIBLE_DEVICES=${gpu_device}" "${vllm_cmd[@]}"
    return
  fi
  if endpoint_ready "${base_url}"; then
    echo "[vllm] reuse queue=${queue_id} gpu=${gpu_device} endpoint=${base_url}"
    return
  fi
  printf '[vllm cmd queue=%s gpu=%s]' "${queue_id}" "${gpu_device}"
  suite_quote_cmd env "CUDA_VISIBLE_DEVICES=${gpu_device}" "${vllm_cmd[@]}"
  env CUDA_VISIBLE_DEVICES="${gpu_device}" \
    nohup "${vllm_cmd[@]}" > "${log_file}" 2>&1 &
  local pid=$!
  vllm_pids+=("${pid}")
  echo "${pid}" > "${LOG_ROOT}/vllm/gpu_${queue_id}.pid"

  local second
  for second in $(seq 1 "${VLLM_WAIT_SECONDS}"); do
    if endpoint_ready "${base_url}"; then
      echo "[vllm] ready queue=${queue_id} gpu=${gpu_device} after=${second}s"
      return
    fi
    if ! kill -0 "${pid}" 2>/dev/null; then
      tail -n 100 "${log_file}" || true
      suite_fail "vLLM queue ${queue_id} exited before becoming ready"
    fi
    sleep 1
  done
  tail -n 100 "${log_file}" || true
  suite_fail "vLLM queue ${queue_id} readiness timeout"
}

run_queue() {
  local queue_id="$1"
  local base_url="$2"
  shift 2
  local case_name status=0
  for case_name in "$@"; do
    if ! (
      SEED_ROOT="${SEED_ROOT}" \
      LOG_ROOT="${LOG_ROOT}" \
      bash "${SCRIPT_DIR}/_run_case.sh" "${case_name}" "${base_url}" "${queue_id}"
    ); then
      status=1
      if ! suite_truthy "${KEEP_GOING}"; then
        break
      fi
    fi
  done
  return "${status}"
}

echo "============================================================"
echo " LiveMath ablations: Qwen3.5-4B + DeepSeek-v4-pro"
echo "============================================================"
echo " seed:                 ${SEED}"
echo " visible GPUs:         ${visible}"
echo " target workers/GPU:   ${TARGET_WORKERS}"
echo " optimizer workers:    ${ANALYST_WORKERS} (four-process ceiling=$((ANALYST_WORKERS * 4)))"
echo " patch-record workers: ${PATCH_RECORD_WORKERS}"
echo " output:               ${SEED_ROOT}"
echo " dry_run:              ${DRY_RUN}"
echo "============================================================"

for queue_id in 0 1 2 3; do
  start_vllm \
    "${queue_id}" \
    "${gpu_devices[$queue_id]}" \
    "$((VLLM_BASE_PORT + queue_id))"
done
if ! suite_truthy "${DRY_RUN}"; then
  for queue_id in 0 1 2 3; do
    qwen_smoke "http://127.0.0.1:$((VLLM_BASE_PORT + queue_id))/v1"
  done
fi

queue_0=(rollout_r8 dynamic_virtual_root rollout_r2 system_prompt_only)
queue_1=(batch_12 full flat_fuse_fixed_real_root init_skill_only)
queue_2=(cluster_random fallback_children merge_concat)
queue_3=(cluster_success_aware fallback_internal fallback_none batch_35)

if suite_truthy "${SMOKE}"; then
  queue_0=(full)
  queue_1=(merge_concat)
  queue_2=(fallback_none)
  queue_3=(cluster_random)
fi

overall_status=0
if suite_truthy "${DRY_RUN}"; then
  # Keep command rendering readable and deterministic.
  run_queue 0 "http://127.0.0.1:${VLLM_BASE_PORT}/v1" "${queue_0[@]}" || overall_status=1
  run_queue 1 "http://127.0.0.1:$((VLLM_BASE_PORT + 1))/v1" "${queue_1[@]}" || overall_status=1
  run_queue 2 "http://127.0.0.1:$((VLLM_BASE_PORT + 2))/v1" "${queue_2[@]}" || overall_status=1
  run_queue 3 "http://127.0.0.1:$((VLLM_BASE_PORT + 3))/v1" "${queue_3[@]}" || overall_status=1
else
  for queue_id in 0 1 2 3; do
    base_url="http://127.0.0.1:$((VLLM_BASE_PORT + queue_id))/v1"
    case "${queue_id}" in
      0) run_queue "${queue_id}" "${base_url}" "${queue_0[@]}" & ;;
      1) run_queue "${queue_id}" "${base_url}" "${queue_1[@]}" & ;;
      2) run_queue "${queue_id}" "${base_url}" "${queue_2[@]}" & ;;
      3) run_queue "${queue_id}" "${base_url}" "${queue_3[@]}" & ;;
    esac
    queue_pids+=("$!")
  done

  for pid in "${queue_pids[@]}"; do
    if ! wait "${pid}"; then
      overall_status=1
    fi
  done
  queue_pids=()
fi

if [[ "${overall_status}" -eq 0 ]]; then
  if ! suite_truthy "${DRY_RUN}"; then
    touch "${SEED_ROOT}/.seed_suite_complete"
  fi
  echo "[suite done] seed=${SEED} output=${SEED_ROOT}"
else
  echo "[suite incomplete] seed=${SEED}; inspect ${LOG_ROOT}" >&2
fi
exit "${overall_status}"

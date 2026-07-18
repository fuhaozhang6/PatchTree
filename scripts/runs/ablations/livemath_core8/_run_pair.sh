#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

for name in \
  PAIR_NAME \
  EXP1_NAME EXP1_BATCH_SIZE EXP1_ROLLOUT_REPEATS EXP1_TREE_DEPTH \
  EXP2_NAME EXP2_BATCH_SIZE EXP2_ROLLOUT_REPEATS EXP2_TREE_DEPTH; do
  [[ -n "${!name:-}" ]] || { echo "ERROR: ${name} is required." >&2; exit 1; }
done

EXP1_TARGET_PEAK=$((EXP1_BATCH_SIZE * EXP1_ROLLOUT_REPEATS))
EXP2_TARGET_PEAK=$((EXP2_BATCH_SIZE * EXP2_ROLLOUT_REPEATS))
PAIR_TARGET_PEAK=$((EXP1_TARGET_PEAK + EXP2_TARGET_PEAK))
VLLM_STABLE_PEAK=128
if (( PAIR_TARGET_PEAK > VLLM_STABLE_PEAK )); then
  echo "ERROR: Pair target peak ${PAIR_TARGET_PEAK} exceeds validated L20/vLLM peak ${VLLM_STABLE_PEAK}." >&2
  exit 1
fi

truthy() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

RUN_STAMP="${ABLATION_RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
PAIR_GROUP="${ABLATION_PAIR_GROUP:-livemath_core8_pairs}"
PAIR_ROOT="${ABLATION_PAIR_ROOT:-${PROJECT_ROOT}/outputs/${PAIR_GROUP}/${PAIR_NAME}_seed${ABLATION_SEED:-42}_${RUN_STAMP}}"
PAIR_LOG_ROOT="${ABLATION_PAIR_LOG_ROOT:-${PROJECT_ROOT}/logs/${PAIR_GROUP}/${PAIR_NAME}_seed${ABLATION_SEED:-42}_${RUN_STAMP}}"
EXP1_LOG_DIR="${PAIR_LOG_ROOT}/${EXP1_NAME}"
EXP2_LOG_DIR="${PAIR_LOG_ROOT}/${EXP2_NAME}"
EXP1_OUT_DIR="${PAIR_ROOT}/${EXP1_NAME}"
EXP2_OUT_DIR="${PAIR_ROOT}/${EXP2_NAME}"

mkdir -p "${EXP1_LOG_DIR}" "${EXP2_LOG_DIR}" "${EXP1_OUT_DIR}" "${EXP2_OUT_DIR}"

PAIR_FINISHED=0
cleanup() {
  local pid pid_file
  if [[ "${PAIR_FINISHED}" != "1" ]]; then
    for pid_file in "${EXP1_LOG_DIR}/pids.txt" "${EXP2_LOG_DIR}/pids.txt"; do
      [[ -f "${pid_file}" ]] || continue
      while read -r _ pid _; do
        if [[ -n "${pid:-}" ]] && kill -0 "${pid}" 2>/dev/null; then
          kill "${pid}" 2>/dev/null || true
        fi
      done < "${pid_file}"
    done
  fi
  pid_file="${EXP1_LOG_DIR}/vllm.pid"
  if [[ -f "${pid_file}" ]]; then
    pid="$(<"${pid_file}")"
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      echo "[cleanup] stopping shared vLLM pid=${pid}"
      kill "${pid}" 2>/dev/null || true
    fi
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

run_experiment() {
  local name="$1"
  local batch_size="$2"
  local repeats="$3"
  local depth="$4"
  local wait_for_jobs="$5"
  local start_vllm="$6"
  local log_dir="$7"
  local out_dir="$8"

  ABLATION_NAME="${name}" \
  ABLATION_BATCH_SIZE="${batch_size}" \
  ABLATION_ROLLOUT_REPEATS="${repeats}" \
  ABLATION_TREE_DEPTH="${depth}" \
  ABLATION_WAIT_FOR_JOBS="${wait_for_jobs}" \
  ABLATION_WORKERS="${ABLATION_WORKERS_PER_RUN:-96}" \
  ABLATION_ANALYST_WORKERS="${ABLATION_ANALYST_WORKERS_PER_RUN:-32}" \
  ABLATION_PATCH_RECORD_WORKERS="${ABLATION_PATCH_RECORD_WORKERS_PER_RUN:-32}" \
  ABLATION_API_MAX_CONCURRENCY="${ABLATION_API_MAX_CONCURRENCY_PER_RUN:-96}" \
  ABLATION_RUN_STAMP="${RUN_STAMP}" \
  ABLATION_LOG_DIR="${log_dir}" \
  ABLATION_OUT_BASE="${out_dir}" \
  START_VLLM="${start_vllm}" \
  STOP_VLLM_ON_EXIT=0 \
  bash "${SCRIPT_DIR}/_run_one.sh"
}

echo "============================================================"
echo "  ${ABLATION_SUITE_LABEL:-LiveMath core-8} pair: ${PAIR_NAME}"
echo "============================================================"
echo "  GPU/vLLM:        1 L20 / 1 shared endpoint"
echo "  concurrent runs: 2"
echo "  per-run workers: ${ABLATION_WORKERS_PER_RUN:-96} target / ${ABLATION_ANALYST_WORKERS_PER_RUN:-32} analyst"
echo "  target peak:     ${EXP1_TARGET_PEAK} + ${EXP2_TARGET_PEAK} = ${PAIR_TARGET_PEAK} / ${VLLM_STABLE_PEAK}"
echo "  run 1:           ${EXP1_NAME} (b=${EXP1_BATCH_SIZE}, r=${EXP1_ROLLOUT_REPEATS}, d=${EXP1_TREE_DEPTH})"
echo "  run 2:           ${EXP2_NAME} (b=${EXP2_BATCH_SIZE}, r=${EXP2_ROLLOUT_REPEATS}, d=${EXP2_TREE_DEPTH})"
echo "  output root:     ${PAIR_ROOT}"
echo "============================================================"

# The first launcher owns the shared vLLM. It returns after detaching its
# training process. The second launcher reuses the ready endpoint and waits for
# its own training process.
run_experiment \
  "${EXP1_NAME}" "${EXP1_BATCH_SIZE}" "${EXP1_ROLLOUT_REPEATS}" "${EXP1_TREE_DEPTH}" \
  0 "${START_VLLM:-1}" "${EXP1_LOG_DIR}" "${EXP1_OUT_DIR}"

set +e
run_experiment \
  "${EXP2_NAME}" "${EXP2_BATCH_SIZE}" "${EXP2_ROLLOUT_REPEATS}" "${EXP2_TREE_DEPTH}" \
  1 0 "${EXP2_LOG_DIR}" "${EXP2_OUT_DIR}"
exp2_status=$?
set -e

echo "[wait] waiting for ${EXP1_NAME} completion marker"
while ! grep -q '^\[exit\] ' "${EXP1_LOG_DIR}/livemath.log" 2>/dev/null; do
  if [[ ! -f "${EXP1_LOG_DIR}/pids.txt" ]]; then
    echo "ERROR: Missing PID file for ${EXP1_NAME}: ${EXP1_LOG_DIR}/pids.txt" >&2
    break
  fi
  if truthy "${DRY_RUN:-0}"; then
    sleep 1
  else
    sleep 5
  fi
done

exp1_status="$(awk '/^\[exit\] / {status=$2} END {print status}' "${EXP1_LOG_DIR}/livemath.log" 2>/dev/null || true)"
if [[ -z "${exp1_status}" ]]; then
  echo "ERROR: Could not determine ${EXP1_NAME} exit status from ${EXP1_LOG_DIR}/livemath.log" >&2
  exp1_status=1
fi

echo "============================================================"
echo "  Pair completed: ${PAIR_NAME}"
echo "  ${EXP1_NAME}: exit=${exp1_status}"
echo "  ${EXP2_NAME}: exit=${exp2_status}"
echo "============================================================"

PAIR_FINISHED=1
if [[ "${exp1_status}" != "0" || "${exp2_status}" != "0" ]]; then
  exit 1
fi

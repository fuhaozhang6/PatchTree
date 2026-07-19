#!/usr/bin/env bash
set -euo pipefail

# Baseline prompt concurrency matrix for SearchQA.
# Measures baseline_current under:
#   dataset/job count: 2, 3
#   workers per job: 96, 128
#
# The script starts or reuses one local Qwen vLLM endpoint, runs each matrix cell,
# then writes a compact recommendation table.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
OUT_BASE="${OUT_BASE:-outputs/searchqa_baseline_concurrency_matrix_${TS}}"
SERVER_OUT_DIR="${SERVER_OUT_DIR:-${OUT_BASE}/server}"

SPLIT_PATH="${SPLIT_PATH:-data/searchqa_split/test/items.json}"
SAMPLE_SIZE="${SAMPLE_SIZE:-0}"
MAX_TOKENS="${MAX_TOKENS:-16384}"
SKILL_PATH="${SKILL_PATH:-skillopt/envs/searchqa/skills/initial.md}"
PROMPT_NAME="${PROMPT_NAME:-baseline_current}"
JOB_COUNTS="${JOB_COUNTS:-2 3}"
WORKERS_LIST="${WORKERS_LIST:-96 128}"
REPEATS="${REPEATS:-1}"

STOP_VLLM_ON_EXIT="${STOP_VLLM_ON_EXIT:-1}"
QWEN_CHAT_BASE_URL="${QWEN_CHAT_BASE_URL:-http://127.0.0.1:${VLLM_PORT:-8000}/v1}"
QWEN_CHAT_API_KEY="${QWEN_CHAT_API_KEY:-dummy}"
QWEN_CHAT_MODEL="${QWEN_CHAT_MODEL:-${SERVED_MODEL_NAME:-Qwen/Qwen3.5-4B}}"
QWEN_CHAT_TIMEOUT_SECONDS="${QWEN_CHAT_TIMEOUT_SECONDS:-240}"
TARGET_QWEN_CHAT_TEMPERATURE="${TARGET_QWEN_CHAT_TEMPERATURE:-0.2}"

MATRIX_SUMMARY_JSONL="${OUT_BASE}/matrix_summary.jsonl"
MATRIX_SUMMARY_MD="${OUT_BASE}/matrix_summary.md"

mkdir -p "${OUT_BASE}"
: > "${MATRIX_SUMMARY_JSONL}"

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

echo "[matrix] project=${PROJECT_ROOT}"
echo "[matrix] out_base=${OUT_BASE}"
echo "[matrix] prompt=${PROMPT_NAME}"
echo "[matrix] split=${SPLIT_PATH}"
echo "[matrix] sample_size=${SAMPLE_SIZE} max_tokens=${MAX_TOKENS}"
echo "[matrix] job_counts=${JOB_COUNTS}"
echo "[matrix] workers_list=${WORKERS_LIST}"
echo "[matrix] repeats=${REPEATS}"

echo "[step 1/3] start or reuse vLLM"
START_ONLY=1 STOP_VLLM_ON_EXIT=0 OUT_DIR="${SERVER_OUT_DIR}" \
  QWEN_CHAT_BASE_URL="${QWEN_CHAT_BASE_URL}" \
  QWEN_CHAT_API_KEY="${QWEN_CHAT_API_KEY}" \
  QWEN_CHAT_MODEL="${QWEN_CHAT_MODEL}" \
  bash experiments/searchqa_prompt_latency/run_with_vllm.sh

cell_status=0
for repeat in $(seq 1 "${REPEATS}"); do
  for jobs in ${JOB_COUNTS}; do
    for workers in ${WORKERS_LIST}; do
      cell_dir="${OUT_BASE}/jobs${jobs}_workers${workers}_rep${repeat}"
      mkdir -p "${cell_dir}"
      echo
      echo "[step 2/3] cell jobs=${jobs} workers=${workers} repeat=${repeat}"

      pids=()
      for idx in $(seq 1 "${jobs}"); do
        job_out_dir="${cell_dir}/job_${idx}"
        job_log="${cell_dir}/job_${idx}.log"
        mkdir -p "${job_out_dir}"
        echo "[start] cell=jobs${jobs}_workers${workers}_rep${repeat} job=${idx}"
        "${PYTHON_BIN}" experiments/searchqa_prompt_latency/run_searchqa_prompt_latency.py \
          --base-url "${QWEN_CHAT_BASE_URL}" \
          --api-key "${QWEN_CHAT_API_KEY}" \
          --model "${QWEN_CHAT_MODEL}" \
          --split "${SPLIT_PATH}" \
          --skill "${SKILL_PATH}" \
          --sample-size "${SAMPLE_SIZE}" \
          --workers "${workers}" \
          --max-tokens "${MAX_TOKENS}" \
          --temperature "${TARGET_QWEN_CHAT_TEMPERATURE}" \
          --timeout "${QWEN_CHAT_TIMEOUT_SECONDS}" \
          --dataset-label "searchqa_baseline_j${idx}" \
          --out-dir "${job_out_dir}" \
          --prompt "${PROMPT_NAME}" \
          > "${job_log}" 2>&1 &
        pids+=("$!")
      done

      status=0
      for pid in "${pids[@]}"; do
        if ! wait "${pid}"; then
          status=1
        fi
      done
      if [[ "${status}" -ne 0 ]]; then
        cell_status=1
        echo "[warn] cell failed or partially failed: jobs=${jobs} workers=${workers} repeat=${repeat}"
      fi

      "${PYTHON_BIN}" - "${cell_dir}" "${jobs}" "${workers}" "${repeat}" "${MATRIX_SUMMARY_JSONL}" <<'PY'
import json
import sys
from pathlib import Path

cell_dir = Path(sys.argv[1])
jobs = int(sys.argv[2])
workers = int(sys.argv[3])
repeat = int(sys.argv[4])
out_path = Path(sys.argv[5])

job_rows = []
for idx in range(1, jobs + 1):
    summary_path = cell_dir / f"job_{idx}" / "summary.json"
    if not summary_path.exists():
        job_rows.append({"job": idx, "missing": True})
        continue
    summaries = json.loads(summary_path.read_text(encoding="utf-8"))
    if not summaries:
        job_rows.append({"job": idx, "missing": True})
        continue
    row = summaries[0]
    row["job"] = idx
    job_rows.append(row)

valid = [r for r in job_rows if not r.get("missing")]
def avg(key):
    vals = [float(r[key]) for r in valid if r.get(key) is not None]
    return sum(vals) / len(vals) if vals else None
def total_finish(reason):
    return sum(int((r.get("finish_reasons") or {}).get(reason, 0)) for r in valid)

cell = {
    "cell_dir": str(cell_dir),
    "jobs": jobs,
    "workers_per_job": workers,
    "total_client_workers": jobs * workers,
    "repeat": repeat,
    "completed_jobs": len(valid),
    "expected_jobs": jobs,
    "hard_mean": avg("hard"),
    "soft_mean": avg("soft"),
    "wall_mean_s": avg("wall_s"),
    "wall_max_s": max((float(r.get("wall_s") or 0) for r in valid), default=None),
    "req_s_mean": avg("requests_per_s"),
    "avg_latency_s_mean": avg("avg_latency_s"),
    "p95_latency_s_mean": avg("p95_latency_s"),
    "avg_completion_tokens_mean": avg("avg_completion_tokens"),
    "p95_completion_tokens_mean": avg("p95_completion_tokens"),
    "max_completion_tokens_max": max((int(r.get("max_completion_tokens") or 0) for r in valid), default=None),
    "length_total": total_finish("length"),
    "error_total": sum(int(r.get("errors") or 0) for r in valid),
    "ok_total": sum(int(r.get("ok") or 0) for r in valid),
    "n_total": sum(int(r.get("n") or 0) for r in valid),
}
with out_path.open("a", encoding="utf-8") as f:
    f.write(json.dumps(cell, ensure_ascii=False) + "\n")
print(json.dumps(cell, ensure_ascii=False), flush=True)
PY
    done
  done
done

echo
echo "[step 3/3] write matrix summary"
"${PYTHON_BIN}" - "${MATRIX_SUMMARY_JSONL}" "${MATRIX_SUMMARY_MD}" <<'PY'
import json
import sys
from collections import defaultdict
from pathlib import Path

jsonl_path = Path(sys.argv[1])
md_path = Path(sys.argv[2])
rows = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip()]

def fmt(value, digits=3):
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)

groups = defaultdict(list)
for row in rows:
    groups[(row["jobs"], row["workers_per_job"])].append(row)

agg_rows = []
for (jobs, workers), vals in sorted(groups.items()):
    def avg(key):
        xs = [float(v[key]) for v in vals if v.get(key) is not None]
        return sum(xs) / len(xs) if xs else None
    agg_rows.append(
        {
            "jobs": jobs,
            "workers_per_job": workers,
            "total_client_workers": jobs * workers,
            "repeats": len(vals),
            "hard": avg("hard_mean"),
            "soft": avg("soft_mean"),
            "wall_mean_s": avg("wall_mean_s"),
            "wall_max_s": avg("wall_max_s"),
            "req_s_mean": avg("req_s_mean"),
            "avg_latency_s": avg("avg_latency_s_mean"),
            "p95_latency_s": avg("p95_latency_s_mean"),
            "avg_comp": avg("avg_completion_tokens_mean"),
            "p95_comp": avg("p95_completion_tokens_mean"),
            "max_comp": max((int(v.get("max_completion_tokens_max") or 0) for v in vals), default=None),
            "length_total": sum(int(v.get("length_total") or 0) for v in vals),
            "errors": sum(int(v.get("error_total") or 0) for v in vals),
            "ok_total": sum(int(v.get("ok_total") or 0) for v in vals),
            "n_total": sum(int(v.get("n_total") or 0) for v in vals),
        }
    )

valid = [r for r in agg_rows if not r["errors"]]
if valid:
    hard_best = max(r["hard"] or 0 for r in valid)
    # Prefer low wall-time and low length tails while requiring near-best hard.
    for r in valid:
        hard_gap = hard_best - (r["hard"] or 0)
        r["_score"] = (r["wall_max_s"] or 1e9) + 20.0 * (r["length_total"] or 0) + 500.0 * max(0.0, hard_gap - 0.005)
    best = min(valid, key=lambda r: r["_score"])
else:
    best = None

headers = [
    "jobs",
    "workers/job",
    "total_workers",
    "repeats",
    "hard",
    "soft",
    "wall_mean_s",
    "wall_max_s",
    "req/s_mean",
    "avg_lat_s",
    "p95_lat_s",
    "avg_comp",
    "p95_comp",
    "max_comp",
    "length",
    "errors",
]

lines = []
lines.append("# SearchQA Baseline Concurrency Matrix")
lines.append("")
lines.append("| " + " | ".join(headers) + " |")
lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
for r in agg_rows:
    lines.append(
        "| "
        + " | ".join(
            [
                str(r["jobs"]),
                str(r["workers_per_job"]),
                str(r["total_client_workers"]),
                str(r["repeats"]),
                fmt(r["hard"]),
                fmt(r["soft"]),
                fmt(r["wall_mean_s"], 2),
                fmt(r["wall_max_s"], 2),
                fmt(r["req_s_mean"], 2),
                fmt(r["avg_latency_s"], 2),
                fmt(r["p95_latency_s"], 2),
                fmt(r["avg_comp"], 1),
                fmt(r["p95_comp"], 0),
                fmt(r["max_comp"], 0),
                str(r["length_total"]),
                str(r["errors"]),
            ]
        )
        + " |"
    )

lines.append("")
if best:
    lines.append(
        "Recommended setting: "
        f"`JOBS={best['jobs']}` and `WORKERS_PER_JOB={best['workers_per_job']}` "
        f"(total client workers={best['total_client_workers']})."
    )
    lines.append("")
    lines.append(
        "Selection rule: keep hard score within 0.5 points of the best observed "
        "setting, then prefer lower max wall time and fewer length-stop tails."
    )
else:
    lines.append("No clean recommendation because every setting had errors.")

lines.append("")
lines.append("Raw per-cell JSONL:")
lines.append(f"`{jsonl_path}`")

md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines), flush=True)
PY

echo "[done] out_base=${OUT_BASE}"
exit "${cell_status}"

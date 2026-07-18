#!/usr/bin/env bash
set -euo pipefail

# Convenience entry point after stage 2. It validates the saved extraction and
# cluster artifacts, runs semantic adjudication, and can optionally launch the
# one-H20 transfer validation. Validation is off by default so the generated
# taxonomy can be reviewed before spending target-model compute.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "${HERE}/_common.sh"

SHARD_COUNT="${SHARD_COUNT:-2}"
RUN_ADJUDICATION="${RUN_ADJUDICATION:-1}"
RUN_TRANSFER_VALIDATION="${RUN_TRANSFER_VALIDATION:-0}"
CLUSTER_PATH="${OUT_BASE}/clusters/candidate_clusters.json"
TAXONOMY_PATH="${TAXONOMY_PATH:-${OUT_BASE}/taxonomy/blind_revision_taxonomy.json}"

# Stage-3 defaults: two independent drafts plus reconciliation. Forty-eight
# cards gives the large contrast cores some all-failure coverage while keeping
# each adjudication prompt bounded.
DRAFTS="${DRAFTS:-2}"
ADJUDICATION_WORKERS="${ADJUDICATION_WORKERS:-12}"
MAX_FIT_CARDS="${MAX_FIT_CARDS:-48}"

# Conservative, previously used single-H20 profile for the optional stage 4.
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-128}"
VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-65536}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
TARGET_WORKERS="${TARGET_WORKERS:-128}"
TARGET_MAX_TOKENS="${TARGET_MAX_TOKENS:-4096}"
REPEATS="${REPEATS:-5}"
MAX_HOLDOUT_PER_TYPE="${MAX_HOLDOUT_PER_TYPE:-40}"
MAX_BOUNDARY_PER_TYPE="${MAX_BOUNDARY_PER_TYPE:-12}"
MIN_HOLDOUT_SAMPLES="${MIN_HOLDOUT_SAMPLES:-10}"
MIN_BOUNDARY_SAMPLES="${MIN_BOUNDARY_SAMPLES:-4}"

export RUN_ID OBSERVED_ROOT OUT_BASE LOG_BASE SHARD_COUNT
export DRAFTS ADJUDICATION_WORKERS MAX_FIT_CARDS
export TAXONOMY_PATH VLLM_MAX_NUM_SEQS VLLM_MAX_NUM_BATCHED_TOKENS
export MAX_MODEL_LEN TARGET_WORKERS TARGET_MAX_TOKENS REPEATS
export MAX_HOLDOUT_PER_TYPE MAX_BOUNDARY_PER_TYPE
export MIN_HOLDOUT_SAMPLES MIN_BOUNDARY_SAMPLES

card_paths=()
for ((index=0; index<SHARD_COUNT; index++)); do
  card_dir="${OUT_BASE}/extract_shard${index}of${SHARD_COUNT}"
  card_path="${card_dir}/usable_mechanism_cards.jsonl"
  label_path="${card_dir}/posthoc_seeded_labels.jsonl"
  [[ -s "${card_path}" ]] || blind_fail "missing or empty blind cards: ${card_path}"
  [[ -f "${label_path}" ]] || blind_fail "missing post-hoc label map: ${label_path}"
  card_paths+=("${card_path}")
done
[[ -s "${CLUSTER_PATH}" ]] || blind_fail "missing or empty stage-2 clusters: ${CLUSTER_PATH}"

"${PYTHON_BIN}" - "${CLUSTER_PATH}" "${card_paths[@]}" <<'PY'
import json
import sys
from pathlib import Path

cluster_path = Path(sys.argv[1])
card_paths = [Path(value) for value in sys.argv[2:]]
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

payload = json.loads(cluster_path.read_text(encoding="utf-8"))
clusters = list(payload.get("clusters") or [])
if not clusters:
    raise SystemExit("ERROR: candidate_clusters.json contains zero clusters")
cluster_ids = set()
assigned = set()
for cluster in clusters:
    cluster_id = str(cluster.get("cluster_id") or "")
    if not cluster_id or cluster_id in cluster_ids:
        raise SystemExit(f"ERROR: invalid/duplicate cluster_id: {cluster_id!r}")
    cluster_ids.add(cluster_id)
    members = set(cluster.get("member_keys") or [])
    fit = set(cluster.get("fit_member_keys") or [])
    holdout = set(cluster.get("holdout_member_keys") or [])
    if not members or not fit or not holdout:
        raise SystemExit(f"ERROR: {cluster_id} has an empty member/fit/holdout set")
    if not fit <= members or not holdout <= members or fit & holdout:
        raise SystemExit(f"ERROR: {cluster_id} has inconsistent fit/holdout membership")
    unknown = members - card_keys
    if unknown:
        raise SystemExit(
            f"ERROR: {cluster_id} references {len(unknown)} unknown card key(s)"
        )
    overlap = assigned & members
    if overlap:
        raise SystemExit(
            f"ERROR: cluster memberships overlap at {sorted(overlap)[0]}"
        )
    assigned.update(members)

print(
    f"[preflight] cards={len(card_keys)} clusters={len(clusters)} "
    f"assigned={len(assigned)} noise={len(card_keys) - len(assigned)}"
)
PY

if blind_truthy "${RUN_ADJUDICATION}"; then
  echo "[stage 3] semantic adjudication and global reconciliation"
  bash "${HERE}/03_adjudicate.sh"
else
  echo "[stage 3] skipped (RUN_ADJUDICATION=${RUN_ADJUDICATION})"
fi

if blind_truthy "${DRY_RUN}" && [[ ! -s "${TAXONOMY_PATH}" ]]; then
  echo "[dry-run] taxonomy was not created; stage-4 checks are skipped"
  exit 0
fi

[[ -s "${TAXONOMY_PATH}" ]] \
  || blind_fail "stage 3 did not produce taxonomy: ${TAXONOMY_PATH}"

"${PYTHON_BIN}" - "${TAXONOMY_PATH}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
types = list(payload.get("types") or [])
if not types:
    raise SystemExit("ERROR: adjudication produced zero final types")
print(
    f"[taxonomy] final_types={len(types)} "
    f"rejected_clusters={len(payload.get('rejected_cluster_ids') or [])}"
)
print(f"[taxonomy] review {path.with_suffix('.md')}")
PY

if ! blind_truthy "${RUN_TRANSFER_VALIDATION}"; then
  echo "[done] stage 4 not requested; review the taxonomy before validation"
  echo "[next] RUN_ADJUDICATION=0 RUN_TRANSFER_VALIDATION=1 CUDA_VISIBLE_DEVICES=0 \\"
  echo "       bash ${HERE}/05_adjudicate_and_optional_validate_h20.sh"
  exit 0
fi

gpu_selection="${CUDA_VISIBLE_DEVICES:-0}"
case "${gpu_selection// /}" in
  "") blind_fail "set CUDA_VISIBLE_DEVICES to one H20 GPU" ;;
  *,*) blind_fail "stage 4 expects one H20 GPU; got CUDA_VISIBLE_DEVICES=${gpu_selection}" ;;
esac
(( TARGET_WORKERS <= VLLM_MAX_NUM_SEQS )) \
  || blind_fail "TARGET_WORKERS=${TARGET_WORKERS} exceeds VLLM_MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS}"

export CUDA_VISIBLE_DEVICES="${gpu_selection}"
echo "[stage 4] one-H20 held-out and boundary transfer validation"
echo "[stage 4] gpu=${CUDA_VISIBLE_DEVICES} workers/seqs=${TARGET_WORKERS}/${VLLM_MAX_NUM_SEQS}"
bash "${HERE}/04_transfer_validate_h20.sh"

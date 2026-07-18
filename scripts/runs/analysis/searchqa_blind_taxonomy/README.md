# SearchQA blind repair-taxonomy discovery

This pipeline discovers `revision_type` from saved repeated trajectories. It
does not feed the old SearchQA question/revision catalog or its few-shots into
mechanism extraction, clustering, or naming.

## Inputs and target-model reuse

The expected input is:

```text
outputs/observed_taxonomy_h20_taxonomy_v1/
  searchqa_shard0of2/
  searchqa_shard1of2/
```

Each shard already contains:

```text
chunks/<split>/<chunk>/rollout_flattened/results.jsonl
chunks/<split>/<chunk>/rollout_flattened/predictions/<attempt-id>/conversation.json
```

Stages 1–3 read those artifacts and make **zero target-Qwen calls**. An empty
conversation file is tolerated because SearchQA `results.jsonl` still records
the question, prediction, gold answers, hard score, response, and failure
reason. A missing `results.jsonl` is a hard error rather than a silent fallback
to fresh inference. The original question/context is restored from
`data/searchqa_split`; this is a data read, not a target-model call.

## Run

First perform a read-only evidence check:

```bash
RUN_ID=blind_v1 \
  bash scripts/runs/analysis/searchqa_blind_taxonomy/00_check_saved_evidence.sh
```

The extraction can use two CPU/API workers in parallel:

```bash
RUN_ID=blind_v1 SHARD_INDEX=0 \
  bash scripts/runs/analysis/searchqa_blind_taxonomy/01_extract_shard.sh

RUN_ID=blind_v1 SHARD_INDEX=1 \
  bash scripts/runs/analysis/searchqa_blind_taxonomy/01_extract_shard.sh
```

Both jobs use DeepSeek official and default to 64 concurrent analysts, for a
combined ceiling of 128. No GPU is needed.

After both finish, run once:

```bash
RUN_ID=blind_v1 \
  bash scripts/runs/analysis/searchqa_blind_taxonomy/02_cluster.sh

RUN_ID=blind_v1 \
  bash scripts/runs/analysis/searchqa_blind_taxonomy/03_adjudicate.sh
```

Key outputs:

```text
outputs/searchqa_blind_taxonomy_blind_v1/
  extract_shard0of2/usable_mechanism_cards.jsonl
  extract_shard1of2/usable_mechanism_cards.jsonl
  clusters/candidate_clusters.json
  clusters/clustering_summary.md
  taxonomy/blind_revision_taxonomy.json
  taxonomy/blind_revision_taxonomy.md
```

The old seeded labels are written to physically separate
`posthoc_seeded_labels.jsonl` files. Stage 3 loads them only after final
cluster membership, names, and global merges are frozen. They are never present
in blind card files, LLM requests, candidate cluster files, or clustering
features.

## Algorithm and defaults

1. Recompute outcome using only infrastructure-valid attempts.
2. Extract no-label mechanism cards. Unstable rows contain a direct
   success/failure contrast; all-failure rows are marked lower-evidence.
3. Form initial density clusters from unstable cards only.
4. Assign compatible all-failure cards to those centers.
5. Cluster repeated residual all-failure mechanisms separately.
6. Split each cluster deterministically into 60% fit and 40% held-out members.
7. Produce two independent DeepSeek adjudications and one reconciliation.
8. Reconcile genuinely interchangeable clusters globally, then name types.

The zero-extra-dependency candidate clusterer is hashed word 1/2-gram TF-IDF
plus cosine DBSCAN. DeepSeek performs the semantic coherence check afterward.

Useful controls:

```bash
SIMILARITY_THRESHOLD=auto        # scans the observed similarity distribution
ASSIGNMENT_THRESHOLD=auto        # keeps the strongest 30% centroid matches
MIN_CLUSTER_SIZE=8               # higher -> stronger support, more noise
DRAFTS=2
ADJUDICATION_WORKERS=12
MAX_FIT_CARDS=36
```

Inspect `clustering_summary.md` before stage 3. The default auto mode sweeps
candidate thresholds and scores silhouette, coverage, and largest-cluster
share. The full sweep is saved under `threshold_diagnostics` in
`candidate_clusters.json`. One giant cluster still means the selected
similarity/assignment thresholds are too low; mostly noise means they are too
high.

## Optional functional validation on one H20

After reviewing the discovered taxonomy:

```bash
RUN_ID=blind_v1 CUDA_VISIBLE_DEVICES=0 \
  bash scripts/runs/analysis/searchqa_blind_taxonomy/04_transfer_validate_h20.sh
```

This is the only stage that starts vLLM. A shared patch derived from fit members
is evaluated on held-out members and nearest outside-cluster boundary samples.
The default acceptance rule is:

```text
held-out in-cluster delta >= +0.05
boundary delta >= -0.02
```

The base accuracy comes from the saved three-attempt audit; only patched
attempts are newly generated.

## Test-set leakage

The default extraction uses `train val test`, matching the explicit taxonomy
construction goal. Such a taxonomy cannot subsequently support a claim of
unbiased SearchQA test performance. To preserve the test split:

```bash
SPLITS="train val" RUN_ID=blind_trainval SHARD_INDEX=0 bash .../01_extract_shard.sh
SPLITS="train val" RUN_ID=blind_trainval SHARD_INDEX=1 bash .../01_extract_shard.sh
```

## Cheap checks

No API or GPU:

```bash
DRY_RUN=1 LIMIT=4 SHARD_INDEX=0 \
  bash scripts/runs/analysis/searchqa_blind_taxonomy/01_extract_shard.sh
```

Stage outputs and per-card/per-cluster API results are cached, so rerunning the
same command resumes rather than repeating completed calls.

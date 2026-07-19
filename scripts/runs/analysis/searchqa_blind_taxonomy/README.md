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

The validator now reruns both the initial and patched Skill on the same
samples with matching repeat counts. The current Qwen chat adapter does not
forward the batch seed as a model-generation seed, so the comparison is paired
by sample but not by identical generation randomness. Acceptance uses these
fresh paired measurements rather than the saved audit accuracy. By default a
type also needs at least 10 held-out members and 4 boundary members.

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

## Convenience runner after clustering

Once `clusters/candidate_clusters.json` exists, the following runner checks
that both extraction shards and all cluster memberships are internally
consistent, then runs stage 3 with DeepSeek official:

```bash
DEEPSEEK_API_KEY=... RUN_ID=blind_v1 \
  bash scripts/runs/analysis/searchqa_blind_taxonomy/05_adjudicate_and_optional_validate_h20.sh
```

Validation is deliberately disabled on that first command. Review
`outputs/searchqa_blind_taxonomy_blind_v1/taxonomy/blind_revision_taxonomy.md`,
then run only stage 4 on one H20:

```bash
RUN_ID=blind_v1 RUN_ADJUDICATION=0 RUN_TRANSFER_VALIDATION=1 \
  CUDA_VISIBLE_DEVICES=0 \
  bash scripts/runs/analysis/searchqa_blind_taxonomy/05_adjudicate_and_optional_validate_h20.sh
```

The H20 defaults are 128 target workers/sequences, 65536 batched tokens,
five paired repeats, at most 40 held-out members and 12 boundary members per
type. A type needs at least 10 held-out and 4 boundary members to pass.
Use `START_VLLM=0 QWEN_CHAT_BASE_URL=http://host:port/v1` to reuse an existing
compatible Qwen endpoint. A combined dry preflight is available with
`DRY_RUN=1`; it makes no DeepSeek or target-model calls.

## Parent/child abstraction-level validation

After the six-type no-global-merge taxonomy is available, compare a broad
parent patch, a narrower child patch, and their composition on exactly the same
samples:

```bash
RUN_ID=blind_v1 DRY_RUN=1 \
  bash scripts/runs/analysis/searchqa_blind_taxonomy/06_validate_hierarchy_h20.sh

RUN_ID=blind_v1 CUDA_VISIBLE_DEVICES=0 \
  bash scripts/runs/analysis/searchqa_blind_taxonomy/06_validate_hierarchy_h20.sh
```

The default pairs are:

```text
R_SEARCH_001:R_SEARCH_002:R_SEARCH_004
  # entity focus -> person-name minimization; containing-entity is the control
R_SEARCH_001:R_SEARCH_004:R_SEARCH_002
  # entity focus -> containing-entity resolution; person-name is the control
```

For each pair, the validator runs `initial`, `parent`, `child`, and
`parent_plus_child` on the child's held-out members and on a stratified sample
of parent-reference members. The optional third type after the second colon
also adds an `unrelated_control` condition. This separates a genuine
child-specific effect from the generic effect of appending any extra rule.
Together these conditions test whether the broad rule, the conditional detail,
or their composition is the useful abstraction level.
Outputs are written to:

```text
outputs/searchqa_blind_taxonomy_blind_v1/hierarchy_validation/
  hierarchy_validation.md
  hierarchy_validation.json
  <parent>__<child>/hierarchy_result.json
```

Useful controls:

```bash
HIERARCHY_PAIRS="R_SEARCH_001:R_SEARCH_002:R_SEARCH_004 R_SEARCH_001:R_SEARCH_004:R_SEARCH_002"
MAX_CHILD_HOLDOUT=40
MAX_PARENT_REFERENCE=20
REPEATS=5
SEED=4242
TARGET_WORKERS=128
VLLM_MAX_NUM_SEQS=128
```

The runner checks all taxonomy type IDs, held-out members, card references, and
duplicate card keys before using the GPU. Set
`START_VLLM=0 QWEN_CHAT_BASE_URL=http://host:port/v1` to reuse an existing
compatible endpoint. It only reuses the endpoint when `/models` exposes the
configured `TARGET_MODEL`.

# Observed taxonomy audit on 4×H20

This workflow does **not train or update a Skill**. It fixes each dataset's
initial Skill, runs the target Qwen model three times for every train/val/test
item, and mines a measured `(question_type, revision_type)` taxonomy.

## Why target trajectories are required

A raw question can support a `question_type` prior, but it cannot prove that a
specific correction was needed. In this audit:

- `q_i = 1`: stable success; classify only `question_type`.
- `0 < q_i < 1`: unstable; analyze the successful/failed contrast.
- `q_i = 0`: stable failure; analyze all failed trajectories.
- `revision_type` exists only when the normal dataset-specific analyst returns
  a valid reusable PatchRecord. `no_patch` is recorded separately.

This keeps task taxonomy, model behavior, and correction taxonomy distinct.

## Four-H20 allocation

Use the same `RUN_ID` on all workers when their project output filesystem is
shared. The three numbered launcher types produce four jobs:

```bash
# H20-1: half of SearchQA
RUN_ID=taxonomy_v1 SHARD_INDEX=0 OPTIMIZER_SOURCE=deepseek \
  bash scripts/runs/analysis/observed_taxonomy_h20/01_searchqa_shard.sh

# H20-2: the other half of SearchQA
RUN_ID=taxonomy_v1 SHARD_INDEX=1 OPTIMIZER_SOURCE=deepseek \
  bash scripts/runs/analysis/observed_taxonomy_h20/01_searchqa_shard.sh

# H20-3: slow interactive dataset
RUN_ID=taxonomy_v1 OPTIMIZER_SOURCE=deepseek \
  bash scripts/runs/analysis/observed_taxonomy_h20/02_alfworld.sh

# H20-4: four similarly sized/throughput-compatible datasets in parallel
RUN_ID=taxonomy_v1 OPTIMIZER_SOURCE=deepseek \
  bash scripts/runs/analysis/observed_taxonomy_h20/03_other_four.sh
```

All four launchers are fixed to DeepSeek official so provider/model differences
cannot contaminate the taxonomy. Set `DEEPSEEK_API_KEY` on every worker. Each
output row records the optimizer source/model, target model, Skill hash, and
repeat count.

The concurrency plan follows the verified safety envelope:

| Worker | Local target workers | vLLM max seqs | Optimizer analysts |
|---|---:|---:|---:|
| SearchQA shard 0 | 128 | 128 | 64 |
| SearchQA shard 1 | 128 | 128 | 64 |
| ALFWorld | 24 (natural batch peak 8) | 96 | 16 |
| Other four datasets | 24 + 16 + 28 + 28 = 96 | 128 | 16 + 16 + 24 + 24 = 80 |

Across all workers the DeepSeek analyst ceiling is 224, below the measured
stable account ceiling of 256. Each local target endpoint stays at or below the
96–128 safe region.

The SearchQA split is deterministic (`items[shard_index::2]`), so the two jobs
have no overlap and exactly cover all items.

All workers may share the same `RUN_ID` and VEPFS log directory. vLLM logs use
job-specific names and therefore do not collide:

- `vllm_searchqa_shard0of2.log`
- `vllm_searchqa_shard1of2.log`
- `vllm_alfworld.log`
- `vllm_other_four.log`

## Evidence ranking and final few-shots

After all jobs finish:

```bash
RUN_ID=taxonomy_v1 OPTIMIZER_SOURCE=deepseek \
  bash scripts/runs/analysis/observed_taxonomy_h20/04_merge_and_synthesize.sh
```

The merge stage rejects conflicting duplicate sample IDs and reports Wilson
95% intervals for stable-success and reusable-revision rates. Candidate
few-shots are ranked by:

1. unstable contrast evidence;
2. replicated support;
3. support across train/val/test;
4. coverage of distinct revision mechanisms.

Two independent LLM adjudications plus one reconciliation merge only genuinely
synonymous labels, sanitize sample-specific details, and select at most eight
dataset-specific examples. Results are written under
`proposed_few_shots/`; prompts are deliberately not overwritten until the
evidence and wording are reviewed.

If workers write to different output roots, copy/mount the outputs together or
pass a whitespace-separated list:

```bash
INPUT_ROOTS="/path/job1 /path/job2 /path/job3 /path/job4" \
  MERGED_DIR=/path/merged \
  bash scripts/runs/analysis/observed_taxonomy_h20/04_merge_and_synthesize.sh
```

For a cheap configuration check:

```bash
DRY_RUN=1 LIMIT_PER_SPLIT=2 SHARD_INDEX=0 \
  bash scripts/runs/analysis/observed_taxonomy_h20/01_searchqa_shard.sh
```

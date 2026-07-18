# Compact PatchRecord and Semantic Merge Runtime

The current V2 runtime uses a compact PatchRecord and dataset-owned type prompts:

```text
repeated rollouts
-> compact PatchRecords
-> global semantic clustering
-> leaf nodes
-> optional mid nodes
-> root candidate
-> normal validation gate / direct-child fallback
```

## Compact PatchRecord

Only failed or unstable samples produce records. `q_i` is used before generation
to decide whether a record is needed, but is not persisted in the record.

```json
{
  "record_id": "R0001",
  "question_type": "near_duplicate_field_selection",
  "revision_type": "near_match_disambiguation",
  "repair_signature": "match field label before extracting value",
  "condition": "several nearby fields contain plausible values",
  "boundary": "do not reject an exact field match",
  "patch": {"op": "append", "content": "..."}
}
```

There is no raw/anchor/canonical type chain. Each environment owns a complete
`type_guided_patch_record.md` prompt containing its type suggestions and few-shot
examples. Type names may overlap across datasets naturally.

## Clustering

When enabled, one global LLM planner receives all records from the current step.
`question_type` and `revision_type` are soft signals; `repair_signature`,
`condition`, and `boundary` determine whether repairs are genuinely compatible.
Invalid planner output falls back to deterministic repair-signature grouping.

## Semantic Node Contract

Leaf, mid, and root merge calls return:

```json
{
  "shared_core": {
    "condition": "...",
    "boundary": "...",
    "source_child_ids": ["L1", "L2"],
    "patch": {"op": "append", "content": "..."}
  },
  "conditional_residuals": [
    {
      "condition": "...",
      "boundary": "...",
      "source_child_ids": ["L1"],
      "patch": {"op": "append", "content": "..."}
    }
  ],
  "preserved_constraints": {"L1": ["..."]},
  "unresolved_conflicts": []
}
```

The runtime compiles the shared core and every valid conditional residual into
the existing `edits` interface. Conditions are written explicitly into Skill
text. Preserved constraints and unresolved conflicts remain artifact metadata;
conflicts are never compiled into the Skill.

## Validation

Leaf self-check and support-sample self-check have been removed. The normal
candidate validation gate remains authoritative. If a root candidate fails,
the existing direct-child fallback may evaluate leaf children (`tree_depth=2`)
or mid children (`tree_depth=3`). Tail-bank candidates also pass the normal gate.

## Caching and Artifacts

PatchRecord generation and global clustering retain model/skill/prompt-version
caches. There is no canonicalization cache or self-check artifact. Per-step
artifacts include records, clustering, leaf nodes, optional mid nodes, root,
merge artifact, cache report, and fallback results.

## Concurrency

PatchRecord generation already uses a bounded worker pool. PatchTree also runs
independent leaf merges concurrently and, for a three-level tree, independent
mid merges concurrently. The root merge remains sequential because it depends
on all child nodes. Configure the two new pools independently:

```yaml
optimizer:
  type_guided_leaf_merge_workers: 4
  type_guided_mid_merge_workers: 4
```

The multi-dataset launcher shares one local Qwen/vLLM server across jobs and
uses DeepSeek only for optimizer calls. vLLM prefix caching, chunked prefill,
maximum concurrent sequences, dataset concurrency, rollout workers, and the
two merge pools can therefore be tuned without changing the PatchTree data
contract.

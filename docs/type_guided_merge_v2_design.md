# Type-Guided Merge Tree V2 Design

> Historical design draft. The current runtime uses compact dataset-specific
> PatchRecords, has removed canonicalization and leaf/support self-check, and
> compiles `shared_core + conditional_residuals` into executable edits. See
> `docs/type_guided_v23_light.md` for the implemented contract.

## 1. Goal

V2 returns to the original method idea more faithfully than V1:

```text
per-question repeated rollouts
-> per-question PatchRecord
-> type-guided leaf clusters
-> leaf patch merge
-> optional leaf self-check
-> conservative root merge
-> root validation
-> one-level child fallback
```

V1 reused SkillOpt's minibatch analyst output and grouped typed edits. V2
changes the unit of evidence: each training question first becomes one
lightweight PatchRecord. Clustering and merging happen over PatchRecords, not
over minibatch-level edits.

The key research claim is:

```text
Skill updates should be selected at the right abstraction level. A local repair
must first be supported by multiple typed question-level records, and a global
root edit should only be accepted if it improves validation.
```

## 2. Relationship to SkillOpt

V2 does not replace all of SkillOpt. It replaces the reflection and aggregation
portion of the loop:

```text
Original SkillOpt:
Rollout -> Minibatch Reflect -> Failure/Success Aggregate -> Select -> Update -> Gate

V2:
Repeated Rollout -> PatchRecord Generation -> Leaf/Root Merge -> Update -> Gate
```

V2 continues to reuse:

- dataset splits and batch construction;
- target rollout execution;
- skill application;
- validation gate;
- logging and output layout;
- final evaluation.

V2 no longer uses the original success/failure patch merge as its core update
mechanism. Instead:

```text
stable_success  -> no PatchRecord
unstable        -> PatchRecord
failure         -> PatchRecord
```

This is a finer distinction than SkillOpt's success/failure split.

## 3. High-Level Algorithm

For each training batch at step `t`:

```text
1. For each question x_i, run K independent rollouts under current skill S_t.
2. Compute empirical success q_i.
3. If q_i >= tau_succ, mark the question as stable_success and skip it.
4. If q_i < tau_succ, generate one lightweight PatchRecord R_i.
5. Cluster PatchRecords into type-guided leaf clusters.
6. Merge records inside each leaf cluster into one leaf patch.
7. Optionally self-check each leaf patch on its support questions.
8. Merge surviving leaf patches into one conservative root patch.
9. Validate root candidate on D_val.
10. If root fails, evaluate direct children, combine positive children once,
    and accept only if the combination passes validation.
```

The V2 tree is intentionally shallow:

```text
Leaf -> Root
```

Root children are leaf patches. There are no internal nodes in the default V2.
The direct-child fallback therefore evaluates leaves, not a deeper recursive
tree.

## 4. PatchRecord

PatchRecord should be lightweight. Its clustering fields must be stable and
short; diagnostic fields may exist but should not drive the primary grouping.

### 4.1 Algorithm-Facing Schema

```json
{
  "record_id": "R0001",
  "sample_id": "202512:37",
  "q_i": 0.3333,
  "status": "unstable",
  "question_type": "comparison_and_selection",
  "revision_type": "overgeneralization_control",
  "question_summary": "The task asks for the strongest valid statement among similar theorem claims.",
  "revision_summary": "Compare overstrong universal claims against weaker qualified claims before selecting.",
  "patch": {
    "op": "append",
    "target": "",
    "content": "When a choice uses universal wording such as every, all, or always, compare it against qualified alternatives before selecting it as the strongest provable statement."
  }
}
```

Required fields:

- `record_id`
- `sample_id`
- `q_i`
- `status`
- `question_type`
- `revision_type`
- `question_summary`
- `revision_summary`
- `patch`

Optional fields:

- `failure_subtype`
- `target_summary`
- `confidence`
- `debug_ref`

`debug_ref` can point to an artifact containing raw rollouts and full analysis.
Raw trajectories should not be embedded in the algorithm-facing record.

### 4.2 Status Values

```text
stable_success: q_i >= tau_succ
unstable:       0 < q_i < tau_succ
failure:        q_i = 0
```

Only `unstable` and `failure` records enter clustering.

### 4.3 Types vs Summaries

V2 uses both type labels and short summaries:

```text
question_type + revision_type:
    stable program-facing grouping signal

question_summary + revision_summary:
    semantic context for LLM clustering, leaf merge, and root merge
```

The primary grouping key is:

```text
(question_type, revision_type)
```

`target` is not part of the grouping key. It is too unstable and tends to
fragment otherwise compatible repairs.

`failure_subtype` may be used as an auxiliary signal inside a type group, but
it should not be required for all environments.

## 5. PatchRecord Generation

For each question, V2 provides the analyzer with K rollouts from the same
question:

```text
Question
Choices / task input
Current skill
K rollout summaries
Per-rollout hard/soft score
Correct answer or evaluator feedback if available
```

The analyzer returns exactly one of:

```json
{"no_patch": true, "reasoning": "..."}
```

or:

```json
{
  "question_type": "...",
  "revision_type": "...",
  "question_summary": "...",
  "revision_summary": "...",
  "patch": {...}
}
```

The prompt should emphasize:

- generate a skill-level repair, not a question-specific answer;
- do not include entity names, numeric constants, or the correct option label;
- prefer short summaries;
- choose labels from an allowed taxonomy;
- produce one minimal patch record.

### 5.1 Sample-Level Prompt Design

V2 should use a dedicated sample-level prompt rather than reusing the
minibatch failure analyst verbatim. The prompt can reuse the current
`analyst_error` taxonomy and edit schema, but the input unit is one question
with K repeated rollouts.

Common prompt inputs:

```text
sample_id
question / task input
choices or answer format constraints
current skill
K rollout summaries
per-rollout hard score and soft score
correct answer or evaluator feedback when available
allowed question_type labels
allowed revision_type labels
```

Common output contract:

```json
{
  "no_patch": false,
  "status": "unstable",
  "q_i": 0.3333,
  "question_type": "comparison_and_selection",
  "revision_type": "overgeneralization_control",
  "question_summary": "...",
  "revision_summary": "...",
  "patch": {
    "op": "append|insert_after|replace|delete",
    "target": "<optional exact target>",
    "content": "<minimal skill-level repair>"
  },
  "applicability": "...",
  "boundary": "..."
}
```

or:

```json
{
  "no_patch": true,
  "reasoning": "The failures do not share a skill-level repair."
}
```

Common prompt rules:

- Generate one PatchRecord at most for this question.
- The patch must repair a reusable behavior of the skill, not memorize this
  question.
- Do not mention sample-specific entity names, numbers, option labels, final
  answers, or theorem names unless they are part of a general format rule.
- Do not duplicate an instruction already present in the current skill.
- Prefer concrete operational rules over generic advice such as "think
  carefully", "verify the answer", or "avoid mistakes".
- If the best patch would be too abstract to change behavior, return
  `no_patch`.
- Keep `question_summary` and `revision_summary` short enough to support later
  clustering.
- Use the same protected-section constraint as the current merge prompts: never
  target content between `<!-- SLOW_UPDATE_START -->` and
  `<!-- SLOW_UPDATE_END -->`.

#### 5.1.1 Mixed Correct/Wrong Rollouts

When `0 < q_i < tau_succ`, the prompt should treat the question as unstable.
This case is especially valuable because correct rollouts reveal behavior that
should be preserved.

Prompt intent:

```text
Some repeated attempts are correct and some are wrong. Compare the successful
and failed attempts. Identify the smallest skill-level rule that would make the
successful behavior reliable without removing it.
```

Additional rules:

- Use successful trajectories as positive evidence for what should be kept.
- Use failed trajectories to locate the missing guard, check, decomposition, or
  boundary.
- Prefer stability repairs: verification, comparison discipline, evidence
  grounding, format enforcement, or ambiguity handling.
- If failures look like stochastic execution noise and the successful behavior
  already follows the skill, return `no_patch`.
- Set `status` to `unstable`.

This prompt should not ask for a broad rewrite. It should ask:

```text
What one concrete instruction would make this question type consistently follow
the successful pattern?
```

#### 5.1.2 All-Wrong Rollouts

When `q_i = 0`, the prompt should treat the question as a failure case. This is
weaker evidence than the mixed case because there is no successful behavior to
contrast against.

Prompt intent:

```text
All repeated attempts failed. Find the minimal skill-level repair that is
supported by the common failure mechanism across these attempts.
```

Additional rules:

- Prefer a patch only if the K failures share a common cause.
- Do not infer a broad new rule from one idiosyncratic wrong path.
- Do not encode the correct answer or any sample-specific cue.
- If the attempts fail for unrelated reasons, return `no_patch`.
- Set `status` to `failure`.

For all-wrong samples, the analyzer should be more conservative than for mixed
samples. A useful implementation detail is to add an internal confidence field
and later rank `unstable` records before `failure` records when the record
budget is exceeded.

## 6. LeafCluster

Leaf clusters group PatchRecords that represent the same local repair direction.

### 6.1 Full LeafCluster Artifact

```json
{
  "leaf_id": "L1",
  "question_type": "comparison_and_selection",
  "revision_type": "overgeneralization_control",
  "support_sample_ids": ["202512:37", "202601:24"],
  "record_ids": ["R0001", "R0007", "R0009"],
  "support_count": 3,
  "question_summary": "These questions ask for the strongest valid statement among close theorem options.",
  "revision_summary": "Avoid choosing overstrong options without checking qualified alternatives.",
  "target_summary": "choice comparison rules",
  "leaf_patch": {
    "op": "append",
    "target": "",
    "content": "..."
  },
  "applicability": "...",
  "boundary": "...",
  "self_check_status": "unchecked",
  "self_check_gain": null
}
```

### 6.2 Leaf Card for Root Merge

Root merge should not read full raw records. It receives compact leaf cards:

```json
{
  "leaf_id": "L1",
  "question_type": "comparison_and_selection",
  "revision_type": "overgeneralization_control",
  "support_count": 3,
  "question_summary": "...",
  "revision_summary": "...",
  "edit_summary": "...",
  "applicability": "...",
  "boundary": "...",
  "self_check_status": "passed",
  "self_check_gain": 0.25
}
```

## 7. Leaf Clustering

V2 uses a conservative two-stage clustering recipe:

```text
1. Group records by (question_type, revision_type).
2. Within each group, optionally split or merge using short summaries.
```

Default implementation can skip the second stage and use exact type groups.
The optional summary-aware stage is useful when a type group is too broad.

Suggested controls:

```yaml
type_guided_target_cluster_size: 6
type_guided_max_patch_records: 24
type_guided_max_leaf_clusters: 8
type_guided_summary_cluster: false
```

If `type_guided_summary_cluster=false`, V2 avoids LLM clustering and uses
deterministic type grouping.

If `type_guided_summary_cluster=true`, the cluster prompt should only see:

```text
record_id
question_type
revision_type
question_summary
revision_summary
patch summary
```

It should not see raw trajectories.

## 8. Leaf Patch Merge

For each leaf cluster, merge its PatchRecords into one leaf patch.

The leaf merge prompt should preserve:

- common repair intent;
- applicability;
- boundary;
- support record ids;
- source question and revision types.

It should avoid:

- memorizing sample facts;
- adding correct labels;
- merging incompatible target locations into a vague global rule.

If target locations conflict, the leaf may contain multiple edits, but the
leaf should still represent one local repair type.

### 8.1 Leaf Merge Prompt Design

The leaf prompt receives PatchRecords from one leaf cluster, not raw
trajectories. It can reuse the current `type_guided_leaf` prompt style, but it
should be explicit that the output must be both generalizable and concrete.

Prompt inputs:

```text
current skill
leaf_id
shared question_type
shared revision_type
PatchRecord cards:
  record_id
  sample_id
  q_i
  status
  question_summary
  revision_summary
  patch
  applicability
  boundary
```

Prompt goals:

- Merge records into one compact leaf patch for the shared repair direction.
- Deduplicate overlapping patches.
- Resolve conflicts conservatively.
- Preserve useful boundaries and applicability conditions.
- Discard records whose patches are task-specific, unsupported, or redundant
  with the current skill.
- If two target locations are incompatible but the repair type is still shared,
  return multiple edits inside the same leaf patch instead of producing a vague
  global edit.
- Make the patch operational: it should tell the agent what to check, compare,
  extract, compute, or constrain.

Anti-abstraction rule:

```text
Do not replace several concrete repairs with a broad sentence that would be
true for almost every task. A leaf patch should be general over its cluster,
but still specific enough to change behavior.
```

Suggested output:

```json
{
  "leaf_id": "L1",
  "reasoning": "<short merge rationale>",
  "question_type": "...",
  "revision_type": "...",
  "support_sample_ids": ["..."],
  "record_ids": ["..."],
  "applicability": "...",
  "boundary": "...",
  "conflict_resolution": "<what was merged, kept separate, or discarded>",
  "discarded_record_ids": ["..."],
  "edits": [
    {
      "op": "append|insert_after|replace|delete",
      "target": "<if needed>",
      "content": "<markdown>",
      "support_count": 3,
      "source_type": "failure",
      "question_type": "...",
      "revision_type": "...",
      "support_sample_ids": ["..."],
      "record_ids": ["..."]
    }
  ]
}
```

## 9. Optional Leaf Self-Check

Self-check should be configurable because it is expensive and may fail when
support items are unavailable. In V2-min, self-check is diagnostic only: it
records whether a leaf patch improves its own support questions, but it does
not decide whether the leaf can enter root merge or fallback.

Recommended config:

```yaml
type_guided_self_check: true
type_guided_self_check_gate: false
type_guided_self_check_required: false
type_guided_self_check_max_items: 5
type_guided_self_check_tau: 0.0
type_guided_leaf_revise: false
```

Self-check statuses:

```text
passed
failed
revised_passed
unchecked
```

Default policy:

```text
passed / revised_passed:
    recorded as positive diagnostic evidence

unchecked:
    recorded as missing diagnostic evidence

failed:
    recorded as negative diagnostic evidence, but not discarded in V2-min
```

Self-check procedure:

```text
1. Build support set X(C_j) from support_sample_ids.
2. Evaluate current skill S_t on X(C_j).
3. Evaluate S_t + leaf_patch on X(C_j).
4. Compute delta_self = score_after - score_before.
5. Mark passed if delta_self > tau_self.
```

V2-min should not use self-check as a hard gate. The root merge prompt may see
`self_check_status` and `self_check_gain` as weak evidence, but it should not
drop a leaf solely because self-check failed. Leaf revision can be added after
the check path is stable.

## 10. Root Merge

Root merge is conservative. It receives leaf cards and leaf patches, then
builds one root candidate.

Root merge should:

- preserve distinct leaves when abstraction would be too broad;
- merge only clearly shared repair principles;
- avoid introducing unsupported behavior;
- carry leaf ids into root edits;
- keep applicability and boundary text.

V2 should not use a free-form multi-level planner by default. The default root
structure is:

```json
{
  "root_patch": {...},
  "children": ["L1", "L2", "L3"]
}
```

### 10.1 Root Merge Prompt Design

The root prompt receives compact leaf cards plus leaf patches. It should not see
raw trajectories and should not invent new repair directions. Its job is to
build the best single candidate skill update that can be validated as a whole.

Prompt inputs:

```text
current skill
leaf cards:
  leaf_id
  question_type
  revision_type
  support_count
  question_summary
  revision_summary
  applicability
  boundary
  self_check_status
  self_check_gain
leaf patches
```

Prompt goals:

- Cover the surviving leaves as completely as possible without weakening their
  boundaries.
- Merge leaves only when they clearly express the same higher-level rule.
- Preserve separate edits when abstraction would become vague or overbroad.
- Resolve contradictions conservatively; prefer the narrower supported rule.
- Carry `leaf_ids`, `support_sample_ids`, `question_type`, and `revision_type`
  into root edits.
- Avoid adding behavior that no leaf supports.
- Avoid redundant filler instructions that would not affect the target agent.

Root-level anti-abstraction rule:

```text
A root patch may be more comprehensive than a leaf patch, but it must still be
testable as a behavioral instruction. If the merged sentence would fit any
dataset or any error, it is too abstract.
```

Suggested output:

```json
{
  "reasoning": "<root merge decisions>",
  "children": ["L1", "L2"],
  "conflict_resolution": "<how conflicts among leaves were handled>",
  "edits": [
    {
      "op": "append|insert_after|replace|delete",
      "target": "<if needed>",
      "content": "<markdown>",
      "support_count": 5,
      "source_type": "failure",
      "question_type": "<dominant or combined type>",
      "revision_type": "<dominant or combined type>",
      "applicability": "...",
      "boundary": "...",
      "support_sample_ids": ["..."],
      "leaf_ids": ["L1", "L2"]
    }
  ]
}
```

## 11. Validation and One-Level Fallback

V2 uses restricted pruning:

```text
1. Apply root_patch to S_t and evaluate on D_val.
2. If root improves over current, accept root.
3. If root fails, evaluate each direct child leaf on D_val.
4. Keep leaves with validation gain > tau_child.
5. Combine kept leaves into one child_candidate.
6. Evaluate child_candidate once.
7. If child_candidate passes, accept it.
8. Otherwise reject the round and keep S_t.
```

There is no recursive fallback:

```text
root -> direct children -> stop
```

For V2's default two-level tree, root children are leaves.

Therefore, in the default design, if root fails, V2 will check every leaf
cluster's candidate on validation one by one. The saved fallback artifact should
contain each leaf's validation score:

```json
{
  "root_status": "rejected",
  "leaf_scores": [
    {
      "leaf_id": "L1",
      "candidate_hash": "...",
      "selection_hard": 0.2222,
      "selection_soft": 0.31,
      "gain": 0.0556,
      "kept": true
    }
  ],
  "kept_leaf_ids": ["L1"],
  "combo_status": "accepted"
}
```

This is expensive but simple and reliable. To control cost, V2 can optionally
cap fallback evaluation:

```yaml
type_guided_fallback_eval_all_leaves: true
type_guided_fallback_top_k: 0       # 0 means no cap
type_guided_fallback_tau_child: 0.0
```

Recommended default:

```text
type_guided_fallback_eval_all_leaves=true
```

For the first implementation, evaluating all leaves is preferable because it
gives clear evidence about whether the root failed due to over-merge or because
the leaf patches themselves were weak. Later, `fallback_top_k` can rank leaves
by self-check gain, support count, and unstable-sample count before validation.

## 12. Success Handling

V2 does not create typed success patches.

Stable successes are handled by skipping PatchRecord generation:

```text
q_i >= tau_succ -> stable_success -> no patch
```

Unstable samples are more useful than ordinary success samples:

```text
0 < q_i < tau_succ -> unstable -> PatchRecord
```

This preserves the useful signal that the current skill sometimes works but is
not reliable enough.

An optional future extension may summarize stable success examples as boundary
context, but this should not be part of V2-min.

## 13. Caching and Parallelism

V2 increases training cost because it introduces repeated rollouts,
sample-level patch generation, optional leaf self-check, root validation, and
leaf fallback validation. The implementation should treat cache and parallelism
as part of the method, not as an afterthought.

### 13.1 Cache Keys

Use deterministic JSON artifacts and cache keys based on the actual inputs to
each stage.

Repeated rollout cache:

```text
(sample_id, skill_hash, repeat_id, split, target_model, max_completion_tokens,
 rollout_prompt_version, environment_version)
```

PatchRecord cache:

```text
(sample_id, skill_hash, rollout_result_hashes, patch_record_prompt_version,
 optimizer_model)
```

Leaf merge cache:

```text
(skill_hash, sorted_record_ids, record_content_hash, leaf_prompt_version,
 optimizer_model)
```

Root merge cache:

```text
(skill_hash, sorted_leaf_ids, leaf_content_hash, root_prompt_version,
 optimizer_model)
```

Leaf self-check cache:

```text
(skill_hash, leaf_patch_hash, sorted_support_sample_ids, target_model,
 max_completion_tokens)
```

Validation cache:

```text
candidate_skill_hash -> (selection_hard, selection_soft)
```

The current trainer already has an in-memory selection cache (`sel_cache`) for
candidate hashes. V2 should reuse and extend that pattern for root, leaf, and
combined-child candidates. If cross-run reuse is desired, the same records can
also be written under `type_guided_cache_dir`.

### 13.2 Parallel Stages

The following stages are naturally parallel:

- repeated rollouts across `(sample_id, repeat_id)`;
- PatchRecord generation across samples;
- leaf merge across clusters;
- optional leaf self-check across leaves;
- fallback leaf validation across leaves, if the environment supports safe
  concurrent evaluation directories.

Recommended worker knobs:

```yaml
type_guided_rollout_workers: null       # default to env workers
type_guided_patch_record_workers: null  # default to analyst_workers
type_guided_leaf_merge_workers: 4
type_guided_self_check_workers: 4
type_guided_fallback_workers: 4
```

The implementation should still respect the target model's API concurrency
limit. If target rollout and analyst calls share one backend quota, use the
smaller of the requested worker count and the configured API limit.

### 13.3 Budget Controls

V2-min should include hard caps so a bad batch cannot explode cost:

```yaml
type_guided_rollout_repeats: 3
type_guided_max_patch_records: 24
type_guided_max_leaf_clusters: 8
type_guided_self_check_max_items: 5
type_guided_fallback_top_k: 0
type_guided_cache_dir: null
type_guided_reuse_rollouts: true
```

When too many PatchRecords are available, rank them before clustering:

```text
1. unstable records before all-wrong failure records;
2. higher support or confidence before lower support;
3. diverse (question_type, revision_type) coverage;
4. shorter, more concrete patch summaries before vague ones.
```

### 13.4 Cache Safety

Cache writes should be atomic:

```text
write temp file -> fsync/close -> rename to final path
```

Do not cache malformed LLM JSON, failed evaluations, or partial rollout files
as valid results. Every cache hit should be logged in the step artifact so a
run can be audited later.

## 14. Configuration

Recommended V2-min config:

```yaml
optimizer:
  type_guided_merge: true
  type_guided_version: v2
  type_guided_rollout_repeats: 3
  type_guided_tau_succ: 1.0
  type_guided_max_patch_records: 24
  type_guided_target_cluster_size: 6
  type_guided_max_leaf_clusters: 8
  type_guided_summary_cluster: false
  type_guided_self_check: true
  type_guided_self_check_gate: false
  type_guided_self_check_required: false
  type_guided_self_check_max_items: 5
  type_guided_self_check_tau: 0.0
  type_guided_leaf_revise: false
  type_guided_tree_depth: 1
  type_guided_leaf_fallback: true
  type_guided_fallback_eval_all_leaves: true
  type_guided_fallback_top_k: 0
  type_guided_fallback_tau_child: 0.0
  type_guided_reuse_rollouts: true
  type_guided_cache_dir: null
  type_guided_patch_record_workers: null
  type_guided_leaf_merge_workers: 4
  type_guided_self_check_workers: 4
  type_guided_fallback_workers: 4
```

Recommended first fair comparison:

```text
A. Original SkillOpt aggregate
B. V1 typed edit aggregate
C. V2 per-question PatchRecord aggregate
```

Keep these matched:

```text
num_epochs
batch_size
target model
optimizer model
selection set
max_completion_tokens
slow_update flag
meta_skill flag
```

## 15. Artifacts

Each step should save:

```text
type_guided_v2_rollouts.jsonl
type_guided_v2_patch_records.json
type_guided_v2_leaf_clusters.json
type_guided_v2_self_check.json
type_guided_v2_root.json
type_guided_v2_fallback.json
type_guided_v2_cache_report.json
```

These artifacts should make the following questions answerable:

- Which questions generated PatchRecords?
- Which stable successes were skipped?
- Which records formed each leaf?
- Which leaves passed self-check?
- Did root fail because it was too broad?
- Which leaves survived fallback?
- Which stages used cache hits?
- Which stages dominated runtime?

## 16. Minimal Implementation Milestones

### Milestone 1: PatchRecord Mode

- Add per-question repeated rollout support.
- Generate one PatchRecord per failed/unstable question.
- Save records.
- No clustering beyond `(question_type, revision_type)`.

### Milestone 2: Leaf and Root Merge

- Merge PatchRecords into leaf patches.
- Merge leaves into conservative root.
- Reuse existing update and gate logic.

### Milestone 3: One-Level Fallback

- If root fails, evaluate leaves.
- Combine positive leaves once.
- Reject if combination fails.

### Milestone 4: Optional Self-Check

- Evaluate leaf patches on support samples.
- Mark `passed`, `failed`, or `unchecked`.
- Record self-check scores and statuses.
- Do not drop failed leaves in V2-min.

### Milestone 5: Summary-Aware Clustering

- Optional LLM clustering within type groups.
- Use short summaries only.
- Keep deterministic grouping as default.

### Milestone 6: Persistent Cache and Parallel Fallback

- Add persistent cache directory.
- Parallelize patch-record generation and leaf merges.
- Optionally parallelize fallback leaf validation.
- Save runtime and cache-hit reports.

## 17. Non-Goals for V2

V2 should not implement:

- deep merge trees;
- recursive validation fallback;
- typed success patches;
- target as a hard clustering key;
- full raw trajectory clustering;
- repeated leaf revision loops.

These are reserved for later versions after V2-min is stable.

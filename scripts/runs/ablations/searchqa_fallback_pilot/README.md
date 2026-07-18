# SearchQA fallback pilot

This is a controlled 2×2 pilot. All four runs use SearchQA train400,
batch40, repeats3, one epoch, seed42, full val200 root gating, and the same
DeepSeek official optimizer endpoint.

| Script | Tree depth | Fallback |
|---|---:|---:|
| `run_01_d2_fallback_off.sh` | 2 | off |
| `run_02_d2_fallback_on.sh` | 2 | on |
| `run_03_d3_fallback_off.sh` | 3 | off |
| `run_04_d3_fallback_on.sh` | 3 | on |
| `run_05_d3_clustering_on_tail_off_fallback_off.sh` | 3 | off |
| `run_06_d3_clustering_on_min_support1_all_clusters_tail_off_fallback_off.sh` | 3 | off |
| `run_07_d3_clustering_on_tail_on_fallback_off.sh` | 3 | off |
| `run_08_d3_min_support1_cap8_validated_fuse_no_edit_clip.sh` | 3 | on |

`run_05` is a tree-shape pilot. It enables global PatchRecord clustering with
target/max cluster sizes `2/4`, and keeps both child fallback and the
epoch-level tail bank off. It isolates whether the revised Mid planner builds
meaningful multi-Leaf abstractions.

`run_06` starts from `run_05`, sets per-step `min_support=1`, and raises the
Leaf cap from 8 to 40. Since each step already caps PatchRecords at 40, this
implements the intended "all clusters enter the tree" treatment. It is not a
strict single-variable ablation: retaining the old cap of 8 could still drop
clusters even when `min_support=1`.

`run_07` is the clean tail-bank comparison against `run_05`: the per-step
tree remains `min_support=2`, Leaf cap 8, clustering target/max 2/4, depth 3,
and fallback off. The tail bank collects low-support records during all ten
steps and runs once at the end of the single SearchQA epoch. Tail groups need
support 2 and occurrences in at least two distinct steps before the tail
tree and full-val gate are attempted.

`run_08` is the first combined tree-plus-fusion pilot after strengthening the
Mid prompts. It keeps low-support leaves but caps them at 8, enables rejected-root
child fallback, uses the LLM `validated_frontier_fuse` mode, and sets a constant
edit budget of 64 so post-root clipping is effectively disabled for this run.

When fallback is on, a rejected root selects at most four children. At each
step, one deterministic random sample of 40 val items is shared by current
and every child. Current is normally sliced from its prior full-val per-item
cache. Kept children are reconciled, then the combination must pass the full
val200 gate.

## TEST policy

The pilot launchers default to `EVAL_TEST=true` and the complete SearchQA
test1400. Set `PILOT_EVAL_TEST=false` only for an explicitly training-only
run. Such a run remains `test_pending` and must be evaluated with
`run_test_three_best_skills_h20.sh` (or an equivalent `eval_only.py` run)
before comparing experiment quality.

The three runs completed on 2026-07-17 with test disabled are listed in the
H20 supplement script. The fourth run should keep the new default so it
tests automatically.

Run one command on each resource-pool machine:

```bash
export DEEPSEEK_API_KEY=...
export CUDA_VISIBLE_DEVICES=0
bash scripts/runs/ablations/searchqa_fallback_pilot/run_01_d2_fallback_off.sh
```

Use `run_02...`, `run_03...`, and `run_04...` on the other three machines.
Use a different `VLLM_PORT` only if two runs share one host.

Render commands without starting vLLM or training:

```bash
DRY_RUN=1 START_VLLM=0 \
  bash scripts/runs/ablations/searchqa_fallback_pilot/run_02_d2_fallback_on.sh
```

## Supplement the three missing tests on one H20

The evaluator starts one Qwen3.5-4B vLLM and evaluates the three completed
best skills sequentially on the full SearchQA test1400:

```bash
export CUDA_VISIBLE_DEVICES=0
bash scripts/runs/ablations/searchqa_fallback_pilot/run_test_three_best_skills_h20.sh
```

The H20 fast profile is 256 workers/sequences, 65536 batched tokens,
32768 model length, thinking off, and a 4096 completion cap. If the specific
machine is unstable, retry the same result directory with:

```bash
WORKERS=192 VLLM_MAX_NUM_SEQS=192 \
RESULT_ROOT=/path/from/the/first/attempt \
bash scripts/runs/ablations/searchqa_fallback_pilot/run_test_three_best_skills_h20.sh
```

`eval_only.py` is resume-aware through each run's `results.jsonl`. The script
writes `three_test_summary.md`, `.csv`, and `.json`. To include the fourth run
later, pass a TSV using `SKILL_MANIFEST`; its columns are
`run_name` and `relative_skill_path` relative to `OUTPUTS_ROOT`.

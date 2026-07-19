# SearchQA fixed-evidence PatchTree verdict

This suite does not train SearchQA again. It reuses one completed P6 output,
freezes its PatchRecords and saved Tree/Leaf nodes, compiles only the missing
Flat Root and frozen-Leaf Root, and performs one paired evaluation.

## 1. Prepare fixed candidates

Use the same optimizer provider that you want recorded in the replay artifact.

DeepSeek official:

```bash
export DEEPSEEK_API_KEY='...'
bash scripts/runs/ablations/searchqa_tree_verdict/run_01_prepare_fixed_candidates.sh
```

Volcano Ark:

```bash
export ARK_API_KEY='...'
OPTIMIZER_PROVIDER=ark \
  bash scripts/runs/ablations/searchqa_tree_verdict/run_01_prepare_fixed_candidates.sh
```

The default frozen directory is:

```text
outputs/searchqa_tree_verdict_p6_fixed
```

If it already contains `replay_manifest.json`, the launcher refuses to compile
another pair. Use the existing candidates.

## 2. Run the one-GPU verdict

```bash
CUDA_VISIBLE_DEVICES=0 \
REPLAY_DIR=/ai-app-vepfs/zhangfuhao/skill/PatchTree/outputs/searchqa_tree_verdict_p6_fixed \
bash scripts/runs/ablations/searchqa_tree_verdict/run_02_eval_one_gpu.sh
```

The evaluator uses one local Qwen3.5-4B vLLM with temperature 0, max output
4096, 128 workers/sequences, full val200, and full test1400.

It evaluates G0 Parent, G1 Flat, G2 frozen-Leaf Root, and G3 saved Full Tree.
It then evaluates the existing natural Root-reject step and only runs the G4
TEST if:

- the Root still fails full val under the fixed protocol;
- at least two selected Mid children have complementary repairs;
- their deterministic combination passes full val.

No seed, subset, top-k, or step is retried after failure.

## Dry run

```bash
DRY_RUN=1 \
  bash scripts/runs/ablations/searchqa_tree_verdict/run_01_prepare_fixed_candidates.sh

DRY_RUN=1 \
  bash scripts/runs/ablations/searchqa_tree_verdict/run_02_eval_one_gpu.sh
```

## Important output files

```text
<REPLAY_DIR>/replay_manifest.json
<RESULT_ROOT>/verdict/main_verdict.json
<RESULT_ROOT>/verdict/main_verdict.md
<RESULT_ROOT>/verdict/topdown_verdict.json
<RESULT_ROOT>/verdict/topdown_verdict.md
```

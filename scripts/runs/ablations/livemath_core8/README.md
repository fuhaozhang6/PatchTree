# LiveMath core-8 ablations: two runs per L20

The eight configurations are packed into four resource-pool launchers. Each
launcher starts exactly one local Qwen3.5-4B vLLM service on one L20 and runs
two LiveMath training processes concurrently against that shared endpoint.

| L20 script | Concurrent experiment 1 | Concurrent experiment 2 |
|---|---|---|
| `run_01_r1_r4.sh` | repeats=1, batch=8, depth=2 | repeats=4, batch=8, depth=2 |
| `run_02_r2_base.sh` | repeats=2, batch=8, depth=2 | baseline: repeats=3, batch=8, depth=2 |
| `run_03_b4_b32.sh` | repeats=3, batch=4, depth=2 | repeats=3, batch=32, depth=2 |
| `run_04_b16_d3.sh` | repeats=3, batch=16, depth=2 | repeats=3, batch=8, depth=3 |

The heavy and light configurations are paired to balance wall-clock time. Each
experiment permits 96 target workers and 32 PatchRecord analyst workers. The
actual LiveMath rollout concurrency is bounded by `batch_size * repeats`, so
the four pair peaks are 40, 40, 108, and 72 requests. All stay below the
validated L20/vLLM throughput peak at `VLLM_MAX_NUM_SEQS=128`.

The 96-worker cap matters for the `batch=32, repeats=3` run: it lets all 96
flattened rollout items run in one client wave. A 48-worker cap would split
that rollout into two waves. PatchRecord workers are 32 because one step can
produce at most 32 records; across all eight simultaneous runs, the theoretical
record-analysis peak is 92, below the official DeepSeek stable point of 256.

Common controls: 4 epochs, seed 42, DeepSeek official `deepseek-v4-pro`,
fallback top-k 4, clustering off, tail bank off, and test evaluation deferred.
Model selection uses the 18-item LiveMath validation split.

The benchmark-derived concurrency policy is documented in
[`docs/throughput_resource_allocation_20260716.md`](../../../../docs/throughput_resource_allocation_20260716.md).

## Launch on four separate one-L20 workers

Run one command on each resource-pool worker:

```bash
export DEEPSEEK_API_KEY='...'
bash scripts/runs/ablations/livemath_core8/run_01_r1_r4.sh
```

```bash
export DEEPSEEK_API_KEY='...'
bash scripts/runs/ablations/livemath_core8/run_02_r2_base.sh
```

```bash
export DEEPSEEK_API_KEY='...'
bash scripts/runs/ablations/livemath_core8/run_03_b4_b32.sh
```

```bash
export DEEPSEEK_API_KEY='...'
bash scripts/runs/ablations/livemath_core8/run_04_b16_d3.sh
```

The resource scheduler normally sets `CUDA_VISIBLE_DEVICES`. Override
`MODEL_PATH` if the local Qwen mount differs from the project default.

## Validation and later seeds

Dry-run a pair without an API key or GPU service:

```bash
DRY_RUN=1 bash scripts/runs/ablations/livemath_core8/run_01_r1_r4.sh
```

Use `ABLATION_SEED=43` or `ABLATION_SEED=44` for later replications. Set
`ABLATION_EVAL_TEST=true` only for intentional final test evaluation.

## Evaluate the eight saved best skills on TEST

Run all eight `best_skill.md` files sequentially on the full 124-item LiveMath
test split, sharing one local Qwen3.5-4B/vLLM service:

```bash
bash scripts/runs/ablations/livemath_core8/run_test_core8_best_skills.sh
```

This is eval-only: it does not call the optimizer or resume training. It
writes an independent rollout directory for every skill and aggregates the
results into `core8_test_summary.md`, `.csv`, and `.json`. The default input
root is `/ai-app-vepfs/zhangfuhao/skill/SkillOpt-Tree/outputs/livemath_core8_pairs`.

Useful overrides:

```bash
# Smoke-test 20 test examples.
TEST_ENV_NUM=20 bash scripts/runs/ablations/livemath_core8/run_test_core8_best_skills.sh

# Reuse an already-running OpenAI-compatible Qwen endpoint.
START_VLLM=0 QWEN_CHAT_BASE_URL=http://127.0.0.1:8000/v1 \
  bash scripts/runs/ablations/livemath_core8/run_test_core8_best_skills.sh
```

The test timeout defaults to 600 seconds because the pilot logs showed that
some 300-second requests were scored as wrong even though vLLM later returned
HTTP 200. Set `TARGET_QWEN_CHAT_TIMEOUT_SECONDS=300` and
`LIVEMATH_EXEC_TIMEOUT=300` to reproduce the training-time deadline exactly.

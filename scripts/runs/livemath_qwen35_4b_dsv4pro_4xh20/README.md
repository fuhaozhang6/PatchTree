# LiveMath: Qwen3.5-4B target + DeepSeek-v4-pro optimizer on 4 H20

One custom resource task uses exactly four H20 GPUs. It starts four independent
single-GPU vLLM servers and runs a serial case queue on each GPU. Submit one
resource task per training seed; do not start all three at once with the default
optimizer concurrency.

## Formal runs

Run these as three separate custom tasks:

```bash
export DEEPSEEK_API_KEY='...'
SEED=42 bash scripts/runs/livemath_qwen35_4b_dsv4pro_4xh20/run_seed_4xh20.sh
SEED=43 bash scripts/runs/livemath_qwen35_4b_dsv4pro_4xh20/run_seed_4xh20.sh
SEED=44 bash scripts/runs/livemath_qwen35_4b_dsv4pro_4xh20/run_seed_4xh20.sh
```

The resource scheduler may set `CUDA_VISIBLE_DEVICES`; the launcher requires
exactly four entries and binds one vLLM server to each entry.

## Queue assignment

| GPU queue | Cases, run serially |
|---:|---|
| 0 | `rollout_r8`, `dynamic_virtual_root`, `rollout_r2`, `system_prompt_only` |
| 1 | `batch_12`, `full`, `flat_fuse_fixed_real_root`, `init_skill_only` |
| 2 | `cluster_random`, `fallback_children`, `merge_concat` |
| 3 | `cluster_success_aware`, `fallback_internal`, `fallback_none`, `batch_35` |

`flat_fuse_fixed_real_root` is one canonical run with two reporting roles:
the flat one-shot leaf fusion ablation and the legacy fixed-real-root reference
are the same mechanism, so the suite does not train it twice.

## Validation before a formal task

Render every command without starting vLLM or calling the API:

```bash
DRY_RUN=1 SEED=42 \
  bash scripts/runs/livemath_qwen35_4b_dsv4pro_4xh20/run_seed_4xh20.sh
```

Run one short case on each GPU:

```bash
export DEEPSEEK_API_KEY='...'
SMOKE=1 SEED=42 \
  bash scripts/runs/livemath_qwen35_4b_dsv4pro_4xh20/run_seed_4xh20.sh
```

The smoke suite uses one epoch and eight train/validation/test examples by
default. Override with `SMOKE_EPOCHS`, `SMOKE_TRAIN_SIZE`,
`SMOKE_SEL_ENV_NUM`, and `SMOKE_TEST_ENV_NUM`.

## Concurrency

The defaults are:

```text
four simultaneous training processes
48 optimizer analyst workers per process (theoretical total 192)
24 PatchRecord workers per process
96 target workers per GPU
128 vLLM maximum sequences per GPU
```

If two four-GPU custom tasks must run simultaneously, set
`ANALYST_WORKERS=24` in both tasks. `API_MAX_CONCURRENCY` is a per-process
sanity ceiling, not an account-wide semaphore.

Successful cases create `.suite_complete` below their output directory.
Rerunning the same seed skips completed cases and lets the trainer resume
partially completed cases. `FORCE_RERUN=1` bypasses only the suite marker; the
trainer's own resume state still applies. Use a new `RUN_TAG` when a completely
fresh rerun is required.

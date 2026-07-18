# LiveMath follow-up ablations: two L20 workers

This suite extends the completed core-8 study with two repeat-count points and
two interaction runs. Start one launcher on each separate one-L20 resource-pool
worker. Each launcher starts one local Qwen3.5-4B vLLM endpoint and runs its two
experiments concurrently against that endpoint.

| L20 | Launcher | Experiment 1 | Experiment 2 | Peak target requests |
|---|---|---|---|---:|
| 1 | `run_01_r5_r4b16d3.sh` | `r5_b8_d3` | `r4_b16_d3` | 40 + 64 = 104 |
| 2 | `run_02_r8_r4b16d2.sh` | `r8_b8_d3` | `r4_b16_d2` | 64 + 64 = 128 |

The first three requested configurations are `r5_b8_d3`, `r8_b8_d3`, and
`r4_b16_d3`. The fourth slot is `r4_b16_d2`: comparing it with `r4_b16_d3`
isolates the depth effect under the same high-repeat, large-batch setting.

Common controls remain identical to core-8: four epochs, seed 42, full train
split, 18-item validation selection, fixed edit budget 4, minimum support 2,
fallback top-k 4, clustering off, tail bank off, and deferred test evaluation.
The PatchRecord cap remains 32: repeated rollouts are grouped by original
sample before record generation, so these runs can produce at most 8 or 16
PatchRecords per step rather than `batch_size * repeats` records.

Run on worker 1:

```bash
export DEEPSEEK_API_KEY='...'
bash scripts/runs/ablations/livemath_followup4/run_01_r5_r4b16d3.sh
```

Run on worker 2:

```bash
export DEEPSEEK_API_KEY='...'
bash scripts/runs/ablations/livemath_followup4/run_02_r8_r4b16d2.sh
```

Outputs are written below `outputs/livemath_followup4_pairs/`, with matching
logs below `logs/livemath_followup4_pairs/`. Override `ABLATION_SEED` for a
replication. Set `DRY_RUN=1` to validate command construction without starting
vLLM or training.

## Evaluate the completed follow-up runs on TEST

The completed seed-42 runs from 2026-07-17 were stored under the earlier
`outputs/livemath_core8_pairs/` root. Evaluate their four saved best skills
sequentially on the full 124-item test split with:

```bash
bash scripts/runs/ablations/livemath_followup4/run_test_followup4_best_skills.sh
```

The evaluator starts one local vLLM service, reuses it for all four skills, and
writes `followup4_test_summary.md`, `.csv`, and `.json`. Use
`TEST_ENV_NUM=20` for a smoke test or `START_VLLM=0` with
`QWEN_CHAT_BASE_URL=...` to reuse an existing endpoint.

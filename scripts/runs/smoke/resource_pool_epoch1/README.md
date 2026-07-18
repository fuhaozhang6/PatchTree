# Minimal one-epoch dataset smoke tests

These scripts validate that the five not-yet-confirmed datasets can complete a
real PatchTree-v4 training epoch. ALFWorld is intentionally excluded because
`scripts/runs/alfworld/run_alfworld_l20_qwen35_4b_smoke.sh` has already been
verified successfully.

`LIMIT=2` truncates each loaded train/val/test split to two items. With
`BATCH_SIZE=2`, `NUM_EPOCHS=1`, and `TRAIN_SIZE=0`, the trainer resolves a
two-item training split and completes exactly one training step: a complete
epoch over the smoke split. Two items are the smallest useful setting for
`type_guided_min_support=2`.

The tests keep repeated rollout, PatchRecord generation, tree aggregation,
validation gate, and top-1 fallback enabled. Test evaluation is disabled.

## Run on three separate one-L20 workers

SearchQA through the official DeepSeek endpoint:

```bash
export DEEPSEEK_API_KEY='...'
bash scripts/runs/smoke/resource_pool_epoch1/01_searchqa_deepseek_official.sh
```

DocVQA and SpreadsheetBench share one L20/vLLM and use Ark:

```bash
export ARK_API_KEY='...'
bash scripts/runs/smoke/resource_pool_epoch1/02_docvqa_spreadsheet_ark.sh
```

LiveMath and OfficeQA share one L20/vLLM and use Ark:

```bash
export ARK_API_KEY='...'
bash scripts/runs/smoke/resource_pool_epoch1/03_livemath_officeqa_ark.sh
```

Each run uses a unique timestamped output directory. After training, the
verifier requires `summary.json`, one completed epoch, one total step, two
loaded train items, non-empty rollout output, and `type_guided_version=v2`.
It also checks DocVQA image metadata and OfficeQA local-document evidence. An
OfficeQA all-zero score with non-empty answers is reported as a warning for
manual inspection; empty answers or missing evidence fail the smoke test.
SpreadsheetBench receives a 1200-second task timeout because its multi-turn
code-generation episodes exceeded 600 seconds on the L20 smoke profile.

## Dry-run

No API key or GPU is required:

```bash
DRY_RUN=1 bash scripts/runs/smoke/resource_pool_epoch1/01_searchqa_deepseek_official.sh
DRY_RUN=1 bash scripts/runs/smoke/resource_pool_epoch1/02_docvqa_spreadsheet_ark.sh
DRY_RUN=1 bash scripts/runs/smoke/resource_pool_epoch1/03_livemath_officeqa_ark.sh
```

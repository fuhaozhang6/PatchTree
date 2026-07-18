# Four-resource L20 launch plan

Before a formal run, follow the unified resource and training checklist in
[`docs/throughput_resource_allocation_20260716.md`](../../../docs/throughput_resource_allocation_20260716.md).
It includes concurrency limits, timeout layering, dataset-specific settings,
smoke-test pass criteria, and the OfficeQA/SpreadsheetBench failure cases found
in the 2026-07-17 validation.

These four scripts are intended to be launched separately on four resource-pool
workers. Each worker must expose exactly one L20 GPU. Every script starts its
own single-GPU vLLM service and then runs its assigned dataset group.

| Script | Datasets | Optimizer source | Local target concurrency |
|---|---|---|---:|
| `01_searchqa_deepseek_official.sh` | SearchQA | DeepSeek official | 128 |
| `02_alfworld_deepseek_official.sh` | ALFWorld | DeepSeek official | 24 workers / 96 max seqs |
| `03_docvqa_spreadsheet_ark.sh` | DocVQA + SpreadsheetBench | Volcano Ark | 48 + 48 workers / 128 max seqs |
| `04_livemath_officeqa_ark.sh` | LiveMath + OfficeQA | Volcano Ark | 48 + 48 workers / 128 max seqs |

## Launch on separate resource-pool workers

SearchQA worker:

```bash
export DEEPSEEK_API_KEY='...'
bash scripts/runs/resource_pool_4x_l20/01_searchqa_deepseek_official.sh
```

ALFWorld worker:

```bash
export DEEPSEEK_API_KEY='...'
bash scripts/runs/resource_pool_4x_l20/02_alfworld_deepseek_official.sh
```

DocVQA + SpreadsheetBench worker:

```bash
export ARK_API_KEY='...'
bash scripts/runs/resource_pool_4x_l20/03_docvqa_spreadsheet_ark.sh
```

LiveMath + OfficeQA worker:

```bash
export ARK_API_KEY='...'
bash scripts/runs/resource_pool_4x_l20/04_livemath_officeqa_ark.sh
```

Resource schedulers normally set `CUDA_VISIBLE_DEVICES` automatically. If a
worker does not, specify the single local GPU explicitly:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/runs/resource_pool_4x_l20/01_searchqa_deepseek_official.sh
```

All scripts default to the model path
`/ai-app-vepfs/modelscope/models/Qwen/Qwen3.5-4B`. Override `MODEL_PATH` when a
resource worker mounts it elsewhere. `VLLM_PORT` defaults to `59317`; because
the scripts run on different hosts they may safely use the same port.

## Dry-run validation

No API key or GPU service is required for dry-run:

```bash
DRY_RUN=1 bash scripts/runs/resource_pool_4x_l20/01_searchqa_deepseek_official.sh
DRY_RUN=1 bash scripts/runs/resource_pool_4x_l20/02_alfworld_deepseek_official.sh
DRY_RUN=1 bash scripts/runs/resource_pool_4x_l20/03_docvqa_spreadsheet_ark.sh
DRY_RUN=1 bash scripts/runs/resource_pool_4x_l20/04_livemath_officeqa_ark.sh
```

The API source is set explicitly by every script. Existing
`AZURE_OPENAI_ENDPOINT` or `AZURE_OPENAI_API_KEY` values in the worker
environment are intentionally replaced to prevent cross-source routing.

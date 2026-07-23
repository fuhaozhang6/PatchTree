# SkillOpt-Tree comparison with DeepSeek-v4 flash (target) + pro (optimizer)

These launchers run the SkillOpt-Tree training pipeline with **both** models
served through the DeepSeek official API — there is no local vLLM service, no
GPU allocation, and no tensor-parallel setup:

- target / student: `deepseek-v4-flash` (`--target_backend openai_chat`);
- optimizer / teacher: `deepseek-v4-pro` (`--optimizer_backend openai_chat`).

Both models share one endpoint (`https://api.deepseek.com`) and one API key;
only the model name differs. SkillOpt method parameters remain inherited from
each dataset's `configs/<dataset>/default.yaml`.

DeepSeek thinking is disabled for both models (`DEEPSEEK_THINKING=disabled`),
since `_deepseek_extra_body()` applies the flag to every `deepseek-*` call. This
keeps flash/pro aligned with the non-thinking baselines.

The scripts reuse materialized data from this repo's own `data/` directory by
default. Override `DATA_PROJECT_ROOT` or `DATA_ROOT` if the data is mounted
elsewhere.

## SkillOpt-Tree CLI note

SkillOpt-Tree's `scripts/train.py` does **not** define `--exec_timeout` /
`--llm_timeout` flags — passing them directly would fail argparse. It accepts
`--cfg-options` and exposes `env.exec_timeout` / `env.llm_timeout` in the
structured config. The shared `_common.sh` intercepts the reference-style
`--exec_timeout N` / `--llm_timeout N` arguments in `run_dataset` and rewrites
them into `--cfg-options env.exec_timeout=N env.llm_timeout=N`. It always also
appends `env.max_completion_tokens=<TARGET_MAX_COMPLETION_TOKENS>` to cap the
student generations. Everything is packed into a single trailing
`--cfg-options` block (which is `nargs="+"`), so per-dataset launchers stay
simple and you can still pass `SEARCHQA_EXEC_TIMEOUT`, `GROUP_EXEC_TIMEOUT`,
etc. as before.

## Resource profile

Because both models are API-served, there is no GPU/vLLM footprint. Wall-time
and cost are governed by the DeepSeek API rate limits and by the worker counts
below. Adjust workers to stay within your account's concurrency quota.

| Script | Datasets | Target workers | Analyst workers | Note |
|---|---|---:|---:|---|
| `01_searchqa.sh` | searchqa | 24 | config default | single dataset |
| `02_alfworld.sh` | alfworld | 32 env workers, 8 API workers | config default | multi-turn agent |
| `03_docvqa_spreadsheet.sh` | docvqa + spreadsheetbench | 16 + 16 | 16 + 16 | parallel pair |
| `04_livemath_officeqa.sh` | livemath + officeqa | 16 + 16 | 16 + 16 | parallel pair |
| `05_spreadsheet_officeqa.sh` | spreadsheetbench + officeqa | 16 + 16 | 16 + 16 | parallel pair |
| `06_spreadsheet_docvqa.sh` | spreadsheetbench + docvqa | 16 + 16 | 16 + 16 | parallel pair |

Because there is no local engine to protect, multiple launchers can run
concurrently as long as the combined worker count stays within your DeepSeek API
concurrency limit. The `03`–`06` launchers already run two datasets in parallel
per launcher (32 target workers total each).

## Launch

Set a single DeepSeek key (either `DEEPSEEK_API_KEY`, `DS_API_KEY`, or
`DEEPSEEK_OFFICIAL_API_KEY`) and run any launcher:

```bash
export DEEPSEEK_API_KEY='...'
bash scripts/runs/resource_pool_dsv4_flash_pro_api/01_searchqa.sh
```

ALFWorld (`02`) needs the environment package and simulator data installed on
the run machine. The launcher self-checks and, with `ALFWORLD_AUTO_INSTALL=1`
(default), installs the missing Python deps automatically before launch:

```bash
export DEEPSEEK_API_KEY='...'
bash scripts/runs/resource_pool_dsv4_flash_pro_api/02_alfworld.sh
```

If you would rather set the environment up manually and only self-check, set
`ALFWORLD_AUTO_INSTALL=0`:

```bash
cd /ai-app-vepfs/zhangfuhao/skill/SkillOpt-Tree
python -m pip install -e ".[alfworld]"
python -m pip install "alfworld[full]"
ALFWORLD_DATA=/ai-app-vepfs/zhangfuhao/skill/SkillOpt-Tree/data/alfworld alfworld-download
```

The paired launchers run two datasets in parallel and wait for both:

```bash
export DEEPSEEK_API_KEY='...'
bash scripts/runs/resource_pool_dsv4_flash_pro_api/03_docvqa_spreadsheet.sh
bash scripts/runs/resource_pool_dsv4_flash_pro_api/04_livemath_officeqa.sh
bash scripts/runs/resource_pool_dsv4_flash_pro_api/05_spreadsheet_officeqa.sh
bash scripts/runs/resource_pool_dsv4_flash_pro_api/06_spreadsheet_docvqa.sh
```

Each launcher scans generated `results.jsonl` files after a dataset finishes; if
enough records exist and the run is all-zero with API/timeout failure
signatures, the launcher fails instead of silently producing a bogus experiment.
Set `RESULT_GUARD=0` only for intentional diagnostic runs.

## Dry-run and common overrides

Dry-run prints the resolved training commands without requiring an API key:

```bash
DRY_RUN=1 bash scripts/runs/resource_pool_dsv4_flash_pro_api/01_searchqa.sh
```

OfficeQA tool calls are handled entirely on the API/rollout side (there is no
vLLM tool-call parser to configure here), so no extra tool-choice flags are
needed.

Common overrides:

- `OPTIMIZER_MODEL` (default `deepseek-v4-pro`), `TARGET_MODEL` (default
  `deepseek-v4-flash`);
- `DEEPSEEK_BASE_URL` to point both models at a different endpoint;
- `TARGET_AZURE_OPENAI_ENDPOINT` / `TARGET_AZURE_OPENAI_API_KEY` to send the
  student to a *different* endpoint/key than the optimizer;
- `TARGET_MAX_COMPLETION_TOKENS` (default `16384`);
- `DEEPSEEK_THINKING` (default `disabled`);
- `GROUP_WORKERS`, `GROUP_ANALYST_WORKERS`, `GROUP_EXEC_TIMEOUT`,
  `SEARCHQA_WORKERS`, `SEARCHQA_EXEC_TIMEOUT`, `ALFWORLD_WORKERS`,
  `ALFWORLD_API_WORKERS`, `ALFWORLD_MAX_STEPS`, `OUT_BASE`.

Additional CLI arguments are appended to every dataset command, for example:

```bash
bash scripts/runs/resource_pool_dsv4_flash_pro_api/01_searchqa.sh --num_epochs 1
```

## LiveMath Volcano Ark dynamic-tree suite

Scripts `15`–`19` use Volcano Ark by default rather than the DeepSeek official
endpoint used by scripts `01`–`14`:

- target: `deepseek-v4-flash-260425`;
- optimizer: `deepseek-v4-pro-260425`;
- target workers: 48;
- analyst workers: 48;
- runs inside a sweep are sequential.

```bash
export ARK_API_KEY='...'

# Dataset system-prompt only; no init skill and no training.
bash scripts/runs/resource_pool_dsv4_flash_pro_api/15_livemath_ark_system_prompt_only.sh

# Dataset system prompt + repository init_skill; no training.
bash scripts/runs/resource_pool_dsv4_flash_pro_api/16_livemath_ark_init_skill_no_train.sh

# Dynamic recursive tree main configuration.
bash scripts/runs/resource_pool_dsv4_flash_pro_api/17_livemath_ark_dynamic_main.sh

# Core dynamic-tree ablations.
bash scripts/runs/resource_pool_dsv4_flash_pro_api/18_livemath_ark_dynamic_ablation.sh

# Both baselines + ablations together.
bash scripts/runs/resource_pool_dsv4_flash_pro_api/19_livemath_ark_all.sh
```

The default ablation rows are `dynamic_auto`, `fixed_real_root`,
`dynamic_real_root`, `dynamic_virtual_root`, `no_recursive_fallback`, and
`min_support_2`. Add the optional fan-out, maximum-depth, and validation-budget
rows with `DO_EXTENDED=1`, or select rows explicitly with a space-separated
`ABLATION_ROWS` value.

All scripts support `DRY_RUN=1`. Override `ARK_TARGET_MODEL`,
`ARK_OPTIMIZER_MODEL`, `LIVEMATH_ARK_WORKERS`, or
`LIVEMATH_ARK_ANALYST_WORKERS` when needed.

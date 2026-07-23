# SkillOpt-Tree comparison with Qwen3.6-35B-A3B on 4xL20

These launchers mirror the dataset combinations in
`../../../SkillOpt-main/scripts/runs/resource_pool_qwen36_35b_a3b_4xl20`, but
run against the SkillOpt-Tree training pipeline. They switch the target/student
model to a local `Qwen3.6-35B-A3B` vLLM service. SkillOpt method parameters
remain inherited from each dataset's `configs/<dataset>/default.yaml`.

The target model assignment is the same for every dataset:

- target: `Qwen3.6-35B-A3B` through local vLLM;
- target temperature: `0.2`;
- target thinking disabled through the Qwen chat client.

Optimizer source is script-specific:

- `01_searchqa_ark.sh`, `02_alfworld_ark.sh`, `05_spreadsheet_officeqa_ark.sh`,
  `06_spreadsheet_docvqa_ark.sh`: Volcano Ark
  `deepseek-v4-pro-260425`.
- `03_docvqa_spreadsheet_ark.sh`, `04_livemath_officeqa_ark.sh`: DeepSeek
  official `deepseek-v4-pro` at `https://api.deepseek.com`.

The scripts reuse materialized data from this repo's own `data/` directory by
default. Override `DATA_PROJECT_ROOT` or `DATA_ROOT` if the data is mounted
elsewhere.

## SkillOpt-Tree CLI note

The only meaningful difference from the SkillOpt-main reference is the training
CLI contract. SkillOpt-Tree's `scripts/train.py` (via `scripts/cli/train.py`)
does **not** define `--exec_timeout` / `--llm_timeout` flags — passing them to
`train.py` directly would fail argparse. Instead it accepts `--cfg-options` and
exposes `env.exec_timeout` / `env.llm_timeout` in the structured config. The
shared `_common.sh` therefore intercepts the reference-style `--exec_timeout N`
/ `--llm_timeout N` arguments in `run_dataset` and rewrites them into
`--cfg-options env.exec_timeout=N env.llm_timeout=N` (appended last, since
`--cfg-options` is `nargs="+"`). The per-dataset launchers stay identical to the
SkillOpt-main reference, so you can pass `SEARCHQA_EXEC_TIMEOUT`,
`GROUP_EXEC_TIMEOUT`, etc. exactly as before.

## Resource profile

This is not a one-script-per-L20 schedule. By default each launcher starts one
shared vLLM engine using four L20 cards:

- `MODEL_PATH=/ai-app-vepfs/models/Qwen3.6-35B-A3B`
- `SERVED_MODEL_NAME=Qwen3.6-35B-A3B`
- `CUDA_VISIBLE_DEVICES=0,1,2,3`
- `VLLM_TENSOR_PARALLEL_SIZE=4`
- `MAX_MODEL_LEN=32768`
- `VLLM_MAX_NUM_SEQS=32`
- `VLLM_MAX_NUM_BATCHED_TOKENS=32768`
- `GPU_MEMORY_UTILIZATION=0.85`

Concurrency by launcher:

| Script | Datasets | Optimizer source | Target workers | Analyst workers | Current note |
|---|---|---|---:|---:|---|
| `01_searchqa_ark.sh` | searchqa | Ark | config default, currently 24 | config default, currently 16 | usable |
| `02_alfworld_ark.sh` | alfworld | Ark | 32 env workers, 8 API workers | config default, currently 16 | usable |
| `03_docvqa_spreadsheet_ark.sh` | docvqa + spreadsheetbench | DeepSeek official | 16 + 16 = 32 | 16 + 16 | preferred paired run |
| `04_livemath_officeqa_ark.sh` | livemath + officeqa | DeepSeek official | 16 + 16 = 32 | 16 + 16 | preferred paired run |
| `05_spreadsheet_officeqa_ark.sh` | spreadsheetbench + officeqa | Ark | 16 + 16 = 32 | 16 + 16 | hold for now |
| `06_spreadsheet_docvqa_ark.sh` | spreadsheetbench + docvqa | Ark | 16 + 16 = 32 | 16 + 16 | hold for now |

Run at most one launcher per four-L20 allocation. Every launcher defaults to
one `TP=4` vLLM engine on `CUDA_VISIBLE_DEVICES=0,1,2,3`. Starting two
launchers on the same four cards starts two 35B-A3B engines and should be
treated as invalid. Reusing one endpoint for multiple launchers is also not
recommended unless you manually reduce total workers across all launchers to
`<= VLLM_MAX_NUM_SEQS` and accept higher optimizer-side API pressure.

## Launch

Run one launcher at a time on a four-L20 allocation:

ALFWorld (`02`) needs the environment package and simulator data installed on
the run machine before launch. The launcher now checks this before starting
vLLM:

```bash
cd /ai-app-vepfs/zhangfuhao/skill/SkillOpt-Tree
python -m pip install -e ".[alfworld]"
python -m pip install "alfworld[full]"
ALFWORLD_DATA=/ai-app-vepfs/zhangfuhao/skill/SkillOpt-Tree/data/alfworld alfworld-download
```

```bash
export ARK_API_KEY='...'
bash scripts/runs/resource_pool_qwen36_35b_a3b_4xl20/01_searchqa_ark.sh
```

```bash
export ARK_API_KEY='...'
bash scripts/runs/resource_pool_qwen36_35b_a3b_4xl20/02_alfworld_ark.sh
```

```bash
export DEEPSEEK_API_KEY='...'
bash scripts/runs/resource_pool_qwen36_35b_a3b_4xl20/03_docvqa_spreadsheet_ark.sh
```

```bash
export DEEPSEEK_API_KEY='...'
bash scripts/runs/resource_pool_qwen36_35b_a3b_4xl20/04_livemath_officeqa_ark.sh
```

The `05` and `06` launchers are kept for completeness but are not part of the
current run plan.

```bash
export ARK_API_KEY='...'
bash scripts/runs/resource_pool_qwen36_35b_a3b_4xl20/05_spreadsheet_officeqa_ark.sh
```

```bash
export ARK_API_KEY='...'
bash scripts/runs/resource_pool_qwen36_35b_a3b_4xl20/06_spreadsheet_docvqa_ark.sh
```

The shared launcher monitors the vLLM process: if the engine exits, dataset
jobs are stopped and the launcher fails instead of continuing to write
zero-score API-failure records. It also scans generated `results.jsonl` files
after each dataset finishes; if enough records exist and the run is all-zero
with API/timeout/tool-parser failure signatures, the launcher fails. Set
`RESULT_GUARD=0` only for intentional diagnostic runs.

## Dry-run and endpoint reuse

Dry-run prints the resolved training commands without requiring an API key,
GPU, or vLLM installation:

```bash
DRY_RUN=1 bash scripts/runs/resource_pool_qwen36_35b_a3b_4xl20/01_searchqa_ark.sh
```

To use an already-running endpoint:

```bash
START_VLLM=0 \
LOCAL_BASE_URL=http://127.0.0.1:8010/v1 \
bash scripts/runs/resource_pool_qwen36_35b_a3b_4xl20/01_searchqa_ark.sh
```

Common overrides include `MODEL_PATH`, `CUDA_VISIBLE_DEVICES`,
`VLLM_TENSOR_PARALLEL_SIZE`, `VLLM_PORT`, `OUT_BASE`,
`TARGET_QWEN_CHAT_TEMPERATURE`, `OPTIMIZER_SOURCE`, and `DEEPSEEK_THINKING`.
Additional CLI arguments are appended to every dataset command, for example:

```bash
bash scripts/runs/resource_pool_qwen36_35b_a3b_4xl20/01_searchqa_ark.sh --num_epochs 1
```

For OfficeQA the launcher enables vLLM auto tool choice with the `qwen3_coder`
parser, matching Qwen3.x XML-style tool calls
(`<tool_call><function=name><parameter=key>value</parameter></function></tool_call>`).
If an externally managed endpoint is reused, start that endpoint with
`--enable-auto-tool-choice --tool-call-parser qwen3_coder` (or `qwen3_xml` on
builds that only register that alias). SpreadsheetBench runs in codegen mode
and does not use tool calls, so it is unaffected by the parser choice.

Target request timeout is 1800 seconds and thinking remains explicitly
disabled. Override `GROUP_WORKERS`, `GROUP_ANALYST_WORKERS`, or
`GROUP_EXEC_TIMEOUT` when needed.

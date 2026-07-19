# SearchQA Prompt Latency Test

This isolated experiment compares SearchQA rollout prompt variants and local
Qwen serving concurrency without changing the main SkillOpt code path.

If the local Qwen vLLM endpoint is already running, call the Python probe
directly:

```bash
python experiments/searchqa_prompt_latency/run_searchqa_prompt_latency.py \
  --base-url http://127.0.0.1:8000/v1 \
  --model Qwen/Qwen3.5-4B \
  --split data/searchqa_split/val/items.json \
  --sample-size 64 \
  --workers 48 96 128 \
  --max-tokens 16384 \
  --out-dir outputs/searchqa_prompt_latency
```

If the endpoint is not running, use the wrapper below. It starts vLLM first,
waits for `/v1/models`, then runs the same probe:

```bash
bash experiments/searchqa_prompt_latency/run_with_vllm.sh
```

Useful overrides:

```bash
MODEL_PATH=/ai-app-vepfs/modelscope/models/Qwen/Qwen3.5-4B \
QWEN_CUDA_VISIBLE_DEVICES=0,1,2,3 \
VLLM_TENSOR_PARALLEL_SIZE=4 \
SAMPLE_SIZE=256 \
WORKERS_LIST="48 96 128" \
bash experiments/searchqa_prompt_latency/run_with_vllm.sh
```

The wrapper waits up to `VLLM_STARTUP_TIMEOUT_SECONDS=900` by default because
the first vLLM start can spend several minutes in torch.compile before
`/v1/models` is available. Increase it if the server is still compiling:

```bash
VLLM_STARTUP_TIMEOUT_SECONDS=1200 bash experiments/searchqa_prompt_latency/run_with_vllm.sh
```

The script writes one JSONL file per prompt/concurrency setting and a combined
`summary.json`. Use a small `--sample-size` first, then increase it if the
endpoint remains healthy.

Full SearchQA test split with exact EM/F1 and full responses:

```bash
SPLIT_PATH=data/searchqa_split/test/items.json \
SAMPLE_SIZE=0 \
WORKERS_LIST="96 128 192 256" \
MAX_TOKENS=16384 \
bash experiments/searchqa_prompt_latency/run_with_vllm.sh
```

Summarize a finished run:

```bash
python experiments/searchqa_prompt_latency/summarize_prompt_latency.py \
  outputs/searchqa_prompt_latency_YYYYMMDD_HHMMSS
```

Simulate several datasets/jobs sharing the same vLLM endpoint:

```bash
JOBS=3 \
WORKERS_PER_JOB=128 \
SPLIT_PATH=data/searchqa_split/test/items.json \
SAMPLE_SIZE=0 \
PROMPT_LIST="baseline_current direct_with_evidence_check" \
bash experiments/searchqa_prompt_latency/run_parallel_load_with_vllm.sh
```

Unified full stress suite. It starts or reuses vLLM once, runs the full 1400
SearchQA test split prompt comparison, writes a markdown summary, then runs the
parallel load simulation and summarizes every simulated job:

```bash
bash experiments/searchqa_prompt_latency/run_full_prompt_stress_suite.sh
```

Useful suite overrides:

```bash
FULL_WORKERS_LIST="96 128 192 256" \
FULL_PROMPT_LIST="baseline_current direct_when_identified direct_with_evidence_check" \
PARALLEL_JOBS=3 \
PARALLEL_WORKERS_PER_JOB=128 \
PARALLEL_PROMPT_LIST="baseline_current direct_with_evidence_check" \
SPLIT_PATH=data/searchqa_split/test/items.json \
SAMPLE_SIZE=0 \
MAX_TOKENS=16384 \
bash experiments/searchqa_prompt_latency/run_full_prompt_stress_suite.sh
```

Skip either phase if needed:

```bash
RUN_PARALLEL_LOAD=0 bash experiments/searchqa_prompt_latency/run_full_prompt_stress_suite.sh
RUN_FULL_TEST=0 bash experiments/searchqa_prompt_latency/run_full_prompt_stress_suite.sh
```

Baseline-only concurrency matrix. This tests the practical choice of simulated
dataset/job count `2/3` and workers per job `96/128` with the `baseline_current`
prompt:

```bash
bash experiments/searchqa_prompt_latency/run_baseline_concurrency_matrix.sh
```

Useful overrides:

```bash
JOB_COUNTS="2 3" \
WORKERS_LIST="96 128" \
SPLIT_PATH=data/searchqa_split/test/items.json \
SAMPLE_SIZE=0 \
MAX_TOKENS=16384 \
bash experiments/searchqa_prompt_latency/run_baseline_concurrency_matrix.sh
```

The main result is written to:

```bash
outputs/searchqa_baseline_concurrency_matrix_<timestamp>/matrix_summary.md
```

ALFWorld speed probe with local Qwen target and workers=128:

```bash
bash experiments/searchqa_prompt_latency/run_alfworld_qwen_speed_workers128.sh
```

By default it runs the ALFWorld `test` split with:

```bash
ALFWORLD_WORKERS=128
ALFWORLD_MAX_API_WORKERS=128
ALFWORLD_TARGET_MAX_COMPLETION_TOKENS=2048
```

For a quick speed smoke, truncate the split:

```bash
ALFWORLD_ENV_NUM=16 \
bash experiments/searchqa_prompt_latency/run_alfworld_qwen_speed_workers128.sh
```

The summary is written to:

```bash
outputs/alfworld_qwen_speed_workers128_<timestamp>/speed_summary.md
```

If you see `Connection refused`, the test process cannot reach vLLM at
`--base-url`. Start vLLM on the same machine, run this script on the vLLM
machine, or point `--base-url` at a reachable endpoint.

One possible local server command:

```bash
mkdir -p logs/searchqa_prompt_latency
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup vllm serve /ai-app-vepfs/modelscope/models/Qwen/Qwen3.5-4B \
  --served-model-name Qwen/Qwen3.5-4B \
  --host 0.0.0.0 \
  --port 8000 \
  --trust-remote-code \
  --dtype bfloat16 \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 32768 \
  --enable-prefix-caching \
  > logs/searchqa_prompt_latency/vllm_qwen.log 2>&1 &
```

After the server starts, check:

```bash
curl http://127.0.0.1:8000/v1/models
```

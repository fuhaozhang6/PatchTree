# Scripts layout

The executable scripts are grouped by purpose:

- `cli/`: canonical Python entry points for training and evaluation.
- `runs/`: experiment launchers, grouped by dataset; cross-dataset launchers
  live under `runs/multi/`, and post-training evaluation launchers under
  `runs/evaluation/`.
- `tools/`: offline data preparation and artifact analysis utilities.
- `benchmarks/`: standalone throughput and concurrency benchmarks.
- `setup/`: environment and dependency installation helpers.

The root-level `train.py` and `eval_only.py` files are compatibility shims for
older commands. New scripts and documentation should use `scripts/cli/train.py`
and `scripts/cli/eval_only.py` directly.

## Throughput benchmark

Use `benchmarks/benchmark_v4pro_and_l20.sh` to sweep concurrency for Volcano
Ark DeepSeek V4 Pro, the official DeepSeek V4 Pro endpoint, and a local
Qwen3.5-4B vLLM server on one L20. It writes one report per backend plus a
best-stable-point comparison under `outputs/`.

```bash
export ARK_API_KEY='...'
export DEEPSEEK_API_KEY='...'
bash scripts/benchmarks/benchmark_v4pro_and_l20.sh
```

Use `TARGETS=ark`, `TARGETS=deepseek`, or `TARGETS=local` for an individual
backend. Before a paid full sweep, a small smoke test is recommended:

```bash
TARGETS='ark deepseek' CONCURRENCY_LEVELS='1 2' \
  MIN_REQUESTS=2 MAX_TOKEN_OPTIONS='64 128' \
  bash scripts/benchmarks/benchmark_v4pro_and_l20.sh
```

By default, individual requests rotate through approximate input sizes
`256 512 1024 2048` and output limits `256 512 1024`. Override them with
`PROMPT_TOKEN_OPTIONS` and `MAX_TOKEN_OPTIONS`. Set each variable to one value
when a fixed-shape benchmark is needed.

## Resource-pool four-L20 launchers

The `runs/resource_pool_4x_l20/` directory contains four independent,
single-GPU launchers for workers allocated on different hosts. SearchQA and
ALFWorld use the official DeepSeek endpoint; the two paired fast-dataset jobs
use Volcano Ark. See that directory's `README.md` for the per-worker commands.

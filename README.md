# PatchTree

PatchTree optimizes an agent Skill as a hierarchy of evidence-grounded edits.
The current training path is intentionally single-method: repeated rollouts are
converted into compact `PatchRecord`s, semantically grouped into leaf nodes,
optionally merged through mid nodes, compiled at the root, ranked, applied, and
accepted only through the validation gate.

```text
rollout repeats
  -> dataset-specific PatchRecord generation
  -> semantic clustering
  -> parallel leaf merge
  -> optional parallel mid merge
  -> root merge
  -> rank / apply
  -> validation gate and child fallback
```

The former generic reflection/aggregation path, rewrite mode, skill-aware
reflection, support/self-check, slow update, and meta-skill path have been
removed from the active package.

## Install

```bash
pip install -e ".[dev,qwen]"
```

Dataset-specific dependencies and data paths are documented under `docs/` and
`configs/`.

## Train one dataset

```bash
python -u scripts/cli/train.py \
  --config configs/searchqa/default.yaml \
  --target_backend qwen_chat \
  --target_qwen_chat_base_url http://127.0.0.1:8000/v1 \
  --type_guided_leaf_merge_workers 4 \
  --type_guided_mid_merge_workers 4
```

`scripts/cli/train.py --help` is the authoritative CLI reference.

## DeepSeek optimizer + shared local Qwen/vLLM

The parallel launcher starts one shared vLLM endpoint and runs several dataset
jobs against it. Optimizer calls use the configured DeepSeek OpenAI-compatible
endpoint; target rollouts use local Qwen.

```bash
DATASETS="searchqa docvqa livemath" \
MAX_PARALLEL=3 \
TYPE_GUIDED_LEAF_MERGE_WORKERS=8 \
TYPE_GUIDED_MID_MERGE_WORKERS=4 \
bash scripts/runs/multi/run_v3_deepseek_local_qwen_parallel.sh
```

Useful throughput controls include `VLLM_MAX_NUM_SEQS`,
`VLLM_MAX_NUM_BATCHED_TOKENS`, `VLLM_ENABLE_CHUNKED_PREFILL`, dataset-level
`WORKERS`, PatchRecord workers, and leaf/mid merge workers. Run with
`DRY_RUN=1 START_VLLM=0` to inspect commands without training.

## Current data contract

The compact PatchRecord and `shared_core + conditional_residuals` node contract
are described in [docs/type_guided_v23_light.md](docs/type_guided_v23_light.md).
The complete code-grounded training flow is documented in
[docs/PatchTree-v4.md](docs/PatchTree-v4.md).

## Verify

```bash
python -m compileall -q -x '(^|/)\._' skillopt scripts tests
pytest -q
find scripts -type f -name '*.sh' -not -name '._*' -print0 \
  | xargs -0 -n1 bash -n
```

The Python import namespace remains `skillopt` for compatibility; the training
method implemented by that package is PatchTree.

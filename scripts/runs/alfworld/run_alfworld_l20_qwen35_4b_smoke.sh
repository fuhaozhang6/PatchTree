#!/usr/bin/env bash
set -euo pipefail

# ALFWorld Smoke Test Launcher for Single L20:
#   1. Installs ALFWorld deps only (vLLM assumed present; INSTALL_VLLM=1 to add it)
#   2. Runs 1 epoch with 1 sample (shortest possible run)
#   3. Uses one L20 GPU (TP=1)
#
# ALFWorld game-file scan progress bars are suppressed downstream via TQDM_DISABLE
# (set in run_alfworld_seed_api_smoke.sh). Export TQDM_DISABLE=0 to show them.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHON_BIN="${PYTHON_BIN:-python3}"
export DRY_RUN="${DRY_RUN:-0}"

# 1. Install Dependencies
# vLLM is assumed to be already installed in this environment, so we only install
# the ALFWorld deps by default (INSTALL_VLLM=0). The setup script always installs
# `.[alfworld]` + omegaconf/json_repair regardless of INSTALL_VLLM; setting it to 0
# just skips (re)installing vLLM. Override with INSTALL_VLLM=1 to install vLLM too.
if [[ "${DRY_RUN}" == "0" ]]; then
  echo ">>> Step 1: Installing ALFWorld dependencies (skipping vLLM)..."
  export INSTALL_VLLM="${INSTALL_VLLM:-0}"
  bash scripts/setup/setup_alfworld_qwen_vllm_deps.sh
else
  echo "[dry-run] Skipping dependency installation."
fi

# 2. Configure Environment for Smoke Test
echo ">>> Step 2: Configuring environment for L20 smoke test..."
export ALFWORLD_DATA="${PROJECT_ROOT}/data/alfworld"
export ALFWORLD_SPLIT_DIR="${PROJECT_ROOT}/data/alfworld_path_split"

# Single L20 Configuration
export QWEN_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export VLLM_TENSOR_PARALLEL_SIZE=1
export VLLM_PORT=49317
export VLLM_MAX_NUM_SEQS=128

# Shortest Smoke Test Parameters
export NUM_EPOCHS=1
export LIMIT=1           # Only 1 sample from train set
export SEL_ENV_NUM=1      # Only 1 sample for gate evaluation
export TEST_ENV_NUM=0     # Skip test evaluation
export EVAL_TEST=false
export BATCH_SIZE=1       # Minimal batch for smoke test
export MINIBATCH_SIZE=1   # Avoid unbound variable error in underlying scripts
export MERGE_BATCH_SIZE=1 # Avoid unbound variable error in underlying scripts
export WORKERS=32         # Reduced workers for smoke test stability
export ANALYST_WORKERS=8

# Optimizer (DeepSeek via Ark or Official)
export DEEPSEEK_THINKING="${DEEPSEEK_THINKING:-disabled}"
# The optimizer connectivity smoke uses a tiny max_tokens (32). A reasoning model
# with reasoning_effort=high burns that budget before emitting any content, so the
# check returns finish_reason=length with empty content and fails. Keep the smoke
# probe as a fast, non-thinking connectivity check. This does NOT affect the real
# training reasoning_effort (REASONING_EFFORT), only the launch-time probe.
export OPTIMIZER_SMOKE_REASONING_EFFORT="${OPTIMIZER_SMOKE_REASONING_EFFORT:-}"

echo "============================================================"
echo "  ALFWorld L20 Smoke Test"
echo "============================================================"
echo "  GPU ID:             ${QWEN_CUDA_VISIBLE_DEVICES}"
echo "  Data Path:          ${ALFWORLD_DATA}"
echo "  Epochs:             ${NUM_EPOCHS}"
echo "  Limit:              ${LIMIT}"
echo "  Batch Size:         ${BATCH_SIZE}"
echo "  Workers:            ${WORKERS}"
echo "  Python:             ${PYTHON_BIN}"
echo "============================================================"

# 3. Launch the main training script
bash "${PROJECT_ROOT}/scripts/runs/alfworld/run_alfworld_qwen35_4b_2xh20_dsv4pro_train.sh" "$@"

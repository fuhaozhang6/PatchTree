#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PAIR_NAME=b4_b32
export EXP1_NAME=r3_b4_d2 EXP1_BATCH_SIZE=4 EXP1_ROLLOUT_REPEATS=3 EXP1_TREE_DEPTH=2
export EXP2_NAME=r3_b32_d2 EXP2_BATCH_SIZE=32 EXP2_ROLLOUT_REPEATS=3 EXP2_TREE_DEPTH=2
exec bash "${SCRIPT_DIR}/_run_pair.sh" "$@"

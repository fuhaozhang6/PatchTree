#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PAIR_NAME=b16_d3
export EXP1_NAME=r3_b16_d2 EXP1_BATCH_SIZE=16 EXP1_ROLLOUT_REPEATS=3 EXP1_TREE_DEPTH=2
export EXP2_NAME=r3_b8_d3 EXP2_BATCH_SIZE=8 EXP2_ROLLOUT_REPEATS=3 EXP2_TREE_DEPTH=3
exec bash "${SCRIPT_DIR}/_run_pair.sh" "$@"

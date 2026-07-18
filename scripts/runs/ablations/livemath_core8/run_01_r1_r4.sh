#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PAIR_NAME=r1_r4
export EXP1_NAME=r1_b8_d2 EXP1_BATCH_SIZE=8 EXP1_ROLLOUT_REPEATS=1 EXP1_TREE_DEPTH=2
export EXP2_NAME=r4_b8_d2 EXP2_BATCH_SIZE=8 EXP2_ROLLOUT_REPEATS=4 EXP2_TREE_DEPTH=2
exec bash "${SCRIPT_DIR}/_run_pair.sh" "$@"

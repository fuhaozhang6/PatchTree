#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PAIR_NAME=r2_base
export EXP1_NAME=r2_b8_d2 EXP1_BATCH_SIZE=8 EXP1_ROLLOUT_REPEATS=2 EXP1_TREE_DEPTH=2
export EXP2_NAME=base_r3_b8_d2 EXP2_BATCH_SIZE=8 EXP2_ROLLOUT_REPEATS=3 EXP2_TREE_DEPTH=2
exec bash "${SCRIPT_DIR}/_run_pair.sh" "$@"

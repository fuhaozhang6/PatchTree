#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# GPU 2: 8*8 + 4*16 = 128 simultaneous target requests at rollout peak.
export ABLATION_PAIR_GROUP=livemath_followup4_pairs
export ABLATION_SUITE_SLUG=livemath_followup4
export ABLATION_SUITE_LABEL="LiveMath follow-up"
export PAIR_NAME=r8_r4b16d2
export EXP1_NAME=r8_b8_d3 EXP1_BATCH_SIZE=8 EXP1_ROLLOUT_REPEATS=8 EXP1_TREE_DEPTH=3
export EXP2_NAME=r4_b16_d2 EXP2_BATCH_SIZE=16 EXP2_ROLLOUT_REPEATS=4 EXP2_TREE_DEPTH=2

exec bash "${SCRIPT_DIR}/../livemath_core8/_run_pair.sh" "$@"
